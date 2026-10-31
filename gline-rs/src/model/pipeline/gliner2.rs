use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use composable::Composable;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::RelationInput;
use crate::model::input::schema::{ExtractionFieldSchema, ExtractionSchema, SpecialTokens};
use crate::model::input::tensors::schema::{
    schema_input_names, ExtractionInput, PreparedSequence, SequenceInput, SequenceTask,
    ATTENTION_MASK, INPUT_IDS, SCHEMA_POSITIONS, SPAN_IDX, TEXT_POSITIONS,
};
use crate::model::input::text::TextInput;
use crate::model::output::classification::{
    ClassificationContext, ClassificationOutput, OutputsToClassification,
};
use crate::model::output::decoded::span_scores::{OutputsToSpans, SequenceContext};
use crate::model::output::decoded::SpanOutput;
use crate::model::output::extraction::{ExtractionContext, ExtractionOutput, OutputsToExtraction};
use crate::model::output::relation::{RelationOutput, SpanOutputToRelationOutput};
use crate::model::pipeline::context::RelationContext;
use crate::model::pipeline::gliner2_multi::{
    GLiNER2MultiPipeline, GLiNER2MultiPipelineOutput, GLiNER2MultiPipelineSchema,
};
use crate::model::{resolve_onnx_path, validate_required_file};
use crate::text::splitter::RegexSplitter;
use crate::text::tokenizer::HFTokenizer;
use crate::util::extract::parse_field_spec;
use crate::util::json::{json_bool_opt, json_f32_opt, json_usize};
use crate::util::ort_session::open_session;
use crate::util::profiling::StageMillis;
use crate::util::result::Result;
use crate::util::time::timed;
use serde_json::Value;

/// Checkpoint settings from `config.json`.
#[derive(Debug, Clone)]
pub struct GLiNER2Config {
    pub max_width: usize,
    pub threshold: f32,
    pub flat_ner: bool,
    pub dup_label: bool,
    pub multi_label: bool,
}

impl GLiNER2Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let value: Value = serde_json::from_reader(File::open(path)?)?;
        Self::from_value(&value)
    }

    fn from_value(value: &Value) -> Result<Self> {
        let max_width = json_usize(value, "max_width")?;
        if max_width == 0 {
            return Err("invalid GLiNER2 config: max_width must be greater than zero".into());
        }
        Ok(Self {
            max_width,
            threshold: json_f32_opt(value, "threshold").unwrap_or(0.5),
            flat_ner: json_bool_opt(value, "flat_ner").unwrap_or(true),
            dup_label: json_bool_opt(value, "dup_label").unwrap_or(false),
            multi_label: json_bool_opt(value, "multi_label").unwrap_or(false),
        })
    }
}

pub struct GLiNER2 {
    config: GLiNER2Config,
    model: Model,
    ner_pipeline: GLiNER2NerPipeline,
    classification_pipeline: GLiNER2ClassificationPipeline,
    extraction_pipeline: GLiNER2ExtractionPipeline,
}

impl GLiNER2 {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        runtime_parameters: RuntimeParameters,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_model_path = resolve_onnx_path(model_dir);
        let config_path = model_dir.join("config.json");

        validate_required_file("tokenizer", &tokenizer_path)?;
        validate_required_file("ONNX model", &onnx_model_path)?;
        validate_required_file("config", &config_path)?;

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        let special_tokens = SpecialTokens::resolve(&tokenizer)?;

        Ok(Self {
            config: GLiNER2Config::from_file(&config_path)?,
            model: Model::new(onnx_model_path, runtime_parameters)?,
            ner_pipeline: GLiNER2NerPipeline::new(tokenizer.clone(), special_tokens.clone()),
            classification_pipeline: GLiNER2ClassificationPipeline::new(
                tokenizer.clone(),
                special_tokens.clone(),
            ),
            extraction_pipeline: GLiNER2ExtractionPipeline::new(tokenizer, special_tokens),
        })
    }

    pub fn get_inner_model(&self) -> &Model {
        &self.model
    }

    pub fn inference(&self, input: TextInput) -> Result<SpanOutput> {
        let TextInput { texts, entities } = input;
        let mut spans = Vec::with_capacity(texts.len());

        for (sequence_index, text) in texts.iter().enumerate() {
            if text.trim().is_empty() {
                spans.push(Vec::new());
                continue;
            }

            let output = self.model.inference(
                SequenceInput {
                    sequence_index,
                    text: text.clone(),
                    labels: entities.clone(),
                    task: SequenceTask::Entities,
                },
                &self.ner_pipeline,
                &self.config,
            )?;

            spans.push(output.spans.into_iter().next().unwrap_or_default());
        }

        Ok(SpanOutput::new(texts, entities, spans))
    }

    /// Runs schema-driven GLiNER2 classification using the monolithic `span_scores` export.
    ///
    /// The current ONNX contract does not expose a dedicated classification head, so this
    /// method scores each candidate label with the best span score returned for that label.
    pub fn classify(&self, text: &str, labels: &[String]) -> Result<ClassificationOutput> {
        if text.trim().is_empty() {
            return Err("invalid input: text contains no tokenizable words".into());
        }
        if labels.is_empty() {
            return Err("invalid input: labels cannot be empty".into());
        }

        self.model.inference(
            SequenceInput {
                sequence_index: 0,
                text: text.to_string(),
                labels: labels.to_vec(),
                task: SequenceTask::Classification,
            },
            &self.classification_pipeline,
            &self.config,
        )
    }

    /// Runs schema-driven extraction using the monolithic `span_scores` export.
    ///
    /// The current ONNX contract exposes span scores only, so extraction is decoded as
    /// thresholded spans grouped by schema fields.
    pub fn extract(&self, text: &str, schema: &ExtractionSchema) -> Result<ExtractionOutput> {
        if text.trim().is_empty() {
            return Err("invalid input: text contains no tokenizable words".into());
        }

        let flattened = schema.flatten_labels()?;

        self.model.inference(
            ExtractionInput {
                sequence_index: 0,
                text: text.to_string(),
                flattened_schema: flattened,
            },
            &self.extraction_pipeline,
            &self.config,
        )
    }

    pub fn extract_json(
        &self,
        text: &str,
        schema: &HashMap<String, Vec<String>>,
    ) -> Result<serde_json::Value> {
        let parsed_schema = parse_extract_json_schema(schema)?;
        let extraction_schema =
            ExtractionSchema::from_fields(parsed_schema.extraction_fields.clone());
        let output = self.extract(text, &extraction_schema)?;
        Ok(extraction_output_to_json(&parsed_schema, output))
    }

    pub fn extract_relations(
        &self,
        input: TextInput,
        schema: &RelationSchema,
    ) -> Result<RelationOutput> {
        let entity_spans = self.inference(input)?;
        let relation_input = RelationInput::from_spans(entity_spans, schema);
        let RelationInput {
            prompts,
            labels,
            entity_labels,
            entity_offsets,
        } = relation_input;

        let relation_spans = self.inference(TextInput::new(prompts, labels)?)?;

        SpanOutputToRelationOutput::new(schema).apply((
            relation_spans,
            RelationContext {
                entity_labels,
                entity_offsets,
            },
        ))
    }

    pub fn create_schema(&self) -> GLiNER2MultiPipelineSchema {
        GLiNER2MultiPipelineSchema::new()
    }

    pub fn extract_with_schema(
        &self,
        text: &str,
        schema: &GLiNER2MultiPipelineSchema,
    ) -> Result<GLiNER2MultiPipelineOutput> {
        GLiNER2MultiPipeline::new(self).extract(text, schema)
    }
}

#[derive(Clone)]
enum JsonFieldMode {
    Single,
    List,
}

#[derive(Clone)]
struct JsonFieldSpec {
    object_name: String,
    field_name: String,
    mode: JsonFieldMode,
}

struct JsonExtractSchema {
    extraction_fields: Vec<ExtractionFieldSchema>,
    fields: Vec<JsonFieldSpec>,
}

fn parse_extract_json_schema(schema: &HashMap<String, Vec<String>>) -> Result<JsonExtractSchema> {
    if schema.is_empty() {
        return Err("invalid JSON schema: must contain at least one object".into());
    }

    let mut extraction_fields = Vec::new();
    let mut fields = Vec::new();
    let mut seen_field_names = HashSet::new();

    for (object_name, specs) in schema {
        if object_name.trim().is_empty() {
            return Err("invalid JSON schema: object name cannot be empty".into());
        }
        if specs.is_empty() {
            return Err(format!(
                "invalid JSON schema: object `{object_name}` must contain at least one field spec"
            )
            .into());
        }

        for spec in specs {
            let (field_name, single) = parse_field_spec(spec)?;
            let mode = if single {
                JsonFieldMode::Single
            } else {
                JsonFieldMode::List
            };
            if !seen_field_names.insert(field_name.clone()) {
                return Err(format!(
                    "invalid JSON schema: duplicate field name `{field_name}` is not supported"
                )
                .into());
            }

            extraction_fields.push(ExtractionFieldSchema::new(
                field_name.clone(),
                vec![field_name.clone()],
            ));
            fields.push(JsonFieldSpec {
                object_name: object_name.clone(),
                field_name,
                mode,
            });
        }
    }

    Ok(JsonExtractSchema {
        extraction_fields,
        fields,
    })
}

fn extraction_output_to_json(
    schema: &JsonExtractSchema,
    output: ExtractionOutput,
) -> serde_json::Value {
    let field_values = output
        .fields
        .into_iter()
        .map(|field| {
            (
                field.name,
                field
                    .values
                    .into_iter()
                    .map(|value| value.text)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut objects: HashMap<String, serde_json::Map<String, serde_json::Value>> = HashMap::new();
    for spec in &schema.fields {
        let values = field_values
            .get(&spec.field_name)
            .cloned()
            .unwrap_or_default();

        let value = match spec.mode {
            JsonFieldMode::Single => values
                .into_iter()
                .next()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
            JsonFieldMode::List => serde_json::Value::Array(
                values
                    .into_iter()
                    .map(serde_json::Value::String)
                    .collect::<Vec<_>>(),
            ),
        };

        objects
            .entry(spec.object_name.clone())
            .or_default()
            .insert(spec.field_name.clone(), value);
    }

    let result = objects
        .into_iter()
        .map(|(object_name, object)| {
            (
                object_name,
                serde_json::Value::Array(vec![serde_json::Value::Object(object)]),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    serde_json::Value::Object(result)
}

pub struct Gliner2Probe {
    session: ort::session::Session,
    ner: GLiNER2NerPipeline,
    classification: GLiNER2ClassificationPipeline,
    extraction: GLiNER2ExtractionPipeline,
    config: GLiNER2Config,
}

impl Gliner2Probe {
    pub fn open(model_dir: &Path, runtime_parameters: RuntimeParameters) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = resolve_onnx_path(model_dir);
        let config_path = model_dir.join("config.json");
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        let special_tokens = SpecialTokens::resolve(&tokenizer)?;
        Ok(Self {
            session: open_session(&onnx_path, &runtime_parameters)?,
            ner: GLiNER2NerPipeline::new(tokenizer.clone(), special_tokens.clone()),
            classification: GLiNER2ClassificationPipeline::new(
                tokenizer.clone(),
                special_tokens.clone(),
            ),
            extraction: GLiNER2ExtractionPipeline::new(tokenizer, special_tokens),
            config: GLiNER2Config::from_file(&config_path)?,
        })
    }

    pub fn time_ner(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            self.ner.pre_processor(&self.config).apply(SequenceInput {
                sequence_index: 0,
                text: text.to_string(),
                labels: labels.to_vec(),
                task: SequenceTask::Entities,
            })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.ner
                .post_processor(&self.config)
                .apply((outputs, context))
        })?;
        let kept = decoded.spans.iter().map(|spans| spans.len()).sum::<usize>();
        std::hint::black_box(kept);
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }

    pub fn time_classify(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            self.classification
                .pre_processor(&self.config)
                .apply(SequenceInput {
                    sequence_index: 0,
                    text: text.to_string(),
                    labels: labels.to_vec(),
                    task: SequenceTask::Classification,
                })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.classification
                .post_processor(&self.config)
                .apply((outputs, context))
        })?;
        std::hint::black_box(decoded.scores.len());
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }

    /// One combined extraction call, the session `extract_with_schema` uses for entities, relations, and structures.
    pub fn time_extract(&self, text: &str, schema: &ExtractionSchema) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            let flattened = schema.flatten_labels()?;
            self.extraction
                .pre_processor(&self.config)
                .apply(ExtractionInput {
                    sequence_index: 0,
                    text: text.to_string(),
                    flattened_schema: flattened,
                })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.extraction
                .post_processor(&self.config)
                .apply((outputs, context))
        })?;
        let kept = decoded
            .fields
            .iter()
            .map(|field| field.values.len())
            .sum::<usize>();
        std::hint::black_box(kept);
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }
}

struct GLiNER2NerPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2NerPipeline {
    fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToSpans::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2NerPipeline {
    type Input = SequenceInput;
    type Output = SpanOutput;
    type Context = SequenceContext;
    type Parameters = GLiNER2Config;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        SequenceToNerTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToSpans::new(
            params.threshold,
            params.max_width,
            params.flat_ner,
            params.dup_label,
            params.multi_label,
        )
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

struct GLiNER2ClassificationPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2ClassificationPipeline {
    fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToClassification::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ClassificationPipeline {
    type Input = SequenceInput;
    type Output = ClassificationOutput;
    type Context = ClassificationContext;
    type Parameters = GLiNER2Config;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        SequenceToClassificationTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToClassification::new(params.max_width)
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

struct GLiNER2ExtractionPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2ExtractionPipeline {
    fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToExtraction::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ExtractionPipeline {
    type Input = ExtractionInput;
    type Output = ExtractionOutput;
    type Context = ExtractionContext;
    type Parameters = GLiNER2Config;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        ExtractionToTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToExtraction::new(
            params.threshold,
            params.max_width,
            params.flat_ner,
            params.dup_label,
            params.multi_label,
        )
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

struct SequenceToNerTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, SequenceContext)>
    for SequenceToNerTensors
{
    fn apply(&self, input: SequenceInput) -> Result<(SessionInputs<'a, 'a>, SequenceContext)> {
        let prepared = prepare(
            input,
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        Ok((
            session_inputs.into(),
            SequenceContext {
                sequence_index: prepared.sequence_index,
                text: prepared.text,
                tokens: prepared.tokens,
                labels: prepared.labels,
            },
        ))
    }
}

struct SequenceToClassificationTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, ClassificationContext)>
    for SequenceToClassificationTensors
{
    fn apply(
        &self,
        input: SequenceInput,
    ) -> Result<(SessionInputs<'a, 'a>, ClassificationContext)> {
        let prepared = prepare(
            input,
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        Ok((
            session_inputs.into(),
            ClassificationContext {
                text: prepared.text,
                num_words: prepared.tokens.len(),
                labels: prepared.labels,
            },
        ))
    }
}

struct ExtractionToTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<ExtractionInput, (SessionInputs<'a, 'a>, ExtractionContext)>
    for ExtractionToTensors
{
    fn apply(&self, input: ExtractionInput) -> Result<(SessionInputs<'a, 'a>, ExtractionContext)> {
        let labels = input.flattened_schema.labels.clone();
        let prepared = prepare(
            SequenceInput {
                sequence_index: input.sequence_index,
                text: input.text,
                labels,
                task: SequenceTask::Extraction,
            },
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        let label_to_field = prepared
            .labels
            .iter()
            .cloned()
            .zip(input.flattened_schema.label_to_field)
            .collect();

        Ok((
            session_inputs.into(),
            ExtractionContext {
                sequence: SequenceContext {
                    sequence_index: prepared.sequence_index,
                    text: prepared.text,
                    tokens: prepared.tokens,
                    labels: prepared.labels,
                },
                field_names: input.flattened_schema.field_names,
                label_to_field,
            },
        ))
    }
}

fn prepare(
    input: SequenceInput,
    splitter: &RegexSplitter,
    tokenizer: &HFTokenizer,
    special_tokens: &SpecialTokens,
    max_width: usize,
) -> Result<PreparedSequence> {
    crate::model::input::tensors::schema::prepare_sequence(
        input,
        splitter,
        tokenizer,
        special_tokens,
        max_width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_reads_max_width_from_json() {
        let config = GLiNER2Config::from_value(&serde_json::json!({
            "max_width": 8,
            "model_type": "extractor"
        }))
        .unwrap();
        assert_eq!(config.max_width, 8);
        assert_eq!(config.threshold, 0.5);
        assert!(config.flat_ner);
        assert!(!config.dup_label);
        assert!(!config.multi_label);
    }

    #[test]
    fn config_rejects_a_missing_max_width() {
        let err = GLiNER2Config::from_value(&serde_json::json!({
            "model_type": "extractor"
        }))
        .unwrap_err();
        assert!(err.to_string().contains("max_width"));
    }
}
