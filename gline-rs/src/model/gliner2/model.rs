use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use composable::*;
use ndarray::{Array1, Array2, Array3};
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::RelationInput;
use crate::model::input::text::TextInput;
use crate::model::output::decoded::SpanOutput;
use crate::model::output::relation::RelationOutput;
use crate::model::params::Parameters;
use crate::model::pipeline::context::RelationContext;
use crate::text::splitter::{RegexSplitter, Splitter};
use crate::text::token::Token;
use crate::util::result::Result;

use super::classification::{ClassificationContext, ClassificationOutput, OutputsToClassification};
use super::decoder::{OutputsToSpans, SequenceContext};
use super::extraction::{
    ExtractionContext, ExtractionFieldSchema, ExtractionOutput, ExtractionSchema,
    FlattenedExtractionSchema, OutputsToExtraction,
};
use super::pipeline::{GLiNER2Pipeline, GLiNER2PipelineOutput, GLiNER2PipelineSchema};
use super::relations::OutputsToRelations;
use super::schema::SchemaPrefix;
use super::spans::build_span_idx;
use super::tokenizer::GLiNER2Tokenizer;

const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
const TEXT_POSITIONS: &str = "text_positions";
const SCHEMA_POSITIONS: &str = "schema_positions";
const SPAN_IDX: &str = "span_idx";
const GLINER2_MAX_WIDTH: usize = 8;

pub struct GLiNER2 {
    params: Parameters,
    model: Model,
    ner_pipeline: GLiNER2NerPipeline,
    classification_pipeline: GLiNER2ClassificationPipeline,
    extraction_pipeline: GLiNER2ExtractionPipeline,
}

impl GLiNER2 {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        parameters: Parameters,
        runtime_parameters: RuntimeParameters,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_model_path = resolve_onnx_path(model_dir);

        validate_required_file("tokenizer", &tokenizer_path)?;
        validate_required_file("ONNX model", &onnx_model_path)?;

        let tokenizer = GLiNER2Tokenizer::from_file(&tokenizer_path)?;
        let special_tokens = resolve_special_tokens(&tokenizer)?;
        // GLiNER2 models currently use a fixed span width baked into the ONNX graph.
        let parameters = parameters.with_max_width(GLINER2_MAX_WIDTH);

        Ok(Self {
            params: parameters,
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
                &self.params,
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
            &self.params,
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
            &self.params,
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

        OutputsToRelations::new(schema).apply((
            relation_spans,
            RelationContext {
                entity_labels,
                entity_offsets,
            },
        ))
    }

    pub fn create_schema(&self) -> GLiNER2PipelineSchema {
        GLiNER2PipelineSchema::new()
    }

    pub fn extract_with_schema(
        &self,
        text: &str,
        schema: &GLiNER2PipelineSchema,
    ) -> Result<GLiNER2PipelineOutput> {
        GLiNER2Pipeline::new(self).extract(text, schema)
    }
}

#[derive(Clone)]
pub struct SpecialTokens {
    pub prompt: String,
    pub entity: String,
    pub sep_text: String,
    pub ids: HashMap<String, i64>,
}

struct SequenceInput {
    sequence_index: usize,
    text: String,
    labels: Vec<String>,
    task: SequenceTask,
}

#[derive(Clone, Copy)]
enum SequenceTask {
    Entities,
    Classification,
    Extraction,
}

struct GLiNER2NerPipeline {
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2NerPipeline {
    fn new(tokenizer: GLiNER2Tokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: expected_input_names(),
            expected_outputs: OutputsToSpans::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2NerPipeline {
    type Input = SequenceInput;
    type Output = SpanOutput;
    type Context = SequenceContext;
    type Parameters = Parameters;

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
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2ClassificationPipeline {
    fn new(tokenizer: GLiNER2Tokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: expected_input_names(),
            expected_outputs: OutputsToClassification::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ClassificationPipeline {
    type Input = SequenceInput;
    type Output = ClassificationOutput;
    type Context = ClassificationContext;
    type Parameters = Parameters;

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
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

struct ExtractionInput {
    sequence_index: usize,
    text: String,
    flattened_schema: FlattenedExtractionSchema,
}

impl GLiNER2ExtractionPipeline {
    fn new(tokenizer: GLiNER2Tokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: expected_input_names(),
            expected_outputs: OutputsToExtraction::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ExtractionPipeline {
    type Input = ExtractionInput;
    type Output = ExtractionOutput;
    type Context = ExtractionContext;
    type Parameters = Parameters;

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
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, SequenceContext)>
    for SequenceToNerTensors
{
    fn apply(&self, input: SequenceInput) -> Result<(SessionInputs<'a, 'a>, SequenceContext)> {
        let prepared = prepare_sequence(
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
    tokenizer: GLiNER2Tokenizer,
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
        let prepared = prepare_sequence(
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
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<ExtractionInput, (SessionInputs<'a, 'a>, ExtractionContext)>
    for ExtractionToTensors
{
    fn apply(&self, input: ExtractionInput) -> Result<(SessionInputs<'a, 'a>, ExtractionContext)> {
        let labels = input.flattened_schema.labels.clone();
        let prepared = prepare_sequence(
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

struct PreparedSequence {
    sequence_index: usize,
    text: String,
    tokens: Vec<Token>,
    labels: Vec<String>,
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    text_positions: Array1<i64>,
    schema_positions: Array1<i64>,
    span_idx: Array3<i64>,
}

fn prepare_sequence(
    input: SequenceInput,
    splitter: &RegexSplitter,
    tokenizer: &GLiNER2Tokenizer,
    special_tokens: &SpecialTokens,
    max_width: usize,
) -> Result<PreparedSequence> {
    let tokens = splitter.split(&input.text, None)?;
    if tokens.is_empty() {
        return Err("invalid input: text contains no tokenizable words".into());
    }
    if input.labels.is_empty() {
        return Err("invalid input: labels cannot be empty".into());
    }

    let schema = match input.task {
        SequenceTask::Entities => SchemaPrefix::build_ner(&input.labels, special_tokens, splitter)?,
        SequenceTask::Classification => {
            SchemaPrefix::build_classification(&input.labels, special_tokens, splitter)?
        }
        SequenceTask::Extraction => {
            SchemaPrefix::build_extraction(&input.labels, special_tokens, splitter)?
        }
    };
    let text_piece_offset = schema.pieces.len();

    let mut pieces = schema.pieces;
    pieces.extend(tokens.iter().map(|token| token.text().to_string()));

    let encoded = tokenizer.encode_pieces(&pieces)?;
    let text_positions = encoded
        .first_piece_positions
        .iter()
        .skip(text_piece_offset)
        .map(|position| *position as i64)
        .collect::<Vec<_>>();
    let schema_positions = schema
        .schema_piece_indices
        .iter()
        .map(|piece_index| encoded.first_piece_positions[*piece_index] as i64)
        .collect::<Vec<_>>();
    let span_idx = build_span_idx(tokens.len(), max_width);

    Ok(PreparedSequence {
        sequence_index: input.sequence_index,
        text: input.text,
        tokens,
        labels: input.labels,
        input_ids: Array2::from_shape_vec((1, encoded.input_ids.len()), encoded.input_ids)?,
        attention_mask: Array2::from_shape_vec(
            (1, encoded.attention_mask.len()),
            encoded.attention_mask,
        )?,
        text_positions: Array1::from_vec(text_positions),
        schema_positions: Array1::from_vec(schema_positions),
        span_idx,
    })
}

fn expected_input_names() -> HashSet<&'static str> {
    [
        INPUT_IDS,
        ATTENTION_MASK,
        TEXT_POSITIONS,
        SCHEMA_POSITIONS,
        SPAN_IDX,
    ]
    .into_iter()
    .collect()
}

fn resolve_special_tokens(tokenizer: &GLiNER2Tokenizer) -> Result<SpecialTokens> {
    let mut ids = HashMap::new();
    for token in ["[P]", "[E]", "[SEP_TEXT]"] {
        let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
            format!("required GLiNER2 special token `{token}` not found in tokenizer.json")
        })?;
        ids.insert(token.to_string(), token_id);
    }

    Ok(SpecialTokens {
        prompt: "[P]".to_string(),
        entity: "[E]".to_string(),
        sep_text: "[SEP_TEXT]".to_string(),
        ids,
    })
}

fn resolve_onnx_path(model_dir: &Path) -> PathBuf {
    let nested = model_dir.join("onnx/model.onnx");
    if nested.is_file() {
        nested
    } else {
        model_dir.join("model.onnx")
    }
}

fn validate_required_file(component: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing required {component} file: {}", path.display()).into())
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
            let (field_name, mode) = parse_field_spec(spec)?;
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

fn parse_field_spec(spec: &str) -> Result<(String, JsonFieldMode)> {
    let (field_name, mode) = if let Some((name, suffix)) = spec.split_once("::") {
        let mode = match suffix.trim() {
            "str" => JsonFieldMode::Single,
            "list" => JsonFieldMode::List,
            other => {
                return Err(format!(
                    "invalid JSON schema field spec `{spec}`: unsupported type suffix `{other}` (expected `str` or `list`)"
                )
                .into())
            }
        };
        (name.trim().to_string(), mode)
    } else {
        (spec.trim().to_string(), JsonFieldMode::List)
    };

    if field_name.is_empty() {
        return Err(format!("invalid JSON schema field spec `{spec}`: empty field name").into());
    }

    Ok((field_name, mode))
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
