use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use composable::Composable;
use orp::model::Model;
use orp::params::RuntimeParameters;

use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::RelationInput;
use crate::model::input::schema::{ExtractionFieldSchema, ExtractionSchema, SpecialTokens};
use crate::model::input::tensors::gliclass::{
    require_gliclass_tokens, GLiClassInput, GLiClassSettings,
};
use crate::model::input::tensors::schema::{ExtractionInput, SequenceInput, SequenceTask};
use crate::model::input::text::TextInput;
use crate::model::output::classification::ClassificationOutput;
use crate::model::output::decoded::SpanOutput;
use crate::model::output::extraction::ExtractionOutput;
use crate::model::output::relation::{RelationOutput, SpanOutputToRelationOutput};
use crate::model::params::Parameters;
use crate::model::pipeline::context::RelationContext;
use crate::model::pipeline::gliclass::GLiClassPipeline;
use crate::model::pipeline::multitask::{
    GLiNER2Pipeline, GLiNER2PipelineOutput, GLiNER2PipelineSchema,
};
use crate::model::pipeline::schema::{
    GLiNER2ClassificationPipeline, GLiNER2ExtractionPipeline, GLiNER2NerPipeline,
};
use crate::model::{input, output, pipeline, GLiNER};
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

const GLINER2_MAX_WIDTH: usize = 8;

/// Runtime-selected GLiNER model (span or token mode).
pub enum InferenceMode {
    Span(GLiNER<pipeline::span::SpanMode>),
    Token(GLiNER<pipeline::token::TokenMode>),
}

impl InferenceMode {
    pub fn get_inner_model(&self) -> &Model {
        match self {
            Self::Span(model) => model.get_inner_model(),
            Self::Token(model) => model.get_inner_model(),
        }
    }

    pub fn inference<'a>(
        &'a self,
        input: input::text::TextInput,
    ) -> Result<output::decoded::SpanOutput> {
        match self {
            Self::Span(model) => model.inference(input),
            Self::Token(model) => model.inference(input),
        }
    }
}

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

        super::validate_required_file("tokenizer", &tokenizer_path)?;
        super::validate_required_file("ONNX model", &onnx_model_path)?;

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        let special_tokens = SpecialTokens::resolve(&tokenizer)?;
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

        SpanOutputToRelationOutput::new(schema).apply((
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

pub struct GLiClass {
    params: Parameters,
    model: Model,
    pipeline: GLiClassPipeline,
}

impl GLiClass {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        parameters: Parameters,
        runtime_parameters: RuntimeParameters,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_model_path = resolve_onnx_path(model_dir);

        super::validate_required_file("tokenizer", &tokenizer_path)?;
        super::validate_required_file("ONNX model", &onnx_model_path)?;

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        require_gliclass_tokens(&tokenizer)?;
        let settings = GLiClassSettings::load(model_dir, parameters.max_length)?;
        let parameters = parameters.with_max_length(Some(settings.max_length));

        Ok(Self {
            params: parameters,
            model: Model::new(onnx_model_path, runtime_parameters)?,
            pipeline: GLiClassPipeline::new(tokenizer, settings.prompt_first),
        })
    }

    pub fn get_inner_model(&self) -> &Model {
        &self.model
    }

    /// Scores `text` against `labels` with a uni-encoder GLiClass ONNX export.
    ///
    /// The graph returns one logit per label. Scores are multi-label sigmoid
    /// probabilities, sorted from highest to lowest.
    pub fn classify(&self, text: &str, labels: &[String]) -> Result<ClassificationOutput> {
        if text.trim().is_empty() {
            return Err("invalid input: text contains no tokenizable words".into());
        }
        if labels.is_empty() {
            return Err("invalid input: labels cannot be empty".into());
        }

        self.model.inference(
            GLiClassInput {
                text: text.to_string(),
                labels: labels.to_vec(),
            },
            &self.pipeline,
            &self.params,
        )
    }
}

fn resolve_onnx_path(model_dir: &Path) -> PathBuf {
    let nested = model_dir.join("onnx/model.onnx");
    if nested.is_file() {
        nested
    } else {
        model_dir.join("model.onnx")
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
