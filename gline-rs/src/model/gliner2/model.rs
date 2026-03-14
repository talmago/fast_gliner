use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use composable::Composable;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::text::TextInput;
use crate::model::output::decoded::SpanOutput;
use crate::model::params::Parameters;
use crate::text::splitter::{RegexSplitter, Splitter};
use crate::util::result::Result;

use super::decoder::{OutputsToSpans, SequenceContext};
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
    pipeline: GLiNER2Pipeline,
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
            pipeline: GLiNER2Pipeline::new(tokenizer, special_tokens),
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
                    entities: entities.clone(),
                },
                &self.pipeline,
                &self.params,
            )?;

            spans.push(output.spans.into_iter().next().unwrap_or_default());
        }

        Ok(SpanOutput::new(texts, entities, spans))
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
    entities: Vec<String>,
}

struct GLiNER2Pipeline {
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2Pipeline {
    fn new(tokenizer: GLiNER2Tokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: [
                INPUT_IDS,
                ATTENTION_MASK,
                TEXT_POSITIONS,
                SCHEMA_POSITIONS,
                SPAN_IDX,
            ]
            .into_iter()
            .collect(),
            expected_outputs: OutputsToSpans::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2Pipeline {
    type Input = SequenceInput;
    type Output = SpanOutput;
    type Context = SequenceContext;
    type Parameters = Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        SequenceToTensors {
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

struct SequenceToTensors {
    splitter: RegexSplitter,
    tokenizer: GLiNER2Tokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, SequenceContext)> for SequenceToTensors {
    fn apply(&self, input: SequenceInput) -> Result<(SessionInputs<'a, 'a>, SequenceContext)> {
        let tokens = self.splitter.split(&input.text, None)?;
        if tokens.is_empty() {
            return Err("invalid input: text contains no tokenizable words".into());
        }

        let schema =
            SchemaPrefix::build_ner(&input.entities, &self.special_tokens, &self.splitter)?;
        let text_piece_offset = schema.pieces.len();

        let mut pieces = schema.pieces;
        pieces.extend(tokens.iter().map(|token| token.text().to_string()));

        let encoded = self.tokenizer.encode_pieces(&pieces)?;
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
        let span_idx = build_span_idx(tokens.len(), self.max_width);

        let input_ids =
            ndarray::Array2::from_shape_vec((1, encoded.input_ids.len()), encoded.input_ids)?;
        let attention_mask = ndarray::Array2::from_shape_vec(
            (1, encoded.attention_mask.len()),
            encoded.attention_mask,
        )?;
        let text_positions = ndarray::Array1::from_vec(text_positions);
        let schema_positions = ndarray::Array1::from_vec(schema_positions);

        let session_inputs = ort::inputs! {
            INPUT_IDS => input_ids,
            ATTENTION_MASK => attention_mask,
            TEXT_POSITIONS => text_positions,
            SCHEMA_POSITIONS => schema_positions,
            SPAN_IDX => span_idx,
        }?;

        Ok((
            session_inputs.into(),
            SequenceContext {
                sequence_index: input.sequence_index,
                text: input.text,
                tokens,
                entities: input.entities,
            },
        ))
    }
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
