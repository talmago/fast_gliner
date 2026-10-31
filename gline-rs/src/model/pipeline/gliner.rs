//! GLiNER runtime: span/token loader and `gliner_config.json`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use composable::*;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::*;
use serde_json::Value;

use super::context::{EntityContext, RelationContext};
use crate::model::input;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::{RelationInputToTextInput, SpanOutputToRelationInput};
use crate::model::output;
use crate::model::output::decoded::SpanOutput;
use crate::model::output::relation::{RelationOutput, SpanOutputToRelationOutput};
use crate::model::{resolve_onnx_path, validate_required_file};
use crate::text::splitter::{RegexSplitter, Splitter};
use crate::text::tokenizer::{HFTokenizer, Tokenizer};
use crate::util::json::{json_bool_opt, json_f32_opt, json_string_opt, json_usize_opt};
use crate::util::result::Result;

const DEFAULT_MAX_WIDTH: usize = 12;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigMode {
    Span,
    Token,
}

/// Checkpoint settings from `gliner_config.json`.
#[derive(Debug, Clone)]
pub struct GLiNERConfig {
    pub mode: ConfigMode,
    pub max_width: usize,
    pub threshold: f32,
    pub flat_ner: bool,
    pub dup_label: bool,
    pub multi_label: bool,
    pub max_length: Option<usize>,
}

impl Default for ConfigMode {
    fn default() -> Self {
        Self::Span
    }
}

impl Default for GLiNERConfig {
    fn default() -> Self {
        Self {
            mode: ConfigMode::Span,
            max_width: DEFAULT_MAX_WIDTH,
            threshold: DEFAULT_THRESHOLD,
            flat_ner: true,
            dup_label: false,
            multi_label: false,
            max_length: Some(DEFAULT_MAX_LENGTH),
        }
    }
}

impl GLiNERConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    fn from_json(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)?;
        let mode = match json_string_opt(&value, "mode").as_deref() {
            None | Some("span") => ConfigMode::Span,
            Some("token") => ConfigMode::Token,
            Some(mode) => {
                return Err(format!("unsupported GLiNER mode `{mode}`").into());
            }
        };
        let max_width = json_usize_opt(&value, "max_width").unwrap_or(DEFAULT_MAX_WIDTH);
        if max_width == 0 {
            return Err("invalid GLiNER config: max_width must be greater than zero".into());
        }
        let max_length = json_usize_opt(&value, "max_len")
            .or_else(|| json_usize_opt(&value, "max_length"))
            .unwrap_or(DEFAULT_MAX_LENGTH);
        if max_length == 0 {
            return Err("invalid GLiNER config: max_len must be greater than zero".into());
        }

        Ok(Self {
            mode,
            max_width,
            threshold: json_f32_opt(&value, "threshold").unwrap_or(DEFAULT_THRESHOLD),
            flat_ner: json_bool_opt(&value, "flat_ner").unwrap_or(true),
            dup_label: json_bool_opt(&value, "dup_label").unwrap_or(false),
            multi_label: json_bool_opt(&value, "multi_label").unwrap_or(false),
            max_length: Some(max_length),
        })
    }
}

/// Basic GLiNER, parametrized by the span or token pipeline.
pub struct GLiNER<P> {
    pub(crate) config: GLiNERConfig,
    pub(crate) model: Model,
    pub(crate) pipeline: P,
}

impl<P> GLiNER<P> {
    pub fn get_inner_model(&self) -> &Model {
        &self.model
    }
}

impl<'a, P: Pipeline<'a, Parameters = GLiNERConfig>> GLiNER<P> {
    pub fn inference(&'a self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, &self.pipeline, &self.config)
    }
}

impl GLiNER<()> {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        runtime_parameters: RuntimeParameters,
    ) -> Result<InferenceMode> {
        Self::from_dir_with(model_dir, runtime_parameters, None, None, None)
    }

    pub fn from_dir_with<P: AsRef<Path>>(
        model_dir: P,
        runtime_parameters: RuntimeParameters,
        tokenizer_path: Option<&str>,
        onnx_model_path: Option<&str>,
        config_path: Option<&str>,
    ) -> Result<InferenceMode> {
        let model_dir = model_dir.as_ref();

        let tokenizer_path = resolve_component_path(model_dir, tokenizer_path, "tokenizer.json");
        let onnx_model_path = match onnx_model_path {
            Some(path) => resolve_component_path(model_dir, Some(path), "onnx/model.onnx"),
            None => resolve_onnx_path(model_dir),
        };
        let config_path = resolve_component_path(model_dir, config_path, "gliner_config.json");

        validate_required_file("tokenizer", &tokenizer_path)?;
        validate_required_file("ONNX model", &onnx_model_path)?;
        validate_required_file("config", &config_path)?;

        let config = GLiNERConfig::from_file(&config_path)?;
        match config.mode {
            ConfigMode::Span => Ok(InferenceMode::Span(GLiNER::<SpanMode>::new(
                config,
                runtime_parameters,
                tokenizer_path,
                onnx_model_path,
            )?)),
            ConfigMode::Token => Ok(InferenceMode::Token(GLiNER::<TokenMode>::new(
                config,
                runtime_parameters,
                tokenizer_path,
                onnx_model_path,
            )?)),
        }
    }
}

fn resolve_component_path(
    model_dir: &Path,
    override_path: Option<&str>,
    default_path: &str,
) -> PathBuf {
    match override_path {
        Some(path) => {
            let path = Path::new(path);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                model_dir.join(path)
            }
        }
        None => model_dir.join(default_path),
    }
}

/// Runtime-selected GLiNER model (span or token mode).
pub enum InferenceMode {
    Span(GLiNER<SpanMode>),
    Token(GLiNER<TokenMode>),
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

pub struct SpanPipeline<S, T> {
    splitter: S,
    tokenizer: T,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl<'a, S: Splitter, T: Tokenizer> Pipeline<'a> for SpanPipeline<S, T> {
    type Input = input::text::TextInput;
    type Output = output::decoded::SpanOutput;
    type Context = EntityContext;
    type Parameters = GLiNERConfig;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::span::EncodedToTensors::new(params.max_width),
            input::tensors::span::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),
            output::decoded::span::TensorsToDecoded::new(params.threshold, params.max_width),
            output::decoded::sort::SpanSort::default(),
            output::decoded::greedy::GreedySearch::new(
                params.flat_ner,
                params.dup_label,
                params.multi_label
            )
        ]
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

impl SpanPipeline<RegexSplitter, HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: RegexSplitter::default(),
            tokenizer: HFTokenizer::from_file(tokenizer_path)?,
            expected_inputs: input::tensors::span::SpanTensors::inputs()
                .into_iter()
                .collect(),
            expected_outputs: output::decoded::span::TensorsToDecoded::outputs()
                .into_iter()
                .collect(),
        })
    }

    pub fn new_from_bytes(tokenizer_bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            splitter: RegexSplitter::default(),
            tokenizer: HFTokenizer::from_bytes(tokenizer_bytes)?,
            expected_inputs: input::tensors::span::SpanTensors::inputs()
                .into_iter()
                .collect(),
            expected_outputs: output::decoded::span::TensorsToDecoded::outputs()
                .into_iter()
                .collect(),
        })
    }
}

pub type SpanMode = SpanPipeline<RegexSplitter, HFTokenizer>;

impl GLiNER<SpanMode> {
    pub fn new<P: AsRef<Path>>(
        config: GLiNERConfig,
        runtime_params: RuntimeParameters,
        tokenizer_path: P,
        model_path: P,
    ) -> Result<Self> {
        Ok(Self {
            model: Model::new(model_path, runtime_params)?,
            pipeline: SpanPipeline::new(tokenizer_path)?,
            config,
        })
    }

    pub fn new_from_bytes(
        config: GLiNERConfig,
        runtime_params: RuntimeParameters,
        tokenizer_bytes: &[u8],
        model_bytes: &[u8],
    ) -> Result<Self> {
        Ok(Self {
            model: Model::new_from_bytes(model_bytes, runtime_params)?,
            pipeline: SpanPipeline::new_from_bytes(tokenizer_bytes)?,
            config,
        })
    }
}

pub struct TokenPipeline<S, T> {
    splitter: S,
    tokenizer: T,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl<'a, S: Splitter, T: Tokenizer> Pipeline<'a> for TokenPipeline<S, T> {
    type Input = input::text::TextInput;
    type Output = output::decoded::SpanOutput;
    type Context = EntityContext;
    type Parameters = GLiNERConfig;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::token::EncodedToTensors::default(),
            input::tensors::token::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),
            output::decoded::token::TensorsToDecoded::new(params.threshold),
            output::decoded::sort::SpanSort::default(),
            output::decoded::greedy::GreedySearch::new(
                params.flat_ner,
                params.dup_label,
                params.multi_label
            )
        ]
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

impl TokenPipeline<RegexSplitter, HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: RegexSplitter::default(),
            tokenizer: HFTokenizer::from_file(tokenizer_path)?,
            expected_inputs: input::tensors::token::TokenTensors::inputs()
                .into_iter()
                .collect(),
            expected_outputs: output::decoded::token::TensorsToDecoded::outputs()
                .into_iter()
                .collect(),
        })
    }
}

pub type TokenMode = TokenPipeline<RegexSplitter, HFTokenizer>;

/// Relation extraction on top of the GLiNER token pipeline.
pub struct GLiNERRelationPipeline<'a, S, T> {
    token_pipeline: TokenPipeline<S, T>,
    relation_schema: &'a RelationSchema,
}

impl<'a, S: Splitter, T: Tokenizer> Pipeline<'a> for GLiNERRelationPipeline<'a, S, T> {
    type Input = SpanOutput;
    type Output = RelationOutput;
    type Context = (RelationContext, EntityContext);
    type Parameters = GLiNERConfig;

    fn pre_processor(
        &self,
        params: &GLiNERConfig,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed_t![
            composed![
                SpanOutputToRelationInput::new(self.relation_schema),
                RelationInputToTextInput::default()
            ],
            self.token_pipeline.pre_processor(params)
        ]
    }

    fn post_processor(
        &self,
        params: &GLiNERConfig,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed_rt![
            self.token_pipeline.post_processor(params),
            SpanOutputToRelationOutput::new(self.relation_schema)
        ]
    }
}

impl<'a, S, T> GLiNERRelationPipeline<'a, S, T> {
    pub fn new(token_pipeline: TokenPipeline<S, T>, relation_schema: &'a RelationSchema) -> Self {
        Self {
            token_pipeline,
            relation_schema,
        }
    }
}

impl<'a> GLiNERRelationPipeline<'a, RegexSplitter, HFTokenizer> {
    pub fn default<P: AsRef<Path>>(
        tokenizer_path: P,
        relation_schema: &'a RelationSchema,
    ) -> Result<Self> {
        Ok(GLiNERRelationPipeline::new(
            TokenPipeline::new(tokenizer_path)?,
            relation_schema,
        ))
    }
}

impl GLiNER<TokenMode> {
    pub fn new<P: AsRef<Path>>(
        config: GLiNERConfig,
        runtime_params: RuntimeParameters,
        tokenizer_path: P,
        model_path: P,
    ) -> Result<Self> {
        Ok(Self {
            config,
            model: Model::new(model_path, runtime_params)?,
            pipeline: TokenPipeline::new(tokenizer_path)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigMode, GLiNERConfig};

    #[test]
    fn config_json_defaults_to_span_mode_and_width() {
        let config = GLiNERConfig::from_json(r#"{}"#).unwrap();
        assert_eq!(config.mode, ConfigMode::Span);
        assert_eq!(config.max_width, 12);
        assert_eq!(config.threshold, 0.5);
        assert!(config.flat_ner);
        assert!(!config.dup_label);
        assert!(!config.multi_label);
        assert_eq!(config.max_length, Some(512));
    }

    #[test]
    fn config_json_reads_token_mode_and_limits() {
        let config = GLiNERConfig::from_json(
            r#"{
                "mode": "token",
                "max_width": 8,
                "max_len": 384,
                "threshold": 0.4,
                "flat_ner": false,
                "dup_label": true,
                "multi_label": true
            }"#,
        )
        .unwrap();

        assert_eq!(config.mode, ConfigMode::Token);
        assert_eq!(config.max_width, 8);
        assert_eq!(config.max_length, Some(384));
        assert_eq!(config.threshold, 0.4);
        assert!(!config.flat_ner);
        assert!(config.dup_label);
        assert!(config.multi_label);
    }

    #[test]
    fn config_json_rejects_unknown_mode() {
        let error = GLiNERConfig::from_json(r#"{ "mode": "decoder" }"#)
            .unwrap_err()
            .to_string();
        assert!(error.contains("decoder"));
    }
}
