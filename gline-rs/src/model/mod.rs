//! The core of `gline-rs`: everything about pre-/post-processing, and inferencing

pub mod config;
pub mod input;
pub mod output;
pub mod params;
pub mod pipeline;
pub mod runtime;

use std::path::{Path, PathBuf};

use crate::util::result::Result;
use config::{ConfigMode, ModelConfig};
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use params::Parameters;
use runtime::InferenceMode;

/// Basic GLiNER, to be parametrized by a specific pipeline (see implementations within the pipeline module)
///
/// This is just a convenience wrapper around a `Model`, a `Pipeline`, and some `Parameters`.
pub struct GLiNER<P> {
    params: Parameters,
    model: Model,
    pipeline: P,
}

impl<P> GLiNER<P> {
    pub fn get_inner_model(&self) -> &Model {
        &self.model
    }
}

impl<'a, P: Pipeline<'a, Parameters = Parameters>> GLiNER<P> {
    pub fn inference(&'a self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, &self.pipeline, &self.params)
    }
}

impl GLiNER<()> {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        parameters: Parameters,
        runtime_parameters: RuntimeParameters,
    ) -> Result<InferenceMode> {
        Self::from_dir_with(model_dir, parameters, runtime_parameters, None, None, None)
    }

    pub fn from_dir_with<P: AsRef<Path>>(
        model_dir: P,
        parameters: Parameters,
        runtime_parameters: RuntimeParameters,
        tokenizer_path: Option<&str>,
        onnx_model_path: Option<&str>,
        config_path: Option<&str>,
    ) -> Result<InferenceMode> {
        let model_dir = model_dir.as_ref();

        let tokenizer_path = resolve_component_path(model_dir, tokenizer_path, "tokenizer.json");
        let onnx_model_path = resolve_component_path(model_dir, onnx_model_path, "onnx/model.onnx");
        let config_path = resolve_component_path(model_dir, config_path, "gliner_config.json");

        validate_required_file("tokenizer", &tokenizer_path)?;
        validate_required_file("ONNX model", &onnx_model_path)?;
        validate_required_file("config", &config_path)?;

        let config = ModelConfig::from_file(&config_path)?;
        let parameters = parameters.with_max_width(config.max_width);

        match config.mode {
            ConfigMode::Span => Ok(InferenceMode::Span(
                GLiNER::<pipeline::span::SpanMode>::new(
                    parameters,
                    runtime_parameters,
                    tokenizer_path,
                    onnx_model_path,
                )?,
            )),
            ConfigMode::Token => Ok(InferenceMode::Token(
                GLiNER::<pipeline::token::TokenMode>::new(
                    parameters,
                    runtime_parameters,
                    tokenizer_path,
                    onnx_model_path,
                )?,
            )),
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

fn validate_required_file(component: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing required {component} file: {}", path.display()).into())
    }
}
