use std::path::{Path, PathBuf};

use composable::*;
use gliner::model::input::{relation::schema::RelationSchema, text::TextInput};
use gliner::model::pipeline::gliner::{GLiNERRelationPipeline, TokenPipeline};
use gliner::model::GLiNERConfig;
use gliner::util::result::Result;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::*;

/// Sample usage of the public API for Relation Extraction
///
/// Also provides an example of direct use of `Model` and `Pipeline`.
///
/// Expected output:
///
/// ```text
/// Entities:
/// 0 | Bill Gates      | person     | 99.9%
/// 0 | Microsoft       | company    | 99.6%
/// Relations:
/// 0 | Bill Gates      | founded    | Microsoft       | 99.7%
/// ```
fn main() -> Result<()> {
    let model_dir = resolve_model_dir(std::env::args().nth(1))?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    let onnx_path = resolve_onnx_path(&model_dir);
    let config_path = model_dir.join("gliner_config.json");

    let params = if config_path.is_file() {
        GLiNERConfig::from_file(&config_path)?
    } else {
        GLiNERConfig::default()
    };
    let runtime_params = RuntimeParameters::default();

    let mut relation_schema = RelationSchema::new();
    relation_schema.push_with_allowed_labels("founded", &["person"], &["company"]);

    let input = TextInput::from_str(
        &["Bill Gates is an American businessman who co-founded Microsoft."],
        &["person", "company"],
    )?;

    println!("Loading model from: {}", model_dir.display());
    let model = Model::new(&onnx_path, runtime_params)?;

    let pipeline = composed![
        TokenPipeline::new(&tokenizer_path)?.to_composable(&model, &params),
        Print::new(Some("Entities:\n"), None),
        GLiNERRelationPipeline::default(&tokenizer_path, &relation_schema)?
            .to_composable(&model, &params),
        Print::new(Some("Relations:\n"), None)
    ];

    pipeline.apply(input)?;

    Ok(())
}

fn resolve_onnx_path(model_dir: &Path) -> PathBuf {
    let nested = model_dir.join("onnx/model.onnx");
    if nested.is_file() {
        nested
    } else {
        model_dir.join("model.onnx")
    }
}

fn resolve_model_dir(arg: Option<String>) -> Result<PathBuf> {
    let requested = arg.unwrap_or_else(|| "models/gliner-multitask-large-v0.5".to_string());
    let path = PathBuf::from(&requested);
    if path.is_dir() {
        return Ok(path);
    }

    let from_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(&requested);
    if from_repo.is_dir() {
        return Ok(from_repo);
    }

    Err(format!(
        "missing GLiNER model directory: {requested} (from gline-rs, use ../models/gliner-multitask-large-v0.5)"
    )
    .into())
}
