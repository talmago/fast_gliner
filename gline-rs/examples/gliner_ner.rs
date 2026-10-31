use std::path::{Path, PathBuf};

use gliner::model::{input::text::TextInput, GLiNER};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a GLiNER model and run NER inference
fn main() -> Result<()> {
    let model_dir = resolve_model_dir(std::env::args().nth(1))?;

    println!("Loading model from: {}", model_dir.display());

    let model = GLiNER::from_dir(&model_dir, RuntimeParameters::default())?;

    let input = TextInput::from_str(
        &[
            "I am James Bond",
            "This is James and I live in Chelsea, London.",
            "My name is Bond, James Bond.",
            "I like to drive my Aston Martin.",
            "The villain in the movie is Auric Goldfinger.",
        ],
        &["person", "location", "vehicle"],
    )?;

    println!("Running inference...");

    let output = model.inference(input)?;

    println!("Results:");

    for spans in output.spans {
        for span in spans {
            println!(
                "{:3} | {:16} | {:10} | {:.1}%",
                span.sequence(),
                span.text(),
                span.class(),
                span.probability() * 100.0
            );
        }
    }

    Ok(())
}

fn resolve_model_dir(arg: Option<String>) -> Result<PathBuf> {
    let requested = arg.unwrap_or_else(|| "models/gliner_multi-v2.1-onnx".to_string());
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
        "missing GLiNER model directory: {requested} (from gline-rs, use ../models/gliner_multi-v2.1-onnx)"
    )
    .into())
}
