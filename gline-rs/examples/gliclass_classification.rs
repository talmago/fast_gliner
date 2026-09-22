use gliner::model::{params::Parameters, GLiClass};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a uni-encoder GLiClass ONNX model and classify a text.
///
/// Scores come from the `logits` head. Each label is an independent sigmoid
/// probability, sorted from highest to lowest.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliclass_classification -- <model_dir>")?;

    println!("Loading model from: {}", model_dir);

    let model = GLiClass::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let labels = vec![
        "computing".to_string(),
        "science".to_string(),
        "programming".to_string(),
        "travel".to_string(),
        "food".to_string(),
        "politics".to_string(),
    ];

    let output = model.classify(
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        &labels,
    )?;

    println!("Text: {}", output.text);
    println!("Scores:");
    for score in &output.scores {
        println!("  {:12} {:.4}", score.label, score.score);
    }

    if let Some(top) = output.top() {
        println!("Top label: {} ({:.4})", top.label, top.score);
    }

    Ok(())
}
