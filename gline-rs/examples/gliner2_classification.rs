use gliner::model::{gliner2::GLiNER2, params::Parameters};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a GLiNER2 model and run schema-driven classification inference.
///
/// This runtime derives classification scores from the monolithic `span_scores` output
/// by taking the best span score for each requested label.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliner2_classification -- <model_dir>")?;

    println!("Loading model from: {}", model_dir);

    let model = GLiNER2::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let labels = vec![
        "shopping".to_string(),
        "work".to_string(),
        "personal".to_string(),
    ];

    let output = model.classify("Buy milk and eggs after work", &labels)?;

    println!("Text: {}", output.text);
    println!("Scores:");
    for score in &output.scores {
        println!("  {:10} {:.4}", score.label, score.score);
    }

    if let Some(top) = output.top() {
        println!("Top label: {} ({:.4})", top.label, top.score);
    }

    Ok(())
}
