use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a GLiNER model and run NER inference
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example ner -- <model_dir>")?;

    println!("Loading model from: {}", model_dir);

    let model = GLiNER::from_dir_with(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
        None,
        Some("model.onnx"),
        None,
    )?;

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
