use orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::token::TokenMode;

/// Sample usage of the public API in token mode
fn main() -> Result<()> {    
    
    println!("Loading model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        "models/gliner-multitask-large-v0.5/tokenizer.json",
        "models/gliner-multitask-large-v0.5/onnx/model.onnx",
    )?;
    
    let input = TextInput::from_str(
        &[ 
            "I am James Bond",
            "This is James and I live in Chelsea, London.",
            "My name is Bond, James Bond.",
            "I like to drive my Aston Martin.",
            "The villain in the movie is Auric Goldfinger."
        ],
        &[
            "person", 
            "location",
            "vehicle",
        ]
    )?;

    println!("Inferencing...");
    let output = model.inference(input)?;

    println!("Results:");
    for spans in output.spans {
        for span in spans {
            println!("{:3} | {:16} | {:10} | {:.1}%", span.sequence(), span.text(), span.class(), span.probability() * 100.0);
        }
    }

    Ok(())

}
