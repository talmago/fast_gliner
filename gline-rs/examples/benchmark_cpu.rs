use orp::params::RuntimeParameters;
use gliner::model::input;
use gliner::model::pipeline::token::TokenMode;
use gliner::util::result::Result;
use gliner::model::{GLiNER, params::Parameters};

fn main() -> Result<()> {    

    const MAX_SAMPLES: usize = 100;
    const THREADS: usize = 12;
    const CSV_PATH: &str = "data/nuner-sample-1k.csv";

    let entities = [
        "person", 
        "location",
        "vehicle",
    ];

    println!("Loading data...");
    let input = input::text::TextInput::new_from_csv(CSV_PATH, 0, MAX_SAMPLES, entities.map(|x| x.to_string()).to_vec())?;
    let nb_samples = input.texts.len();
    
    println!("Loading model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default().with_threads(THREADS),
        std::path::Path::new("models/gliner-multitask-large-v0.5/tokenizer.json"),
        std::path::Path::new("models/gliner-multitask-large-v0.5/onnx/model.onnx")
    )?;

    println!("Inferencing...");
    let inference_start = std::time::Instant::now();
    let _output = model.inference(input)?;
    
    let inference_time = inference_start.elapsed();
    println!("Inference took {} seconds on {} samples ({:.2} samples/sec)", inference_time.as_secs(), nb_samples, nb_samples as f32 / inference_time.as_secs() as f32);

    Ok(())
}
