use orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::model::pipeline::token::TokenMode;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};


/// Sample usage of the public API using GPU for inferencing
/// 
/// This example will try two execution providers in that order:
/// * CUDA
/// * CoreML
/// 
/// To leverage one of them you need to enable the appropriate feature, for exemple:
/// 
/// ```console
/// $ cargo run --example gpu --features=cuda
/// ```
/// 
/// See `Readme.md` and `doc/ORT.md` for more information.
fn main() -> Result<()> {    

    const MAX_SAMPLES: usize = 1000;
    const CSV_PATH: &str = "data/nuner-sample-1k.csv";

    let entities = [
        "person", 
        "location",
        "vehicle",
    ];

    println!("Loading data...");
    let input = TextInput::new_from_csv(CSV_PATH, 0, MAX_SAMPLES, entities.map(|x| x.to_string()).to_vec())?;
    let nb_samples = input.texts.len();
    
    println!("Loading model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default().with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),            
        ]),
        "models/gliner-multitask-large-v0.5/tokenizer.json",
        "models/gliner-multitask-large-v0.5/onnx/model.onnx",
    )?;

    println!("Inferencing...");
    let inference_start = std::time::Instant::now();
    let _output = model.inference(input)?;
    
    let inference_time = inference_start.elapsed();
    println!("Inference took {} seconds on {} samples ({:.2} samples/sec)", inference_time.as_secs(), nb_samples, nb_samples as f32 / inference_time.as_secs() as f32);

    Ok(())
}

