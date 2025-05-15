use composable::*;
use orp::model::Model;
use orp::pipeline::*;
use orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::params::Parameters;
use gliner::model::pipeline::{token::TokenPipeline, relation::RelationPipeline};
use gliner::model::input::{text::TextInput, relation::schema::RelationSchema};


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
    // Set model and tokenizer paths    
    const MODEL_PATH: &str = "models/gliner-multitask-large-v0.5/onnx/model.onnx";
    const TOKENIZER_PATH: &str = "models/gliner-multitask-large-v0.5/tokenizer.json";
    
    // Use default parameters
    let params: Parameters = Parameters::default();
    let runtime_params = RuntimeParameters::default();

    // Define a relation schema.
    // We declare a "founded" relation which subject has to be a "person" and object has to be a "company"
    let mut relation_schema = RelationSchema::new();
    relation_schema.push_with_allowed_labels("founded", &["person"], &["company"]);

    // Sample input text and entity labels
    let input = TextInput::from_str(
        &["Bill Gates is an American businessman who co-founded Microsoft."],
        &["person", "company"],
    )?;
    
    // Load the model that will be leveraged for the pipeline below
    println!("Loading model...");      
    let model = Model::new(MODEL_PATH, runtime_params)?;
    
    // Relation Extraction needs Named Entity Recognition to be applied first.
    // Here we combine the two pipelines: one for NER, and one for RE.
    // For testing purposes we also insert printing functions.
    let pipeline = composed![
        TokenPipeline::new(TOKENIZER_PATH)?.to_composable(&model, &params),
        Print::new(Some("Entities:\n"), None),
        RelationPipeline::default(TOKENIZER_PATH, &relation_schema)?.to_composable(&model, &params),
        Print::new(Some("Relations:\n"), None)
    ];

    // Actually perform inferences using the pipeline defined above
    pipeline.apply(input)?;
    
    Ok(())
}
