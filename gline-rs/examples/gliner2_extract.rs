use gliner::model::{
    gliner2::{ExtractionFieldSchema, ExtractionSchema, GLiNER2},
    params::Parameters,
};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a GLiNER2 model and run schema-driven extraction inference.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliner2_extract -- <model_dir>")?;

    println!("Loading model from: {}", model_dir);

    let model = GLiNER2::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let text = "Contact: John Smith\nEmail: john@example.com\nPhones: 555-1234, 555-5678\nAddress: 123 Main St, NYC";

    let schema = ExtractionSchema::from_fields(vec![
        ExtractionFieldSchema::new("name", vec!["person".to_string()]),
        ExtractionFieldSchema::new("email", vec!["email".to_string()]),
        ExtractionFieldSchema::new("phone", vec!["phone".to_string()]),
        ExtractionFieldSchema::new("address", vec!["address".to_string()]),
    ]);

    let output = model.extract(text, &schema)?;

    println!("Text: {}", output.text);
    for field in output.fields {
        println!("Field: {}", field.name);
        for value in field.values {
            println!("  {}", value.text);
        }
        println!();
    }

    Ok(())
}
