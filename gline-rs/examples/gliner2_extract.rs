use std::collections::HashMap;

use gliner::model::{gliner2::GLiNER2, params::Parameters};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a GLiNER2 model and run JSON-schema extraction via `extract_json`.
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

    let text =
        "Contact: John Smith\nEmail: john@example.com\n\nPhones: 555-1234, 555-5678\nAddress: 123 Main St, NYC";

    let schema = HashMap::from([(
        "contact".to_string(),
        vec![
            "name::str".to_string(),
            "email::str".to_string(),
            "phone::list".to_string(),
            "address".to_string(),
        ],
    )]);

    let output = model.extract_json(text, &schema)?;

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
