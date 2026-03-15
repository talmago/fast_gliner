use gliner::model::{
    gliner2::GLiNER2,
    input::{relation::schema::RelationSchema, text::TextInput},
    params::Parameters,
};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Sample usage of GLiNER2 relation extraction.
///
/// This mirrors the `relation_extraction.rs` flow while loading the model via `GLiNER2::from_dir`.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliner2_relations -- <model_dir>")?;

    let model = GLiNER2::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let mut relation_schema = RelationSchema::new();
    relation_schema.push_with_allowed_labels("founded", &["person"], &["company"]);

    let input = TextInput::from_str(
        &["Bill Gates is an American businessman who co-founded Microsoft."],
        &["person", "company"],
    )?;

    let output = model.extract_relations(input, &relation_schema)?;
    println!("Relations:\n{}", output);

    Ok(())
}
