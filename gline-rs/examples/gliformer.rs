use std::collections::HashMap;

use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::params::Parameters;
use gliner::model::GLiFormer;
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Texts and labels follow the usage examples in the GLiFormer README.
///
/// This checkpoint scores relations with the joint head, so the open-relation
/// example is expressed as entity labels plus relation labels. Structures are flat.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliformer -- <models/gliformer-base-v1>")?;

    let model = GLiFormer::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    println!("entities");
    let entities = model.predict_entities(
        "Marie Curie worked at the University of Paris in France.",
        &["person".into(), "organization".into(), "location".into()],
    )?;
    for entity in entities {
        println!("{} => {}", entity.text(), entity.class());
    }

    println!("classification");
    let classification = model.classify(
        "The new search feature is fast and easy to use.",
        &["positive".into(), "negative".into(), "neutral".into()],
    )?;
    for score in classification.scores {
        println!("{} | {:.3}", score.label, score.score);
    }

    println!("relations");
    let mut relations = RelationSchema::new();
    relations.push_with_allowed_labels("works_at", &["person"], &["organization"]);
    relations.push_with_allowed_labels("lives_in", &["person"], &["location"]);
    let triples = model.extract_relations(
        "Alice works at Acme and lives in London.",
        &["person".into(), "organization".into(), "location".into()],
        &relations,
    )?;
    for relation in triples.relations.into_iter().flatten() {
        println!(
            "{} => {} => {}",
            relation.subject().text,
            relation.class(),
            relation.object().text
        );
    }

    println!("structure");
    let mut structure = HashMap::new();
    structure.insert(
        "employee".to_string(),
        vec!["name::str".to_string(), "company::str".to_string()],
    );
    let records = model.extract_json("Alice joined Acme as a software engineer.", &structure)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&records).map_err(|err| err.to_string())?
    );

    println!("multi-task");
    let schema = model
        .create_schema()
        .entities(vec!["person", "organization"])
        .classification("topic", vec!["business", "sports", "technology"])
        .structure("employee")
        .field("name")
        .field("company");
    let output = model.extract_with_schema("Alice joined Acme as a software engineer.", &schema)?;
    for entity in &output.entities {
        println!("{} => {}", entity.text(), entity.class());
    }
    if let Some(topic) = output.classifications.get("topic") {
        for score in &topic.scores {
            println!("{} | {:.3}", score.label, score.score);
        }
    }
    if let Some(employee) = output.structures.get("employee") {
        for field in &employee.fields {
            let values = field
                .values
                .iter()
                .map(|value| value.text.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}: {values}", field.name);
        }
    }

    Ok(())
}
