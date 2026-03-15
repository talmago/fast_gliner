use gliner::model::{
    gliner2::{GLiNER2, GLiNER2Pipeline, GLiNER2PipelineOutput, GLiNER2PipelineSchema},
    input::{relation::schema::RelationSchema, text::TextInput},
    params::Parameters,
};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliner2_pipeline -- <model_dir>")?;

    let model = GLiNER2::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let pipeline = GLiNER2Pipeline::new(&model);

    let relations_only_schema = GLiNER2PipelineSchema::new()
        .entities(vec!["person", "company"])
        .relation_with_labels("founded", vec!["person"], vec!["company"]);

    let relations_only_text = "Bill Gates is an American businessman who co-founded Microsoft.";
    let relations_only_output = pipeline.extract(relations_only_text, &relations_only_schema)?;
    print_case(
        "Relations + Entities",
        relations_only_text,
        relations_only_output,
    );
    print_dedicated_relations(
        &model,
        "Relations + Entities (Dedicated API)",
        relations_only_text,
        &["person", "company"],
        &[("founded", &["person"], &["company"])],
    )?;

    let full_schema = GLiNER2PipelineSchema::new()
        .classification(
            "document_type",
            vec!["news".to_string(), "report".to_string()],
        )
        .entities(vec!["person".to_string(), "company".to_string()])
        .relation_with_labels("founded", vec!["person"], vec!["company"])
        .structure("event")
        .field("date")
        .field("description");

    let full_text = "NEWS REPORT: Bill Gates founded Microsoft on April 4, 1975. \
The announcement was repeated in a press briefing.";
    let full_output = pipeline.extract(full_text, &full_schema)?;
    print_case("Full Pipeline", full_text, full_output);
    print_dedicated_relations(
        &model,
        "Full Pipeline (Dedicated API)",
        full_text,
        &["person", "company"],
        &[("founded", &["person"], &["company"])],
    )?;

    Ok(())
}

fn print_dedicated_relations(
    model: &GLiNER2,
    name: &str,
    text: &str,
    entity_labels: &[&str],
    relation_specs: &[(&str, &[&str], &[&str])],
) -> Result<()> {
    let mut relation_schema = RelationSchema::new();
    for (relation, subjects, objects) in relation_specs {
        relation_schema.push_with_allowed_labels(relation, subjects, objects);
    }

    let input = TextInput::from_str(&[text], entity_labels)?;
    let output = model.extract_relations(input, &relation_schema)?;

    println!("=== {name} ===");
    if let Some(relations) = output.relations.first() {
        if relations.is_empty() {
            println!("  (none)");
        } else {
            for relation in relations {
                println!(
                    "  - {} --{}--> {} ({:.4})",
                    relation.subject().text,
                    relation.class(),
                    relation.object().text,
                    relation.probability()
                );
            }
        }
    } else {
        println!("  (none)");
    }
    println!();

    Ok(())
}

fn print_case(name: &str, text: &str, output: GLiNER2PipelineOutput) {
    println!("=== {name} ===");
    println!("Text: {text}");
    println!();

    println!("Classifications:");
    if output.classifications.is_empty() {
        println!("  (none)");
    } else {
        for (task_name, classification) in output.classifications {
            println!("  Task: {task_name}");
            for score in classification.scores {
                println!("    - {}: {:.4}", score.label, score.score);
            }
        }
    }
    println!();

    println!("Entities:");
    if output.entities.is_empty() {
        println!("  (none)");
    } else {
        for entity in output.entities {
            let (start, end) = entity.offsets();
            println!(
                "  - {} [{}] {:.4} ({}..{})",
                entity.text(),
                entity.class(),
                entity.probability(),
                start,
                end
            );
        }
    }
    println!();

    println!("Relations:");
    if output.relations.is_empty() {
        println!("  (none)");
    } else {
        for relation in output.relations {
            println!(
                "  - {} --{}--> {} ({:.4})",
                relation.subject().text,
                relation.class(),
                relation.object().text,
                relation.probability()
            );
        }
    }
    println!();

    println!("Structures:");
    if output.structures.is_empty() {
        println!("  (none)");
    } else {
        for (structure_name, structure_output) in output.structures {
            println!("  Structure: {structure_name}");
            for field in structure_output.fields {
                println!("    Field: {}", field.name);
                if field.values.is_empty() {
                    println!("      (none)");
                } else {
                    for value in field.values {
                        println!(
                            "      - {} [{}] {:.4} ({}..{})",
                            value.text, value.label, value.score, value.start, value.end
                        );
                    }
                }
            }
        }
    }
    println!();
}
