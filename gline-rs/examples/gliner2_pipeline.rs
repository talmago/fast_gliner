use gliner::model::{
    gliner2::{GLiNER2, GLiNER2PipelineOutput},
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

    let schema = model
        .create_schema()
        .classification("document_type", vec!["news", "report", "announcement"])
        .entities(vec!["person", "company"])
        .relation("founded", vec!["person"], vec!["company"])
        .relation("works_for", vec!["person"], vec!["company"])
        .structure("event")
        .field("date")
        .field("description");

    let text = "NEWS REPORT: Bill Gates founded Microsoft on October 26, 2018. \
Satya Nadella works for Microsoft.";

    let output = model.extract_with_schema(text, &schema)?;

    println!("Text:");
    println!("  {text}");
    println!();

    print_output(output);

    Ok(())
}

fn print_output(output: GLiNER2PipelineOutput) {
    println!("Classifications:");
    if output.classifications.is_empty() {
        println!("  (none)");
    } else {
        let mut tasks = output.classifications.into_iter().collect::<Vec<_>>();
        tasks.sort_by(|left, right| left.0.cmp(&right.0));

        for (task_name, classification) in tasks {
            println!("  {task_name}:");
            for score in classification.scores {
                println!("    {}: {:.4}", score.label, score.score);
            }
        }
    }
    println!();

    println!("Entities:");
    if output.entities.is_empty() {
        println!("  (none)");
    } else {
        for entity in output.entities {
            println!("  {} ({})", entity.text(), entity.class());
        }
    }
    println!();

    println!("Relations:");
    if output.relations.is_empty() {
        println!("  (none)");
    } else {
        for relation in output.relations {
            println!(
                "  {} --{}--> {}",
                relation.subject().text,
                relation.class(),
                relation.object().text,
            );
        }
    }
    println!();

    println!("Structures:");
    if output.structures.is_empty() {
        println!("  (none)");
    } else {
        let mut structures = output.structures.into_iter().collect::<Vec<_>>();
        structures.sort_by(|left, right| left.0.cmp(&right.0));

        for (structure_name, structure_output) in structures {
            println!("  {structure_name}:");
            for field in structure_output.fields {
                if field.values.is_empty() {
                    println!("    {}: (none)", field.name);
                } else {
                    let values = field
                        .values
                        .into_iter()
                        .map(|value| value.text)
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("    {}: {}", field.name, values);
                }
            }
        }
    }
}
