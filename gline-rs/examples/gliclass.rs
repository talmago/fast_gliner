use gliner::model::output::classification::ClassificationScore;
use gliner::model::params::Parameters;
use gliner::model::{
    nest_gliclass_scores, GLiClass, GLiClassExample, GLiClassLabelNode, GLiClassLabels,
    GLiClassRequest, HierarchicalValue,
};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

/// Example: load a uni-encoder GLiClass ONNX model and classify a text.
///
/// Scores come from the `logits` head. Each label is an independent sigmoid
/// probability. The second call adds a label hierarchy, a task prompt, and
/// few-shot examples.
fn main() -> Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("Usage: cargo run --example gliclass -- <model_dir>")?;

    println!("Loading model from: {}", model_dir);

    let model = GLiClass::from_dir(
        model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let labels = vec![
        "computing".to_string(),
        "science".to_string(),
        "programming".to_string(),
        "travel".to_string(),
        "food".to_string(),
        "politics".to_string(),
    ];

    let output = model.classify(
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        &labels,
    )?;

    println!("Text: {}", output.text);
    println!("Scores:");
    print_scores(&output.scores);

    if let Some(top) = output.top() {
        println!("Top label: {} ({:.4})", top.label, top.score);
    }

    let hierarchy = GLiClassLabels::Hierarchical(GLiClassLabelNode::Group(vec![
        (
            "sentiment".to_string(),
            GLiClassLabelNode::Leaves(vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ]),
        ),
        (
            "topic".to_string(),
            GLiClassLabelNode::Leaves(vec![
                "product".to_string(),
                "service".to_string(),
                "shipping".to_string(),
            ]),
        ),
    ]));
    let examples = vec![
        GLiClassExample {
            text: "Love this item, great quality!".to_string(),
            labels: vec!["positive".to_string(), "product".to_string()],
        },
        GLiClassExample {
            text: "Customer support was unhelpful".to_string(),
            labels: vec!["negative".to_string(), "service".to_string()],
        },
    ];

    let mut request = GLiClassRequest {
        text: "The product quality is amazing but delivery was slow".to_string(),
        labels: hierarchy.clone(),
        examples,
        prompt: Some("Classify this customer review by sentiment and topic:".to_string()),
    };
    let advanced = match model.classify_with(request.clone()) {
        Ok(output) => output,
        Err(err) if err.to_string().contains("<<EXAMPLE>>") => {
            println!();
            println!("This tokenizer has no <<EXAMPLE>> token, so few-shot examples are skipped.");
            println!("Scoring the label hierarchy and task prompt.");
            request.examples.clear();
            model.classify_with(request)?
        }
        Err(err) => return Err(err),
    };

    println!();
    println!("Advanced text: {}", advanced.text);
    println!("Dotted scores:");
    print_scores(&advanced.scores);
    println!("Hierarchical scores:");
    print_hierarchical(&nest_gliclass_scores(&hierarchy, &advanced.scores), 0);

    Ok(())
}

fn print_scores(scores: &[ClassificationScore]) {
    for score in scores {
        println!("  {:24} {:.4}", score.label, score.score);
    }
}

fn print_hierarchical(value: &HierarchicalValue, indent: usize) {
    match value {
        HierarchicalValue::Score(score) => println!("{score:.4}"),
        HierarchicalValue::Group(entries) => {
            for (key, child) in entries {
                print!("{:indent$}{key}: ", "", key = key);
                match child {
                    HierarchicalValue::Score(score) => println!("{score:.4}"),
                    HierarchicalValue::Group(_) => {
                        println!();
                        print_hierarchical(child, indent + 2);
                    }
                }
            }
        }
    }
}
