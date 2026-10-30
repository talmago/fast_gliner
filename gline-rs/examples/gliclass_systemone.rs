use std::path::{Path, PathBuf};

use gliner::model::params::Parameters;
use gliner::model::{GLiClass, SystemOneAnswer, SystemOneQuestion, SystemOneRequest};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;
use serde_json::json;

/// Example: answer typed System One questions with a GLiClass v4 checkpoint.
///
/// From `gline-rs/`, the checkpoints live in `../models`.
fn main() -> Result<()> {
    let model_dir = resolve_model_dir(std::env::args().nth(1))?;
    println!("Loading model from: {}", model_dir.display());

    let model = GLiClass::from_dir(
        &model_dir,
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    let state = json!({
        "ticket": "Checkout errors hit 12% after the 14:05 deploy.",
        "facts": [
            "Rolling back takes two minutes.",
            "No incident is open yet."
        ]
    });
    let questions_json = json!({
        "action": {
            "type": "choice",
            "instructions": "What should the on-call engineer do right now?",
            "criteria": {
                "rollback": "Roll back to the last healthy build.",
                "wait": "Watch error rates for another hour.",
                "escalate": "Page the payments team lead."
            }
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this need action within the next hour?",
            "criteria": {
                "true": "Yes, customers are affected right now.",
                "false": "No, it can wait."
            }
        },
        "severity": {
            "type": "score",
            "instructions": "Rate the incident severity.",
            "criteria": ["minor", "moderate", "major", "critical"]
        }
    });

    let mut questions = Vec::new();
    let object = questions_json
        .as_object()
        .ok_or("invalid input: questions must be an object")?;
    for (name, question) in object {
        questions.push((name.clone(), SystemOneQuestion::from_value(question)?));
    }

    let output = model.system_one(SystemOneRequest {
        state,
        questions,
        max_length: None,
    })?;

    println!("Answers:");
    for (name, answer) in &output.answers {
        print_answer(name, answer);
    }
    println!(
        "Usage: {} input tokens, {} output tokens",
        output.usage.input_tokens, output.usage.output_tokens
    );

    Ok(())
}

fn resolve_model_dir(arg: Option<String>) -> Result<PathBuf> {
    let requested = arg.unwrap_or_else(|| "models/gliclass-v4-edge".to_string());
    let path = PathBuf::from(&requested);
    if path.is_dir() {
        return Ok(path);
    }

    let from_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(&requested);
    if from_repo.is_dir() {
        return Ok(from_repo);
    }

    Err(format!(
        "missing GLiClass model directory: {requested} (from gline-rs, use ../models/gliclass-v4-edge)"
    )
    .into())
}

fn print_answer(name: &str, answer: &SystemOneAnswer) {
    match answer {
        SystemOneAnswer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            println!("  {name}: choice={choice} confidence={confidence:.4}");
            for (key, score) in probabilities {
                println!("    {key}: {score:.4}");
            }
        }
        SystemOneAnswer::Noul { noul } => {
            println!("  {name}: noul={noul:.4}");
        }
        SystemOneAnswer::Score {
            score,
            confidence,
            probabilities,
        } => {
            println!("  {name}: score={score:.4} confidence={confidence:.4}");
            for (key, value) in probabilities {
                println!("    {key}: {value:.4}");
            }
        }
    }
}
