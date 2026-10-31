//! Typed System One questions on top of GLiClass and GLiFormer.

use serde_json::Value;

use crate::util::result::Result;

/// A System One request: a state plus named typed questions.
#[derive(Debug, Clone)]
pub struct SystemOneRequest {
    pub state: Value,
    pub questions: Vec<(String, SystemOneQuestion)>,
    pub max_length: Option<usize>,
}

/// One typed question. `instructions` become the GLiClass task prompt or the
/// GLiFormer classification group name.
#[derive(Debug, Clone)]
pub struct SystemOneQuestion {
    pub kind: SystemOneQuestionKind,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SystemOneQuestionKind {
    Choice {
        criteria: Vec<(String, Option<String>)>,
    },
    Noul {
        true_label: Option<String>,
        false_label: Option<String>,
    },
    Score {
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemOneAnswer {
    Choice {
        choice: String,
        confidence: f32,
        probabilities: Vec<(String, f32)>,
    },
    Noul {
        noul: f32,
    },
    Score {
        score: f32,
        confidence: f32,
        probabilities: Vec<(String, f32)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemOneUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneOutput {
    pub answers: Vec<(String, SystemOneAnswer)>,
    pub usage: SystemOneUsage,
}

impl SystemOneQuestion {
    pub fn from_value(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or("invalid input: each System One question must be an object")?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or("invalid input: a System One question needs a type")?;
        let instructions = object
            .get("instructions")
            .map(as_text)
            .filter(|text| !text.is_empty());
        let criteria = object.get("criteria");

        let kind = match kind {
            "choice" => {
                let criteria = criteria
                    .and_then(Value::as_object)
                    .ok_or("a choice question needs at least two criteria")?;
                if criteria.len() < 2 {
                    return Err("a choice question needs at least two criteria".into());
                }
                let criteria = criteria
                    .iter()
                    .map(|(key, value)| {
                        let label = if value.is_null() {
                            None
                        } else {
                            Some(as_text(value))
                        };
                        (key.clone(), label)
                    })
                    .collect();
                SystemOneQuestionKind::Choice { criteria }
            }
            "noul" => {
                let criteria = criteria.and_then(Value::as_object);
                let true_label = criteria
                    .and_then(|map| map.get("true"))
                    .and_then(truthy_text);
                let false_label = criteria
                    .and_then(|map| map.get("false"))
                    .and_then(truthy_text);
                SystemOneQuestionKind::Noul {
                    true_label,
                    false_label,
                }
            }
            "score" => {
                let criteria = criteria
                    .and_then(Value::as_array)
                    .ok_or("a score question needs at least two levels")?;
                if criteria.len() < 2 {
                    return Err("a score question needs at least two levels".into());
                }
                SystemOneQuestionKind::Score {
                    criteria: criteria.clone(),
                }
            }
            other => {
                return Err(format!("unsupported question type `{other}`").into());
            }
        };

        Ok(Self { kind, instructions })
    }
}

impl SystemOneAnswer {
    pub(crate) fn from_probabilities(
        question: &SystemOneQuestion,
        probabilities: &[(String, f32)],
    ) -> Result<Self> {
        if probabilities.is_empty() {
            return Err("invalid input: a System One question produced no probabilities".into());
        }
        let best = probabilities
            .iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("invalid input: a System One question produced no probabilities")?;

        Ok(match &question.kind {
            SystemOneQuestionKind::Choice { .. } => SystemOneAnswer::Choice {
                choice: best.0.clone(),
                confidence: best.1,
                probabilities: probabilities.to_vec(),
            },
            SystemOneQuestionKind::Noul { .. } => {
                let noul = probabilities
                    .iter()
                    .find(|(key, _)| key == "true")
                    .map(|(_, score)| *score)
                    .ok_or("invalid input: a noul question is missing the true option")?;
                SystemOneAnswer::Noul { noul }
            }
            SystemOneQuestionKind::Score { .. } => {
                let score = probabilities
                    .iter()
                    .map(|(key, value)| {
                        key.parse::<f32>().map(|index| index * value).unwrap_or(0.0)
                    })
                    .sum();
                SystemOneAnswer::Score {
                    score,
                    confidence: best.1,
                    probabilities: probabilities.to_vec(),
                }
            }
        })
    }
}

pub fn render_state(state: &Value) -> Result<String> {
    if is_empty_state(state) {
        return Ok(String::new());
    }
    match state {
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => Ok(items
            .iter()
            .map(|item| format!("- {}", as_text(item)))
            .collect::<Vec<_>>()
            .join("\n")),
        Value::Object(map) => {
            let mut lines = Vec::new();
            for (key, value) in map {
                if let Some(items) = string_list(value) {
                    lines.push(format!("{key}:"));
                    for item in items {
                        lines.push(format!("- {item}"));
                    }
                } else {
                    lines.push(format!("{key}: {}", as_text(value)));
                }
            }
            Ok(lines.join("\n"))
        }
        other => Ok(as_text(other)),
    }
}

pub(crate) fn system_one_options(
    question: &SystemOneQuestion,
) -> Result<(Vec<String>, Vec<String>)> {
    match &question.kind {
        SystemOneQuestionKind::Choice { criteria } => {
            if criteria.len() < 2 {
                return Err("a choice question needs at least two criteria".into());
            }
            let keys = criteria.iter().map(|(key, _)| key.clone()).collect();
            let labels = criteria
                .iter()
                .map(|(key, value)| value.clone().unwrap_or_else(|| key.clone()))
                .collect();
            Ok((keys, labels))
        }
        SystemOneQuestionKind::Noul {
            true_label,
            false_label,
        } => Ok((
            vec!["true".to_string(), "false".to_string()],
            vec![
                true_label.clone().unwrap_or_else(|| "yes".to_string()),
                false_label.clone().unwrap_or_else(|| "no".to_string()),
            ],
        )),
        SystemOneQuestionKind::Score { criteria } => {
            if criteria.len() < 2 {
                return Err("a score question needs at least two levels".into());
            }
            let keys = (0..criteria.len()).map(|index| index.to_string()).collect();
            let labels = criteria.iter().map(as_text).collect();
            Ok((keys, labels))
        }
    }
}

/// GLiFormer `[CAT]` labels: short keys, with a description folded as `key: description`.
pub(crate) fn gliformer_system_one_options(
    question: &SystemOneQuestion,
) -> Result<(Vec<String>, Vec<String>)> {
    let (keys, _) = system_one_options(question)?;
    let labels = match &question.kind {
        SystemOneQuestionKind::Choice { criteria } => criteria
            .iter()
            .map(|(key, value)| fold_cat_label(key, value.as_deref()))
            .collect(),
        SystemOneQuestionKind::Noul {
            true_label,
            false_label,
        } => vec![
            fold_cat_label("yes", true_label.as_deref()),
            fold_cat_label("no", false_label.as_deref()),
        ],
        SystemOneQuestionKind::Score { criteria } => criteria
            .iter()
            .enumerate()
            .map(|(index, value)| fold_cat_label(&index.to_string(), Some(&as_text(value))))
            .collect(),
    };
    Ok((keys, labels))
}

fn fold_cat_label(key: &str, description: Option<&str>) -> String {
    match description {
        Some(description) if !description.is_empty() && description != key => {
            format!("{key}: {description}")
        }
        _ => key.to_string(),
    }
}

fn is_empty_state(state: &Value) -> bool {
    match state {
        Value::Null => true,
        Value::String(text) if text.is_empty() => true,
        Value::Array(items) if items.is_empty() => true,
        Value::Object(map) if map.is_empty() => true,
        _ => false,
    }
}

fn string_list(value: &Value) -> Option<Vec<&str>> {
    let items = value.as_array()?;
    if items.is_empty() {
        return None;
    }
    items
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<&str>>>()
}

fn truthy_text(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let text = as_text(value);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

pub fn as_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => dumps(other),
    }
}

fn dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(flag) => {
            if *flag {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Number(number) => number.to_string(),
        Value::String(text) => format!("\"{}\"", escape_json_string(text)),
        Value::Array(items) => {
            let inner = items.iter().map(dumps).collect::<Vec<_>>().join(", ");
            format!("[{inner}]")
        }
        Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(key, value)| format!("\"{}\": {}", escape_json_string(key), dumps(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
    }
}

fn escape_json_string(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::{
        gliformer_system_one_options, render_state, system_one_options, SystemOneAnswer,
        SystemOneQuestion, SystemOneQuestionKind,
    };
    use serde_json::json;

    #[test]
    fn empty_states_render_as_empty_text() {
        for state in [json!(null), json!(""), json!([]), json!({})] {
            assert_eq!(render_state(&state).unwrap(), "");
        }
    }

    #[test]
    fn dict_state_renders_string_lists_as_bullets() {
        let state = json!({
            "ticket": "Checkout errors hit 12% after the 14:05 deploy.",
            "facts": ["Rolling back takes two minutes.", "No incident is open yet."]
        });
        assert_eq!(
            render_state(&state).unwrap(),
            "ticket: Checkout errors hit 12% after the 14:05 deploy.\nfacts:\n- Rolling back takes two minutes.\n- No incident is open yet."
        );
    }

    #[test]
    fn list_state_renders_dashed_items() {
        assert_eq!(
            render_state(&json!(["a", {"k": 1}])).unwrap(),
            "- a\n- {\"k\": 1}"
        );
    }

    #[test]
    fn choice_options_use_descriptions() {
        let question = SystemOneQuestion::from_value(&json!({
            "type": "choice",
            "instructions": "What next?",
            "criteria": {"rollback": "Roll back.", "wait": null}
        }))
        .unwrap();
        let (keys, labels) = system_one_options(&question).unwrap();
        assert_eq!(keys, vec!["rollback", "wait"]);
        assert_eq!(labels, vec!["Roll back.", "wait"]);
        assert_eq!(question.instructions.as_deref(), Some("What next?"));

        let (keys, labels) = gliformer_system_one_options(&question).unwrap();
        assert_eq!(keys, vec!["rollback", "wait"]);
        assert_eq!(labels, vec!["rollback: Roll back.", "wait"]);
    }

    #[test]
    fn noul_and_score_options_match_the_reference_mapping() {
        let noul = SystemOneQuestion::from_value(&json!({
            "type": "noul",
            "criteria": {"true": "Yes, now.", "false": "No, wait."}
        }))
        .unwrap();
        let (keys, labels) = system_one_options(&noul).unwrap();
        assert_eq!(keys, vec!["true", "false"]);
        assert_eq!(labels, vec!["Yes, now.", "No, wait."]);
        let (keys, labels) = gliformer_system_one_options(&noul).unwrap();
        assert_eq!(keys, vec!["true", "false"]);
        assert_eq!(labels, vec!["yes: Yes, now.", "no: No, wait."]);

        let score = SystemOneQuestion::from_value(&json!({
            "type": "score",
            "criteria": ["minor", "moderate", "major"]
        }))
        .unwrap();
        let (keys, labels) = system_one_options(&score).unwrap();
        assert_eq!(keys, vec!["0", "1", "2"]);
        assert_eq!(labels, vec!["minor", "moderate", "major"]);
        let (keys, labels) = gliformer_system_one_options(&score).unwrap();
        assert_eq!(keys, vec!["0", "1", "2"]);
        assert_eq!(labels, vec!["0: minor", "1: moderate", "2: major"]);
    }

    #[test]
    fn choice_rejects_a_single_criterion() {
        let error = SystemOneQuestion::from_value(&json!({
            "type": "choice",
            "criteria": {"only": "one"}
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("at least two criteria"));
    }

    #[test]
    fn score_answer_is_the_expected_level() {
        let question = SystemOneQuestion {
            kind: SystemOneQuestionKind::Score {
                criteria: vec![json!("minor"), json!("major")],
            },
            instructions: None,
        };
        let answer = SystemOneAnswer::from_probabilities(
            &question,
            &[("0".to_string(), 0.25), ("1".to_string(), 0.75)],
        )
        .unwrap();
        match answer {
            SystemOneAnswer::Score {
                score, confidence, ..
            } => {
                assert!((score - 0.75).abs() < 1e-6);
                assert!((confidence - 0.75).abs() < 1e-6);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn noul_answer_is_p_true() {
        let question = SystemOneQuestion {
            kind: SystemOneQuestionKind::Noul {
                true_label: None,
                false_label: None,
            },
            instructions: None,
        };
        let answer = SystemOneAnswer::from_probabilities(
            &question,
            &[("true".to_string(), 0.8), ("false".to_string(), 0.2)],
        )
        .unwrap();
        assert_eq!(answer, SystemOneAnswer::Noul { noul: 0.8 });
    }
}
