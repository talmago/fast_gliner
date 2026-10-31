use std::collections::HashSet;

use ndarray::Array2;

use crate::model::input::prompt::gliclass::{
    build_uniencoder_prompt, GLICLASS_EXAMPLE_TOKEN, GLICLASS_LABEL_TOKEN, GLICLASS_SEP_TOKEN,
};
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

pub(crate) const INPUT_IDS: &str = "input_ids";
pub(crate) const ATTENTION_MASK: &str = "attention_mask";
const LABEL_SEPARATOR: &str = ".";

/// How GLiClass turns logits into scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GLiClassClassificationType {
    /// Independent sigmoid per label. This is the default and matches v1/v2.
    #[default]
    MultiLabel,
    /// Softmax over the label set. Used for mutually exclusive decisions.
    SingleLabel,
}

impl GLiClassClassificationType {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "multi" | "multi-label" | "multi_label" => Ok(Self::MultiLabel),
            "single" | "single-label" | "single_label" => Ok(Self::SingleLabel),
            other => Err(format!(
                "unsupported classification_type `{other}`; use 'multi-label' or 'single-label'"
            )
            .into()),
        }
    }
}

/// One in-context example. Its labels are prompt text, not extra logits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLiClassExample {
    pub text: String,
    pub labels: Vec<String>,
}

/// One node of a hierarchical label set.
///
/// A group is a mapping from a name to a child node. Leaves are the label
/// strings scored by the model. A single leaf is a string value in that mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GLiClassLabelNode {
    Group(Vec<(String, GLiClassLabelNode)>),
    Leaves(Vec<String>),
    Leaf(String),
}

/// Labels passed to classification, either flat or hierarchical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GLiClassLabels {
    Flat(Vec<String>),
    Hierarchical(GLiClassLabelNode),
}

/// A GLiClass classification call.
///
/// `prompt` is a task description inserted after the label separator.
/// `examples` are few-shot blocks appended after the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GLiClassRequest {
    pub text: String,
    pub labels: GLiClassLabels,
    pub examples: Vec<GLiClassExample>,
    pub prompt: Option<String>,
    pub classification_type: GLiClassClassificationType,
}

impl Default for GLiClassRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            labels: GLiClassLabels::Flat(Vec::new()),
            examples: Vec::new(),
            prompt: None,
            classification_type: GLiClassClassificationType::MultiLabel,
        }
    }
}

#[derive(Clone)]
pub struct GLiClassInput {
    pub text: String,
    pub labels: Vec<String>,
    pub prompt: Option<String>,
    pub examples: Vec<GLiClassExample>,
    pub classification_type: GLiClassClassificationType,
    pub sort_scores: bool,
    pub refuse_overlength: bool,
    pub max_length: Option<usize>,
}

pub(crate) struct PreparedGLiClass {
    pub text: String,
    pub labels: Vec<String>,
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub classification_type: GLiClassClassificationType,
    pub sort_scores: bool,
    pub input_tokens: usize,
}

pub(crate) fn gliclass_input_names() -> HashSet<&'static str> {
    [INPUT_IDS, ATTENTION_MASK].into_iter().collect()
}

pub(crate) fn require_gliclass_tokens(tokenizer: &HFTokenizer) -> Result<()> {
    for token in [GLICLASS_LABEL_TOKEN, GLICLASS_SEP_TOKEN] {
        if tokenizer.token_to_id(token).is_none() {
            return Err(format!(
                "missing required GLiClass token in tokenizer vocabulary: {token}"
            )
            .into());
        }
    }
    Ok(())
}

pub(crate) fn flatten_gliclass_labels(labels: &GLiClassLabels) -> Result<Vec<String>> {
    let mut flattened = Vec::new();
    match labels {
        GLiClassLabels::Flat(labels) => {
            if labels.is_empty() {
                return Err("invalid input: labels cannot be empty".into());
            }
            for label in labels {
                if label.is_empty() {
                    return Err("invalid input: labels cannot contain an empty label".into());
                }
                flattened.push(label.clone());
            }
        }
        GLiClassLabels::Hierarchical(node) => {
            flatten_label_node(node, "", &mut flattened)?;
        }
    }

    if flattened.is_empty() {
        return Err("invalid input: labels cannot be empty".into());
    }
    Ok(flattened)
}

fn flatten_label_node(
    node: &GLiClassLabelNode,
    prefix: &str,
    flattened: &mut Vec<String>,
) -> Result<()> {
    match node {
        GLiClassLabelNode::Leaves(labels) => {
            if labels.is_empty() {
                return Err("invalid input: labels cannot be empty".into());
            }
            for label in labels {
                push_flattened_label(prefix, label, flattened)?;
            }
        }
        GLiClassLabelNode::Leaf(label) => {
            push_flattened_label(prefix, label, flattened)?;
        }
        GLiClassLabelNode::Group(entries) => {
            if entries.is_empty() {
                return Err("invalid input: labels cannot be empty".into());
            }
            for (key, child) in entries {
                if key.is_empty() {
                    return Err("invalid input: labels cannot contain an empty label".into());
                }
                let child_prefix = join_label(prefix, key);
                flatten_label_node(child, &child_prefix, flattened)?;
            }
        }
    }
    Ok(())
}

fn push_flattened_label(prefix: &str, label: &str, flattened: &mut Vec<String>) -> Result<()> {
    if label.is_empty() {
        return Err("invalid input: labels cannot contain an empty label".into());
    }
    flattened.push(join_label(prefix, label));
    Ok(())
}

fn join_label(prefix: &str, label: &str) -> String {
    if prefix.is_empty() {
        label.to_string()
    } else {
        format!("{prefix}{LABEL_SEPARATOR}{label}")
    }
}

pub(crate) fn prepare_gliclass(
    input: GLiClassInput,
    tokenizer: &HFTokenizer,
    prompt_first: bool,
    sep_per_label: bool,
    default_max_length: Option<usize>,
) -> Result<PreparedGLiClass> {
    if input.labels.is_empty() {
        return Err("invalid input: labels cannot be empty".into());
    }
    if input.labels.iter().any(|label| label.is_empty()) {
        return Err("invalid input: labels cannot contain an empty label".into());
    }
    let max_length = input.max_length.or(default_max_length);
    if max_length == Some(0) {
        return Err("invalid input: max length must be greater than zero".into());
    }
    if !input.examples.is_empty() && tokenizer.token_to_id(GLICLASS_EXAMPLE_TOKEN).is_none() {
        return Err(format!(
            "missing required GLiClass token in tokenizer vocabulary: {GLICLASS_EXAMPLE_TOKEN}"
        )
        .into());
    }

    let prompt = build_uniencoder_prompt(
        &input.text,
        &input.labels,
        prompt_first,
        input.prompt.as_deref(),
        &input.examples,
        sep_per_label,
    )?;
    let encoding = tokenizer.encode(prompt.as_str(), true)?;
    let mut input_ids = encoding
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    let mut attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|mask| i64::from(*mask))
        .collect::<Vec<_>>();

    if input_ids.len() != attention_mask.len() {
        return Err("tokenizer returned mismatched input ids and attention mask".into());
    }
    let input_tokens = input_ids.len();
    if let Some(max_length) = max_length {
        if input_tokens > max_length {
            if input.refuse_overlength {
                return Err(format!(
                    "input of {input_tokens} tokens exceeds the {max_length}-token limit"
                )
                .into());
            }
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
        }
    }
    if input_ids.is_empty() {
        return Err("invalid input: text contains no tokenizable words".into());
    }

    let length = input_ids.len();
    Ok(PreparedGLiClass {
        text: input.text,
        labels: input.labels,
        input_ids: Array2::from_shape_vec((1, length), input_ids)?,
        attention_mask: Array2::from_shape_vec((1, length), attention_mask)?,
        classification_type: input.classification_type,
        sort_scores: input.sort_scores,
        input_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_gliclass_labels, prepare_gliclass, GLiClassClassificationType, GLiClassInput,
        GLiClassLabelNode, GLiClassLabels,
    };
    use crate::text::tokenizer::HFTokenizer;

    #[test]
    fn hierarchical_labels_flatten_in_walk_order() {
        let labels = GLiClassLabels::Hierarchical(GLiClassLabelNode::Group(vec![
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
                GLiClassLabelNode::Group(vec![(
                    "product".to_string(),
                    GLiClassLabelNode::Leaf("phone".to_string()),
                )]),
            ),
        ]));

        assert_eq!(
            flatten_gliclass_labels(&labels).unwrap(),
            vec![
                "sentiment.positive",
                "sentiment.negative",
                "sentiment.neutral",
                "topic.product.phone",
            ]
        );
    }

    #[test]
    fn flat_labels_reject_an_empty_label() {
        let error =
            flatten_gliclass_labels(&GLiClassLabels::Flat(vec!["ok".to_string(), String::new()]))
                .unwrap_err()
                .to_string();

        assert!(error.contains("empty label"));
    }

    #[test]
    fn classification_type_parses_official_aliases() {
        assert_eq!(
            GLiClassClassificationType::parse("single-label").unwrap(),
            GLiClassClassificationType::SingleLabel
        );
        assert_eq!(
            GLiClassClassificationType::parse("multi_label").unwrap(),
            GLiClassClassificationType::MultiLabel
        );
        assert!(GLiClassClassificationType::parse("ranking")
            .unwrap_err()
            .to_string()
            .contains("unsupported classification_type"));
    }

    #[test]
    fn prepare_refuses_overlength_when_requested() {
        let tokenizer = HFTokenizer::from_file("../models/gliclass-small-v1.0/tokenizer.json")
            .expect("local gliclass tokenizer");
        let prepared = prepare_gliclass(
            GLiClassInput {
                text: "hello world".to_string(),
                labels: vec!["a".to_string(), "b".to_string()],
                prompt: None,
                examples: Vec::new(),
                classification_type: GLiClassClassificationType::SingleLabel,
                sort_scores: false,
                refuse_overlength: true,
                max_length: Some(1),
            },
            &tokenizer,
            true,
            false,
            None,
        );
        let error = match prepared {
            Ok(_) => panic!("expected overlength error"),
            Err(err) => err.to_string(),
        };

        assert!(error.contains("exceeds the 1-token limit"));
        assert!(error.contains("input of"));
    }
}
