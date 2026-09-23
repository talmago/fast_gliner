use std::collections::HashSet;
use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

pub(crate) const INPUT_IDS: &str = "input_ids";
pub(crate) const ATTENTION_MASK: &str = "attention_mask";
pub(crate) const GLICLASS_LABEL_TOKEN: &str = "<<LABEL>>";
pub(crate) const GLICLASS_SEP_TOKEN: &str = "<<SEP>>";
pub(crate) const GLICLASS_EXAMPLE_TOKEN: &str = "<<EXAMPLE>>";
const LABEL_SEPARATOR: &str = ".";

const DEFAULT_MAX_LENGTH: usize = 512;
const UNI_ENCODER_ARCHITECTURE: &str = "uni-encoder";

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
}

pub(crate) struct GLiClassInput {
    pub text: String,
    pub labels: Vec<String>,
    pub prompt: Option<String>,
    pub examples: Vec<GLiClassExample>,
}

pub(crate) struct PreparedGLiClass {
    pub text: String,
    pub labels: Vec<String>,
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
}

#[derive(Debug)]
pub(crate) struct GLiClassSettings {
    pub prompt_first: bool,
    pub max_length: usize,
}

#[derive(Deserialize)]
struct GLiClassFileConfig {
    #[serde(default)]
    architecture_type: Option<String>,
    #[serde(default)]
    prompt_first: Option<bool>,
    #[serde(default)]
    encoder_config: Option<GLiClassEncoderConfig>,
}

#[derive(Deserialize)]
struct GLiClassEncoderConfig {
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

impl GLiClassSettings {
    pub(crate) fn load(model_dir: &Path, requested_max_length: Option<usize>) -> Result<Self> {
        let path = model_dir.join("config.json");
        if !path.is_file() {
            return Self::resolve(true, None, requested_max_length);
        }

        let raw = std::fs::read_to_string(&path)?;
        Self::from_config_json(&raw, requested_max_length)
    }

    fn from_config_json(raw: &str, requested_max_length: Option<usize>) -> Result<Self> {
        let config: GLiClassFileConfig = serde_json::from_str(raw)?;
        if let Some(architecture) = config.architecture_type.as_deref() {
            if architecture != UNI_ENCODER_ARCHITECTURE {
                return Err(format!(
                    "unsupported GLiClass architecture `{architecture}`; this runtime supports uni-encoder ONNX exports"
                )
                .into());
            }
        }

        let model_max_length = config
            .encoder_config
            .and_then(|encoder| encoder.max_position_embeddings);
        Self::resolve(
            config.prompt_first.unwrap_or(true),
            model_max_length,
            requested_max_length,
        )
    }

    fn resolve(
        prompt_first: bool,
        model_max_length: Option<usize>,
        requested_max_length: Option<usize>,
    ) -> Result<Self> {
        let model_max_length = model_max_length.unwrap_or(DEFAULT_MAX_LENGTH);
        if model_max_length == 0 {
            return Err(
                "invalid GLiClass config: max_position_embeddings must be greater than zero".into(),
            );
        }

        let max_length = requested_max_length
            .unwrap_or(model_max_length)
            .min(model_max_length);
        if max_length == 0 {
            return Err("invalid input: max length must be greater than zero".into());
        }

        Ok(Self {
            prompt_first,
            max_length,
        })
    }
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

fn format_examples(examples: &[GLiClassExample]) -> Result<String> {
    if examples.is_empty() {
        return Ok(String::new());
    }

    let mut formatted = String::new();
    for example in examples {
        if example.text.is_empty() {
            return Err("invalid input: examples cannot contain an empty text".into());
        }
        if example.labels.is_empty() {
            return Err("invalid input: examples cannot contain an empty label list".into());
        }
        if example.labels.iter().any(|label| label.is_empty()) {
            return Err("invalid input: examples cannot contain an empty label".into());
        }

        formatted.push_str(GLICLASS_EXAMPLE_TOKEN);
        formatted.push_str(&example.text);
        formatted.push_str(" \nLabels:\n ");
        formatted.push_str(&example.labels.join(", "));
    }
    formatted.push_str(GLICLASS_SEP_TOKEN);
    Ok(formatted)
}

pub(crate) fn build_uniencoder_prompt(
    text: &str,
    labels: &[String],
    prompt_first: bool,
    task_prompt: Option<&str>,
    examples: &[GLiClassExample],
) -> Result<String> {
    let mut labels_and_sep = String::new();
    for label in labels {
        labels_and_sep.push_str(GLICLASS_LABEL_TOKEN);
        labels_and_sep.push_str(label);
    }
    labels_and_sep.push_str(GLICLASS_SEP_TOKEN);

    if let Some(task_prompt) = task_prompt.filter(|prompt| !prompt.is_empty()) {
        labels_and_sep.push_str(task_prompt);
    }

    let examples = format_examples(examples)?;
    let mut prompt = String::with_capacity(text.len() + labels_and_sep.len() + examples.len());
    if prompt_first {
        prompt.push_str(&labels_and_sep);
        prompt.push_str(text);
    } else {
        prompt.push_str(text);
        prompt.push_str(&labels_and_sep);
    }
    prompt.push_str(&examples);
    Ok(prompt)
}

pub(crate) fn prepare_gliclass(
    input: GLiClassInput,
    tokenizer: &HFTokenizer,
    prompt_first: bool,
    max_length: usize,
) -> Result<PreparedGLiClass> {
    if input.text.trim().is_empty() {
        return Err("invalid input: text contains no tokenizable words".into());
    }
    if input.labels.is_empty() {
        return Err("invalid input: labels cannot be empty".into());
    }
    if input.labels.iter().any(|label| label.is_empty()) {
        return Err("invalid input: labels cannot contain an empty label".into());
    }
    if max_length == 0 {
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
    if input_ids.len() > max_length {
        input_ids.truncate(max_length);
        attention_mask.truncate(max_length);
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
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_uniencoder_prompt, flatten_gliclass_labels, GLiClassExample, GLiClassLabelNode,
        GLiClassLabels, GLiClassSettings,
    };

    fn prompt(
        text: &str,
        labels: &[&str],
        prompt_first: bool,
        task_prompt: Option<&str>,
        examples: &[GLiClassExample],
    ) -> String {
        build_uniencoder_prompt(
            text,
            &labels
                .iter()
                .map(|label| (*label).to_string())
                .collect::<Vec<_>>(),
            prompt_first,
            task_prompt,
            examples,
        )
        .unwrap()
    }

    #[test]
    fn prompt_first_puts_labels_before_the_text() {
        let built = prompt(
            "Buy milk and eggs after work",
            &["shopping", "work", "personal"],
            true,
            None,
            &[],
        );

        assert_eq!(
            built,
            "<<LABEL>>shopping<<LABEL>>work<<LABEL>>personal<<SEP>>Buy milk and eggs after work"
        );
    }

    #[test]
    fn text_first_puts_labels_after_the_text() {
        assert_eq!(
            prompt("hello", &["a"], false, None, &[]),
            "hello<<LABEL>>a<<SEP>>"
        );
    }

    #[test]
    fn empty_task_prompt_leaves_the_basic_string_unchanged() {
        assert_eq!(
            prompt("hello", &["a"], true, Some(""), &[]),
            "<<LABEL>>a<<SEP>>hello"
        );
    }

    #[test]
    fn task_prompt_sits_after_the_separator() {
        assert_eq!(
            prompt(
                "The battery life is incredible",
                &["positive", "negative"],
                true,
                Some("Classify the sentiment:"),
                &[],
            ),
            "<<LABEL>>positive<<LABEL>>negative<<SEP>>Classify the sentiment:The battery life is incredible"
        );
        assert_eq!(
            prompt("hello", &["a"], false, Some("Do this:"), &[]),
            "hello<<LABEL>>a<<SEP>>Do this:"
        );
    }

    #[test]
    fn examples_are_appended_once_after_the_text() {
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

        assert_eq!(
            prompt(
                "Fast delivery",
                &["positive", "negative"],
                true,
                Some("Classify customer feedback:"),
                &examples,
            ),
            "<<LABEL>>positive<<LABEL>>negative<<SEP>>Classify customer feedback:Fast delivery<<EXAMPLE>>Love this item, great quality! \nLabels:\n positive, product<<EXAMPLE>>Customer support was unhelpful \nLabels:\n negative, service<<SEP>>"
        );
    }

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
    fn missing_config_uses_label_first_prompts_and_the_requested_length() {
        let settings = GLiClassSettings::resolve(true, None, Some(256)).unwrap();

        assert!(settings.prompt_first);
        assert_eq!(settings.max_length, 256);
    }

    #[test]
    fn encoder_position_limit_caps_the_requested_length() {
        let settings = GLiClassSettings::resolve(false, Some(512), Some(1024)).unwrap();

        assert!(!settings.prompt_first);
        assert_eq!(settings.max_length, 512);
    }

    #[test]
    fn config_json_reads_prompt_order_and_position_limit() {
        let settings = GLiClassSettings::from_config_json(
            r#"{
                "architecture_type": "uni-encoder",
                "prompt_first": true,
                "encoder_config": { "max_position_embeddings": 512 }
            }"#,
            Some(1024),
        )
        .unwrap();

        assert!(settings.prompt_first);
        assert_eq!(settings.max_length, 512);
    }

    #[test]
    fn config_json_rejects_other_architectures() {
        let error =
            GLiClassSettings::from_config_json(r#"{ "architecture_type": "bi-encoder" }"#, None)
                .unwrap_err()
                .to_string();

        assert!(error.contains("bi-encoder"));
    }
}
