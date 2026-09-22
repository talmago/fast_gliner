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

const DEFAULT_MAX_LENGTH: usize = 512;
const UNI_ENCODER_ARCHITECTURE: &str = "uni-encoder";

pub(crate) struct GLiClassInput {
    pub text: String,
    pub labels: Vec<String>,
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

pub(crate) fn build_uniencoder_prompt(text: &str, labels: &[String], prompt_first: bool) -> String {
    let mut labels_and_sep = String::new();
    for label in labels {
        labels_and_sep.push_str(GLICLASS_LABEL_TOKEN);
        labels_and_sep.push_str(label);
    }
    labels_and_sep.push_str(GLICLASS_SEP_TOKEN);

    if prompt_first {
        labels_and_sep.push_str(text);
        labels_and_sep
    } else {
        let mut prompt = String::with_capacity(text.len() + labels_and_sep.len());
        prompt.push_str(text);
        prompt.push_str(&labels_and_sep);
        prompt
    }
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

    let prompt = build_uniencoder_prompt(&input.text, &input.labels, prompt_first);
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
    use super::{build_uniencoder_prompt, GLiClassSettings};

    #[test]
    fn prompt_first_puts_labels_before_the_text() {
        let prompt = build_uniencoder_prompt(
            "Buy milk and eggs after work",
            &[
                "shopping".to_string(),
                "work".to_string(),
                "personal".to_string(),
            ],
            true,
        );

        assert_eq!(
            prompt,
            "<<LABEL>>shopping<<LABEL>>work<<LABEL>>personal<<SEP>>Buy milk and eggs after work"
        );
    }

    #[test]
    fn text_first_puts_labels_after_the_text() {
        let prompt = build_uniencoder_prompt("hello", &["a".to_string()], false);

        assert_eq!(prompt, "hello<<LABEL>>a<<SEP>>");
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
