//! Prompt construction for the GLiFormer text path.
//!
//! One task group is encoded at a time. The prompt is pretokenized in the same
//! order as the official processor: `[SEQ]`, a `[SCHEMA]` group, a final `[SEP]`,
//! then whitespace words.

use std::fs::File;
use std::path::Path;

use serde_json::Value;

use crate::text::splitter::{RegexSplitter, Splitter};
use crate::text::token::Token;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

#[derive(Debug, Clone)]
pub struct GLiFormerConfig {
    pub max_len: usize,
    pub hidden_size: usize,
    pub class_token_index: i64,
    pub parent_token_index: i64,
    pub cat_token_index: i64,
    pub rel_token_index: i64,
    pub child_token_index: i64,
    pub seq_token: String,
    pub schema_token: String,
    pub sep_token: String,
    pub entity_token: String,
    pub class_token: String,
    pub relation_token: String,
    pub field_token: String,
}

impl GLiFormerConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let value: Value = serde_json::from_reader(File::open(path)?)?;
        Ok(Self {
            max_len: json_usize(&value, "max_len")?,
            hidden_size: json_usize(&value, "hidden_size")?,
            class_token_index: json_i64(&value, "class_token_index")?,
            parent_token_index: json_i64(&value, "parent_token_index")?,
            cat_token_index: json_i64(
                value.get("classification_config").unwrap_or(&Value::Null),
                "cat_token_index",
            )?,
            rel_token_index: json_i64(
                value.get("joint_relex_config").unwrap_or(&Value::Null),
                "rel_token_index",
            )?,
            child_token_index: json_i64(
                value.get("structuring_config").unwrap_or(&Value::Null),
                "child_token_index",
            )?,
            seq_token: json_string(&value, "seq_token")?,
            schema_token: json_string(&value, "parent_token")?,
            sep_token: json_string(&value, "sep_token")?,
            entity_token: json_string(&value, "ent_token")?,
            class_token: json_string(&value, "cat_token")?,
            relation_token: json_string(&value, "rel_token")?,
            field_token: json_string(&value, "child_token")?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum GLiFormerPrompt {
    Entities(Vec<String>),
    Classes(Vec<String>),
    Relations {
        entities: Vec<String>,
        relations: Vec<String>,
    },
    Fields(Vec<String>),
}

pub struct PreparedPrompt {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub words_mask: Vec<i64>,
    pub words: Vec<Token>,
}

pub fn prepare_prompt(
    tokenizer: &HFTokenizer,
    config: &GLiFormerConfig,
    text: &str,
    prompt: &GLiFormerPrompt,
) -> Result<PreparedPrompt> {
    let words = RegexSplitter::default().split(text, None)?;
    if words.is_empty() {
        return Err("invalid input: text contains no tokenizable words".into());
    }

    let mut pieces = vec![config.seq_token.clone(), config.schema_token.clone()];
    match prompt {
        GLiFormerPrompt::Entities(labels) => push_marked(&mut pieces, &config.entity_token, labels),
        GLiFormerPrompt::Classes(labels) => push_marked(&mut pieces, &config.class_token, labels),
        GLiFormerPrompt::Relations {
            entities,
            relations,
        } => {
            push_marked(&mut pieces, &config.entity_token, entities);
            push_marked(&mut pieces, &config.relation_token, relations);
        }
        GLiFormerPrompt::Fields(labels) => push_marked(&mut pieces, &config.field_token, labels),
    }
    pieces.push(config.sep_token.clone());
    pieces.push(config.sep_token.clone());
    let prompt_words = pieces.len();
    pieces.extend(words.iter().map(|word| word.text().to_string()));

    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let encoding = tokenizer.encode(piece_refs.as_slice(), true)?;
    let limit = config.max_len.max(1);
    let length = encoding.get_ids().len().min(limit);
    let input_ids = encoding
        .get_ids()
        .iter()
        .take(length)
        .map(|id| i64::from(*id))
        .collect();
    let attention_mask = vec![1_i64; length];
    let words_mask = word_mask(encoding.get_word_ids(), prompt_words, length);

    Ok(PreparedPrompt {
        input_ids,
        attention_mask,
        words_mask,
        words,
    })
}

fn push_marked(pieces: &mut Vec<String>, marker: &str, labels: &[String]) {
    for label in labels {
        pieces.push(format!("{marker} {label}"));
    }
}

pub fn word_mask(word_ids: &[Option<u32>], skip_words: usize, length: usize) -> Vec<i64> {
    let mut mask = Vec::with_capacity(length);
    let mut previous: Option<u32> = None;
    let mut seen_words = 0_usize;
    for word_id in word_ids.iter().take(length) {
        let Some(word_id) = word_id else {
            mask.push(0);
            previous = None;
            continue;
        };
        let is_first = previous != Some(*word_id);
        if is_first {
            seen_words += 1;
        }
        if seen_words <= skip_words || !is_first {
            mask.push(0);
        } else {
            mask.push((seen_words - skip_words) as i64);
        }
        previous = Some(*word_id);
    }
    mask
}

fn json_usize(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|number| number as usize)
        .ok_or_else(|| format!("gliner_config.json is missing {key}").into())
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("gliner_config.json is missing {key}").into())
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("gliner_config.json is missing {key}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_mask_skips_prompt_and_keeps_first_subtoken() {
        let word_ids = [
            None,
            Some(0),
            Some(0),
            Some(1),
            None,
            Some(2),
            Some(3),
            None,
        ];
        assert_eq!(
            word_mask(&word_ids, 2, word_ids.len()),
            vec![0, 0, 0, 0, 0, 1, 2, 0]
        );
    }
}
