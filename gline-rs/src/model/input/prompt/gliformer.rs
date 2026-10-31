//! Prompt construction for the GLiFormer text path.
//!
//! One task group is encoded at a time. The prompt is pretokenized in the same
//! order as the official processor: `[SEQ]`, a `[SCHEMA]` group, a final `[SEP]`,
//! then whitespace words.

use crate::model::pipeline::gliformer::GLiFormerConfig;
use crate::text::splitter::{RegexSplitter, Splitter};
use crate::text::token::Token;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

#[derive(Debug, Clone)]
pub enum GLiFormerPrompt {
    Entities(Vec<String>),
    /// Optional `name` is the classification group, placed after `[SCHEMA]`.
    Classes {
        labels: Vec<String>,
        name: Option<String>,
    },
    Relations {
        entities: Vec<String>,
        relations: Vec<String>,
    },
    Fields(Vec<String>),
    /// Record name, then hierarchy pieces placed after the schema token.
    Hierarchy {
        name: String,
        pieces: Vec<String>,
    },
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
    prepare_prompt_with(tokenizer, config, text, prompt, false, None)
}

pub fn prepare_prompt_with(
    tokenizer: &HFTokenizer,
    config: &GLiFormerConfig,
    text: &str,
    prompt: &GLiFormerPrompt,
    refuse_overlength: bool,
    max_length: Option<usize>,
) -> Result<PreparedPrompt> {
    let words = RegexSplitter::default().split(text, None)?;
    if words.is_empty() && !matches!(prompt, GLiFormerPrompt::Classes { .. }) {
        return Err("invalid input: text contains no tokenizable words".into());
    }

    let mut pieces = prompt_pieces(config, prompt);
    let prompt_words = pieces.len();
    pieces.extend(words.iter().map(|word| word.text().to_string()));

    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let encoding = tokenizer.encode(piece_refs.as_slice(), true)?;
    let limit = max_length.unwrap_or(config.max_len).max(1);
    let encoded_len = encoding.get_ids().len();
    let length = bounded_length(encoded_len, limit, refuse_overlength)?;
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

pub(crate) fn prompt_pieces(config: &GLiFormerConfig, prompt: &GLiFormerPrompt) -> Vec<String> {
    let mut pieces = vec![config.seq_token.clone(), config.schema_token.clone()];
    match prompt {
        GLiFormerPrompt::Entities(labels) => push_marked(&mut pieces, &config.entity_token, labels),
        GLiFormerPrompt::Classes { labels, name } => {
            if let Some(name) = name.as_deref().filter(|name| !name.is_empty()) {
                pieces.push(name.to_string());
            }
            push_marked(&mut pieces, &config.class_token, labels);
        }
        GLiFormerPrompt::Relations {
            entities,
            relations,
        } => {
            push_marked(&mut pieces, &config.entity_token, entities);
            push_marked(&mut pieces, &config.relation_token, relations);
        }
        GLiFormerPrompt::Fields(labels) => push_marked(&mut pieces, &config.field_token, labels),
        GLiFormerPrompt::Hierarchy { name, pieces: body } => {
            pieces.push(name.clone());
            pieces.extend(body.iter().cloned());
        }
    }
    pieces.push(config.sep_token.clone());
    pieces.push(config.sep_token.clone());
    pieces
}

pub(crate) fn bounded_length(
    encoded_len: usize,
    max_length: usize,
    refuse_overlength: bool,
) -> Result<usize> {
    if encoded_len > max_length {
        if refuse_overlength {
            return Err(format!(
                "input of {encoded_len} tokens exceeds the {max_length}-token limit"
            )
            .into());
        }
        return Ok(max_length);
    }
    Ok(encoded_len)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pipeline::gliformer::GLiFormerConfig;

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

    fn test_config() -> GLiFormerConfig {
        GLiFormerConfig {
            max_len: 512,
            hidden_size: 8,
            threshold: 0.5,
            class_token_index: 1,
            parent_token_index: 2,
            cat_token_index: 3,
            rel_token_index: 4,
            child_token_index: 5,
            multi_level: false,
            seq_token: "[SEQ]".into(),
            schema_token: "[SCHEMA]".into(),
            sep_token: "[SEP]".into(),
            entity_token: "[ENT]".into(),
            class_token: "[CAT]".into(),
            relation_token: "[REL]".into(),
            field_token: "[FIELD]".into(),
            nest_token: None,
            end_token: None,
        }
    }

    #[test]
    fn class_prompt_without_name_keeps_marker_order() {
        let pieces = prompt_pieces(
            &test_config(),
            &GLiFormerPrompt::Classes {
                labels: vec!["positive".into(), "negative".into()],
                name: None,
            },
        );
        assert_eq!(
            pieces,
            vec![
                "[SEQ]",
                "[SCHEMA]",
                "[CAT] positive",
                "[CAT] negative",
                "[SEP]",
                "[SEP]",
            ]
        );
    }

    #[test]
    fn class_prompt_inserts_group_name_after_schema() {
        let pieces = prompt_pieces(
            &test_config(),
            &GLiFormerPrompt::Classes {
                labels: vec!["rollback".into(), "wait".into()],
                name: Some("What should the on-call engineer do?".into()),
            },
        );
        assert_eq!(
            pieces,
            vec![
                "[SEQ]",
                "[SCHEMA]",
                "What should the on-call engineer do?",
                "[CAT] rollback",
                "[CAT] wait",
                "[SEP]",
                "[SEP]",
            ]
        );
    }

    #[test]
    fn class_prompt_schema_exists_without_text_words() {
        let pieces = prompt_pieces(
            &test_config(),
            &GLiFormerPrompt::Classes {
                labels: vec!["yes".into(), "no".into()],
                name: Some("Is this urgent?".into()),
            },
        );
        assert!(pieces.starts_with(&["[SEQ]".to_string(), "[SCHEMA]".to_string()]));
        assert!(pieces.ends_with(&["[SEP]".to_string(), "[SEP]".to_string()]));
    }

    #[test]
    fn refuse_overlength_errors_instead_of_clipping() {
        let err = bounded_length(20, 8, true).unwrap_err();
        assert!(err.to_string().contains("exceeds the 8-token limit"));
    }

    #[test]
    fn truncate_overlength_clips_to_the_limit() {
        assert_eq!(bounded_length(20, 8, false).unwrap(), 8);
        assert_eq!(bounded_length(4, 8, true).unwrap(), 4);
    }
}
