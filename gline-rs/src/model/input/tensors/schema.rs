use std::collections::HashSet;

use ndarray::{Array1, Array2, Array3};

use crate::model::input::schema::{FlattenedExtractionSchema, SchemaPrefix, SpecialTokens};
use crate::text::splitter::{RegexSplitter, Splitter};
use crate::text::token::Token;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

pub(crate) const INPUT_IDS: &str = "input_ids";
pub(crate) const ATTENTION_MASK: &str = "attention_mask";
pub(crate) const TEXT_POSITIONS: &str = "text_positions";
pub(crate) const SCHEMA_POSITIONS: &str = "schema_positions";
pub(crate) const SPAN_IDX: &str = "span_idx";

pub(crate) fn schema_input_names() -> HashSet<&'static str> {
    [
        INPUT_IDS,
        ATTENTION_MASK,
        TEXT_POSITIONS,
        SCHEMA_POSITIONS,
        SPAN_IDX,
    ]
    .into_iter()
    .collect()
}

pub(crate) struct SequenceInput {
    pub sequence_index: usize,
    pub text: String,
    pub labels: Vec<String>,
    pub task: SequenceTask,
}

#[derive(Clone, Copy)]
pub(crate) enum SequenceTask {
    Entities,
    Classification,
    Extraction,
}

pub(crate) struct ExtractionInput {
    pub sequence_index: usize,
    pub text: String,
    pub flattened_schema: FlattenedExtractionSchema,
}

pub(crate) struct PreparedSequence {
    pub sequence_index: usize,
    pub text: String,
    pub tokens: Vec<Token>,
    pub labels: Vec<String>,
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub text_positions: Array1<i64>,
    pub schema_positions: Array1<i64>,
    pub span_idx: Array3<i64>,
}

pub(crate) fn prepare_sequence(
    input: SequenceInput,
    splitter: &RegexSplitter,
    tokenizer: &HFTokenizer,
    special_tokens: &SpecialTokens,
    max_width: usize,
) -> Result<PreparedSequence> {
    let tokens = splitter.split(&input.text, None)?;
    if tokens.is_empty() {
        return Err("invalid input: text contains no tokenizable words".into());
    }
    if input.labels.is_empty() {
        return Err("invalid input: labels cannot be empty".into());
    }

    let schema = match input.task {
        SequenceTask::Entities => SchemaPrefix::build_ner(&input.labels, special_tokens, splitter)?,
        SequenceTask::Classification => {
            SchemaPrefix::build_classification(&input.labels, special_tokens, splitter)?
        }
        SequenceTask::Extraction => {
            SchemaPrefix::build_extraction(&input.labels, special_tokens, splitter)?
        }
    };
    let text_start_offset = schema.pieces.len();

    let mut pieces = schema.pieces;
    pieces.extend(tokens.iter().map(|token| token.text().to_string()));

    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let encoding = tokenizer.encode(piece_refs.as_slice(), false)?;
    let first_piece_positions = first_subword_positions(encoding.get_word_ids(), pieces.len())?;
    let input_ids = encoding
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|mask| i64::from(*mask))
        .collect::<Vec<_>>();
    let text_positions = first_piece_positions
        .iter()
        .skip(text_start_offset)
        .map(|position| *position as i64)
        .collect::<Vec<_>>();
    let schema_positions = schema
        .schema_piece_indices
        .iter()
        .map(|piece_index| first_piece_positions[*piece_index] as i64)
        .collect::<Vec<_>>();
    let span_idx = build_span_idx(tokens.len(), max_width);

    Ok(PreparedSequence {
        sequence_index: input.sequence_index,
        text: input.text,
        tokens,
        labels: input.labels,
        input_ids: Array2::from_shape_vec((1, input_ids.len()), input_ids)?,
        attention_mask: Array2::from_shape_vec((1, attention_mask.len()), attention_mask)?,
        text_positions: Array1::from_vec(text_positions),
        schema_positions: Array1::from_vec(schema_positions),
        span_idx,
    })
}

/// Index of the first subword for each pretokenized piece.
///
/// `word_ids` comes from `tokenizers::Encoding::get_word_ids`. A missing piece means the
/// tokenizer emitted no tokens for that schema or text span.
pub(crate) fn first_subword_positions(
    word_ids: &[Option<u32>],
    piece_count: usize,
) -> Result<Vec<usize>> {
    let mut positions = vec![usize::MAX; piece_count];
    for (token_index, piece_index) in word_ids.iter().enumerate() {
        let Some(piece_index) = piece_index else {
            continue;
        };

        let piece_index = *piece_index as usize;
        if piece_index < positions.len() && positions[piece_index] == usize::MAX {
            positions[piece_index] = token_index;
        }
    }

    if let Some((missing_index, _)) = positions
        .iter()
        .enumerate()
        .find(|(_, position)| **position == usize::MAX)
    {
        return Err(
            format!("tokenizer produced no tokens for schema/text piece #{missing_index}").into(),
        );
    }

    Ok(positions)
}

pub fn build_span_idx(num_words: usize, max_width: usize) -> Array3<i64> {
    let mut span_idx = ndarray::Array::zeros((1, num_words * max_width, 2));

    for start_word in 0..num_words {
        let remaining_width = num_words.saturating_sub(start_word);
        let valid_width = std::cmp::min(max_width, remaining_width);

        for width in 0..valid_width {
            let flat_index = start_word * max_width + width;
            span_idx[[0, flat_index, 0]] = start_word as i64;
            span_idx[[0, flat_index, 1]] = (start_word + width) as i64;
        }
    }

    span_idx
}

#[cfg(test)]
mod tests {
    use super::{build_span_idx, first_subword_positions};

    #[test]
    fn pads_invalid_tail_widths_with_zeroes() {
        let span_idx = build_span_idx(3, 4);

        assert_eq!(span_idx.shape(), &[1, 12, 2]);
        assert_eq!(span_idx[[0, 0, 0]], 0);
        assert_eq!(span_idx[[0, 0, 1]], 0);
        assert_eq!(span_idx[[0, 1, 0]], 0);
        assert_eq!(span_idx[[0, 1, 1]], 1);
        assert_eq!(span_idx[[0, 2, 0]], 0);
        assert_eq!(span_idx[[0, 2, 1]], 2);
        assert_eq!(span_idx[[0, 3, 0]], 0);
        assert_eq!(span_idx[[0, 3, 1]], 0);
        assert_eq!(span_idx[[0, 10, 0]], 0);
        assert_eq!(span_idx[[0, 10, 1]], 0);
    }

    #[test]
    fn first_subword_positions_keep_the_first_token_of_each_piece() {
        let word_ids = [Some(0), Some(0), Some(1), None, Some(2), Some(2)];

        let positions = first_subword_positions(&word_ids, 3).unwrap();

        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn first_subword_positions_fail_when_a_piece_has_no_tokens() {
        let error = first_subword_positions(&[Some(0), Some(2)], 3).unwrap_err();

        assert!(error.to_string().contains("piece #1"));
    }
}
