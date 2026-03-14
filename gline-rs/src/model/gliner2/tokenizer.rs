use std::path::Path;

use crate::util::result::Result;

pub struct EncodedPieces {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub first_piece_positions: Vec<usize>,
}

#[derive(Clone)]
pub struct GLiNER2Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl GLiNER2Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_file(path)?,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<i64> {
        self.inner.token_to_id(token).map(i64::from)
    }

    pub fn encode_pieces(&self, pieces: &[String]) -> Result<EncodedPieces> {
        let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let encoding = self.inner.encode(piece_refs.as_slice(), false)?;

        let mut first_piece_positions = vec![usize::MAX; pieces.len()];
        for (token_index, piece_index) in encoding.get_word_ids().iter().enumerate() {
            let Some(piece_index) = piece_index else {
                continue;
            };

            let piece_index = *piece_index as usize;
            if piece_index < first_piece_positions.len()
                && first_piece_positions[piece_index] == usize::MAX
            {
                first_piece_positions[piece_index] = token_index;
            }
        }

        if let Some((missing_index, _)) = first_piece_positions
            .iter()
            .enumerate()
            .find(|(_, position)| **position == usize::MAX)
        {
            return Err(format!(
                "tokenizer produced no tokens for schema/text piece #{missing_index}"
            )
            .into());
        }

        Ok(EncodedPieces {
            input_ids: encoding.get_ids().iter().map(|id| i64::from(*id)).collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|mask| i64::from(*mask))
                .collect(),
            first_piece_positions,
        })
    }
}
