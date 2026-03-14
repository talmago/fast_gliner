use crate::text::splitter::Splitter;
use crate::util::result::Result;

use super::model::SpecialTokens;

pub struct SchemaPrefix {
    pub pieces: Vec<String>,
    pub schema_piece_indices: Vec<usize>,
}

impl SchemaPrefix {
    pub fn build_ner(
        labels: &[String],
        special_tokens: &SpecialTokens,
        splitter: &impl Splitter,
    ) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut schema_piece_indices = Vec::with_capacity(1 + labels.len());

        pieces.push("(".to_string());

        schema_piece_indices.push(pieces.len());
        pieces.push(special_tokens.prompt.clone());
        pieces.push("entities".to_string());
        pieces.push("(".to_string());

        for label in labels {
            schema_piece_indices.push(pieces.len());
            pieces.push(special_tokens.entity.clone());

            let label_tokens = splitter.split(label, None)?;
            if label_tokens.is_empty() {
                return Err(format!("invalid entity label: `{label}`").into());
            }

            pieces.extend(
                label_tokens
                    .into_iter()
                    .map(|token| token.text().to_string()),
            );
        }

        pieces.push(")".to_string());
        pieces.push(")".to_string());
        pieces.push(special_tokens.sep_text.clone());

        Ok(Self {
            pieces,
            schema_piece_indices,
        })
    }
}
