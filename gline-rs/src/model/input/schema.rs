use std::collections::{HashMap, HashSet};

use crate::text::splitter::Splitter;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

#[derive(Clone)]
pub struct SpecialTokens {
    pub prompt: String,
    pub classification: String,
    pub entity: String,
    pub relation: String,
    pub label: String,
    pub mask: String,
    pub sep_struct: String,
    pub sep_text: String,
    pub description: String,
    pub example: String,
    pub output: String,
    pub ids: HashMap<String, i64>,
}

impl SpecialTokens {
    pub(crate) fn resolve(tokenizer: &HFTokenizer) -> Result<Self> {
        let mut ids = HashMap::new();
        for token in [
            "[P]",
            "[C]",
            "[E]",
            "[R]",
            "[L]",
            "[MASK]",
            "[SEP_STRUCT]",
            "[SEP_TEXT]",
            "[DESCRIPTION]",
            "[EXAMPLE]",
            "[OUTPUT]",
        ] {
            let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
                format!("required GLiNER2 special token `{token}` not found in tokenizer.json")
            })?;
            ids.insert(token.to_string(), i64::from(token_id));
        }

        Ok(Self {
            prompt: "[P]".to_string(),
            classification: "[C]".to_string(),
            entity: "[E]".to_string(),
            relation: "[R]".to_string(),
            label: "[L]".to_string(),
            mask: "[MASK]".to_string(),
            sep_struct: "[SEP_STRUCT]".to_string(),
            sep_text: "[SEP_TEXT]".to_string(),
            description: "[DESCRIPTION]".to_string(),
            example: "[EXAMPLE]".to_string(),
            output: "[OUTPUT]".to_string(),
            ids,
        })
    }
}

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
        Self::build_task("entities", labels, special_tokens, splitter)
    }

    pub fn build_classification(
        labels: &[String],
        special_tokens: &SpecialTokens,
        splitter: &impl Splitter,
    ) -> Result<Self> {
        Self::build_task("classification", labels, special_tokens, splitter)
    }

    pub fn build_extraction(
        labels: &[String],
        special_tokens: &SpecialTokens,
        splitter: &impl Splitter,
    ) -> Result<Self> {
        Self::build_task("extraction", labels, special_tokens, splitter)
    }

    pub fn build_task(
        task_name: &str,
        labels: &[String],
        special_tokens: &SpecialTokens,
        splitter: &impl Splitter,
    ) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut schema_piece_indices = Vec::with_capacity(1 + labels.len());

        pieces.push("(".to_string());

        schema_piece_indices.push(pieces.len());
        pieces.push(special_tokens.prompt.clone());
        pieces.push(task_name.to_string());
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

#[derive(Debug, Clone)]
pub struct ExtractionFieldSchema {
    pub name: String,
    pub labels: Vec<String>,
}

impl ExtractionFieldSchema {
    pub fn new(name: impl Into<String>, labels: Vec<String>) -> Self {
        Self {
            name: name.into(),
            labels,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExtractionSchema {
    fields: Vec<ExtractionFieldSchema>,
}

impl ExtractionSchema {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn from_fields(fields: Vec<ExtractionFieldSchema>) -> Self {
        Self { fields }
    }

    pub fn push(&mut self, field: ExtractionFieldSchema) {
        self.fields.push(field);
    }

    pub fn fields(&self) -> &[ExtractionFieldSchema] {
        &self.fields
    }

    pub fn flatten_labels(&self) -> Result<FlattenedExtractionSchema> {
        if self.fields.is_empty() {
            return Err("invalid extraction schema: must contain at least one field".into());
        }

        let mut field_names = Vec::with_capacity(self.fields.len());
        let mut labels = Vec::new();
        let mut label_to_field = Vec::new();
        let mut seen_labels = HashSet::new();

        for (field_index, field) in self.fields.iter().enumerate() {
            if field.name.trim().is_empty() {
                return Err("invalid extraction schema: field name cannot be empty".into());
            }
            if field.labels.is_empty() {
                return Err(format!(
                    "invalid extraction schema: field `{}` has no labels",
                    field.name
                )
                .into());
            }

            field_names.push(field.name.clone());

            for label in &field.labels {
                if label.trim().is_empty() {
                    return Err(format!(
                        "invalid extraction schema: field `{}` contains an empty label",
                        field.name
                    )
                    .into());
                }
                if !seen_labels.insert(label.clone()) {
                    return Err(format!(
                        "invalid extraction schema: duplicate label `{label}` across fields is not supported"
                    )
                    .into());
                }

                labels.push(label.clone());
                label_to_field.push(field_index);
            }
        }

        Ok(FlattenedExtractionSchema {
            field_names,
            labels,
            label_to_field,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FlattenedExtractionSchema {
    pub field_names: Vec<String>,
    pub labels: Vec<String>,
    pub label_to_field: Vec<usize>,
}
