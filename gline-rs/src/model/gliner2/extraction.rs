use std::collections::{HashMap, HashSet};

use composable::Composable;
use ort::session::SessionOutputs;

use crate::text::span::Span;
use crate::util::result::Result;

use super::decoder::{OutputsToSpans, SequenceContext};

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

#[derive(Debug, Clone)]
pub struct ExtractedValue {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

impl ExtractedValue {
    fn from_span(span: Span) -> Self {
        let (start, end) = span.offsets();
        Self {
            text: span.text().to_string(),
            label: span.class().to_string(),
            start,
            end,
            score: span.probability(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtractedField {
    pub name: String,
    pub values: Vec<ExtractedValue>,
}

#[derive(Debug, Clone)]
pub struct ExtractionOutput {
    pub text: String,
    pub fields: Vec<ExtractedField>,
}

pub struct ExtractionContext {
    pub sequence: SequenceContext,
    pub field_names: Vec<String>,
    pub label_to_field: HashMap<String, usize>,
}

pub struct OutputsToExtraction {
    span_decoder: OutputsToSpans,
}

impl OutputsToExtraction {
    pub fn new(
        threshold: f32,
        max_width: usize,
        flat_ner: bool,
        dup_label: bool,
        multi_label: bool,
    ) -> Self {
        Self {
            span_decoder: OutputsToSpans::new(
                threshold,
                max_width,
                flat_ner,
                dup_label,
                multi_label,
            ),
        }
    }

    pub fn outputs() -> [&'static str; 1] {
        OutputsToSpans::outputs()
    }

    fn decode(
        &self,
        outputs: SessionOutputs<'_, '_>,
        context: ExtractionContext,
    ) -> Result<ExtractionOutput> {
        let span_output = self.span_decoder.apply((outputs, context.sequence))?;
        let text = span_output.texts.into_iter().next().unwrap_or_default();
        let spans = span_output.spans.into_iter().next().unwrap_or_default();
        group_spans_by_field(text, spans, context.label_to_field, context.field_names)
    }
}

impl Composable<(SessionOutputs<'_, '_>, ExtractionContext), ExtractionOutput>
    for OutputsToExtraction
{
    fn apply(
        &self,
        input: (SessionOutputs<'_, '_>, ExtractionContext),
    ) -> Result<ExtractionOutput> {
        self.decode(input.0, input.1)
    }
}

fn group_spans_by_field(
    text: String,
    spans: Vec<Span>,
    label_to_field: HashMap<String, usize>,
    field_names: Vec<String>,
) -> Result<ExtractionOutput> {
    let mut fields = field_names
        .into_iter()
        .map(|name| ExtractedField {
            name,
            values: Vec::new(),
        })
        .collect::<Vec<_>>();

    for span in spans {
        let label = span.class().to_string();
        let field_index = label_to_field.get(&label).ok_or_else(|| {
            format!("unexpected label `{label}` while building extraction output")
        })?;
        fields[*field_index]
            .values
            .push(ExtractedValue::from_span(span));
    }

    Ok(ExtractionOutput { text, fields })
}
