use composable::Composable;
use ort::session::SessionOutputs;

use crate::model::output::decoded::{greedy::GreedySearch, sort::SpanSort, SpanOutput};
use crate::text::{span::Span, token::Token};
use crate::util::result::Result;

const OUTPUT_SPAN_SCORES: &str = "span_scores";

pub struct SequenceContext {
    pub sequence_index: usize,
    pub text: String,
    pub tokens: Vec<Token>,
    pub labels: Vec<String>,
}

pub struct OutputsToSpans {
    threshold: f32,
    max_width: usize,
    flat_ner: bool,
    dup_label: bool,
    multi_label: bool,
}

impl OutputsToSpans {
    pub fn new(
        threshold: f32,
        max_width: usize,
        flat_ner: bool,
        dup_label: bool,
        multi_label: bool,
    ) -> Self {
        Self {
            threshold,
            max_width,
            flat_ner,
            dup_label,
            multi_label,
        }
    }

    pub fn outputs() -> [&'static str; 1] {
        [OUTPUT_SPAN_SCORES]
    }

    fn decode(
        &self,
        outputs: SessionOutputs<'_, '_>,
        context: SequenceContext,
    ) -> Result<SpanOutput> {
        let scores = outputs
            .get(OUTPUT_SPAN_SCORES)
            .ok_or("span_scores not found in model output")?;
        let scores = scores.try_extract_tensor::<f32>()?;
        let shape = scores.shape();

        if shape.len() != 4 {
            return Err("unexpected span_scores rank".into());
        }
        if shape[0] != 1 {
            return Err("GLiNER2 runtime expects a batch size of 1 per ONNX invocation".into());
        }
        if shape[1] != context.labels.len() {
            return Err("unexpected number of labels in span_scores".into());
        }
        if shape[2] != context.tokens.len() {
            return Err("unexpected number of words in span_scores".into());
        }

        let mut spans = Vec::new();
        let width_limit = std::cmp::min(shape[3], self.max_width);

        for label_index in 0..shape[1] {
            for start_word in 0..shape[2] {
                for width in 0..width_limit {
                    let end_word = start_word + width;
                    if end_word >= context.tokens.len() {
                        continue;
                    }

                    let score = scores[[0, label_index, start_word, width]];
                    if score < self.threshold {
                        continue;
                    }

                    spans.push(make_span(
                        &context,
                        start_word,
                        end_word,
                        label_index,
                        score,
                    )?);
                }
            }
        }

        let output = SpanOutput::new(vec![context.text], context.labels, vec![spans]);
        let output = SpanSort::default().apply(output)?;
        GreedySearch::new(self.flat_ner, self.dup_label, self.multi_label).apply(output)
    }
}

impl Composable<(SessionOutputs<'_, '_>, SequenceContext), SpanOutput> for OutputsToSpans {
    fn apply(&self, input: (SessionOutputs<'_, '_>, SequenceContext)) -> Result<SpanOutput> {
        self.decode(input.0, input.1)
    }
}

fn make_span(
    context: &SequenceContext,
    start_word: usize,
    end_word: usize,
    label_index: usize,
    score: f32,
) -> Result<Span> {
    let start = context
        .tokens
        .get(start_word)
        .ok_or("invalid start word index during GLiNER2 decoding")?
        .start();
    let end = context
        .tokens
        .get(end_word)
        .ok_or("invalid end word index during GLiNER2 decoding")?
        .end();
    let class = context
        .labels
        .get(label_index)
        .ok_or("invalid label index during GLiNER2 decoding")?
        .to_string();

    Ok(Span::new(
        context.sequence_index,
        start,
        end,
        context.text[start..end].to_string(),
        class,
        score,
    ))
}
