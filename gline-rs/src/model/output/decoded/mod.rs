//! Span decoding steps

pub mod span;
pub mod token;
pub mod token_flat;
pub mod sort;
pub mod greedy;

use crate::text::span::Span;

/// Represents the final output of the post-processing steps, as a list of spans for each input sequence
#[derive(Debug)]
pub struct SpanOutput {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
    pub spans: Vec<Vec<Span>>,
}


impl SpanOutput {
    pub fn new(texts: Vec<String>, entities: Vec<String>, spans: Vec<Vec<Span>>) -> Self {
        Self {
            texts, entities, spans
        }
    }
}


impl std::fmt::Display for SpanOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for spans in &self.spans {
            for span in spans {
                writeln!(f, "{:3} | {:15} | {:10} | {:.1}%", span.sequence(), span.text(), span.class(), span.probability() * 100.0)?;
            }
        }
        Ok(())
    }
}