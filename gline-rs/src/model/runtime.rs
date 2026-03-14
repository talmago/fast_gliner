use orp::model::Model;

use crate::model::{input, output, pipeline, GLiNER};
use crate::util::result::Result;

/// Runtime-selected GLiNER model (span or token mode).
///
/// GLiNER2 runtime variants will likely be added here later.
pub enum InferenceMode {
    Span(GLiNER<pipeline::span::SpanMode>),
    Token(GLiNER<pipeline::token::TokenMode>),
}

impl InferenceMode {
    pub fn get_inner_model(&self) -> &Model {
        match self {
            Self::Span(model) => model.get_inner_model(),
            Self::Token(model) => model.get_inner_model(),
        }
    }

    pub fn inference<'a>(
        &'a self,
        input: input::text::TextInput,
    ) -> Result<output::decoded::SpanOutput> {
        match self {
            Self::Span(model) => model.inference(input),
            Self::Token(model) => model.inference(input),
        }
    }
}
