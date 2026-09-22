use std::collections::HashSet;

use composable::*;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::tensors::gliclass::{
    gliclass_input_names, prepare_gliclass, GLiClassInput, ATTENTION_MASK, INPUT_IDS,
};
use crate::model::output::classification::ClassificationOutput;
use crate::model::output::gliclass::{GLiClassContext, OutputsToGLiClassLogits};
use crate::model::params::Parameters;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

pub(crate) struct GLiClassPipeline {
    tokenizer: HFTokenizer,
    prompt_first: bool,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiClassPipeline {
    pub(crate) fn new(tokenizer: HFTokenizer, prompt_first: bool) -> Self {
        Self {
            tokenizer,
            prompt_first,
            expected_inputs: gliclass_input_names(),
            expected_outputs: OutputsToGLiClassLogits::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiClassPipeline {
    type Input = GLiClassInput;
    type Output = ClassificationOutput;
    type Context = GLiClassContext;
    type Parameters = Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        GLiClassToTensors {
            tokenizer: self.tokenizer.clone(),
            prompt_first: self.prompt_first,
            max_length: params.max_length.unwrap_or(512),
        }
    }

    fn post_processor(
        &self,
        _params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToGLiClassLogits
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

struct GLiClassToTensors {
    tokenizer: HFTokenizer,
    prompt_first: bool,
    max_length: usize,
}

impl<'a> Composable<GLiClassInput, (SessionInputs<'a, 'a>, GLiClassContext)> for GLiClassToTensors {
    fn apply(&self, input: GLiClassInput) -> Result<(SessionInputs<'a, 'a>, GLiClassContext)> {
        let prepared =
            prepare_gliclass(input, &self.tokenizer, self.prompt_first, self.max_length)?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
        }?;

        Ok((
            session_inputs.into(),
            GLiClassContext {
                text: prepared.text,
                labels: prepared.labels,
            },
        ))
    }
}
