use std::collections::HashSet;

use composable::*;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::schema::SpecialTokens;
use crate::model::input::tensors::schema::{
    schema_input_names, ExtractionInput, PreparedSequence, SequenceInput, SequenceTask,
    ATTENTION_MASK, INPUT_IDS, SCHEMA_POSITIONS, SPAN_IDX, TEXT_POSITIONS,
};
use crate::model::output::classification::{
    ClassificationContext, ClassificationOutput, OutputsToClassification,
};
use crate::model::output::decoded::span_scores::{OutputsToSpans, SequenceContext};
use crate::model::output::decoded::SpanOutput;
use crate::model::output::extraction::{ExtractionContext, ExtractionOutput, OutputsToExtraction};
use crate::model::params::Parameters;
use crate::text::splitter::RegexSplitter;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

pub(crate) struct GLiNER2NerPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2NerPipeline {
    pub(crate) fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToSpans::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2NerPipeline {
    type Input = SequenceInput;
    type Output = SpanOutput;
    type Context = SequenceContext;
    type Parameters = Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        SequenceToNerTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToSpans::new(
            params.threshold,
            params.max_width,
            params.flat_ner,
            params.dup_label,
            params.multi_label,
        )
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

pub(crate) struct GLiNER2ClassificationPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2ClassificationPipeline {
    pub(crate) fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToClassification::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ClassificationPipeline {
    type Input = SequenceInput;
    type Output = ClassificationOutput;
    type Context = ClassificationContext;
    type Parameters = Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        SequenceToClassificationTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToClassification::new(params.max_width)
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

pub(crate) struct GLiNER2ExtractionPipeline {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl GLiNER2ExtractionPipeline {
    pub(crate) fn new(tokenizer: HFTokenizer, special_tokens: SpecialTokens) -> Self {
        Self {
            tokenizer,
            special_tokens,
            expected_inputs: schema_input_names(),
            expected_outputs: OutputsToExtraction::outputs().into_iter().collect(),
        }
    }
}

impl<'a> Pipeline<'a> for GLiNER2ExtractionPipeline {
    type Input = ExtractionInput;
    type Output = ExtractionOutput;
    type Context = ExtractionContext;
    type Parameters = Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        ExtractionToTensors {
            splitter: RegexSplitter::default(),
            tokenizer: self.tokenizer.clone(),
            special_tokens: self.special_tokens.clone(),
            max_width: params.max_width,
        }
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl orp::pipeline::PostProcessor<'a, Self::Output, Self::Context> {
        OutputsToExtraction::new(
            params.threshold,
            params.max_width,
            params.flat_ner,
            params.dup_label,
            params.multi_label,
        )
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_inputs)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        Some(&self.expected_outputs)
    }
}

struct SequenceToNerTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, SequenceContext)>
    for SequenceToNerTensors
{
    fn apply(&self, input: SequenceInput) -> Result<(SessionInputs<'a, 'a>, SequenceContext)> {
        let prepared = prepare(
            input,
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        Ok((
            session_inputs.into(),
            SequenceContext {
                sequence_index: prepared.sequence_index,
                text: prepared.text,
                tokens: prepared.tokens,
                labels: prepared.labels,
            },
        ))
    }
}

struct SequenceToClassificationTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<SequenceInput, (SessionInputs<'a, 'a>, ClassificationContext)>
    for SequenceToClassificationTensors
{
    fn apply(
        &self,
        input: SequenceInput,
    ) -> Result<(SessionInputs<'a, 'a>, ClassificationContext)> {
        let prepared = prepare(
            input,
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        Ok((
            session_inputs.into(),
            ClassificationContext {
                text: prepared.text,
                num_words: prepared.tokens.len(),
                labels: prepared.labels,
            },
        ))
    }
}

struct ExtractionToTensors {
    splitter: RegexSplitter,
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    max_width: usize,
}

impl<'a> Composable<ExtractionInput, (SessionInputs<'a, 'a>, ExtractionContext)>
    for ExtractionToTensors
{
    fn apply(&self, input: ExtractionInput) -> Result<(SessionInputs<'a, 'a>, ExtractionContext)> {
        let labels = input.flattened_schema.labels.clone();
        let prepared = prepare(
            SequenceInput {
                sequence_index: input.sequence_index,
                text: input.text,
                labels,
                task: SequenceTask::Extraction,
            },
            &self.splitter,
            &self.tokenizer,
            &self.special_tokens,
            self.max_width,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
            TEXT_POSITIONS => prepared.text_positions,
            SCHEMA_POSITIONS => prepared.schema_positions,
            SPAN_IDX => prepared.span_idx,
        }?;

        let label_to_field = prepared
            .labels
            .iter()
            .cloned()
            .zip(input.flattened_schema.label_to_field)
            .collect();

        Ok((
            session_inputs.into(),
            ExtractionContext {
                sequence: SequenceContext {
                    sequence_index: prepared.sequence_index,
                    text: prepared.text,
                    tokens: prepared.tokens,
                    labels: prepared.labels,
                },
                field_names: input.flattened_schema.field_names,
                label_to_field,
            },
        ))
    }
}

fn prepare(
    input: SequenceInput,
    splitter: &RegexSplitter,
    tokenizer: &HFTokenizer,
    special_tokens: &SpecialTokens,
    max_width: usize,
) -> Result<PreparedSequence> {
    crate::model::input::tensors::schema::prepare_sequence(
        input,
        splitter,
        tokenizer,
        special_tokens,
        max_width,
    )
}
