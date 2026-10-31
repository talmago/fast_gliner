use std::collections::HashSet;
use std::fs;
use std::path::Path;

use composable::*;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use ort::session::SessionInputs;

use crate::model::input::tensors::gliclass::{
    flatten_gliclass_labels, gliclass_input_names, prepare_gliclass, require_gliclass_tokens,
    GLiClassClassificationType, GLiClassInput, GLiClassLabels, GLiClassRequest, ATTENTION_MASK,
    INPUT_IDS,
};
use crate::model::output::classification::ClassificationOutput;
use crate::model::output::gliclass::{GLiClassContext, OutputsToGLiClassLogits};
use crate::model::pipeline::system_one::{
    system_one_options, SystemOneAnswer, SystemOneOutput, SystemOneRequest, SystemOneUsage,
};
use crate::model::{resolve_onnx_path, validate_required_file};
use crate::text::tokenizer::HFTokenizer;
use crate::util::json::{json_bool, json_string, json_usize};
use crate::util::ort_session::open_session;
use crate::util::profiling::StageMillis;
use crate::util::result::Result;
use crate::util::time::timed;
use serde_json::Value;

const UNI_ENCODER_ARCHITECTURE: &str = "uni-encoder";

/// Checkpoint settings from `config.json`.
#[derive(Debug, Clone)]
pub struct GLiClassConfig {
    pub architecture_type: String,
    pub prompt_first: bool,
    pub max_length: Option<usize>,
}

impl GLiClassConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    fn from_json(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)?;
        let architecture_type = json_string(&value, "architecture_type")?;
        if architecture_type != UNI_ENCODER_ARCHITECTURE {
            return Err(format!(
                "unsupported GLiClass architecture `{architecture_type}`; this runtime supports uni-encoder ONNX exports"
            )
            .into());
        }

        let max_length = json_usize(
            value.get("encoder_config").unwrap_or(&Value::Null),
            "max_position_embeddings",
        )?;
        if max_length == 0 {
            return Err("config is missing max_position_embeddings".into());
        }

        Ok(Self {
            architecture_type,
            prompt_first: json_bool(&value, "prompt_first")?,
            max_length: Some(max_length),
        })
    }
}

pub type UniEncoderMode = GLiClassPipeline<false>;
pub type V4Mode = GLiClassPipeline<true>;

pub struct GLiClass<P> {
    config: GLiClassConfig,
    model: Model,
    pipeline: P,
}

impl GLiClass<()> {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        runtime_parameters: RuntimeParameters,
    ) -> Result<InferenceMode> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_model_path = resolve_onnx_path(model_dir);
        let config_path = model_dir.join("config.json");

        validate_required_file("tokenizer", &tokenizer_path)?;
        validate_required_file("ONNX model", &onnx_model_path)?;
        validate_required_file("config", &config_path)?;

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        require_gliclass_tokens(&tokenizer)?;
        let mut config = GLiClassConfig::from_file(&config_path)?;
        if model_dir.to_string_lossy().contains("v4") {
            config.max_length = None;
            Ok(InferenceMode::V4(GLiClass::<V4Mode>::new(
                config,
                runtime_parameters,
                tokenizer,
                onnx_model_path,
            )?))
        } else {
            Ok(InferenceMode::UniEncoder(GLiClass::<UniEncoderMode>::new(
                config,
                runtime_parameters,
                tokenizer,
                onnx_model_path,
            )?))
        }
    }
}

impl<const SEP_PER_LABEL: bool> GLiClass<GLiClassPipeline<SEP_PER_LABEL>> {
    fn new(
        config: GLiClassConfig,
        runtime_parameters: RuntimeParameters,
        tokenizer: HFTokenizer,
        onnx_model_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Ok(Self {
            pipeline: GLiClassPipeline::new(tokenizer, &config),
            model: Model::new(onnx_model_path, runtime_parameters)?,
            config,
        })
    }

    pub fn get_inner_model(&self) -> &Model {
        &self.model
    }

    /// Scores `text` against a flat label list.
    ///
    /// The graph returns one logit per label. Scores are multi-label sigmoid
    /// probabilities, sorted from highest to lowest.
    pub fn classify(&self, text: &str, labels: &[String]) -> Result<ClassificationOutput> {
        self.classify_with(GLiClassRequest {
            text: text.to_string(),
            labels: GLiClassLabels::Flat(labels.to_vec()),
            examples: Vec::new(),
            prompt: None,
            classification_type: GLiClassClassificationType::MultiLabel,
        })
    }

    /// Scores one text, optionally with a label hierarchy, few-shot examples, and a task prompt.
    ///
    /// Hierarchical labels are flattened to dotted names before the forward pass.
    /// Those dotted names are the labels on the returned scores. Few-shot example
    /// labels are prompt text and do not add logits.
    ///
    /// `classification_type` selects sigmoid (multi-label, default) or softmax
    /// (single-label). Scores are sorted from highest to lowest.
    pub fn classify_with(&self, request: GLiClassRequest) -> Result<ClassificationOutput> {
        let labels = flatten_gliclass_labels(&request.labels)?;
        self.inference(GLiClassInput {
            text: request.text,
            labels,
            prompt: request.prompt,
            examples: request.examples,
            classification_type: request.classification_type,
            sort_scores: true,
            refuse_overlength: false,
            max_length: None,
        })
    }

    /// Answer typed System One questions about a state.
    ///
    /// Each question is one single-label GLiClass pass. Long inputs are refused
    /// instead of truncated.
    pub fn system_one(&self, request: SystemOneRequest) -> Result<SystemOneOutput> {
        let text = crate::model::pipeline::system_one::render_state(&request.state)?;
        let max_length = request.max_length.or(self.config.max_length);
        let mut answers = Vec::with_capacity(request.questions.len());
        let mut input_tokens = 0usize;

        for (name, question) in &request.questions {
            let (keys, labels) = system_one_options(question)?;
            let prompt = question
                .instructions
                .clone()
                .filter(|value| !value.is_empty());
            let output = self.inference(GLiClassInput {
                text: text.clone(),
                labels,
                prompt,
                examples: Vec::new(),
                classification_type: GLiClassClassificationType::SingleLabel,
                sort_scores: false,
                refuse_overlength: true,
                max_length,
            })?;
            input_tokens += output.input_tokens;
            let probabilities = keys
                .iter()
                .zip(output.scores.iter())
                .map(|(key, score)| (key.clone(), score.score))
                .collect::<Vec<_>>();
            answers.push((
                name.clone(),
                SystemOneAnswer::from_probabilities(question, &probabilities)?,
            ));
        }

        Ok(SystemOneOutput {
            answers,
            usage: SystemOneUsage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }

    fn inference(&self, input: GLiClassInput) -> Result<ClassificationOutput> {
        self.model.inference(input, &self.pipeline, &())
    }
}

/// Runtime-selected GLiClass model (legacy uni-encoder or v4).
pub enum InferenceMode {
    UniEncoder(GLiClass<UniEncoderMode>),
    V4(GLiClass<V4Mode>),
}

impl InferenceMode {
    pub fn get_inner_model(&self) -> &Model {
        match self {
            Self::UniEncoder(model) => model.get_inner_model(),
            Self::V4(model) => model.get_inner_model(),
        }
    }

    pub fn classify(&self, text: &str, labels: &[String]) -> Result<ClassificationOutput> {
        match self {
            Self::UniEncoder(model) => model.classify(text, labels),
            Self::V4(model) => model.classify(text, labels),
        }
    }

    pub fn classify_with(&self, request: GLiClassRequest) -> Result<ClassificationOutput> {
        match self {
            Self::UniEncoder(model) => model.classify_with(request),
            Self::V4(model) => model.classify_with(request),
        }
    }

    pub fn system_one(&self, request: SystemOneRequest) -> Result<SystemOneOutput> {
        match self {
            Self::UniEncoder(model) => model.system_one(request),
            Self::V4(model) => model.system_one(request),
        }
    }
}

pub struct GLiClassPipeline<const SEP_PER_LABEL: bool> {
    tokenizer: HFTokenizer,
    prompt_first: bool,
    max_length: Option<usize>,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl<const SEP_PER_LABEL: bool> GLiClassPipeline<SEP_PER_LABEL> {
    fn new(tokenizer: HFTokenizer, config: &GLiClassConfig) -> Self {
        Self {
            tokenizer,
            prompt_first: config.prompt_first,
            max_length: config.max_length,
            expected_inputs: gliclass_input_names(),
            expected_outputs: OutputsToGLiClassLogits::outputs().into_iter().collect(),
        }
    }
}

impl<'a, const SEP_PER_LABEL: bool> Pipeline<'a> for GLiClassPipeline<SEP_PER_LABEL> {
    type Input = GLiClassInput;
    type Output = ClassificationOutput;
    type Context = GLiClassContext;
    type Parameters = ();

    fn pre_processor(
        &self,
        _params: &Self::Parameters,
    ) -> impl orp::pipeline::PreProcessor<'a, Self::Input, Self::Context> {
        GLiClassToTensors::<SEP_PER_LABEL> {
            tokenizer: self.tokenizer.clone(),
            prompt_first: self.prompt_first,
            max_length: self.max_length,
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

struct GLiClassToTensors<const SEP_PER_LABEL: bool> {
    tokenizer: HFTokenizer,
    prompt_first: bool,
    max_length: Option<usize>,
}

impl<'a, const SEP_PER_LABEL: bool>
    Composable<GLiClassInput, (SessionInputs<'a, 'a>, GLiClassContext)>
    for GLiClassToTensors<SEP_PER_LABEL>
{
    fn apply(&self, input: GLiClassInput) -> Result<(SessionInputs<'a, 'a>, GLiClassContext)> {
        let prepared = prepare_gliclass(
            input,
            &self.tokenizer,
            self.prompt_first,
            SEP_PER_LABEL,
            self.max_length,
        )?;

        let session_inputs = ort::inputs! {
            INPUT_IDS => prepared.input_ids,
            ATTENTION_MASK => prepared.attention_mask,
        }?;

        Ok((
            session_inputs.into(),
            GLiClassContext {
                text: prepared.text,
                labels: prepared.labels,
                classification_type: prepared.classification_type,
                sort_scores: prepared.sort_scores,
                input_tokens: prepared.input_tokens,
            },
        ))
    }
}

enum ProbePipeline {
    UniEncoder(UniEncoderMode),
    V4(V4Mode),
}

pub struct GliclassProbe {
    session: ort::session::Session,
    pipeline: ProbePipeline,
}

impl GliclassProbe {
    pub fn open(model_dir: &Path, runtime_parameters: RuntimeParameters) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = resolve_onnx_path(model_dir);
        let config_path = model_dir.join("config.json");
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        require_gliclass_tokens(&tokenizer)?;
        let mut config = GLiClassConfig::from_file(&config_path)?;
        let is_v4 = model_dir.to_string_lossy().contains("v4");
        if is_v4 {
            config.max_length = None;
        }
        let pipeline = if is_v4 {
            ProbePipeline::V4(GLiClassPipeline::new(tokenizer, &config))
        } else {
            ProbePipeline::UniEncoder(GLiClassPipeline::new(tokenizer, &config))
        };
        Ok(Self {
            session: open_session(&onnx_path, &runtime_parameters)?,
            pipeline,
        })
    }

    pub fn time_classify(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let input = GLiClassInput {
            text: text.to_string(),
            labels: labels.to_vec(),
            prompt: None,
            examples: Vec::new(),
            classification_type: Default::default(),
            sort_scores: true,
            refuse_overlength: false,
            max_length: None,
        };
        let (pre_ms, (prepared, context)) = timed(|| match &self.pipeline {
            ProbePipeline::UniEncoder(pipeline) => pipeline.pre_processor(&()).apply(input.clone()),
            ProbePipeline::V4(pipeline) => pipeline.pre_processor(&()).apply(input.clone()),
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| match &self.pipeline {
            ProbePipeline::UniEncoder(pipeline) => {
                pipeline.post_processor(&()).apply((outputs, context))
            }
            ProbePipeline::V4(pipeline) => pipeline.post_processor(&()).apply((outputs, context)),
        })?;
        std::hint::black_box(decoded.scores.len());
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::GLiClassConfig;

    #[test]
    fn config_json_reads_prompt_order_and_position_limit() {
        let config = GLiClassConfig::from_json(
            r#"{
                "architecture_type": "uni-encoder",
                "prompt_first": true,
                "encoder_config": { "max_position_embeddings": 512 }
            }"#,
        )
        .unwrap();

        assert!(config.prompt_first);
        assert_eq!(config.max_length, Some(512));
    }

    #[test]
    fn config_json_rejects_other_architectures() {
        let error = GLiClassConfig::from_json(r#"{ "architecture_type": "bi-encoder" }"#)
            .unwrap_err()
            .to_string();

        assert!(error.contains("bi-encoder"));
    }
}
