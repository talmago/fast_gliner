//! Stage timings for the newer runtimes.
//!
//! These probes open the same CPU session the loaders use and call the same
//! pre- and post-processors. They do not change inference results.

use std::path::Path;
use std::time::Instant;

use composable::Composable;
use orp::pipeline::Pipeline;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::model::input::schema::{ExtractionSchema, SpecialTokens};
use crate::model::input::tensors::gliclass::{
    require_gliclass_tokens, GLiClassInput, GLiClassSettings,
};
use crate::model::input::tensors::schema::{ExtractionInput, SequenceInput, SequenceTask};
use crate::model::params::Parameters;
use crate::model::pipeline::gliclass::GLiClassPipeline;
use crate::model::pipeline::schema::{
    GLiNER2ClassificationPipeline, GLiNER2ExtractionPipeline, GLiNER2NerPipeline,
};
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

const GLINER2_MAX_WIDTH: usize = 8;

/// Wall-clock stages for one single-text ONNX call, in milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageMillis {
    pub pre_ms: f64,
    pub onnx_ms: f64,
    pub post_ms: f64,
}

/// Wall-clock stages for one GLiFormer NER request, in milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct GliformerStages {
    pub pre_ms: f64,
    pub encoder_ms: f64,
    pub between_ms: f64,
    pub head_ms: f64,
    pub post_ms: f64,
}

/// Wall-clock stages for one GLiFormer `structure` request, in milliseconds.
///
/// A nested schema is still one encoder run plus the NER head and the structuring head.
#[derive(Debug, Clone, Copy, Default)]
pub struct GliformerStructureStages {
    pub pre_ms: f64,
    pub encoder_ms: f64,
    pub between_ms: f64,
    pub ner_head_ms: f64,
    pub structure_head_ms: f64,
    pub post_ms: f64,
}

pub struct Gliner2Probe {
    session: Session,
    ner: GLiNER2NerPipeline,
    classification: GLiNER2ClassificationPipeline,
    extraction: GLiNER2ExtractionPipeline,
    params: Parameters,
}

impl Gliner2Probe {
    pub fn open(model_dir: &Path, threads: usize) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = resolve_onnx_path(model_dir);
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        let special_tokens = SpecialTokens::resolve(&tokenizer)?;
        Ok(Self {
            session: open_session(&onnx_path, threads)?,
            ner: GLiNER2NerPipeline::new(tokenizer.clone(), special_tokens.clone()),
            classification: GLiNER2ClassificationPipeline::new(
                tokenizer.clone(),
                special_tokens.clone(),
            ),
            extraction: GLiNER2ExtractionPipeline::new(tokenizer, special_tokens),
            params: Parameters::default().with_max_width(GLINER2_MAX_WIDTH),
        })
    }

    pub fn time_ner(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            self.ner.pre_processor(&self.params).apply(SequenceInput {
                sequence_index: 0,
                text: text.to_string(),
                labels: labels.to_vec(),
                task: SequenceTask::Entities,
            })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.ner
                .post_processor(&self.params)
                .apply((outputs, context))
        })?;
        let kept = decoded.spans.iter().map(|spans| spans.len()).sum::<usize>();
        std::hint::black_box(kept);
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }

    pub fn time_classify(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            self.classification
                .pre_processor(&self.params)
                .apply(SequenceInput {
                    sequence_index: 0,
                    text: text.to_string(),
                    labels: labels.to_vec(),
                    task: SequenceTask::Classification,
                })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.classification
                .post_processor(&self.params)
                .apply((outputs, context))
        })?;
        std::hint::black_box(decoded.scores.len());
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }

    /// One combined extraction call, the session `extract_with_schema` uses for entities, relations, and structures.
    pub fn time_extract(&self, text: &str, schema: &ExtractionSchema) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            let flattened = schema.flatten_labels()?;
            self.extraction
                .pre_processor(&self.params)
                .apply(ExtractionInput {
                    sequence_index: 0,
                    text: text.to_string(),
                    flattened_schema: flattened,
                })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.extraction
                .post_processor(&self.params)
                .apply((outputs, context))
        })?;
        let kept = decoded
            .fields
            .iter()
            .map(|field| field.values.len())
            .sum::<usize>();
        std::hint::black_box(kept);
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }
}

pub struct GliclassProbe {
    session: Session,
    pipeline: GLiClassPipeline,
    params: Parameters,
}

impl GliclassProbe {
    pub fn open(model_dir: &Path, threads: usize) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = resolve_onnx_path(model_dir);
        let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;
        require_gliclass_tokens(&tokenizer)?;
        let settings = GLiClassSettings::load(model_dir, None)?;
        Ok(Self {
            session: open_session(&onnx_path, threads)?,
            pipeline: GLiClassPipeline::new(tokenizer, settings.prompt_first),
            params: Parameters::default().with_max_length(Some(settings.max_length)),
        })
    }

    pub fn time_classify(&self, text: &str, labels: &[String]) -> Result<StageMillis> {
        let (pre_ms, (prepared, context)) = timed(|| {
            self.pipeline
                .pre_processor(&self.params)
                .apply(GLiClassInput {
                    text: text.to_string(),
                    labels: labels.to_vec(),
                    prompt: None,
                    examples: Vec::new(),
                })
        })?;
        let (onnx_ms, outputs) = timed(|| Ok(self.session.run(prepared)?))?;
        let (post_ms, decoded) = timed(|| {
            self.pipeline
                .post_processor(&self.params)
                .apply((outputs, context))
        })?;
        std::hint::black_box(decoded.scores.len());
        Ok(StageMillis {
            pre_ms,
            onnx_ms,
            post_ms,
        })
    }
}

fn open_session(path: &Path, threads: usize) -> Result<Session> {
    Ok(Session::builder()?
        .with_intra_threads(threads)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?)
}

fn resolve_onnx_path(model_dir: &Path) -> std::path::PathBuf {
    let nested = model_dir.join("onnx/model.onnx");
    if nested.is_file() {
        nested
    } else {
        model_dir.join("model.onnx")
    }
}

fn timed<T>(run: impl FnOnce() -> Result<T>) -> Result<(f64, T)> {
    let started = Instant::now();
    let value = run()?;
    Ok((started.elapsed().as_secs_f64() * 1000.0, value))
}
