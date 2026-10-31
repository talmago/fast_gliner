//! The core of `gline-rs`: everything about pre-/post-processing, and inferencing

pub mod input;
pub mod output;
pub mod pipeline;

pub use crate::util::profiling::StageMillis;
pub use input::schema::{ExtractionFieldSchema, ExtractionSchema};
pub use input::tensors::gliclass::{
    GLiClassClassificationType, GLiClassExample, GLiClassLabelNode, GLiClassLabels, GLiClassRequest,
};
pub use output::classification::{ClassificationOutput, ClassificationScore};
pub use output::extraction::{ExtractedField, ExtractedValue, ExtractionOutput};
pub use output::gliclass::{nest_gliclass_scores, HierarchicalValue};
pub use pipeline::gliclass::{
    GLiClass, GLiClassConfig, GliclassProbe, InferenceMode as GLiClassMode,
};
pub use pipeline::gliformer::{
    GLiFormer, GLiFormerConfig, GliformerStages, GliformerStructureStages,
};
pub use pipeline::gliner::{GLiNER, GLiNERConfig, InferenceMode};
pub use pipeline::gliner2::{GLiNER2, GLiNER2Config, Gliner2Probe};
pub use pipeline::gliner2_multi::{
    GLiNER2MultiPipeline, GLiNER2MultiPipelineOutput, GLiNER2MultiPipelineSchema,
};
pub use pipeline::structure::{SchemaNode, StructureSchema};
pub use pipeline::system_one::{
    SystemOneAnswer, SystemOneOutput, SystemOneQuestion, SystemOneQuestionKind, SystemOneRequest,
    SystemOneUsage,
};

use std::path::{Path, PathBuf};

use crate::util::result::Result;

pub(crate) fn resolve_onnx_path(model_dir: &Path) -> PathBuf {
    let nested = model_dir.join("onnx/model.onnx");
    if nested.is_file() {
        nested
    } else {
        model_dir.join("model.onnx")
    }
}

pub(crate) fn validate_required_file(component: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing required {component} file: {}", path.display()).into())
    }
}
