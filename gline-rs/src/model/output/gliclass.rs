use composable::Composable;
use ort::session::SessionOutputs;

use crate::model::output::classification::{ClassificationOutput, ClassificationScore};
use crate::util::math::sigmoid;
use crate::util::result::Result;

const OUTPUT_LOGITS: &str = "logits";

pub struct GLiClassContext {
    pub text: String,
    pub labels: Vec<String>,
}

pub struct OutputsToGLiClassLogits;

impl OutputsToGLiClassLogits {
    pub fn outputs() -> [&'static str; 1] {
        [OUTPUT_LOGITS]
    }

    fn decode(
        outputs: SessionOutputs<'_, '_>,
        context: GLiClassContext,
    ) -> Result<ClassificationOutput> {
        let logits = outputs
            .get(OUTPUT_LOGITS)
            .ok_or("logits not found in model output")?;
        let logits = logits.try_extract_tensor::<f32>()?;
        let shape = logits.shape();

        if shape.len() != 2 {
            return Err("unexpected logits rank".into());
        }
        if shape[0] != 1 {
            return Err("GLiClass runtime expects a batch size of 1 per ONNX invocation".into());
        }
        if shape[1] < context.labels.len() {
            return Err("unexpected number of labels in logits".into());
        }

        let mut label_scores = Vec::with_capacity(context.labels.len());
        for (label_index, label) in context.labels.iter().enumerate() {
            label_scores.push(ClassificationScore {
                label: label.clone(),
                score: sigmoid(logits[[0, label_index]]),
            });
        }

        label_scores.sort_by(|left, right| right.score.total_cmp(&left.score));

        Ok(ClassificationOutput {
            text: context.text,
            scores: label_scores,
        })
    }
}

impl Composable<(SessionOutputs<'_, '_>, GLiClassContext), ClassificationOutput>
    for OutputsToGLiClassLogits
{
    fn apply(
        &self,
        input: (SessionOutputs<'_, '_>, GLiClassContext),
    ) -> Result<ClassificationOutput> {
        Self::decode(input.0, input.1)
    }
}
