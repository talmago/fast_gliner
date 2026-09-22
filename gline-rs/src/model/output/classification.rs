use composable::Composable;
use ort::session::SessionOutputs;

use crate::util::result::Result;

const OUTPUT_SPAN_SCORES: &str = "span_scores";

#[derive(Debug, Clone)]
pub struct ClassificationScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    pub text: String,
    pub scores: Vec<ClassificationScore>,
}

impl ClassificationOutput {
    pub fn top(&self) -> Option<&ClassificationScore> {
        self.scores.first()
    }
}

pub struct ClassificationContext {
    pub text: String,
    pub num_words: usize,
    pub labels: Vec<String>,
}

pub struct OutputsToClassification {
    max_width: usize,
}

impl OutputsToClassification {
    pub fn new(max_width: usize) -> Self {
        Self { max_width }
    }

    pub fn outputs() -> [&'static str; 1] {
        [OUTPUT_SPAN_SCORES]
    }

    fn decode(
        &self,
        outputs: SessionOutputs<'_, '_>,
        context: ClassificationContext,
    ) -> Result<ClassificationOutput> {
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
        if shape[2] != context.num_words {
            return Err("unexpected number of words in span_scores".into());
        }

        let width_limit = std::cmp::min(shape[3], self.max_width);
        let mut label_scores = Vec::with_capacity(context.labels.len());

        for label_index in 0..shape[1] {
            let mut best_score = f32::NEG_INFINITY;

            for start_word in 0..shape[2] {
                for width in 0..width_limit {
                    let end_word = start_word + width;
                    if end_word >= context.num_words {
                        continue;
                    }

                    let score = scores[[0, label_index, start_word, width]];
                    if score > best_score {
                        best_score = score;
                    }
                }
            }

            if !best_score.is_finite() {
                best_score = 0.0;
            }

            label_scores.push(ClassificationScore {
                label: context.labels[label_index].clone(),
                score: best_score,
            });
        }

        label_scores.sort_by(|left, right| right.score.total_cmp(&left.score));

        Ok(ClassificationOutput {
            text: context.text,
            scores: label_scores,
        })
    }
}

impl Composable<(SessionOutputs<'_, '_>, ClassificationContext), ClassificationOutput>
    for OutputsToClassification
{
    fn apply(
        &self,
        input: (SessionOutputs<'_, '_>, ClassificationContext),
    ) -> Result<ClassificationOutput> {
        self.decode(input.0, input.1)
    }
}
