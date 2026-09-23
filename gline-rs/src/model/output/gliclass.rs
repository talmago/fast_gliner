use std::collections::HashMap;

use composable::Composable;
use ort::session::SessionOutputs;

use crate::model::input::tensors::gliclass::{GLiClassLabelNode, GLiClassLabels};
use crate::model::output::classification::{ClassificationOutput, ClassificationScore};
use crate::util::math::sigmoid;
use crate::util::result::Result;

const LABEL_SEPARATOR: &str = ".";

/// Nested scores matching the caller's label tree.
///
/// A group preserves the input key order. A score is one flattened leaf.
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalValue {
    Group(Vec<(String, HierarchicalValue)>),
    Score(f32),
}

/// Rebuilds `scores` into the shape of `labels`.
///
/// Flat labels become one map of label to score, in input order. A hierarchy
/// keeps its groups, and each leaf is filled from the flattened label name.
/// A label missing from `scores` is `0.0`.
pub fn nest_gliclass_scores(
    labels: &GLiClassLabels,
    scores: &[ClassificationScore],
) -> HierarchicalValue {
    let lookup = scores
        .iter()
        .map(|score| (score.label.as_str(), score.score))
        .collect::<HashMap<_, _>>();

    match labels {
        GLiClassLabels::Flat(labels) => leaves_to_group(labels, "", &lookup),
        GLiClassLabels::Hierarchical(node) => nest_node(node, "", &lookup),
    }
}

fn nest_node(
    node: &GLiClassLabelNode,
    prefix: &str,
    lookup: &HashMap<&str, f32>,
) -> HierarchicalValue {
    match node {
        GLiClassLabelNode::Leaves(labels) => leaves_to_group(labels, prefix, lookup),
        GLiClassLabelNode::Leaf(label) => {
            leaves_to_group(std::slice::from_ref(label), prefix, lookup)
        }
        GLiClassLabelNode::Group(entries) => {
            let nested = entries
                .iter()
                .map(|(key, child)| {
                    let child_prefix = join_label(prefix, key);
                    (key.clone(), nest_node(child, &child_prefix, lookup))
                })
                .collect();
            HierarchicalValue::Group(nested)
        }
    }
}

fn leaves_to_group(
    labels: &[String],
    prefix: &str,
    lookup: &HashMap<&str, f32>,
) -> HierarchicalValue {
    let entries = labels
        .iter()
        .map(|label| {
            let full_label = join_label(prefix, label);
            let score = lookup.get(full_label.as_str()).copied().unwrap_or(0.0);
            (label.clone(), HierarchicalValue::Score(score))
        })
        .collect();
    HierarchicalValue::Group(entries)
}

fn join_label(prefix: &str, label: &str) -> String {
    if prefix.is_empty() {
        label.to_string()
    } else {
        format!("{prefix}{LABEL_SEPARATOR}{label}")
    }
}

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

#[cfg(test)]
mod tests {
    use super::{nest_gliclass_scores, HierarchicalValue};
    use crate::model::input::tensors::gliclass::{GLiClassLabelNode, GLiClassLabels};
    use crate::model::output::classification::ClassificationScore;

    fn score(label: &str, value: f32) -> ClassificationScore {
        ClassificationScore {
            label: label.to_string(),
            score: value,
        }
    }

    #[test]
    fn flat_labels_nest_in_input_order() {
        let labels = GLiClassLabels::Flat(vec!["shopping".to_string(), "work".to_string()]);
        let nested = nest_gliclass_scores(&labels, &[score("work", 0.2), score("shopping", 0.9)]);

        assert_eq!(
            nested,
            HierarchicalValue::Group(vec![
                ("shopping".to_string(), HierarchicalValue::Score(0.9)),
                ("work".to_string(), HierarchicalValue::Score(0.2)),
            ])
        );
    }

    #[test]
    fn hierarchical_labels_nest_onto_dotted_leaves() {
        let labels = GLiClassLabels::Hierarchical(GLiClassLabelNode::Group(vec![
            (
                "sentiment".to_string(),
                GLiClassLabelNode::Leaves(vec!["positive".to_string(), "negative".to_string()]),
            ),
            (
                "topic".to_string(),
                GLiClassLabelNode::Leaf("product".to_string()),
            ),
        ]));

        let nested = nest_gliclass_scores(
            &labels,
            &[
                score("sentiment.positive", 0.8),
                score("topic.product", 0.4),
            ],
        );

        assert_eq!(
            nested,
            HierarchicalValue::Group(vec![
                (
                    "sentiment".to_string(),
                    HierarchicalValue::Group(vec![
                        ("positive".to_string(), HierarchicalValue::Score(0.8)),
                        ("negative".to_string(), HierarchicalValue::Score(0.0)),
                    ])
                ),
                (
                    "topic".to_string(),
                    HierarchicalValue::Group(vec![(
                        "product".to_string(),
                        HierarchicalValue::Score(0.4)
                    )])
                ),
            ])
        );
    }
}
