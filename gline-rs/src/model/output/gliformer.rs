use crate::model::input::tensors::gliclass::GLiClassClassificationType;
use crate::model::output::classification::ClassificationOutput;
use crate::model::output::gliclass::scores_from_logits;

pub(crate) fn classification_from_logits(
    text: String,
    logits: &[f32],
    labels: &[String],
    classification_type: GLiClassClassificationType,
    sort_scores: bool,
    input_tokens: usize,
) -> ClassificationOutput {
    ClassificationOutput {
        text,
        scores: scores_from_logits(logits, labels, classification_type, sort_scores),
        input_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::classification_from_logits;
    use crate::model::input::tensors::gliclass::GLiClassClassificationType;

    #[test]
    fn softmax_stays_in_input_order_when_unsorted() {
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let output = classification_from_logits(
            "state".to_string(),
            &[1.0, 2.0, 3.0],
            &labels,
            GLiClassClassificationType::SingleLabel,
            false,
            12,
        );
        let total: f32 = output.scores.iter().map(|score| score.score).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(
            output
                .scores
                .iter()
                .map(|score| score.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(output.scores[2].score > output.scores[1].score);
        assert_eq!(output.input_tokens, 12);
    }
}
