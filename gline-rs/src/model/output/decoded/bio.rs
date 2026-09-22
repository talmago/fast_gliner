//! BIO span decoding for GLiFormer NER logits.
//!
//! Logits have shape `(words, classes, 3)` with channels start, end, and inside.
//! A span is kept when its start, end, and every inside token exceed the threshold.
//! The span score is the minimum of those probabilities and the outside-boundary
//! penalties used by the official decoder.

#[derive(Debug, Clone, PartialEq)]
pub struct BioSpan {
    pub start: usize,
    pub end: usize,
    pub class_index: usize,
    pub score: f32,
}

pub fn decode_bio(logits: &[f32], words: usize, classes: usize, threshold: f32) -> Vec<BioSpan> {
    if words == 0 || classes == 0 || logits.len() < words * classes * 3 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for class_index in 0..classes {
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for word in 0..words {
            if probability(logits, words, classes, word, class_index, 0) > threshold {
                starts.push(word);
            }
            if probability(logits, words, classes, word, class_index, 1) > threshold {
                ends.push(word);
            }
        }

        for start in starts {
            for &end in ends.iter().filter(|end| **end >= start) {
                let mut inside_ok = true;
                let mut score = probability(logits, words, classes, start, class_index, 0)
                    .min(probability(logits, words, classes, end, class_index, 1));
                for word in start..=end {
                    let inside = probability(logits, words, classes, word, class_index, 2);
                    if inside <= threshold {
                        inside_ok = false;
                        break;
                    }
                    score = score.min(inside);
                }
                if !inside_ok {
                    continue;
                }
                if start > 0 {
                    let outside =
                        1.0 - probability(logits, words, classes, start - 1, class_index, 2);
                    score = score.min(outside);
                }
                if end + 1 < words {
                    let outside =
                        1.0 - probability(logits, words, classes, end + 1, class_index, 2);
                    score = score.min(outside);
                }
                candidates.push(BioSpan {
                    start,
                    end,
                    class_index,
                    score,
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    for candidate in candidates {
        let overlaps = selected.iter().any(|kept: &BioSpan| {
            if candidate.start == kept.start && candidate.end == kept.end {
                return true;
            }
            !(candidate.start > kept.end || kept.start > candidate.end)
        });
        if !overlaps {
            selected.push(candidate);
        }
    }
    selected.sort_by_key(|span| span.start);
    selected
}

fn probability(
    logits: &[f32],
    words: usize,
    classes: usize,
    word: usize,
    class_index: usize,
    channel: usize,
) -> f32 {
    let index = ((word * classes) + class_index) * 3 + channel;
    debug_assert!(word < words);
    sigmoid(logits[index])
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logit(probability: f32) -> f32 {
        let probability = probability.clamp(1e-4, 1.0 - 1e-4);
        (probability / (1.0 - probability)).ln()
    }

    #[test]
    fn keeps_a_single_word_span_and_drops_the_overlap() {
        let words = 2;
        let classes = 1;
        let mut logits = vec![logit(0.1); words * classes * 3];
        let write = |logits: &mut [f32], word: usize, channel: usize, probability: f32| {
            logits[(word * classes) * 3 + channel] = logit(probability);
        };
        write(&mut logits, 0, 0, 0.95);
        write(&mut logits, 0, 1, 0.95);
        write(&mut logits, 0, 2, 0.95);
        write(&mut logits, 1, 0, 0.9);
        write(&mut logits, 1, 1, 0.9);
        write(&mut logits, 1, 2, 0.2);

        let spans = decode_bio(&logits, words, classes, 0.5);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 0);
        assert!(spans[0].score > 0.5);
    }
}
