use ndarray::NdFloat;

pub fn sigmoid<T: NdFloat>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        let even = 1.0 / logits.len() as f32;
        return vec![even; logits.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::{sigmoid, softmax};

    #[test]
    fn softmax_is_a_distribution() {
        let scores = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn sigmoid_is_independent_per_logit() {
        assert!((sigmoid(0.0_f32) - 0.5).abs() < 1e-6);
    }
}
