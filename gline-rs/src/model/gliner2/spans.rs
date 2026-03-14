pub fn build_span_idx(num_words: usize, max_width: usize) -> ndarray::Array3<i64> {
    let mut span_idx = ndarray::Array::zeros((1, num_words * max_width, 2));

    for start_word in 0..num_words {
        let remaining_width = num_words.saturating_sub(start_word);
        let valid_width = std::cmp::min(max_width, remaining_width);

        for width in 0..valid_width {
            let flat_index = start_word * max_width + width;
            span_idx[[0, flat_index, 0]] = start_word as i64;
            span_idx[[0, flat_index, 1]] = (start_word + width) as i64;
        }
    }

    span_idx
}

#[cfg(test)]
mod tests {
    use super::build_span_idx;

    #[test]
    fn pads_invalid_tail_widths_with_zeroes() {
        let span_idx = build_span_idx(3, 4);

        assert_eq!(span_idx.shape(), &[1, 12, 2]);
        assert_eq!(span_idx[[0, 0, 0]], 0);
        assert_eq!(span_idx[[0, 0, 1]], 0);
        assert_eq!(span_idx[[0, 1, 0]], 0);
        assert_eq!(span_idx[[0, 1, 1]], 1);
        assert_eq!(span_idx[[0, 2, 0]], 0);
        assert_eq!(span_idx[[0, 2, 1]], 2);
        assert_eq!(span_idx[[0, 3, 0]], 0);
        assert_eq!(span_idx[[0, 3, 1]], 0);
        assert_eq!(span_idx[[0, 10, 0]], 0);
        assert_eq!(span_idx[[0, 10, 1]], 0);
    }
}
