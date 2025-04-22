#[derive(Debug, Clone)]
pub struct Span {
    /// Input index in the batch
    sequence: usize, 
    /// Start offset
    start: usize,
    /// End offset
    end: usize,
    /// Entity text
    text: String,
    /// Entity class
    class: String,
    /// Probability
    probability: f32,
}

impl Span {
    pub fn new(sequence: usize, start: usize, end: usize, text: String, class: String, probability: f32) -> Self {
        assert!(end > start);
        Self { sequence, start, end, text, class, probability }
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    pub fn offsets(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn class(&self) -> &str {
        &self.class
    }

    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// returns `true` iif this span is nested inside (or equals) the given span
    pub fn is_nested_in(&self, other: &Span) -> bool {
        self.start >= other.start && self.end <= other.end
    }

    /// returns `true` iif this span overlaps with the given one (symetric)
    pub fn overlaps(&self, other: &Span) -> bool {
        !(other.start > self.end || other.end < self.start)
    }

    /// returns `true` iif the spans do not overlap
    pub fn is_disjoint(&self, other: &Span) -> bool {
        !self.overlaps(other)
    }

    /// returns `true` iif this span has the same offsets as the given one
    pub fn same_offsets(&self, other: &Span) -> bool {
        self.start == other.start && self.end == other.end
    }


}

