//! Processing parameters


/// Represents the set of parameters for the whole pipeline
/// 
/// * pre-processing
/// * post-processing
/// 
/// The easiest way to instanciate sound parameters is to use the
/// `default()` constructor and then use individual setters as needed.
pub struct Parameters {
    /// Probability threshold (default: 0.5)
    pub threshold: f32,    
    /// Setting this parameter to `true` means that no entity can overlap with another one (default: true)
    pub flat_ner: bool,
    /// If `flat_ner=false`, setting this parameter to `true` means that overlapping spans can belong to the *same* class (default: false)
    pub dup_label: bool,
    /// If `flat_ner=false`, setting this parameter to `true` means that overlapping spans can belong to *different* classes (default: false)
    pub multi_label: bool,    
    /// For span mode, maximum span width (default: 12)
    pub max_width: usize,
    /// Maximum sequence length (default: 512)
    pub max_length: Option<usize>,
}

impl Default for Parameters {
    /// Default configuration, which can be safely used in most cases
    fn default() -> Self {
        Self::new(
            0.5, 
            12, 
            Some(512),
            true,             
            false,
            false,
        )
    }
}

impl Parameters {
    /// New configuration specifying every parameter
    pub fn new(threshold: f32, max_width: usize, max_length: Option<usize>, flat_ner: bool, dup_label: bool, multi_label: bool) -> Self {
        Self { 
            threshold, 
            max_width, 
            max_length,
            flat_ner,
            dup_label,
            multi_label,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_max_width(mut self, max_width: usize) -> Self {
        self.max_width = max_width;
        self
    }

    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_flat_ner(mut self, flat_ner: bool) -> Self {
        self.flat_ner = flat_ner;
        self
    }

    pub fn with_dup_label(mut self, dup_label: bool) -> Self {
        self.dup_label = dup_label;
        self
    }

    pub fn with_multi_label(mut self, multi_label: bool) -> Self {
        self.multi_label = multi_label;
        self
    }

}
