use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::util::result::Result;

const DEFAULT_MAX_WIDTH: usize = 12;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConfigMode {
    Span,
    Token,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub mode: ConfigMode,
    #[serde(default = "default_max_width")]
    pub max_width: usize,
}

impl Default for ConfigMode {
    fn default() -> Self {
        Self::Span
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            mode: ConfigMode::default(),
            max_width: default_max_width(),
        }
    }
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let config = serde_json::from_str(&raw)?;
        Ok(config)
    }
}

fn default_max_width() -> usize {
    DEFAULT_MAX_WIDTH
}
