use std::path::Path;
use crate::util::result::Result;


/// Sub-word tokenization (aka encoding)
pub trait Tokenizer {
    fn encode(&self, input: &str) -> Result<Vec<u32>>;
}


/// Implement `Tokenizer` as a wrapper around Hugging Face tokenizers
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}


impl HFTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_file(path)?
        })
    }

    pub fn from_pretrained(identifier: &str) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_pretrained(identifier, None)?
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_bytes(bytes)?
        })
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(input, false)?;
        Ok(encoding.get_ids().to_vec())
    }
}