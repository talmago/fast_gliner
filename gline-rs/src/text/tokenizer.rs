use crate::util::result::Result;
use std::path::Path;

/// Sub-word tokenization (aka encoding)
pub trait Tokenizer {
    fn encode(&self, input: &str) -> Result<Vec<u32>>;
}

/// Implement `Tokenizer` as a wrapper around Hugging Face tokenizers
#[derive(Clone)]
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HFTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_file(path)?,
        })
    }

    pub fn from_pretrained(identifier: &str) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_pretrained(identifier, None)?,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            inner: tokenizers::Tokenizer::from_bytes(bytes)?,
        })
    }

    /// Resolve a vocabulary token to its id.
    ///
    /// This matches `tokenizers::Tokenizer::token_to_id`.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode raw text or pretokenized pieces.
    ///
    /// This matches `tokenizers::Tokenizer::encode`. A `&[&str]` input is treated as
    /// `InputSequence::PreTokenized`. Callers read `get_ids()`, `get_attention_mask()`,
    /// and `get_word_ids()` from the returned `Encoding`.
    ///
    /// The `Tokenizer` trait method of the same name stays the v1 path: one string in,
    /// token ids out. It calls the inner tokenizer directly so the two methods do not recurse.
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<tokenizers::Encoding>
    where
        E: Into<tokenizers::EncodeInput<'s>>,
    {
        Ok(self.inner.encode(input, add_special_tokens)?)
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(input, false)?;
        Ok(encoding.get_ids().to_vec())
    }
}
