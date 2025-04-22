//! Pre-processing steps
//! 
//! For NER, they will normally be applied in that order:
//! - Text input (raw entities and texts)
//! - Tokenized input (unchanged entities with tokenized texts)
//! - Prompts with entities tokens + text word-level tokens
//! - Encoded prompts applying sub-word tokenization to text tokens
//! - Ready for inference tensors
//! 
//! Other steps are for use in pipelines for other applications:
//! - Input for relation extraction
//! - ...

pub mod text;
pub mod tokenized; 
pub mod prompt; 
pub mod encoded;
pub mod tensors;
pub mod relation;
