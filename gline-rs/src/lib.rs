//! An inference engine for [GLiNER](https://github.com/urchade/GLiNER/tree/main) models, which can perform 
//! zero-shot [Named Entity Recognition](https://paperswithcode.com/task/cg) (NER) and many other tasks such 
//! as [Relation Extraction](https://paperswithcode.com/task/relation-extraction).
//!
//! This implementation supports both span- and token-oriented models (inference only). 
//! 
//! It aims to provide an efficient, production-grade, user-friendly API in a modern and safe programming language, 
//! as well as a clean and maintainable implementation of the mechanics surrounding the model itself.

pub mod model;
pub mod text;
pub mod util;

