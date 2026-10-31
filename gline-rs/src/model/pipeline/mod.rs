//! Family pipelines and runtimes.
//!
//! GLiNER span/token, GLiNER2, GLiClass, and GLiFormer each live here. Shared
//! System One types are in `system_one`. Nested GLiFormer records are in
//! `structure`. Combined GLiNER2 schemas are in `gliner2_multi`.

pub mod context;
pub mod gliclass;
pub mod gliformer;
pub mod gliner;
pub mod gliner2;
pub mod gliner2_multi;
pub mod structure;
pub mod system_one;
