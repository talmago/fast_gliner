# ARCHITECTURE.md

This document describes the internal architecture of **fast_gliner** and
its Rust inference engine.

The goal of the project is to provide **fast CPU/GPU inference for
GLiNER models** through a Python API while keeping heavy computation
inside Rust.

------------------------------------------------------------------------

## High-Level Architecture

The system is organized as layered components.

    Python API (fast_gliner)
            │
            │  PyO3 bindings
            ▼
    Rust Extension Layer
    (PyFastGliNER / PyFastGliNER2 / PyFastGLiClass / PyFastGLiFormer)
            │
            │
            ▼
    Rust Inference Engine
    (gline-rs)
        │
        ├── GLiNER v1 runtime
        │       prompt-based NER pipeline
        │
        ├── GLiNER2 runtime
        │       schema-driven multi-task pipeline
        │
        ├── GLiClass runtime
        │       uni-encoder sequence classification
        │
        └── GLiFormer runtime
                local encoder plus task-head ONNX graphs
                │
                ▼
        ONNX Runtime
                │
                ▼
        GLiNER / GLiNER2 / GLiClass / GLiFormer ONNX model

The Python layer exposes a simple API while Rust performs all
performance-critical operations including tokenization, tensor
preparation, inference, and decoding.

------------------------------------------------------------------------

# Repository Structure

    fast_gliner
    ├── ARCHITECTURE.md
    │
    ├── bindings/python
    │   Python package and PyO3 bindings
    │
    └── gline-rs
        Rust inference engine

The repository vendors the `gline-rs` project so the Python bindings and
Rust inference engine can evolve together.

------------------------------------------------------------------------

# Python Layer

Location:

    bindings/python/py_src/fast_gliner

Public runtime classes:

    FastGLiNER
    FastGLiNER2
    FastGLiClass
    FastGLiFormer

Responsibilities:

• model loading (`from_pretrained`)
• API interface for users\
• input validation and normalization
• calling Rust extension classes (`PyFastGliNER`, `PyFastGliNER2`, `PyFastGLiClass`, `PyFastGLiFormer`)
• formatting outputs

The Python layer should remain **thin**. All heavy computation must
remain inside Rust.

------------------------------------------------------------------------

# Rust Layer

Location:

    gline-rs/src

Top‑level modules:

    model/
    text/
    util/

The Rust crate implements the model runtimes. How each family constructs
inputs, runs ONNX, and decodes outputs is documented in
[`docs/MODELING.md`](./docs/MODELING.md).

------------------------------------------------------------------------

# Models

| Runtime | Rust type | Role |
|------|------|------|
| GLiNER v1 | `InferenceMode` | span and token NER, relations |
| GLiNER2 | `GLiNER2` | schema-driven multi-task extraction |
| GLiClass | `GLiClass` | uni-encoder sequence classification |
| GLiFormer | `GLiFormer` | encoder plus task-head graphs |

See [`docs/MODELING.md`](./docs/MODELING.md) for prompts, tensors, loaders,
and the module map.

------------------------------------------------------------------------

# Text Module

The `text` module provides shared primitives:

    Token
    Span
    Tokenizer
    Splitter
    Prompt

These maintain alignment between tokens and original text offsets.

------------------------------------------------------------------------

# Utility Module

The `util` module provides shared helpers:

• error handling\
• math utilities\
• shared result types

------------------------------------------------------------------------

# Execution Flow

Typical GLiNER v1 call:

    Python FastGLiNER
          ↓
    Rust PyFastGliNER
          ↓
    GLiNER pipeline
          ↓
    ONNX Runtime
          ↓
    decoded spans
          ↓
    Python result formatting

Typical GLiNER2 or GLiFormer call:

    Python FastGLiNER2 / FastGLiFormer
          ↓
    Rust PyFastGliNER2 / PyFastGLiFormer
          ↓
    GLiNER2 or GLiFormer runtime
          ↓
    ONNX Runtime
          ↓
    decoded spans, classes, or structures
          ↓
    Python result formatting

Typical GLiClass call:

    Python FastGLiClass
          ↓
    Rust PyFastGLiClass
          ↓
    GLiClass runtime
          ↓
    ONNX Runtime
          ↓
    label scores
          ↓
    Python result formatting

------------------------------------------------------------------------

# Performance Goals

The engine prioritizes:

• minimal allocations\
• efficient tensor operations\
• safe Rust code\
• CPU and GPU inference

------------------------------------------------------------------------

# Extension Points

Future features should primarily modify:

    gline-rs/src/model/input
    gline-rs/src/model/output
    gline-rs/src/model/pipeline
    gline-rs/src/text
