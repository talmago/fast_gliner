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
    (PyFastGliNER / PyFastGliNER2)
            │
            │
            ▼
    Rust Inference Engine
    (gline-rs)
        │
        ├── GLiNER v1 runtime
        │       prompt-based NER pipeline
        │
        └── GLiNER2 runtime
                schema-driven multi-task pipeline
                │
                ▼
        ONNX Runtime
                │
                ▼
        GLiNER / GLiNER2 ONNX model

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

Responsibilities:

• model loading (`from_pretrained`)
• API interface for users\
• input validation and normalization
• calling Rust extension classes (`PyFastGliNER`, `PyFastGliNER2`)
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

The Rust crate implements the full GLiNER inference pipeline and runtime
implementations.

------------------------------------------------------------------------

# GLiNER v1 Architecture

GLiNER v1 uses a **prompt‑based architecture** where entity labels are
embedded into the input text.

Inference is implemented as a **composable processing pipeline**.

High‑level flow:

    TextInput
      ↓
    TokenizedInput
      ↓
    PromptInput
      ↓
    EncodedInput
      ↓
    Tensor preparation
      ↓
    ONNX Runtime inference
      ↓
    TensorOutput
      ↓
    Span decoding
      ↓
    Greedy filtering

------------------------------------------------------------------------

# Input Pipeline

## TextInput

Represents raw input text and entity labels.

## TokenizedInput

Word tokenization using `RegexSplitter`.

## PromptInput

Constructs GLiNER prompts:

    <<ENT>> entity1 <<ENT>> entity2 <<SEP>> text tokens

## EncodedInput

Subword tokenization using a HuggingFace tokenizer.

Produces tensors such as:

    input_ids
    attention_mask
    word_mask
    text_lengths

------------------------------------------------------------------------

# Tensor Preparation

Two inference modes exist.

## Span Mode

Used for most GLiNER NER models.

Additional tensors:

    span_idx
    span_mask

## Token Mode

Used for multitask GLiNER models.

Works with token-level logits.

------------------------------------------------------------------------

# Model Inference

The ONNX model is executed through:

    orp::model::Model

Which wraps **ONNX Runtime**.

Execution providers include:

• CPU\
• CUDA

------------------------------------------------------------------------

# Output Pipeline

After inference, the model produces:

    TensorOutput

This tensor is decoded into entity spans.

------------------------------------------------------------------------

# Span Decoding

Expected tensor shape:

    (batch_size, num_words, max_width, num_classes)

Converted into spans containing:

    text
    label
    probability
    start_offset
    end_offset

------------------------------------------------------------------------

# Token Decoding

Token mode uses three logits:

    start
    end
    inside

These logits are combined to reconstruct spans.

------------------------------------------------------------------------

# Span Filtering

Decoded spans are filtered using **greedy search**.

Filtering rules:

    flat_ner
    dup_label
    multi_label

------------------------------------------------------------------------

# Relation Extraction

Relation extraction builds on top of NER results.

Pipeline:

    NER spans
       ↓
    relation prompts
       ↓
    token pipeline
       ↓
    relation decoding

Relations are validated against a schema.

------------------------------------------------------------------------


# GLiNER2 Runtime

GLiNER2 uses the same stage layout as GLiNER v1. It is not a separate
source tree. `GLiNER2` in `model/runtime.rs` loads the model and
dispatches to pipeline compositions.

Key differences from v1:

• does **not use prompt-based label encoding**  
• does **not rely on `gliner_config.json`**  
• builds schema representations directly in tokenizer space  
• supports **schema‑driven multi‑task inference**

------------------------------------------------------------------------

## Stage Modules

    text/tokenizer.rs
        HFTokenizer. Raw text and pretokenized pieces both go through
        encode(input, add_special_tokens), which returns
        tokenizers::Encoding. Special tokens use token_to_id.

    model/input/schema.rs
        Schema prefix pieces, SpecialTokens, and extraction schemas.

    model/input/tensors/schema.rs
        input_ids, attention_mask, text_positions, schema_positions,
        span_idx. First-subword positions come from Encoding::get_word_ids.

    model/output/decoded/span_scores.rs
        Decodes the monolithic span_scores head into spans.

    model/output/classification.rs
        Classification scores from the best span score per label.

    model/output/extraction.rs
        Groups decoded spans into extraction fields.

    model/pipeline/schema.rs
        Single-task NER, classification, and extraction pipelines.

    model/pipeline/multitask.rs
        Multi-task schema builder and combined execution.

    model/runtime.rs
        GLiNER2::from_dir and the task methods.

Relation decoding reuses `model/output/relation.rs`.

Public APIs:

    GLiNER2::from_dir(...)
    GLiNER2::inference(...)
    GLiNER2::extract(...)
    GLiNER2::classify(...)
    GLiNER2::extract_relations(...)
    GLiNER2::create_schema(...)

A later model that shares this contract should load through
`GLiNER2::from_dir`. A model that changes one stage should add or
replace that stage under `input`, `output`, or `pipeline`.

------------------------------------------------------------------------

## GLiNER2 Loader

    GLiNER2::from_dir(model_dir, params, runtime_params)

Loader flow:

    model_dir
       ↓
    resolve tokenizer.json
    resolve ONNX model path
       ↓
    initialize HFTokenizer
       ↓
    resolve special tokens
       ↓
    initialize ONNX runtime

Required files:

    tokenizer.json
    model.onnx

`onnx/model.onnx` is used when that path exists.

------------------------------------------------------------------------

## Special Token Resolution

GLiNER2 resolves runtime control tokens directly from the tokenizer
vocabulary.

Required tokens:

    [P]
    [C]
    [E]
    [R]
    [L]
    [MASK]
    [SEP_STRUCT]
    [SEP_TEXT]
    [DESCRIPTION]
    [EXAMPLE]
    [OUTPUT]

Tokens are resolved using:

    tokenizer.token_to_id(...)

If any required token is missing, runtime initialization fails.

------------------------------------------------------------------------

## GLiNER2 Inference Pipeline

High‑level flow:

    TextInput
       ↓
    schema prefix construction
       ↓
    HFTokenizer::encode(pieces, false)
       ↓
    tensor preparation
       ↓
    ONNX inference
       ↓
    span decoding
       ↓
    greedy filtering

Example tensors:

    input_ids
    attention_mask
    text_positions
    schema_positions
    span_idx

------------------------------------------------------------------------

## Multi‑Task Pipeline

GLiNER2 supports **schema‑driven multi‑task pipelines** where several
tasks can be executed together.

Supported tasks:

• entity extraction  
• relation extraction  
• classification  
• structured extraction

Example schema:

    schema = (
        model.create_schema()
        .entities(["person", "company"])
        .relation("works_for", ["person"], ["company"])
        .classification("sentiment", ["positive", "neutral", "negative"])
        .structure("event")
            .field("date")
            .field("description")
    )

Pipeline execution strategy:

    entity-only schema
        ↓
    fast NER inference path

    multi-task schema
        ↓
    combined schema prefix
        ↓
    model.extract(...)
        ↓
    span pool
        ↓
    decoding

Decoding steps:

    spans → entities
    spans → relations
    spans → structures

Classification is executed using the dedicated runtime:

    model.classify(...)

Classification cannot be derived from spans because
classification labels are **not span-based**.

------------------------------------------------------------------------

## Entity Fast Path

For performance reasons the pipeline uses a specialized path when the
schema only contains entity extraction.

Flow:

    TextInput
       ↓
    tokenizer encoding
       ↓
    model.inference(...)
       ↓
    span decoding

This avoids constructing the full multi‑task schema.

------------------------------------------------------------------------

## Python API

GLiNER2 is exposed through the Python bindings as:

    FastGLiNER2

Example:

    model = FastGLiNER2.from_pretrained("lion-ai/gliner2-multi-v1-onnx")

Schemas are built using a fluent builder:

    schema = (
        model.create_schema()
        .entities(["person", "company"])
        .relation("founded", ["person"], ["company"])
    )

Inference:

    result = model.extract(text, schema)

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

Typical GLiNER2 call:

    Python FastGLiNER2
          ↓
    Rust PyFastGliNER2
          ↓
    GLiNER2 runtime
          ↓
    ONNX Runtime
          ↓
    decoded spans
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
