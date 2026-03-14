# ARCHITECTURE.md

This document describes the internal architecture of **fast_gliner** and its Rust inference engine.

The goal of the project is to provide **fast CPU/GPU inference for GLiNER models** through a Python API while keeping heavy computation inside Rust.

---

# System Overview

The system consists of four major layers:

Python API
    ↓
PyO3 bindings
    ↓
Rust GLiNER inference engine
    ↓
ONNX Runtime
    ↓
GLiNER model

Python exposes a simple API for users, while Rust handles the full inference pipeline and performance‑critical operations.

---

# Repository Structure

```
fast_gliner
│
├── AGENTS.md
├── ARCHITECTURE.md
│
├── bindings/python
│   Python package and PyO3 bindings
│
└── gline-rs
    Rust inference engine
```

The repository intentionally vendors the `gline-rs` project instead of using it as an external dependency.

This allows the Python bindings and Rust inference engine to evolve together.

---

# Python Layer

Location:

```
bindings/python/py_src/fast_gliner
```

Main class:

```
FastGLiNER
```

Responsibilities:

• model loading  
• API interface for users  
• validating inputs  
• calling Rust extension functions  
• formatting outputs  

The Python layer should remain **thin**.

All heavy computation must remain inside Rust.

---

# Rust Engine

Location:

```
gline-rs/src
```

Main modules:

```
model/
text/
util/
```

The Rust crate implements the full GLiNER inference pipeline.

---

# Pipeline Architecture

Inference is implemented as a **composable processing pipeline**.

Each stage converts data into a new representation until the final entity spans are produced.

High‑level flow:

```
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
```

---

# Input Pipeline

## TextInput

Represents raw input text and entity labels.

## TokenizedInput

Word tokenization using RegexSplitter.

## PromptInput

Constructs GLiNER prompts:

```
<<ENT>> entity1 <<ENT>> entity2 <<SEP>> text tokens
```

## EncodedInput

Subword tokenization via HuggingFace tokenizer.

Produces tensors:

```
input_ids  
attention_mask  
word_mask  
text_lengths
```

---

# Tensor Preparation

Two inference modes exist.

## Span Mode

Used for most GLiNER NER models.

Additional tensors:

```
span_idx  
span_mask
```

## Token Mode

Used for multitask GLiNER models.

Works with token-level logits.

---

# Model Inference

The ONNX model is executed through:

```
orp::model::Model
```

Which wraps **ONNX Runtime**.

Execution providers include:

• CPU  
• CUDA

---

# Output Pipeline

After inference, the model produces:

TensorOutput

This tensor is decoded into entity spans.

---

# Span Decoding

Expected tensor shape:

```
(batch_size, num_words, max_width, num_classes)
```

Converted into spans containing:

```
text  
label  
probability  
start_offset  
end_offset
```

---

# Token Decoding

Token mode uses three token logits:

```
start  
end  
inside
```

These logits are combined to reconstruct spans.

---

# Span Filtering

Decoded spans are filtered using **greedy search**.

Filtering rules:

```
flat_ner  
dup_label  
multi_label
```

---

# Relation Extraction

Relation extraction builds on top of NER results.

Pipeline:

```
NER spans
   ↓
relation prompts
   ↓
token pipeline
   ↓
relation decoding

Relations are validated against a schema.
```

---

# Text Module

The `text` module provides core structures:

```
Token  
Span  
Tokenizer  
Splitter  
Prompt
```

These maintain alignment between tokens and original text offsets.

---

# Utility Module

The `util` module provides shared helpers such as:

• error handling  
• math utilities  
• shared result types  

---

# Execution Flow

Typical inference call:

```
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
```

---

# Performance Goals

The engine prioritizes:

• minimal allocations  
• safe Rust code  
• efficient tensor operations  
• batch processing  

---

# Extension Points

Future features should primarily modify:

```
gline-rs/src/model/input
gline-rs/src/model/output
gline-rs/src/model/pipeline
```

These areas define model architecture behavior.

---

# Future Development

Planned improvements include:

• GLiNER2 model support  
• improved CUDA execution  
• additional decoding strategies  
• improved batching support
