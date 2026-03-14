# AGENTS.md

This repository contains **fast_gliner**, Python bindings for the Rust inference engine **gline-rs**, which runs GLiNER models.

The project enables fast CPU/GPU inference for GLiNER models through a Python API while keeping heavy computation in Rust.

This document defines how coding agents should operate within this repository.

---

# Repository Overview

The repository consists of two main components:

```
fast_gliner
│
├── bindings/python
│   Python package + PyO3 bindings
│
└── gline-rs
    Rust inference engine
```

Agents must understand the responsibility boundary:

| Layer | Responsibility |
|------|---------------|
| Python | User API |
| PyO3 | FFI bridge |
| Rust | Inference logic |

---

# Core Principle

Inference logic must remain in Rust.

Python must remain a thin wrapper.

Do NOT implement model logic in Python.

---

# Development Environment

Development setup, build commands, and release procedures are documented in:

[`docs/DEVELOPMENT.md`](./docs/DEVELOPMENT.md)

Agents should consult that document instead of redefining build instructions here.

---

# Python Layer

Location:

```
bindings/python
```

Key Python source:

```
bindings/python/py_src/fast_gliner
```

Main class:

```
FastGLiNER
```

Responsibilities of the Python layer:

- Loading models (`from_pretrained`)
- Validating inputs
- Calling the Rust extension
- Formatting outputs for users

Python should not perform inference logic.

All heavy computation must remain in Rust.

---

# Rust Inference Engine

Location:

```
gline-rs/src
```

Top-level modules:

```
model/
text/
util/
```

The Rust crate implements the complete GLiNER inference pipeline.

Agents modifying model behavior should work inside this crate.

---

# Inference Architecture

The GLiNER inference pipeline and system design are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

Agents modifying model logic should read that document before changing pipeline code.

---

# Inference Mode

`GLiNER` supports multiple inference modes (e.g. span mode and token mode).

Details about these modes and their pipelines are documented in: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

# Model Runtime

Inference runs through ONNX Runtime via:

```
orp::model::Model
```

Agents should not modify runtime behavior unless necessary.

---

# Safe Modification Areas

Agents may safely modify:

```
gline-rs/src/model/pipeline
gline-rs/src/model/input
gline-rs/src/model/output
```

Agents should avoid modifying:

```
gline-rs/src/text
gline-rs/src/util
```

unless fixing bugs.

---

# Coding Guidelines

Rust

Prefer:

- iterators
- zero-copy data flow
- minimal allocations

Avoid:

- cloning large tensors
- unnecessary heap allocations

Python

Keep Python logic minimal.

Python should:

- load models
- call Rust functions
- format results

---

# Backwards Compatibility

Changes must not break the existing Python API:

```
FastGLiNER.from_pretrained()
FastGLiNER.predict_entities()
FastGLiNER.extract_relations()
```

---

# Upstream Relationship

The Rust engine originates from:

```
https://github.com/fbilhaut/gline-rs
```

Agents should inspect upstream implementation before rewriting core logic.

---

# Future Development

Upcoming development plans are tracked in:

[`docs/ROADMAP.md`][`docs/ROADMAP.md`](./docs/ROADMAP.md)

---

# Agent Behavior Guidelines

When implementing features:

1. Modify the Rust engine first
2. Keep the Python API stable
3. Avoid duplicating logic across layers
4. Preserve pipeline composability
5. Maintain performance characteristics

If uncertain about architecture changes, propose a design before implementing.
