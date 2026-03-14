# DEVELOPMENT.md

This document describes how to develop and modify the `fast_gliner` project.

---

# Project Layout

```
fast_gliner
│
├── AGENTS.md
├── ARCHITECTURE.md
│
├── bindings/python
│   Python package and PyO3 bindings
│
├── gline-rs
│   Rust inference engine
│
└── docs
```

---

# Development Environment

The project requires:

- Rust
- Python 3.10+
- Poetry
- Maturin

Install Python dependencies:

```sh
cd bindings/python
poetry install
```

---

# Building the Python Extension

To build and install the Rust extension locally:

```sh
cd bindings/python
make dev
```

This command will:

1. Install Python dependencies via Poetry
2. Compile the Rust extension with `maturin`
3. Install the module into the Poetry environment

---

# Formatting

Python formatting and linting:

```sh
make style
```

This runs `ruff`.

---

# Building Wheels

To build a distributable wheel:

```sh
make build
```

---

# CUDA Builds

CUDA support can be enabled with:

```sh
make FEATURES=cuda build
```

---

# Rust Development

Rust code lives in:

```
gline-rs/src
```

Key areas:

model/      → inference pipelines
text/       → tokenization and offsets
util/       → shared helpers

---

# Testing Changes

When modifying inference logic, test:

1. Entity extraction
2. Relation extraction
3. CPU execution
4. CUDA execution (if available)

Example Python test:

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner_multi-v2.1-onnx"
)

model.predict_entities("Barack Obama lives in Washington", ["person", "location"])
```

---

# Modifying the Rust Engine

When implementing new model features:

1. Update pipeline stages in `model/pipeline`
2. Modify tensor input/output logic if needed
3. Keep Python bindings minimal

---

# Contribution Guidelines

Before implementing large changes:

- propose a design
- ensure compatibility with the Python API
- maintain performance characteristics
