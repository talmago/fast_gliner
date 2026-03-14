# ROADMAP.md

This document describes the future development plan for `fast_gliner`.

---

# Phase 1 (Completed)

Initial release of Python bindings for the Rust inference engine `gline-rs`.

Features delivered:

- Named Entity Recognition (NER) inference
- Relation extraction
- CPU execution support
- CUDA execution support
- PyPI distribution

The project currently supports **GLiNER v1 models** exported to ONNX.

---

# ROADMAP.md

This document describes the future development plan for `fast_gliner`.

---

# Phase 1 (Completed)

Initial release of Python bindings for the Rust inference engine `gline-rs`.

Features delivered:

- Named Entity Recognition (NER) inference
- Relation extraction
- CPU execution support
- CUDA execution support
- PyPI distribution

The project currently supports **GLiNER v1 models** exported to ONNX.

---

# Phase 2 (Current) — GLiNER2 Support

The next development phase introduces support for **GLiNER2 models**.

GLiNER2 expands the original GLiNER architecture into a **schema-driven multi-task information extraction system** capable of performing:

- Named Entity Recognition (NER)
- Classification
- Schema-driven structured extraction
- Relation extraction

Background information about GLiNER and GLiNER2 is documented in:

- [`docs/GLINER_OVERVIEW.md`](./GLINER_OVERVIEW.md)

That document also contains references to the official repositories and research papers.

---

## Implementation Strategy

Phase 2 is divided into **two independent implementation tasks**.

Each task can be executed separately and validated using Rust examples and the Python bindings.

---

## Task 1 — Model Loader Refactor (Completed)

Before adding GLiNER2 support, the Rust inference engine had to support **directory-based model loading**.

This refactor introduces a unified model loading interface used by both **GLiNER v1 and GLiNER2 models**.

The existing API loads models using individual file paths:

```rs
GLiNER::<SpanMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "models/gliner_small-v2.1/tokenizer.json",
    "models/gliner_small-v2.1/onnx/model.onnx",
)
```

The new API now loads models from a directory:

```rs
GLiNER::<SpanMode>::from_dir(
    "models/gliner_small-v2.1",
    Parameters::default(),
    RuntimeParameters::default()
)
```


Optional overrides allow specifying custom paths for tokenizer, ONNX model, and config.

Default layout expected inside a model directory:

```
model_dir/
├── tokenizer.json
├── gliner_config.json
└── onnx/
└── model.onnx
```

Full implementation details and validation examples are documented in [`docs/MODEL_LOADING_REFACTOR.md`](./docs/MODEL_LOADING_REFACTOR.md).

This task is complete and unlocks GLiNER2 runtime implementation.

## Task 2 — GLiNER2 Runtime Support (Next)

Once directory-based loading is implemented, the next task introduces runtime support for **GLiNER2 models**.

GLiNER2 models use a different ONNX interface and a schema-driven input format.

Supporting GLiNER2 requires extending the Rust inference engine to:

- construct schema-based inputs
- support new ONNX tensor inputs
- decode span scores into structured outputs
- implement pipelines for multiple tasks

The GLiNER2 runtime will support:

- Named Entity Recognition
- Classification
- Schema-driven extraction
- Relation extraction

Most changes will occur inside the Rust inference engine:

```
gline-rs/src/model
```

A new module will be introduced:

```
gline-rs/src/model/gliner2
```

Full implementation instructions are documented in [`docs/GLINER2_RUNTIME.md`](./docs/GLINER2_RUNTIME.md).


This document includes:

- architecture notes
- Rust module structure
- example implementations
- validation examples

---

# Future work

Possible future improvements for `fast_gliner`:

- dynamic batching
- streaming inference
- improved GPU execution
- multi-model loading
- improved memory management
