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

GLiNER2 expands the original GLiNER architecture into a **schema-driven multi-task information extraction system**.

Supporting GLiNER2 may require updates to the Rust inference engine.

Background information about GLiNER and GLiNER2 is documented in [`docs/GLINER_OVERVIEW.md`](./docs/GLINER_OVERVIEW.md).

That document also contains references to the official repositories and research papers.

---

## Investigation Tasks

Before implementing support, the following areas must be analyzed:

1. GLiNER2 model architecture
2. ONNX graph structure
3. tensor input formats
4. tensor output formats
5. tokenization differences
6. decoding logic for structured outputs

---

## Expected Impact on fast_gliner

Most changes will occur inside the Rust inference engine:

```
gline-rs/src/model/input
gline-rs/src/model/output
gline-rs/src/model/pipeline
```


Possible modifications include:

- schema parsing
- additional model output heads
- modified tensor shapes
- updated span decoding logic

The Python bindings are expected to require **minimal changes**.

---

## Planned Implementation Steps

1. Inspect the GLiNER2 repository
2. Export a GLiNER2 model to ONNX
3. Compare ONNX graphs with GLiNER v1 models
4. Identify differences in input tensors
5. Identify differences in output tensors
6. Update the Rust inference pipeline
7. Validate compatibility with the Python API

---

# Phase 3 (Future)

Possible future improvements for `fast_gliner`:

- dynamic batching
- streaming inference
- improved GPU execution
- multi-model loading
- improved memory management