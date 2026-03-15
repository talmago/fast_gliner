# fast_gliner

![PyPI](https://img.shields.io/pypi/v/fast_gliner)
![Python](https://img.shields.io/pypi/pyversions/fast_gliner)
![License](https://img.shields.io/github/license/fbilhaut/gline-rs)
![Rust](https://img.shields.io/badge/runtime-rust-orange)

Python bindings for the Rust inference engine  [gline-rs](https://github.com/fbilhaut/gline-rs) — providing fast CPU/GPU inference for [GLiNER](https://github.com/urchade/GLiNER) and [GLiNER2](https://huggingface.co/papers/2507.18546) models.

`fast_gliner` exposes a simple Python API while delegating all heavy computation to a Rust runtime powered by **ONNX Runtime**.

---

## ✨ Features

- 🚀 High-performance inference using Rust
- 🧠 Supports **GLiNER** and **GLiNER2** models
- ⚡ ~4× faster CPU inference than the PyTorch implementation
- 🐍 Simple Python API
- 🖥 Optional **CUDA execution** through ONNX Runtime

---

## ⏳ Installation

### Pre-built wheel (CPU)

```bash
$ pip install fast_gliner
```

### Building from source

```
$ pip install --no-binary=:all: fast_gliner
```

### Building with CUDA

```
$ pip install --no-binary=:all: fast_gliner[cuda]
```

---

## 🚀 Quickstart

### Named Entity Recognition

#### GLiNER2 (recommended)

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    "lion-ai/gliner2-multi-v1-onnx"
)

model.predict_entities(
    "I am James Bond",
    ["person"]
)
```

#### GLiNER (legacy runtime)

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner_multi-v2.1-onnx",
    execution_provider="cpu",
)

model.predict_entities("I am James Bond", ["person"])
```

Output:

```
[
    {
        'text': 'James Bond',
        'label': 'person',
        'score': 0.9012733697891235,
        'start': 5,
        'end': 15
    }
]
```

### Classification

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    "lion-ai/gliner2-multi-v1-onnx"
)

model.classify("Buy milk and eggs after work", ["shopping", "work", "personal"])
```

Output:

```
[
    ('shopping', 0.93), 
    ('personal', 0.61), 
    ('work', 0.44)
]
```

### Structured Extraction

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    "lion-ai/gliner2-multi-v1-onnx"
)

text = """Contact: John Smith
Email: john@example.com
Phones: 555-1234, 555-5678
Address: 123 Main St, NYC"""

schema = {
    "name": ["person"],
    "email": ["email"],
    "phone": ["phone"],
    "address": ["address"],
}

model.extract(text, schema)
```

Output:

```
{   'address': [   {   'end': 96,
                       'label': 'address',
                       'score': 0.9998444318771362,
                       'start': 80,
                       'text': '123 Main St, NYC'}],
    'email': [   {   'end': 43,
                     'label': 'email',
                     'score': 0.9998226165771484,
                     'start': 27,
                     'text': 'john@example.com'}],
    'name': [   {   'end': 19,
                    'label': 'person',
                    'score': 0.9999531507492065,
                    'start': 9,
                    'text': 'John Smith'}],
    'phone': [   {   'end': 60,
                     'label': 'phone',
                     'score': 0.9996894001960754,
                     'start': 52,
                     'text': '555-1234'},
                 {   'end': 70,
                     'label': 'phone',
                     'score': 0.9997830390930176,
                     'start': 62,
                     'text': '555-5678'}]}
```

#### Relation Extraction

#### GLiNER2

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    "lion-ai/gliner2-multi-v1-onnx"
)

text = "Bill Gates founded Microsoft."

labels = ["person", "organization"]

schema = [
    {
        "relation": "founded",
        "subject_labels": ["person"],
        "object_labels": ["organization"]
    }
]

model.extract_relations(text, labels, schema)
```

#### GLiNER (gliner-multitask-large)

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner-multitask-large-v0.5",
    onnx_path="onnx/model.onnx"
)

text = "Bill Gates is the founder of Microsoft."

labels = ["person", "organization"]

schema = [
    {
        "relation": "founder",
        "subject_labels": ["person"],
        "object_labels": ["organization"]
    }
]

model.extract_relations(text, labels, schema)
```

Output:

```
[{'relation': 'founder',
  'score': 0.9981993436813354,
  'subject': {'text': 'Bill Gates',
   'label': 'person',
   'score': 0.9981993436813354,
   'start': 85,
   'end': 94},
  'object': {'text': 'Microsoft',
   'label': 'organization',
   'score': 0.9981993436813354,
   'start': 85,
   'end': 94}}]
```

---

## Supported Models

| Model | Runtime | Task | Multilingual |
|------|------|------|------|
| **GLiNER v2.1** | | | |
| [`onnx-community/gliner_small-v2.1`](https://huggingface.co/onnx-community/gliner_small-v2.1) | `FastGLiNER` | NER | ❌ |
| [`onnx-community/gliner_medium-v2.1`](https://huggingface.co/onnx-community/gliner_medium-v2.1) | `FastGLiNER` | NER | ❌ |
| [`onnx-community/gliner_large-v2.1`](https://huggingface.co/onnx-community/gliner_large-v2.1) | `FastGLiNER` | NER | ❌ |
| [`onnx-community/gliner_multi-v2.1-onnx`](https://huggingface.co/onnx-community/gliner_multi-v2.1-onnx) | `FastGLiNER` | NER | ✅ |
| [`juampahc/gliner_multi-v2.1-onnx`](https://huggingface.co/juampahc/gliner_multi-v2.1-onnx) | `FastGLiNER` | NER | ✅ |
| **GLiNER multitask** | | | |
| [`onnx-community/gliner-multitask-large-v0.5`](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | `FastGLiNER` | NER, Relation Extraction | ❌ |
| **GLiNER2** | | | |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | NER, Classification, Structured Extraction, Relation Extraction | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | NER, Classification, Structured Extraction, Relation Extraction | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | NER, Classification, Structured Extraction, Relation Extraction | ✅ |

---

## Performance

`fast_gliner` uses the Rust engine **gline-rs** and ONNX Runtime to accelerate inference.

Benchmarks show **~4× faster CPU inference** compared to the original PyTorch implementation.

See the benchmark results in the [gline-rs README](https://github.com/fbilhaut/gline-rs?tab=readme-ov-file#cpu).

---

## Development

Set up environment

```sh
$ cd fast_gliner/bindings/python
$ make dev
```

Run code formatting

```sh
$ make style
```

Release package to PyPI

```sh
$ make
$ make release
```

---

## For Contributors

If you're planning to contribute to `fast_gliner`, the following documents provide useful context:

1. **Start here:**
   [`docs/GLINER_OVERVIEW.md`](./docs/GLINER_OVERVIEW.md) — background on GLiNER and GLiNER2 models.

2. **Understand the system design:**
   [`ARCHITECTURE.md`](./ARCHITECTURE.md) — explains how the Python API, Rust inference engine, and ONNX Runtime interact.

3. **Set up your development environment:**
   [`docs/DEVELOPMENT.md`](./docs/DEVELOPMENT.md) — instructions for building the project and running it locally.

4. **Planned future work:**
   [`docs/ROADMAP.md`](./docs/ROADMAP.md) — outlines upcoming development phases, including GLiNER2 support.

Coding agents working in this repository should also follow the rules described in:

* [`AGENTS.md`](./AGENTS.md)

---

## References

[1] [GLiNER](https://github.com/urchade/GLiNER): Generalist Model for Named Entity Recognition using Bidirectional Transformer.

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
  title   = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
  author  = "Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Charnois, Thierry",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)",
  year    = "2024",
  url     = "https://aclanthology.org/2024.naacl-long.300"
}
```


[2] GLiNER2: Schema-Driven Multi-Task Learning for Structured Information Extraction

```bibtex
@inproceedings{zaratiana-etal-2025-gliner2,
    title = "{GL}i{NER}2: Schema-Driven Multi-Task Learning for Structured Information Extraction",
    author = "Zaratiana, Urchade and Pasternak, Gil and Boyd, Oliver and Hurn-Maloney, George and Lewis, Ash",
    booktitle = "EMNLP 2025 System Demonstrations",
    year = "2025"
}
```
