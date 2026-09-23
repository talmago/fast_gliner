# fast_gliner

![PyPI](https://img.shields.io/pypi/v/fast_gliner)
![Python](https://img.shields.io/pypi/pyversions/fast_gliner)
![License](https://img.shields.io/github/license/fbilhaut/gline-rs)
![Rust](https://img.shields.io/badge/runtime-rust-orange)

Python bindings for the Rust inference engine [gline-rs](https://github.com/fbilhaut/gline-rs), providing fast CPU/GPU inference for:

- [GLiNER](https://github.com/urchade/GLiNER)
- [GLiNER2](https://huggingface.co/papers/2507.18546)
- [GLiClass](https://github.com/Knowledgator/GLiClass)
- [GLiFormer](https://github.com/Knowledgator/GLiFormer)

`fast_gliner` exposes a simple Python API while delegating all heavy computation to a Rust runtime powered by **ONNX Runtime**.

---

## ✨ Features

- 🚀 High-performance inference using Rust
- 🧠 Supports **GLiNER**, **GLiNER2**, **GLiClass**, and **GLiFormer** models
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

#### GLiNER

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

---

### Classification

GLiNER2 classifies through the span-score head:

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

GLiClass scores one text against a label list. `prompt_first` is read from the checkpoint `config.json`. The return value is `(label, score)` pairs, highest score first.

```python
from fast_gliner import FastGLiClass

model = FastGLiClass.from_pretrained(
    "knowledgator/gliclass-small-v1.0"
)

model.classify(
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    ["computing", "science", "programming", "travel", "food", "politics"],
)
```

Output:

```
[
    ('programming', 1.0),
    ('computing', 1.0),
    ('science', 0.9983),
    ('travel', 0.7708),
    ('politics', 0.4666),
    ('food', 0.3758)
]
```

Hierarchical labels are a dict. Scores use dotted names such as `sentiment.positive`.

```python
model.classify(
    "The product quality is amazing but delivery was slow",
    {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["product", "service", "shipping"],
    },
)
```

Output:

```
[
    ('topic.product', 1.0),
    ('topic.shipping', 1.0),
    ('topic.service', 1.0),
    ('sentiment.positive', 1.0),
    ('sentiment.neutral', 1.0),
    ('sentiment.negative', 1.0)
]
```

`return_hierarchical=True` returns a dict in the shape of the labels instead of a sorted list.

```python
model.classify(
    "The product quality is amazing but delivery was slow",
    {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["product", "service", "shipping"],
    },
    return_hierarchical=True,
)
```

Output:

```
{
    'sentiment': {'positive': 1.0, 'negative': 1.0, 'neutral': 1.0},
    'topic': {'product': 1.0, 'service': 1.0, 'shipping': 1.0}
}
```

Few-shot examples are in-context text. They do not add scored labels. The tokenizer must contain `<<EXAMPLE>>`.

```python
model.classify(
    "Fast delivery and the item works perfectly!",
    ["positive", "negative", "product", "service", "shipping"],
    examples=[
        {"text": "Love this item, great quality!", "labels": ["positive", "product"]},
        {"text": "Customer support was unhelpful", "labels": ["negative", "service"]},
    ],
)
```

Output:

```
missing required GLiClass token in tokenizer vocabulary: <<EXAMPLE>>
```

A task prompt is inserted after the label separator.

```python
model.classify(
    "The battery life on this phone is incredible",
    ["positive", "negative", "neutral"],
    prompt="Classify the sentiment of this product review:",
)
```

Output:

```
[
    ('positive', 1.0),
    ('neutral', 0.9999),
    ('negative', 0.9314)
]
```

---

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

result = model.extract_json(
    text,
    {
        "contact": [
            "name::str",
            "email::str",
            "phone::list",
            "address"
        ]
    }
)
```

Output:

```
{
    'contact': [
        {
            'address': ['123 Main St, NYC'],
            'email': 'john@example.com',
            'name': 'John Smith',
            'phone': ['555-1234', '555-5678']
        }
    ]
}
```

---

### Relation Extraction

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

#### GLiNER

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

## Multi-Task Pipeline

### entities + classification + structured extraction

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    "lion-ai/gliner2-multi-v1-onnx"
)

schema = (
    model.create_schema()
    # Extract entities
    .entities(["person", "company", "location"])
    
    # Classify sentiment
    .classification("sentiment", ["positive", "negative", "neutral"])
    
    # Extract structured product information
    .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
        .field("category", dtype="str", choices=["electronics", "software", "service"])
)

text = """
Apple CEO Tim Cook announced the iPhone 15 for $999 with amazing new features.
This is exciting!
"""

result = model.extract(text, schema)

print(result)
```

Output:

```
{
    "classifications": {
        "sentiment": [
            {"label": "positive", "score": 0.9232913255691528},
            {"label": "neutral", "score": 0.19288331270217896},
            {"label": "negative", "score": 0.005759984254837036},
        ]
    },
    "entities": [
        {
            "text": "Apple",
            "label": "company",
            "score": 0.9991476535797119,
            "start": 1,
            "end": 6,
        },
        {
            "text": "Tim Cook",
            "label": "person",
            "score": 0.999701738357544,
            "start": 11,
            "end": 19,
        },
    ],
    "relations": [],
    "structures": {
        "product": {
            "name": ["iPhone 15"],
            "price": ["$999"],
            "features": ["amazing new features"],
            "category": [],
        }
    },
}
```

### entities + relation extraction

```python
schema = (
    model.create_schema()
    .entities(["person", "company"])
    .relation("founded", ["person"], ["company"])
    .relation("works_for", ["person"], ["company"])
)

text = """
Bill Gates founded Microsoft.
Satya Nadella works for Microsoft.
"""

model.extract(text, schema)
```

Output:

```json
{
  "entities": [
    {"text": "Bill Gates", "label": "person"},
    {"text": "Microsoft", "label": "company"},
    {"text": "Satya Nadella", "label": "person"},
    {"text": "Microsoft", "label": "company"}
  ],
  "relations": [
    {
      "relation": "founded",
      "subject": {"text": "Bill Gates", "label": "person"},
      "object": {"text": "Microsoft", "label": "company"}
    },
    {
      "relation": "works_for",
      "subject": {"text": "Satya Nadella", "label": "person"},
      "object": {"text": "Microsoft", "label": "company"}
    }
  ]
}
```

---

## GLiFormer

`FastGLiFormer` uses the same schema builder and return values as `FastGLiNER2`. 

Relations come from the joint head, and structures are flat.

```python
from fast_gliner import FastGLiFormer

model = FastGLiFormer.from_pretrained("talmago/gliformer-base-v1-onnx")

schema = (
    model.create_schema()
    .entities(["person", "organization", "location"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relation("works_at", ["person"], ["organization"])
    .relation("lives_in", ["person"], ["location"])
    .structure("employee")
        .field("name")
        .field("company")
)

result = model.extract("Alice works at Acme and lives in London.", schema)
```

Output:

```
{
    "classifications": {
        "sentiment": [
            {"label": "neutral", "score": 0.9998},
            {"label": "positive", "score": 0.0001},
            {"label": "negative", "score": 0.0},
        ]
    },
    "entities": [
        {"text": "Alice", "label": "person", "score": 0.998633, "start": 0, "end": 5},
        {"text": "Acme", "label": "organization", "score": 0.999901, "start": 15, "end": 19},
        {"text": "London", "label": "location", "score": 0.999399, "start": 33, "end": 39},
    ],
    "relations": [
        {
            "relation": "works_at",
            "score": 0.892890,
            "subject": {"text": "Alice", "label": "person", "score": 0.996878, "start": 0, "end": 5},
            "object": {"text": "Acme", "label": "organization", "score": 0.999975, "start": 15, "end": 19},
        },
        {
            "relation": "lives_in",
            "score": 0.897462,
            "subject": {"text": "Alice", "label": "person", "score": 0.996878, "start": 0, "end": 5},
            "object": {"text": "London", "label": "location", "score": 0.999963, "start": 33, "end": 39},
        },
    ],
    "structures": {
        "employee": {
            "name": ["Alice"],
            "company": ["Acme"],
        }
    },
}
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
| **GLiClass** | | | |
| [`knowledgator/gliclass-small-v1.0`](https://huggingface.co/knowledgator/gliclass-small-v1.0) | `FastGLiClass` | Classification | ❌ |
| [`knowledgator/gliclass-base-v1.0`](https://huggingface.co/knowledgator/gliclass-base-v1.0) | `FastGLiClass` | Classification | ❌ |
| [`knowledgator/gliclass-large-v1.0`](https://huggingface.co/knowledgator/gliclass-large-v1.0) | `FastGLiClass` | Classification | ❌ |
| [`knowledgator/gliclass-modern-base-v2.0-init`](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init) | `FastGLiClass` | Classification | ❌ |
| [`knowledgator/gliclass-modern-large-v2.0`](https://huggingface.co/knowledgator/gliclass-modern-large-v2.0) | `FastGLiClass` | Classification | ❌ |
| **GLiFormer** | | | |
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | NER, Classification, Relations, Flat structuring | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | NER, Classification, Relations, Flat structuring | ❌ |

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

Coding agents working in this repository should also follow the rules described in:

* [`AGENTS.md`](./AGENTS.md)

---

## Acknowledgements

This repository is a fork of [gline-rs](https://github.com/fbilhaut/gline-rs), the Rust engine that runs the inference. Thanks as well to the authors of the original GLiNER paper [1], which the models build on.

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


[3] [GLiClass](https://arxiv.org/abs/2508.07662): Generalist Lightweight Model for Sequence Classification Tasks.

```bibtex
@misc{stepanov2025gliclassgeneralistlightweightmodel,
    title = "{GL}i{C}lass: Generalist Lightweight Model for Sequence Classification Tasks",
    author = "Stepanov, Ihor and Shtopko, Mykhailo and Vodianytskyi, Dmytro and Lukashov, Oleksandr and Yavorskyi, Alexander and Yaroshenko, Mykyta",
    year = "2025",
    eprint = "2508.07662",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    url = "https://arxiv.org/abs/2508.07662"
}
```


[4] [GLiFormer](https://www.knowledgator.com/research/gliformer): A Generalist Multitask Transformer Encoder.

```bibtex
@misc{stepanov2026gliformer,
    title = "{GL}i{F}ormer: A Generalist Multitask Transformer Encoder",
    author = "Stepanov, Ihor and Shtopko, Mykhailo and Vodianytskyi, Dmytro and Lukashov, Oleksandr and Yaroshenko, Mykyta",
    year = "2026",
    url = "https://www.knowledgator.com/research/gliformer"
}
```
