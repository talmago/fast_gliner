# fast_gliner

![PyPI](https://img.shields.io/pypi/v/fast_gliner)
![Python](https://img.shields.io/pypi/pyversions/fast_gliner)
![License](https://img.shields.io/github/license/fbilhaut/gline-rs)
![Rust](https://img.shields.io/badge/runtime-rust-orange)

Python bindings for the Rust inference engine [gline-rs](https://github.com/fbilhaut/gline-rs). They run fast CPU and GPU inference for:

- [GLiNER](https://github.com/urchade/GLiNER)
- [GLiNER2](https://huggingface.co/papers/2507.18546)
- [GLiClass](https://github.com/Knowledgator/GLiClass)
- [GLiFormer](https://github.com/Knowledgator/GLiFormer)

`fast_gliner` keeps the Python API small. Rust does the inference. The runtime is **ONNX Runtime**.

---

## ✨ Features

- 🚀 High-performance inference using Rust
- 🧠 Supports **GLiNER**, **GLiNER2**, **GLiClass**, **GLiFormer**, and **System One**
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

```python
from fast_gliner import FastGLiFormer

model = FastGLiFormer.from_pretrained(
    model_id="talmago/gliformer-base-v1-onnx",
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

#### Supported Models

| Model | Class | Batched | Multilingual |
|------|------|------|------|
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | ❌ | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | ❌ | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | ❌ | ✅ |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | ❌ | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | ❌ | ❌ |
| [`onnx-community/gliner_multi-v2.1-onnx`](https://huggingface.co/onnx-community/gliner_multi-v2.1-onnx) | `FastGLiNER` | ✅ | ✅ |
| [`juampahc/gliner_multi-v2.1-onnx`](https://huggingface.co/juampahc/gliner_multi-v2.1-onnx) | `FastGLiNER` | ✅ | ✅ |
| [`onnx-community/gliner_small-v2.1`](https://huggingface.co/onnx-community/gliner_small-v2.1) | `FastGLiNER` | ✅ | ❌ |
| [`onnx-community/gliner_medium-v2.1`](https://huggingface.co/onnx-community/gliner_medium-v2.1) | `FastGLiNER` | ✅ | ❌ |
| [`onnx-community/gliner_large-v2.1`](https://huggingface.co/onnx-community/gliner_large-v2.1) | `FastGLiNER` | ✅ | ❌ |
| [`onnx-community/gliner-multitask-large-v0.5`](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | `FastGLiNER` | ✅ | ❌ |

---

### Classification

```python
from fast_gliner import FastGLiClass

model = FastGLiClass.from_pretrained(
    model_id="knowledgator/gliclass-v4-edge",
    execution_provider="cpu",
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

#### Hierarchical

Nested `labels` flatten to dotted names such as `sentiment.positive`.

`return_hierarchical=True` returns a dict in the same shape as `labels`.

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

#### Task prompt

`prompt` is inserted after the label separator.

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

#### Few-shot

`examples` are extra prompt text, not scored labels.

The tokenizer must contain `<<EXAMPLE>>`.

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

#### Single-label

`classification_type="single-label"` applies a softmax instead of independent sigmoids.

```python
model.classify(
    "The battery life on this phone is incredible",
    ["positive", "negative", "neutral"],
    classification_type="single-label",
)
```

#### Supported Models

| Model | Class | Hierarchical | Few-shot | Task prompt | Single-label | Multilingual |
|------|------|------|------|------|------|------|
| [`knowledgator/gliclass-v4-edge`](https://huggingface.co/knowledgator/gliclass-v4-edge) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`knowledgator/gliclass-modern-large-v2.0`](https://huggingface.co/knowledgator/gliclass-modern-large-v2.0) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`knowledgator/gliclass-modern-base-v2.0-init`](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`knowledgator/gliclass-large-v1.0`](https://huggingface.co/knowledgator/gliclass-large-v1.0) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`knowledgator/gliclass-base-v1.0`](https://huggingface.co/knowledgator/gliclass-base-v1.0) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`knowledgator/gliclass-small-v1.0`](https://huggingface.co/knowledgator/gliclass-small-v1.0) | `FastGLiClass` | ✅ | ✅ | ✅ | ✅ | ❌ |
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | ❌ | ❌ | ❌ | ❌ | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | ❌ | ❌ | ❌ | ❌ | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ❌ | ❌ | ✅ |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ❌ | ❌ | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ❌ | ❌ | ❌ |

---

### System One

[System One](https://docs.system-one.dev/en/docs/primitives) answers typed questions about a state.

Each question is one single-label pass.

`FastGLiFormer.system_one` takes the same arguments and return value.

Question types:

- `choice` — pick one key from `criteria`
- `noul` — probability of `true`
- `score` — expected level over ordered `criteria`

```python
from fast_gliner import Choice, FastGLiClass, Noul, Score

model = FastGLiClass.from_pretrained(
    model_id="knowledgator/gliclass-v4-edge",
    execution_provider="cpu",
)

model.system_one(
    {
        "ticket": "Checkout errors hit 12% after the 14:05 deploy.",
        "facts": [
            "Rolling back takes two minutes.",
            "No incident is open yet.",
        ],
    },
    {
        "action": Choice(
            instructions="What should the on-call engineer do right now?",
            criteria={
                "rollback": "Roll back to the last healthy build.",
                "wait": "Watch error rates for another hour.",
                "escalate": "Page the payments team lead.",
            },
        ),
        "urgent": Noul(
            instructions="Does this need action within the next hour?",
            criteria={
                "true": "Yes, customers are affected right now.",
                "false": "No, it can wait.",
            },
        ),
        "severity": Score(
            instructions="Rate the incident severity.",
            criteria=["minor", "moderate", "major", "critical"],
        ),
    },
    max_length=16384,
)
```

Output:

```
SystemOneOutput(
    answers={
        'action': ChoiceAnswer(
            type='choice',
            choice='rollback',
            confidence=0.8819,
            probabilities={
                'rollback': 0.8819,
                'wait': 0.1119,
                'escalate': 0.0062,
            },
        ),
        'urgent': NoulAnswer(type='noul', noul=0.2908),
        'severity': ScoreAnswer(
            type='score',
            score=0.2822,
            confidence=0.8133,
            probabilities={
                '0': 0.8133,
                '1': 0.1312,
                '2': 0.0156,
                '3': 0.0399,
            },
        ),
    },
    usage=Usage(input_tokens=178, output_tokens=0),
)
```

`result.answers["action"].choice` is the selected key. `result.model_dump()` is the same payload as a dict.

#### Supported Models

| Model | Class | Multilingual |
|------|------|------|
| [`knowledgator/gliclass-v4-edge`](https://huggingface.co/knowledgator/gliclass-v4-edge) | `FastGLiClass` | ❌ |
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | ❌ |

---

### Structured Extraction

#### Nested

Pass a dict schema to `structure`.

A list of field names is a flat record.

Nested schemas need a multi-level structuring head.

Each name maps to a list of records.

Nested lists stay on their parent.

```python
from fast_gliner import FastGLiFormer

model = FastGLiFormer.from_pretrained(
    model_id="talmago/gliformer-base-v1-onnx",
    execution_provider="cpu",
)

text = (
    "At Acme, Engineering includes Alice, a software engineer, and Bob, "
    "a designer. Sales includes Carol, an account manager."
)

records = model.structure(text, {
    "company": {
        "name": "str",
        "departments": [{
            "name": "str",
            "employees": [{"name": "str", "role": "str"}],
        }],
    }
})
```

Output:

```
{
    'company': [
        {
            'departments': [
                {
                    'employees': [
                        {'name': 'Alice', 'role': 'software engineer'},
                        {'name': 'Bob', 'role': 'designer'}
                    ],
                    'name': 'Engineering'
                },
                {
                    'employees': [
                        {'name': 'Carol', 'role': 'account manager'}
                    ],
                    'name': 'Sales'
                }
            ],
            'name': 'Acme'
        }
    ]
}
```

#### Pydantic

The same `structure` call accepts a Pydantic model.

Nested models stay nested.

The return value is instances of that model.

```python
from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    role: str

class Department(BaseModel):
    name: str
    employees: list[Employee]

class Company(BaseModel):
    name: str
    departments: list[Department]

records = model.structure(text, {"company": Company})
```

Output:

```
{
    'company': [
        Company(
            name='Acme',
            departments=[
                Department(
                    name='Engineering',
                    employees=[
                        Employee(name='Alice', role='software engineer'),
                        Employee(name='Bob', role='designer')
                    ]
                ),
                Department(
                    name='Sales',
                    employees=[
                        Employee(name='Carol', role='account manager')
                    ]
                )
            ]
        )
    ]
}
```

#### Legacy JSON

`FastGLiNER2.extract_json` uses `name::str` / `name::list` field specs.

```python
from fast_gliner import FastGLiNER2

model = FastGLiNER2.from_pretrained(
    model_id="lion-ai/gliner2-multi-v1-onnx",
    execution_provider="cpu",
)

text = """Contact: John Smith
Email: john@example.com
Phones: 555-1234, 555-5678
Address: 123 Main St, NYC"""

model.extract_json(
    text,
    {"contact": ["name::str", "email::str", "phone::list", "address"]},
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

#### Supported Models

| Model | Class | Nested | Pydantic | Legacy JSON | Multilingual |
|------|------|------|------|------|------|
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | ✅ | ✅ | ❌ | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | ✅ | ✅ | ❌ | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ✅ | ✅ |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ✅ | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | ❌ | ❌ | ✅ | ❌ |

---

### Relation Extraction

```python
from fast_gliner import FastGLiFormer

model = FastGLiFormer.from_pretrained(
    model_id="talmago/gliformer-base-v1-onnx",
    execution_provider="cpu",
)

model.extract_relations(
    "Alice works at Acme and lives in London.",
    ["person", "organization", "location"],
    [
        {
            "relation": "works_at",
            "subject_labels": ["person"],
            "object_labels": ["organization"],
        },
        {
            "relation": "lives_in",
            "subject_labels": ["person"],
            "object_labels": ["location"],
        },
    ],
)
```

Output:

```
[
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
]
```

#### Supported Models

| Model | Class | Notes | Multilingual |
|------|------|------|------|
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | Joint relation head | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | Joint relation head | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | | ✅ |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | | ❌ |
| [`onnx-community/gliner-multitask-large-v0.5`](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | `FastGLiNER` | Token-mode; `onnx_path="onnx/model.onnx"` | ❌ |

---

### Multi-Task

`create_schema()` selects which tasks run in one `extract` call.

Schema-builder `structure` fields stay flat.

Nested records use `structure()` from Structured Extraction.

```python
from fast_gliner import FastGLiFormer

model = FastGLiFormer.from_pretrained(
    model_id="talmago/gliformer-base-v1-onnx",
    execution_provider="cpu",
)

schema = (
    model.create_schema()
    .entities(["person", "organization", "location"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relation("works_at", ["person"], ["organization"])
    .relation("lives_in", ["person"], ["location"])
    .structure("employee")
        .field("name", dtype="str")
        .field("company", dtype="str")
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

#### Supported Models

| Model | Class | Nested records | Multilingual |
|------|------|------|------|
| [`talmago/gliformer-base-v1-onnx`](https://huggingface.co/talmago/gliformer-base-v1-onnx) | `FastGLiFormer` | `structure()` | ❌ |
| [`talmago/gliformer-large-v1-onnx`](https://huggingface.co/talmago/gliformer-large-v1-onnx) | `FastGLiFormer` | `structure()` | ❌ |
| [`lion-ai/gliner2-multi-v1-onnx`](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) | `FastGLiNER2` | `extract_json()` | ✅ |
| [`lion-ai/gliner2-base-v1-onnx`](https://huggingface.co/lion-ai/gliner2-base-v1-onnx) | `FastGLiNER2` | `extract_json()` | ❌ |
| [`lion-ai/gliner2-large-v1-onnx`](https://huggingface.co/lion-ai/gliner2-large-v1-onnx) | `FastGLiNER2` | `extract_json()` | ❌ |

---

## Performance

`fast_gliner` runs on the Rust engine **gline-rs** and ONNX Runtime.

CPU inference is about **4× faster** than the PyTorch implementation.

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
$ make build
$ make release
```

---

## For Contributors

Start with these documents:

1. **Inference:**
   [`docs/MODELING.md`](./docs/MODELING.md) — how each model family runs.

2. **System design:**
   [`ARCHITECTURE.md`](./ARCHITECTURE.md) — Python API, Rust engine, and ONNX Runtime.

3. **Development setup:**
   [`docs/DEVELOPMENT.md`](./docs/DEVELOPMENT.md) — how to build and run the project.

Agents in this repository should also follow [`AGENTS.md`](./AGENTS.md).

---

## Acknowledgements

This repository is a fork of [gline-rs](https://github.com/fbilhaut/gline-rs). That Rust engine runs the inference. Thanks to the authors of the original GLiNER paper [1].

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


[2] [GLiNER2](https://arxiv.org/abs/2507.18546): Schema-Driven Multi-Task Learning for Structured Information Extraction

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
