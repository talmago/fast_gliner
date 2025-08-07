# fast_gliner: Python bindings for [gline-rs](https://github.com/fbilhaut/gline-rs)

Python binding to *gline-rs*, the inference engine for [GLiNER](https://github.com/urchade/GLiNER) Models written in Rust.

✨ Features
  - Simple python interface to GLiNER models.
  - [~4x speedup](https://github.com/fbilhaut/gline-rs?tab=readme-ov-file#cpu) compared to the PyTorch implementation.

## ⏳ Installation

### Pre-built wheel (CPU-only)

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

## 🚀 Quickstart

### Named Entity Recognition

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner_multi-v2.1-onnx",
    execution_provider="cpu",  # or "cuda"
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

### Relation Extraction

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

results = model.extract_relations(text, labels, schema)

from pprint import pprint
pprint(results)

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

## References

[1] [GLiNER](https://github.com/urchade/GLiNER): Generalist Model for Named Entity Recognition using Bidirectional Transformer.

```
@inproceedings{zaratiana-etal-2024-gliner,
    title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author = "Zaratiana, Urchade  and
      Tomeh, Nadi  and
      Holat, Pierre  and
      Charnois, Thierry",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.300",
    doi = "10.18653/v1/2024.naacl-long.300",
    pages = "5364--5376",
    abstract = "Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks.",
}
```