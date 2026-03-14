# GLiNER and GLiNER2 Overview

This document provides a high-level overview of the GLiNER family of models and the key architectural differences between **GLiNER (v1)** and **GLiNER2**.

The goal of this document is to help developers and coding agents understand how GLiNER works and what changes may affect the `fast_gliner` inference engine.

---

# What is GLiNER?

GLiNER (Generalist Language Interface for Named Entity Recognition) is a lightweight transformer-based model designed for **zero-shot Named Entity Recognition (NER)**.

Unlike traditional NER systems that require predefined entity types during training, GLiNER allows users to **define entity labels at inference time**.

Instead of generating text like large language models, GLiNER uses a **bidirectional transformer encoder** to score spans against user-provided entity labels.

This allows:

- flexible entity definitions
- parallel inference
- fast CPU execution

GLiNER was introduced in:

Zaratiana et al. (2024)  
"GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"

---

# GLiNER Architecture (v1)

GLiNER uses a **prompt-based span extraction architecture**.

At inference time the user provides:

- input text
- entity labels

The model constructs a prompt combining both and performs span classification.

Example prompt format:

<<ENT>> person <<ENT>> organization <<SEP>> input text

The model then predicts which spans correspond to each entity type.

---

# GLiNER v1 Inference Tasks

GLiNER v1 primarily focuses on **Named Entity Recognition**.

Additional capabilities such as relation extraction can be implemented on top of the NER pipeline.

Core characteristics:

- span-based entity detection
- zero-shot entity types
- bidirectional transformer encoder
- efficient CPU inference
- ONNX export compatibility

---

# GLiNER2 Overview

GLiNER2 extends the original architecture into a **general-purpose information extraction system**.

Instead of focusing only on NER, GLiNER2 unifies several NLP tasks into a single model.

These tasks include:

- Named Entity Recognition
- Text Classification
- Structured Data Extraction
- Relation Extraction

The model still uses a **transformer encoder architecture**, but introduces a **schema-driven interface** to describe extraction tasks.

---

# Schema-Based Extraction

One of the key innovations of GLiNER2 is the use of **schemas**.

Instead of providing a flat list of entity labels, users define a structured schema describing the desired output.

Example schema:

```python
schema = {
    "entities": ["person", "organization"],
    "relations": [
        {
            "relation": "founder",
            "subject": "person",
            "object": "organization"
        }
    ],
    "classification": {
        "sentiment": ["positive", "negative", "neutral"]
    }
}
```

The model can process this schema and perform **multiple tasks in a single forward pass**.

---

# Key Improvements in GLiNER2

GLiNER2 introduces several major improvements over GLiNER v1.

## 1. Multi‑Task Information Extraction

GLiNER2 supports multiple tasks simultaneously:

- NER
- classification
- relation extraction
- structured extraction

These tasks are executed **within a single model inference**.

---

## 2. Schema‑Driven Interface

GLiNER2 replaces label lists with **structured schemas** describing the desired output.

Benefits:

- more expressive extraction tasks
- unified interface across tasks
- easier integration with downstream systems

---

## 3. Unified Extraction Pipeline

GLiNER2 consolidates multiple NLP tasks into a **single encoder architecture**.

Traditional pipelines often require separate models for:

- NER
- classification
- relation extraction

GLiNER2 performs them **in a single forward pass**.

---

## 4. Structured Output

GLiNER2 can output structured JSON-like data instead of simple entity spans.

Example output:

```json
{
  "entities": {
    "person": ["Tim Cook"],
    "company": ["Apple"]
  },
  "sentiment": "positive"
}
```

This allows the model to act as a **general information extraction engine**.

---

## 5. CPU‑Optimized Inference

GLiNER2 continues the GLiNER design goal of **fast CPU inference**.

Typical characteristics:

- ~205M parameter base model
- real-time inference on CPU hardware
- no dependency on large GPU infrastructure

---

# Comparison: GLiNER vs GLiNER2

| Feature | GLiNER | GLiNER2 |
|------|------|------|
| Primary Task | NER | Multi-task extraction |
| Input Format | label list | schema definition |
| Output | entity spans | structured results |
| Tasks | NER | NER + classification + relations + structured extraction |
| Inference | single-task pipeline | unified multi-task pipeline |
| Architecture | encoder span model | encoder with task heads |

---

# Implications for fast_gliner

The current `fast_gliner` project implements inference for **GLiNER v1 models** using the Rust engine from `gline-rs`.

Supporting GLiNER2 may require changes in:

gline-rs/src/model/input
gline-rs/src/model/output
gline-rs/src/model/pipeline

Potential changes include:

- schema parsing
- additional output heads
- new tensor shapes
- new decoding logic

The Python bindings are expected to require **minimal changes**.

---

# References

## Papers

[1] Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois.  
**GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.**  
Proceedings of NAACL 2024.

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
  title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
  author = "Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Charnois, Thierry",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
  year = "2024",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.naacl-long.300",
  doi = "10.18653/v1/2024.naacl-long.300",
  pages = "5364--5376"
}
```

[2] Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, and Ash Lewis.
GLiNER2: Schema-Driven Multi-Task Learning for Structured Information Extraction.
Proceedings of EMNLP 2025 System Demonstrations.

```bibtex
@inproceedings{zaratiana-etal-2025-gliner2,
  title = "{GL}i{NER}2: Schema-Driven Multi-Task Learning for Structured Information Extraction",
  author = "Zaratiana, Urchade and Pasternak, Gil and Boyd, Oliver and Hurn-Maloney, George and Lewis, Ash",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year = "2025",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.emnlp-demos.10/",
  pages = "130--140"
}
```

## Project Repositories

  - [GLiNER](https://github.com/urchade/GLiNER) (original implementation)
  - [GLiNER2](https://github.com/fastino-ai/GLiNER2) (next-generation framework)