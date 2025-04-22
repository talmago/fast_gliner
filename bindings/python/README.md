# fast_gliner: Python bindings for [gline-rs](https://github.com/fbilhaut/gline-rs) (Inference Engine for GLiNER Models, written in Rust)

## ⏳ Installation

```bash
$ pip install fast_gliner
```

## 🚀 Quickstart

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner_multi-v2.1",
    onnx_path="onnx/model.onnx"
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