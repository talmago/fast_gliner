# gline-rs

Inference engine for GLiNER-family models, in Rust.

This crate is the engine behind [fast_gliner](https://github.com/talmago/fast_gliner). It is a fork of [gline-rs](https://github.com/fbilhaut/gline-rs) by Frédérik Bilhaut. The library name is `gliner`.

It runs ONNX checkpoints for:

- [GLiNER](https://github.com/urchade/GLiNER) span-mode and token-mode models
- [GLiNER2](https://huggingface.co/papers/2507.18546) NER, classification, relations, and structured extraction
- [GLiClass](https://github.com/Knowledgator/GLiClass) sequence classification
- [GLiFormer](https://github.com/Knowledgator/GLiFormer) text tasks: NER, classification, joint relations, and flat structures

Inference stays in Rust. The Python package is a thin wrapper.

## Background

GLiNER models do zero-shot [named entity recognition](https://paperswithcode.com/task/cg) and related extraction tasks with a bidirectional transformer, using less compute than a generative model. The papers describe the prompt and the decoder. The code around the ONNX session, tokenization, span gathering, and decoding is what this crate implements.

The original engine covered GLiNER span mode and token mode. This fork adds GLiNER2, GLiClass, and GLiFormer beside that pipeline. Each family has its own loader and decoder. They share the tokenizer utilities and the ONNX Runtime session setup from [`orp`](https://github.com/fbilhaut/orp) and [`ort`](https://ort.pyke.io).

## Public API

Load a directory that contains `tokenizer.json` and an ONNX graph.

```rust
use gliner::model::{params::Parameters, GLiNER, GLiNER2, GLiClass, GLiFormer};
use gliner::model::input::text::TextInput;
use orp::params::RuntimeParameters;

let model = GLiNER::from_dir_with(
    "models/gliner_small-v2.1",
    Parameters::default(),
    RuntimeParameters::default(),
    None,
    Some("model.onnx"),
    None,
)?;

let input = TextInput::from_str(
    &["My name is James Bond.", "I like to drive my Aston Martin."],
    &["person", "vehicle"],
)?;

let output = model.inference(input)?;
```

GLiNER2, GLiClass, and GLiFormer use `from_dir`:

```rust
let gliner2 = GLiNER2::from_dir(model_dir, Parameters::default(), RuntimeParameters::default())?;
let gliclass = GLiClass::from_dir(model_dir, Parameters::default(), RuntimeParameters::default())?;
let gliformer = GLiFormer::from_dir(model_dir, Parameters::default(), RuntimeParameters::default())?;
```

`GLiNER2` and `GLiFormer` accept one sequence at a time for `predict_entities`. `GLiClass::classify` and `GLiFormer::classify` return label scores sorted from highest to lowest. `GLiFormer` relations use the joint head, and its structures are flat records. Working calls are in `examples/`.

## Models

The checkpoints are ONNX. GLiNER, GLiNER2, and GLiClass load `onnx/model.onnx` (or the single `.onnx` file in the directory). GLiFormer loads a split graph: `onnx/encoder.onnx` plus `ner.onnx`, `classification.onnx`, `relations.onnx`, and `structuring.onnx`.

| Family | Example checkpoint |
|--------|--------------------|
| GLiNER span | [onnx-community/gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1) |
| GLiNER token / relations | [onnx-community/gliner-multitask-large-v0.5](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) |
| GLiNER2 | [lion-ai/gliner2-multi-v1-onnx](https://huggingface.co/lion-ai/gliner2-multi-v1-onnx) |
| GLiClass | [knowledgator/gliclass-small-v1.0](https://huggingface.co/knowledgator/gliclass-small-v1.0) |
| GLiFormer | [talmago/gliformer-base-v1-onnx](https://huggingface.co/talmago/gliformer-base-v1-onnx) |

GLiFormer ONNX weights are exported from the Knowledgator PyTorch checkpoints with `scripts/export_gliformer_onnx.py` in the parent repository. A large export lives at `models/gliformer-large-v1/` when that script is run with `knowledgator/gliformer-large-v1`.

Place a GLiNER checkpoint like this to run `examples/gliner_ner.rs` unchanged apart from the directory argument:

```text
models/gliner_small-v2.1/tokenizer.json
models/gliner_small-v2.1/onnx/model.onnx
```

## Examples

```bash
cargo run --example gliner_ner -- models/gliner_small-v2.1
cargo run --example gliner_relations -- models/gliner-multitask-large-v0.5
cargo run --example gliner2_ner -- models/gliner2-multi-v1
cargo run --example gliner2_pipeline -- models/gliner2-multi-v1
cargo run --example gliclass_classification -- models/gliclass-small-v1.0
cargo run --example gliformer -- models/gliformer-base-v1
```

`benchmark-cpu` and `benchmark-gpu` measure GLiNER token-mode throughput.

## GPU and other execution providers

Pass an execution provider through `RuntimeParameters`. Without the matching crate feature, ONNX Runtime stays on CPU.

```rust
let runtime = RuntimeParameters::default().with_execution_providers([
    CUDAExecutionProvider::default().build(),
]);
```

```bash
cargo run --example benchmark-gpu --features=cuda
```

Provider setup is described in `doc/ORT.md`.

## Crate features

The features mirror `ort`:

- `load-dynamic` loads the ONNX Runtime library at runtime
- execution providers: `cuda`, `tensorrt`, `directml`, `coreml`, `rocm`, `openvino`, `onednn`, `xnnpack`, `qnn`, `cann`, `nnapi`, `tvm`, `acl`, `armnn`, `migraphx`, `vitis`, `rknpu`

## Performance

These figures are from the original engine on GLiNER token mode. They are not measurements of GLiNER2, GLiClass, or GLiFormer.

### CPU

| Implementation | sequences/second |
|----------------|------------------|
| gline-rs       | 6.67             |
| GLiNER.py      | 1.61             |

- Dataset: first 100 entries of [NuNER](https://huggingface.co/datasets/numind/NuNER)
- Mode: token, `flat_ner: true`
- Entity classes: 3
- Threshold: 0.5
- Model: [gliner-multitask-large-v0.5](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5)
- CPU: Intel Core i9 at 2.3 GHz, 8 cores
- gline-rs 0.9.0

### GPU

| Implementation | sequences/second |
|----------------|------------------|
| gline-rs       | 248.75           |

Same setup, with the first 1000 NuNER entries, CUDA, an NVIDIA RTX 4080, and gline-rs 0.9.1.

## Status

The crate version in this repository is `0.9.5-SNAPSHOT`. ONNX Runtime is used through `ort` 2.0.0-rc.9. The Python API in the parent repository is the supported way to call these models.

## Design

The crate is safe Rust aside from the ONNX Runtime itself. The main dependencies are `orp`, `ort`, Hugging Face `tokenizers`, `ndarray`, and `regex`.

Pre-processing and decoding implement the `Pipeline` trait from `orp`. `Splitter` and `Tokenizer` can be replaced when a task needs a different text front end. GLiNER span mode and token mode stay in `model::pipeline`. GLiNER2 schema tasks live next to them. GLiClass is a uni-encoder classifier. GLiFormer is a separate runtime: its prompt, gather, and BIO decoder are not folded into the GLiNER span or token pipeline.

`doc/Processing.typ` documents the original GLiNER processing pipeline. `doc/ORT.md` covers execution providers.

## References

- [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://aclanthology.org/2024.naacl-long.300/) by Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois (2024).
- [GLiNER2: Schema-Driven Multi-Task Learning for Structured Information Extraction](https://huggingface.co/papers/2507.18546) by Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, and Ash Lewis (2025).
- [GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks](https://arxiv.org/abs/2406.12925) by Ihor Stepanov and Mykhailo Shtopko (2024).
- [GLiClass: Generalist Lightweight Model for Sequence Classification Tasks](https://arxiv.org/abs/2508.07662) by Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, Alexander Yavorskyi, and Mykyta Yaroshenko (2025).
- [GLiFormer: A Generalist Multitask Transformer Encoder](https://www.knowledgator.com/research/gliformer) by Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, and Mykyta Yaroshenko (2026).
- [Named Entity Recognition as Structured Span Prediction](https://aclanthology.org/2022.umios-1.1/) by Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois (2022).

Thanks to the GLiNER authors, and to Frédérik Bilhaut for the original `gline-rs` engine.
