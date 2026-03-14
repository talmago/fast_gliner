# GLINER2_RUNTIME_TASK.md

# Objective

Implement runtime support for **GLiNER2 models** inside the Rust inference engine `gline-rs`.

GLiNER2 extends the original GLiNER architecture into a **schema-driven multi-task information extraction system**.

Supported tasks:

- Named Entity Recognition (NER)
- Classification
- Schema-driven structured extraction
- Relation extraction

This task assumes the **directory-based model loader refactor has already been completed**.

The Python API must remain unchanged while extending capabilities through Rust.

---

# Repository Context

```
fast_gliner
├── AGENTS.md
├── ARCHITECTURE.md
├── bindings/python
│   Python API + PyO3 bindings
├── gline-rs
│   Rust inference engine
└── docs
```

All inference logic must live inside `gline-rs`.

The Python layer must remain a **thin wrapper**.

Relevant docs to read before implementation:

- `AGENTS.md`
- `ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/GLINER_OVERVIEW.md`
- `docs/ROADMAP.md`
- `docs/MODEL_LOADING_REFACTOR.md`

---

# Architecture Notes

The original GLiNER v1 implementation in `gline-rs` is built around a generic `GLiNER<P>` wrapper where the pipeline defines the preprocessing and postprocessing path, and inference is delegated through the model runtime.

Current v1 runtime paths:

- `SpanPipeline` for span-mode NER
- `TokenPipeline` for token-mode models
- `RelationPipeline` layered on top of token inference

GLiNER2 support should be implemented **additively**. Do not rewrite or destabilize the v1 paths.

The monolithic GLiNER2 ONNX export is architecturally closer to the existing span-style runtime than to the earlier experimental multi-session GLiNER2 prototype. It uses a **single ONNX model** and requires new tensor construction and decoding logic, but does not require a split encoder/head runtime for the target model family described here.

Important baseline difference from v1:

- **GLiNER v1** is label-list-driven and prompt-based using the existing `<<ENT>> ... <<SEP>>` flow
- **GLiNER2** is schema-driven and uses explicit schema/token position tensors

This means GLiNER2 should be implemented as a **new runtime family** rather than by forcing it into the old prompt construction logic.

---

# GLiNER2 Model Architecture

GLiNER2 introduces a schema-driven input format and uses a monolithic ONNX model.

Schema example:

( [P] entities ( [E] person [E] company ) ) [SEP_TEXT] Steve Jobs founded Apple

This schema allows the model to support multiple tasks without retraining.

The model directory includes:

- `tokenizer.json`
- `gliner_config.json`
- `onnx/model.onnx`

Example `gliner_config.json` fields relevant to runtime:

- `max_width`
- `special_tokens`
- `onnx_files`

Special tokens include values such as:

- `[P]`
- `[E]`
- `[L]`
- `[SEP_TEXT]`

These token ids must be read from configuration rather than hardcoded.

---

# GLiNER2 ONNX Interface

The target monolithic ONNX model uses the following inputs:

| Input | Shape | Description |
|------|------|-------------|
| input_ids | (1, seq_len) | token ids |
| attention_mask | (1, seq_len) | mask |
| text_positions | (num_words) | first token index of each word |
| schema_positions | (1 + num_fields) | index of `[P]` and `[E]` tokens |
| span_idx | (1, num_words * max_width, 2) | span start/end pairs |

Output:

| Output | Shape | Description |
|------|------|-------------|
| span_scores | (1, num_fields, num_words, max_width) | span label scores |

Observed ONNX signature for the target monolithic model:

Inputs:
- `input_ids [1, ?]`
- `attention_mask [1, ?]`
- `text_positions [?]`
- `schema_positions [?]`
- `span_idx [1, ?, 2]`

Output:
- `span_scores [?, ?, ?, ?]`

Semantics from the model documentation:
- batch is effectively 1
- `text_positions` is a 1D vector of first-token positions for each text word
- `schema_positions` is a 1D vector containing the position of `[P]`, then one entry per `[E]`
- `span_idx` is a batched tensor of `(start_word, end_word)` pairs
- `span_scores` is semantically `(1, num_labels, num_words, max_width)`

Entities are extracted by thresholding `span_scores`.

---

# Input Construction Notes

For GLiNER2, input construction differs materially from v1.

Required steps:

1. Split raw text into words using the GLiNER2 word-splitting regex / equivalent splitter
2. Build schema prefix
3. Tokenize the schema pieces and words using pre-tokenized mode and without automatic special tokens
4. Compute:
   - `input_ids`
   - `attention_mask`
   - `text_positions`
   - `schema_positions`
   - `span_idx`

Important notes:

- `text_positions` must contain the token index of the **first token of each text word**
- `schema_positions` must contain the token index of `[P]`, then the token indices of each `[E]`
- `span_idx` must enumerate valid `(start_word, end_word)` pairs up to `max_width`
- `schema_positions` must be computed at the **tokenized sequence level**, not at the raw word-string level

Do not reuse the old v1 prompt builders (`PromptInput`, `EncodedInput`) directly for GLiNER2. The v1 pipeline is based on a different prompt format and different tensor assumptions.

Good reuse targets from the existing codebase:

- tokenizer wrapper patterns
- splitter utilities where compatible
- `Span`
- `Result`
- general ndarray/ORT integration patterns

---

# Rust Module Structure

Add a new module:

```
gline-rs/src/model/gliner2
```

Recommended structure:

```
model/gliner2
├── mod.rs
├── config.rs
├── model.rs
├── schema.rs
├── tokenize.rs
├── spans.rs
├── decode.rs
├── ner.rs
├── classification.rs
├── extraction.rs
└── relations.rs
```

Responsibilities:

| Module | Purpose |
|------|------|
| config.rs | parse `gliner_config.json` and expose GLiNER2 config |
| model.rs | runtime wrapper and top-level task dispatch |
| schema.rs | schema prefix builder |
| tokenize.rs | tokenizer integration for schema + text |
| spans.rs | generate `span_idx` |
| decode.rs | convert span scores to outputs |
| ner.rs | NER pipeline |
| classification.rs | classification pipeline |
| extraction.rs | structured extraction |
| relations.rs | relation extraction |

Suggested shared runtime shape:

```rs
pub struct GLiNER2Model {
    session: orp::model::Model,
    tokenizer: crate::text::tokenizer::HFTokenizer,
    max_width: usize,
    special_tokens: std::collections::HashMap<String, i64>,
}
```

The first implementation slice may keep this runtime focused and self-contained rather than forcing it into the existing generic `Pipeline` abstraction immediately.

---

# Implementation Strategy

Implement GLiNER2 **incrementally** with runnable validation examples.

Order of work:

1. GLiNER2 model loader
2. NER pipeline
3. Classification
4. Structured extraction
5. Relation extraction
6. Python binding integration adjustments if needed

Each milestone must include:
- Rust code
- a runnable Rust example
- a Python validation snippet
- clear expected behavior

---

# Milestone 1 — GLiNER2 Model Loader

Implement runtime wrapper:

`GLiNER2Model::from_dir()`

Responsibilities:

- load tokenizer
- load ONNX model
- parse `gliner_config.json`
- store `max_width`
- store special token ids

Rust example file:

`gline-rs/examples/gliner2_load.rs`

Example:

```rs
use gliner_rs::model::gliner2::GLiNER2Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Model::from_dir("models/gliner2-multi-v1-onnx")?;
    println!("GLiNER2 model loaded");
    Ok(())
}
```

Run:

```sh
cargo run --example gliner2_load
```

Python validation:

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained("lion-ai/gliner2-multi-v1-onnx")
print(model)
```

Expected result:
- model loads successfully from a directory-based layout
- no regression to v1 loading paths

---

# Milestone 2 — NER Pipeline

Implement:

`predict_entities(text, labels, threshold)`

Pipeline steps:

1. split text into words
2. build schema prefix using `[P]`, task token(s), and `[E]` labels
3. tokenize words and schema pieces
4. compute `text_positions`
5. compute `schema_positions`
6. generate `span_idx`
7. run ONNX inference
8. decode `span_scores` into entity spans
9. apply overlap filtering / deduplication consistent with current behavior

Concrete decoding notes:

- assume output semantics `(1, num_labels, num_words, max_width)`
- for each label index
- for each `start_word`
- for each `width`
- compute `end_word = start_word + width`
- threshold score
- map word offsets to character offsets
- produce `Span`

Rust example file:

`gline-rs/examples/gliner2_ner.rs`

Example:

```rs
use gliner_rs::model::gliner2::GLiNER2Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Model::from_dir("models/gliner2-multi-v1-onnx")?;

    let labels = vec![
        "person".to_string(),
        "company".to_string(),
    ];

    let entities = model.predict_entities(
        "Steve Jobs founded Apple",
        &labels,
        0.5,
    )?;

    println!("{entities:#?}");
    Ok(())
}
```

Run:

```sh
cargo run --example gliner2_ner
```

Python validation:

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained("lion-ai/gliner2-multi-v1-onnx")

entities = model.predict_entities(
    "Steve Jobs founded Apple",
    ["person", "company"]
)

print(entities)
```

Expected behavior:
- detects `"Steve Jobs"` as `person`
- detects `"Apple"` as `company`
- returns correct character offsets

This milestone is the **first required end-to-end working slice**.

---

# Milestone 3 — Classification

Implement:

`classify(text, labels)`

Architecture note:
GLiNER2 conceptually supports classification, but the coding agent must confirm whether classification is supported by the same monolithic export or requires a different model/export contract. If classification is not supported by the exact monolithic NER export, the implementation should:

1. document that fact clearly
2. implement classification against the correct GLiNER2 export if available
3. keep the API shape consistent

Rust example file:

`gline-rs/examples/gliner2_classification.rs`

Example target:

```rs
use gliner_rs::model::gliner2::GLiNER2Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Model::from_dir("models/gliner2-multi-v1-onnx")?;

    let labels = vec![
        "shopping".to_string(),
        "work".to_string(),
    ];

    let result = model.classify(
        "Buy milk",
        &labels,
    )?;

    println!("{result:#?}");
    Ok(())
}
```

Run:

```sh
cargo run --example gliner2_classification
```

Python validation:

```python
result = model.classify(
    "Buy milk",
    ["shopping", "work"]
)
print(result)
```

Expected behavior:
- returns a classification result consistent with the chosen export
- if multi-label classification is supported, document the behavior
- if only single-label classification is supported, document that clearly

---

# Milestone 4 — Structured Extraction

Implement:

`extract(text, schema)`

Architecture note:
Structured extraction is a core GLiNER2 capability conceptually, but the coding agent must confirm the export contract for the target ONNX model family. If a separate export or schema serialization rule is required, document it and implement against the confirmed interface.

Rust example file:

`gline-rs/examples/gliner2_extract.rs`

Example target:

```rust
use gliner_rs::model::gliner2::GLiNER2Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Model::from_dir("models/gliner2-multi-v1-onnx")?;

    // Replace with the actual schema type introduced by the implementation.
    let schema = "...";

    let result = model.extract(
        "Tim Cook is the CEO of Apple.",
        schema,
    )?;

    println!("{result:#?}");
    Ok(())
}
```

Run:

```sh
cargo run --example gliner2_extract
```

Python validation target:

```python
schema = ...
result = model.extract(
    "Tim Cook is the CEO of Apple.",
    schema
)
print(result)
```

Expected behavior:
- structured extraction API exists
- output format is documented and deterministic
- implementation matches the actual supported export semantics

---

# Milestone 5 — Relation Extraction

Implement:

`extract_relations(text, labels, schema)`

Architecture note:
As with classification and extraction, relation extraction should be implemented against the actual confirmed GLiNER2 export contract. If the chosen monolithic ONNX export does not support relations directly, that must be documented clearly and the correct model/export should be used.

Rust example file:

`gline-rs/examples/gliner2_relations.rs`

Example target:

```rs
use gliner_rs::model::gliner2::GLiNER2Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Model::from_dir("models/gliner2-multi-v1-onnx")?;

    let labels = vec![
        "person".to_string(),
        "organization".to_string(),
    ];

    // Replace with the actual relation schema type introduced by the implementation.
    let schema = "...";

    let result = model.extract_relations(
        "Bill Gates founded Microsoft.",
        &labels,
        schema,
    )?;

    println!("{result:#?}");
    Ok(())
}
```

Run:

```sh
cargo run --example gliner2_relations
```

Python validation target:

```python
labels = ["person", "organization"]
schema = ...
result = model.extract_relations(
    "Bill Gates founded Microsoft.",
    labels,
    schema
)
print(result)
```

Expected behavior:
- relation extraction API exists
- output format is documented
- relation extraction is implemented against a confirmed GLiNER2 export path

---

# Python API Target

The public Python interface must remain unchanged in spirit.

Target usage:

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained("lion-ai/gliner2-multi-v1-onnx")

entities = model.predict_entities(
    "Steve Jobs founded Apple",
    ["person", "company"]
)

classification = model.classify(
    "Buy milk",
    ["shopping", "work"]
)
```

Additional target methods:

```python
model.extract(...)
model.extract_relations(...)
```

Rust performs all inference.

Python remains a thin wrapper over Rust.

The coding agent should minimize Python-side logic changes and keep the Python API additive and backward compatible.

---

# Backward Compatibility

GLiNER v1 must remain functional.

Do not break:
- existing v1 Rust runtime paths
- existing Python calls:
  - `FastGLiNER.from_pretrained()`
  - `FastGLiNER.predict_entities()`
  - `FastGLiNER.extract_relations()`

The GLiNER2 runtime must be introduced **additively**.

---

# Acceptance Criteria

Implementation is complete when:

- GLiNER2 models load successfully
- the Rust `gliner2_load` example runs
- the Rust `gliner2_ner` example runs end-to-end
- classification example runs, or the exact required export difference is clearly documented and implemented against the correct export
- extraction example runs, or the exact required export difference is clearly documented and implemented against the correct export
- relation extraction example runs, or the exact required export difference is clearly documented and implemented against the correct export
- Python validation snippets work for the supported tasks
- existing GLiNER v1 functionality remains unchanged