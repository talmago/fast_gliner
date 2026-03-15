# TASK: GLiNER2 Schema-Driven Multi-Task Pipeline

## Goal

Implement a high-level **GLiNER2 pipeline API** that allows running multiple information extraction tasks from a single schema definition with optimized inference.

The pipeline should:

- Allow users to define **classification, NER, relations, and structured extraction** in one schema.
- Execute tasks using **fewer model forward passes** than running them separately.
- Provide a **simple schema-driven public API**.

This mirrors the design described in the GLiNER2 paper.

---

# Motivation

The current GLiNER2 runtime exposes individual task APIs:

```
GLiNER2
 ├─ classify()
 ├─ inference()        # NER
 ├─ extract()
 └─ extract_relations()
```

Running multiple tasks requires multiple calls:

classify(text)
predict_entities(text)
extract_relations(text)
extract(text)

Each call repeats:

- tokenization
- tensor construction
- ONNX inference

This is inefficient.

The new pipeline should enable:

schema = extractor.create_schema() \
    .classification(...)    .entities(...)    .relations(...)    .structure(...)

result = extractor.extract(text, schema)

With optimized inference.

---

# High Level Architecture

GLiNER2Pipeline
      │
      ▼
Combined Schema
      │
      ▼
Single GLiNER2 forward pass
      │
      ▼
Shared span decoding
      │
      ├── classification decoding
      ├── entity decoding
      ├── relation decoding
      └── structured extraction decoding

---

# Core Components

## 1. Pipeline Schema

GLiNER2PipelineSchema

Holds all task definitions.

Example internal structure:

```rust
pub struct GLiNER2PipelineSchema {
    pub classifications: Vec<ClassificationTask>,
    pub entity_labels: Vec<String>,
    pub relation_schema: Option<RelationSchema>,
    pub extraction_schema: Option<ExtractionSchema>,
}
```

---

## 2. Pipeline

GLiNER2Pipeline

Responsible for:

- building the combined schema
- executing optimized inference
- dispatching results to task decoders

```rust
pub struct GLiNER2Pipeline {
    model: GLiNER2
}
```

---

## 3. Pipeline Output

GLiNER2PipelineOutput

Example structure:

```rust
pub struct GLiNER2PipelineOutput {
    pub classifications: HashMap<String, ClassificationResult>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub structures: HashMap<String, ExtractionOutput>,
}
```

---

# Public Rust API

```rust
let schema = GLiNER2PipelineSchema::new()
    .classification("document_type", ["news", "report"])
    .entities(["person", "company"])
    .relations(["works_for", "acquired"])
    .structure("event")
        .field("date")
        .field("description");

let pipeline = GLiNER2Pipeline::new(model);

let result = pipeline.extract(text, schema);
```

---

# Public Python API

The Python API mirrors the Rust builder.

```python
schema = (
    extractor.create_schema()
        .classification("document_type", ["news", "report", "announcement"])
        .entities(["person", "company"])
        .relations(["works_for", "acquired"])
        .structure("event")
            .field("date", dtype="str")
            .field("description", dtype="str")
)

result = extractor.extract(text, schema)
```

---

# Inference Strategy

## Optimized Single-Pass Execution

The pipeline should attempt to run:

- classification
- NER
- structured extraction

in **one model pass**.

Relations may require a second pass depending on schema structure.

### Pass 1

Text → tokenizer → tensors → ONNX

Decode:

- classification
- entities
- extraction

### Pass 2 (optional)

relations

Using entities detected in pass 1.

---

# Design Constraints

The pipeline must:

- reuse existing GLiNER2 runtime components
- not modify existing APIs
- remain fully backward compatible

Existing APIs must remain functional:

GLiNER2::classify()
GLiNER2::inference()
GLiNER2::extract()
GLiNER2::extract_relations()

---

# File Structure

New module:

src/model/gliner2/pipeline.rs

Possible structure:

pipeline/
    schema.rs
    pipeline.rs
    output.rs

---

# Example

Rust example:

```
examples/gliner2_pipeline.rs
```

Python example:

```
README.md
```

---

# Future Extensions

The pipeline design should allow adding:

- event extraction
- hierarchical schemas
- schema validation
- batch pipelines
