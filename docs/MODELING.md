# Modeling

This document describes the model families implemented by `fast_gliner` and how each one turns text into structured output.

System layering, the Python/Rust boundary, and ONNX Runtime wiring are in [`ARCHITECTURE.md`](../ARCHITECTURE.md).

Inference logic lives in `gline-rs`. Python classes are thin wrappers.

---

# Model Families

| Family | Python class | Rust type | Tasks | ONNX layout |
|------|------|------|------|------|
| GLiNER | `FastGLiNER` | `GLiNER` / `InferenceMode` | NER, relation extraction | one graph, span or token mode |
| GLiNER2 | `FastGLiNER2` | `GLiNER2` | NER, classification, structured extraction, relations, multi-task schemas | one monolithic `span_scores` graph |
| GLiClass | `FastGLiClass` | `GLiClass` | sequence classification | one uni-encoder graph with a logits head |
| GLiFormer | `FastGLiFormer` | `GLiFormer` | NER, classification, relations, nested structured extraction | encoder plus separate task-head graphs |

GLiNER and GLiNER2 share the stage layout under `model/input`, `model/output`, and `model/pipeline`. GLiClass uses that layout for its own prompt and logits head. GLiFormer is a separate runtime in `model/gliformer.rs` because its checkpoint is split across several ONNX sessions.

---

# GLiNER

GLiNER is a bidirectional encoder for zero-shot named entity recognition. Labels are supplied at inference time and embedded in a prompt. The model scores spans. It does not generate text.

The prompt format is:

```
<<ENT>> person <<ENT>> organization <<SEP>> input text
```

`gliner_config.json` selects the pipeline:

| `mode` | Pipeline | Use |
|------|------|------|
| `span` (default) | `SpanPipeline` | span-level NER |
| `token` | `TokenPipeline` | token-level multitask models |

`max_width` comes from the same config. The default is 12 when the field is absent.

## Loader

`GLiNER::from_dir` requires:

```
tokenizer.json
onnx/model.onnx
gliner_config.json
```

The config mode chooses `InferenceMode::Span` or `InferenceMode::Token`.

## Span pipeline

```
TextInput
  → word split (RegexSplitter)
  → prompt (<<ENT>> … <<SEP>>)
  → Hugging Face encode
  → tensors
  → ONNX
  → span decode
  → greedy filter
```

Inputs:

| Tensor | Role |
|------|------|
| `input_ids` | token ids |
| `attention_mask` | attention mask |
| `words_mask` | which tokens start a word |
| `text_lengths` | text length |
| `span_idx` | candidate `(start, end)` pairs up to `max_width` |
| `span_mask` | valid span mask |

Output:

| Tensor | Role |
|------|------|
| `logits` | span scores, decoded as `(batch, num_words, max_width, num_classes)` |

Each kept span has text, label, score, and character offsets.

## Token pipeline

Token mode reads start, end, and inside logits and rebuilds spans from those positions. Relation extraction reuses this pipeline.

## Span filtering

Decoded spans pass through greedy search controlled by:

```
flat_ner
dup_label
multi_label
```

## Relation extraction

`extract_relations` runs NER, builds relation prompts from the detected spans, runs the token pipeline again, and validates pairs against a schema of relation name, subject labels, and object labels.

## Code

```
model/config.rs                  gliner_config.json
model/mod.rs                     GLiNER::from_dir
model/runtime.rs                 InferenceMode
model/pipeline/span.rs           span NER
model/pipeline/token.rs          token NER
model/pipeline/relation.rs       relations on top of token mode
model/input/prompt.rs            <<ENT>> prompt
model/input/tensors/span.rs      span tensors
model/input/tensors/token.rs     token tensors
model/output/decoded/span.rs     span logits
model/output/decoded/token.rs    token logits
model/output/decoded/greedy.rs   overlap filtering
model/output/relation.rs         relation decoding
```

---

# GLiNER2

GLiNER2 is a schema-driven encoder for several extraction tasks. A schema describes the requested output. The runtime builds that schema in tokenizer space and reads a single `span_scores` head.

GLiNER2 does not use the GLiNER `<<ENT>>` prompt and does not read `gliner_config.json`. Span width is fixed at 8, matching the exported graph.

Supported tasks:

- entity extraction
- classification
- structured extraction
- relation extraction
- a combined schema that runs those tasks together

## Loader

`GLiNER2::from_dir` requires:

```
tokenizer.json
model.onnx
```

`onnx/model.onnx` is used when that path exists.

Special tokens are resolved with `tokenizer.token_to_id`. Initialization fails if any of these are missing:

```
[P] [C] [E] [R] [L] [MASK]
[SEP_STRUCT] [SEP_TEXT]
[DESCRIPTION] [EXAMPLE] [OUTPUT]
```

## Schema prefix

A single-task prefix looks like:

```
( [P] entities ( [E] person [E] company ) ) [SEP_TEXT] Steve Jobs founded Apple
```

The task name is `entities`, `classification`, or `extraction`. Label text is split with the same word splitter as the input. `[P]` marks the task, and each `[E]` marks a label. `schema_positions` stores those token indexes after encoding, in order: `[P]`, then one entry per `[E]`.

## ONNX contract

The graph is a batch of one sequence.

| Input | Shape | Role |
|------|------|------|
| `input_ids` | `(1, seq_len)` | token ids |
| `attention_mask` | `(1, seq_len)` | mask |
| `text_positions` | `(num_words,)` | first subword of each text word |
| `schema_positions` | `(1 + num_fields,)` | `[P]` then each `[E]` |
| `span_idx` | `(1, num_words * max_width, 2)` | `(start_word, end_word)` pairs |

| Output | Shape | Role |
|------|------|------|
| `span_scores` | `(1, num_labels, num_words, max_width)` | score for each label, start word, and width |

`text_positions` comes from `Encoding::get_word_ids`. `span_idx` enumerates widths up to `max_width`. Decoding thresholds scores, maps word offsets back to characters, and applies the same greedy filters as GLiNER.

## Single-task methods

| Method | Task token | Decoding |
|------|------|------|
| `inference` | `entities` | spans |
| `classify` | `classification` | best span score per label, as `ClassificationOutput` |
| `extract` | `extraction` | spans grouped into schema fields |
| `extract_json` | `extraction` | same spans, emitted as JSON |
| `extract_relations` | entities, then a second entity pass over relation prompts | pairs validated by `RelationSchema` |

The exported graph has no separate classification head. `classify` scores each label by the best span score for that label.

`extract_json` accepts a map of object name to field specs. A spec is `name`, `name::str` (one value), or `name::list` (many values). A bare name is a list. Field names must be unique across the schema.

`extract_relations` detects entities first, builds relation prompts, and runs entity inference again. Relation decoding reuses `model/output/relation.rs`.

Calls are one sequence at a time. The Python wrapper rejects a batch longer than one string.

## Multi-task schema

`create_schema()` returns a builder. `extract(text, schema)` runs it.

```python
schema = (
    model.create_schema()
    .entities(["person", "company"])
    .relation("works_for", ["person"], ["company"])
    .classification("sentiment", ["positive", "neutral", "negative"])
    .structure("event")
        .field("date")
        .field("description")
)
result = model.extract(text, schema)
```

The result is:

```
{
  "classifications": { name: scores },
  "entities": [spans],
  "relations": [relations],
  "structures": { name: fields }
}
```

Execution:

```
entity labels only
  → inference()

any other schema
  → one extract() over a combined field list
       entities, relation prompts, and structure fields
  → decode spans into entities, relations, and structures
  → one classify() call per classification task
```

Classification stays on its own forward pass. Its labels are scored as categories, and the combined extraction pass does not emit them.

An entity-only schema skips the combined schema and calls `inference` directly.

## Code

```
model/runtime.rs                     GLiNER2::from_dir and task methods
model/input/schema.rs                special tokens and schema prefixes
model/input/tensors/schema.rs        GLiNER2 tensors
model/output/decoded/span_scores.rs  span_scores → spans
model/output/classification.rs       classification scores
model/output/extraction.rs           field grouping
model/pipeline/schema.rs             NER, classification, extraction
model/pipeline/multitask.rs          schema builder and combined execution
```

---

# GLiClass

GLiClass is a zero-shot sequence classifier. One forward pass scores one text against a caller-supplied label set. The supported export is a uni-encoder graph that already includes pooling and the MLP scorer.

Official Knowledgator checkpoints ship `model.safetensors`. This runtime loads ONNX exports such as the community `cnmoro/gliclass-*-onnx` repositories. Bi-encoder, fused bi-encoder, encoder-decoder, and decoder-kv exports use a different input contract and are outside this runtime.

## Loader

`GLiClass::from_dir` requires:

```
tokenizer.json
model.onnx
```

`onnx/model.onnx` is used when that path exists.

The vocabulary must contain `<<LABEL>>` and `<<SEP>>`. `<<EXAMPLE>>` is required only when the call includes few-shot examples.

`config.json` is optional. When it is present, the runtime reads `prompt_first`, `architecture_type`, and `encoder_config.max_position_embeddings`. The supported architecture is `uni-encoder`. The sequence cap is the smaller of `Parameters.max_length` and the encoder position limit.

When `config.json` is absent, labels are placed before the text and sequences are capped at 512 tokens.

## Prompt

Labels may be a flat list or a nested dict. A dict is flattened to dotted leaves such as `sentiment.positive` before scoring. Logits stay one sigmoid per candidate, in that flattened order.

Example, with `prompt_first` and no extras:

```
<<LABEL>>shopping<<LABEL>>work<<LABEL>>personal<<SEP>>Buy milk and eggs after work
```

A task prompt is concatenated immediately after `<<SEP>>`. Few-shot examples are appended after the text, and one `<<SEP>>` follows the example block:

```
<<LABEL>>positive<<LABEL>>negative<<SEP>>Classify the sentiment:The battery life is incredible<<EXAMPLE>>Love this item \nLabels:\n positive<<SEP>>
```

Text-first checkpoints put the text before the label block. The task prompt still follows `<<SEP>>`, and the examples still sit at the end.

Example labels are prompt text. They do not add logits.

## Tensors and scores

```
input_ids
attention_mask
```

`logits` has shape `[1, num_labels]`. Each score is an independent sigmoid probability, sorted from highest to lowest.

`classify(..., return_hierarchical=True)` rebuilds those scores into the caller's label tree. A missing leaf is `0.0`. A flat label list becomes `{label: score}` in input order.

## Code

```
model/runtime.rs                 GLiClass::from_dir, classify, classify_with
model/input/tensors/gliclass.rs  prompt, input_ids, attention_mask
model/output/gliclass.rs         sigmoid scores and nested trees
model/pipeline/gliclass.rs       classification pipeline
```

---

# GLiFormer

GLiFormer is a multitask encoder with separate task heads. NER, classification, relations, and the flat multi-task schema match `FastGLiNER2`. Nested records use `structure`, which GLiNER2 does not provide. `classify` returns `(label, score)` pairs sorted from highest to lowest, the same contract as `FastGLiClass`.

The checkpoint is split. `from_pretrained` downloads `*.json` and `onnx/*.onnx`, and loading fails unless `onnx/encoder.onnx` is present.

## Loader

`GLiFormer::from_dir` opens five sessions:

```
tokenizer.json
gliner_config.json
onnx/encoder.onnx
onnx/ner.onnx
onnx/classification.onnx
onnx/relations.onnx
onnx/structuring.onnx
```

`gliner_config.json` supplies `max_len`, `hidden_size`, and the token strings and token indexes used to gather embeddings (`seq_token`, `parent_token`, `sep_token`, `ent_token`, `cat_token`, `rel_token`, `child_token`, plus the head-specific indexes). `structuring_config.multi_level` says whether nested records are available.

## Prompt

One task group is encoded at a time. Pieces are pretokenized in processor order: `[SEQ]`, a `[SCHEMA]` group, a final `[SEP]`, then whitespace words.

```
[SEQ] [SCHEMA] [ENT] person [ENT] company [SEP] [SEP] Steve Jobs founded Apple
```

The marker depends on the task: entity, class, relation, or field. Relation prompts place entity markers and relation markers in the same schema group.

The encoder returns token embeddings. Each head gathers the word embeddings and the embeddings at the task's marker token, then emits logits. NER and structuring decode those logits as BIO tags. Classification applies a sigmoid per label.

## Structured extraction

`structure(text, schema)` is the nested record API. The schema is an object, array, and scalar tree. Python accepts either a nested dict or a Pydantic model class and compiles both to that tree before the call enters Rust. A flat field list is the same method:

```python
model.structure(text, {"employee": ["name", "company"]})

model.structure(text, {
    "company": {
        "name": "str",
        "departments": [{
            "name": "str",
            "employees": [{"name": "str", "role": "str"}],
        }],
    }
})
```

Scalar field labels are dotted paths, so `name` on a company and `name` on an employee stay distinct (`name` and `departments.employees.name`). A nested schema is prompted depth-first: the record name, a field marker for each scalar, and a child marker plus an end marker around each nested object. Each active structuring anchor becomes one record. Spans join the anchor with the highest membership score. When one anchor contains spans from several schema levels, those levels become separate records and adjacent levels from that anchor are linked. A nested schema also reads `anchor_relations`, which are probabilities, and attaches a child record to the parent of the matching schema path. When a slot was split, a child with no relation above the threshold attaches to the nearest preceding parent of that path. The exported graph includes that tensor only when the structuring head is multi-level. A nested schema on a flat checkpoint returns an error.

Each top-level schema name maps to a list of records. Field values are the extracted text. A missing scalar is null, and a missing list is empty. A Pydantic schema returns instances of that model. A dict schema returns dicts.

`extract_json` stays on `FastGLiNER2`. It is not a GLiFormer method.

## Multi-task schema

`extract(text, schema)` runs each requested task on its own head:

```
entities        → ner head
classifications → classification head, once per task
relations       → relations head
structures      → structuring head, once per structure
```

`.structure().field()` on that schema stays a flat field list. Nested records go through `structure()`. There is no combined `span_scores` pass. An entity-only schema still goes through the NER head.

## Code

```
model/gliformer.rs        sessions, task methods, BIO decode
model/structure.rs        schema tree and record assembly
model/input/gliformer.rs  config and prompt
```

---

# References

[1] Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois.
**GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.**
Proceedings of NAACL 2024.
<https://aclanthology.org/2024.naacl-long.300>

[2] Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, and Ash Lewis.
**GLiNER2: Schema-Driven Multi-Task Learning for Structured Information Extraction.**
Proceedings of EMNLP 2025 System Demonstrations.
<https://aclanthology.org/2025.emnlp-demos.10/>

[3] Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, Alexander Yavorskyi, and Mykyta Yaroshenko.
**GLiClass: Generalist Lightweight Model for Sequence Classification Tasks.**
arXiv:2508.07662.
<https://arxiv.org/abs/2508.07662>

[4] Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, and Mykyta Yaroshenko.
**GLiFormer: A Generalist Multitask Transformer Encoder.**
<https://www.knowledgator.com/research/gliformer>

Implementations:

- [GLiNER](https://github.com/urchade/GLiNER)
- [GLiNER2](https://github.com/fastino-ai/GLiNER2)
- [gline-rs](https://github.com/fbilhaut/gline-rs)
