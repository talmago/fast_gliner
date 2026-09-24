# CPU performance

This note covers **GLiNER2**, **GLiClass**, and **GLiFormer**. Original GLiNER span and token mode is unchanged and is not a work item.

Numbers below are medians of 5 runs after 2 warmups, release build, 4 intra-op threads (the `RuntimeParameters` default). A "batch" is eight copies of the same sentence, still one `session.run` per text, which is what these runtimes do today.

Reproduce with:

```bash
cargo run --release --example profile-cpu --manifest-path gline-rs/Cargo.toml -- models
```

## Measured

| Runtime | Case | Pre-ONNX | Session | Post-ONNX | Total |
|---|---|---:|---:|---:|---:|
| GLiNER2 NER | sentence | 66.68 ms (72%) | 26.13 ms | 0.04 ms | 92.85 ms |
| GLiNER2 NER | paragraph | 69.96 ms (50%) | 70.21 ms | 0.15 ms | 140.32 ms |
| GLiNER2 NER | 8 sentences | 519.35 ms (71%) | 212.21 ms | 0.49 ms | 732.06 ms |
| GLiNER2 classify | sentence | 65.72 ms (72%) | 25.54 ms | 0.00 ms | 91.26 ms |
| GLiNER2 extract | multi-task schema, one call | 93.34 ms (69%) | 41.07 ms | 0.03 ms | 134.45 ms |
| GLiNER2 multi-task | extract plus one classify | 157.84 ms (69%) | 72.45 ms | 0.05 ms | 230.34 ms |
| GLiClass | sentence | 48.07 ms (86%) | 8.08 ms | 0.00 ms | 56.15 ms |
| GLiClass | paragraph | 48.61 ms (71%) | 20.17 ms | 0.00 ms | 68.78 ms |
| GLiClass | 8 sentences | 394.78 ms (86%) | 64.94 ms | 0.13 ms | 459.85 ms |

GLiFormer NER splits the session into the encoder and the NER head. Host work between those two runs is the gather of word and label embeddings.

| Case | Pre | Encoder | Between | NER head | Post | Total |
|---|---:|---:|---:|---:|---:|---:|
| sentence | 0.72 ms | 19.03 ms (84%) | 0.01 ms | 2.97 ms (13%) | 0.00 ms | 22.73 ms |
| paragraph | 0.99 ms | 42.97 ms (79%) | 0.01 ms | 10.19 ms (19%) | 0.01 ms | 54.18 ms |
| 8 sentences | 5.54 ms | 154.03 ms (84%) | 0.02 ms | 23.37 ms (13%) | 0.03 ms | 182.99 ms |

Checkpoints: `models/gliner2-multi-v1-onnx`, `models/gliclass-small-v1.0` (`onnx/model.onnx`), `models/gliformer-base-v1`.

GLiNER2 multi-task is the schema from the pipeline example: one classification, two entity labels, two relations, and a two-field structure. `extract_with_schema` runs that as one extraction call, then one extra classification call. The row above is those two calls added together. A second classification task would add another classify, about 91 ms on this sentence.

GLiFormer structured extraction is one nested company schema (one top-level record, so one encoder run). The structuring head is a third session, `onnx/structuring.onnx`.

| Case | Pre | Encoder | NER head | Structuring head | Post | Total |
|---|---:|---:|---:|---:|---:|---:|
| nested structure, one record | 0.76 ms | 32.52 ms (29%) | 4.64 ms (4%) | 76.18 ms (67%) | 0.03 ms | 114.13 ms |

Post-ONNX decode is under 0.5 ms in every case, including structure assembly. Host gathers inside GLiFormer stay under 0.05 ms. Those copies are not a milestone.

## Milestones

**1. Cut GLiNER2 and GLiClass preprocess.** This is the first code change. On a short sentence the host path before `session.run` is 72% of GLiNER2 NER, 72% of GLiNER2 classification, 69% of the multi-task pair, and 86% of GLiClass. That path is `prepare_sequence` in [gline-rs/src/model/input/tensors/schema.rs](../gline-rs/src/model/input/tensors/schema.rs) and `prepare_gliclass` in [gline-rs/src/model/input/tensors/gliclass.rs](../gline-rs/src/model/input/tensors/gliclass.rs): schema or prompt assembly, one tokenizer encode, then tensor packing. Multi-task pays it once per session, so the same schema is about 158 ms of preprocess before either graph runs. Split that stage in the profile before rewriting it, and keep outputs identical. A paragraph does not cost much more preprocess than a sentence (about 70 ms either way on GLiNER2 NER), so the cost is not "more words" in a simple way.

**2. Batch inference.** Eight texts cost about eight times one text, because each text is a full preprocess plus a session. [gline-rs/src/model/runtime.rs](../gline-rs/src/model/runtime.rs) `GLiNER2::inference` loops one text per `model.inference`. `GLiClass::classify` takes one string. [gline-rs/src/model/gliformer.rs](../gline-rs/src/model/gliformer.rs) `predict_entities` takes one string. The Python wrappers reject a list longer than one.

Batching the session is the main lever for GLiFormer NER, where the encoder is about 80% of the request and preprocess is about 1 ms. Structured extraction is different: the structuring head is 67% of that request and the encoder is 29%, and each top-level record is another full pass. For GLiNER2 and GLiClass, batching the session only removes the smaller share until milestone 1 lands; do both. A multi-task schema stays two kinds of call (one extract, then one classify per classification task); batch each kind on its own. Relation extraction stays two passes (NER, then relation prompts). GLiFormer batches the encoder and each head the same way.

Done when a 32-text GLiNER2 NER call is one session run and matches 32 single-text calls, and the same check holds for GLiClass scores and GLiFormer NER.

**3. Length bucketing, after batches exist.** Pad within length groups so a short text is not padded to the longest text in the request. Apply that to the GLiNER2, GLiClass, and GLiFormer input builders. Measure against the unbucketed batch.

**4. Thread sweep on the session-heavy paths.** Record 1, 4, and physical-core runs for the GLiFormer encoder, the structuring head, and GLiNER2 on a paragraph, where the session is already about half the request. Leave graph rewrites and execution-provider changes out of this item.

## Out of scope

Original GLiNER span and token mode. No pipeline edits, no batching work, and no tokenizer changes on that path. Quantized weights are not part of this plan.
