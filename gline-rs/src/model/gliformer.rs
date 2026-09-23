//! GLiFormer text runtime.
//!
//! The checkpoint is split into an encoder and task heads. This module builds
//! the whitespace prompt, gathers word and label embeddings, and decodes the
//! head outputs into the existing GLiNER2 result types.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};
use orp::params::RuntimeParameters;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::model::input::gliformer::{
    prepare_prompt, GLiFormerConfig, GLiFormerPrompt, PreparedPrompt,
};
use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::schema::ExtractionSchema;
use crate::model::output::classification::{ClassificationOutput, ClassificationScore};
use crate::model::output::decoded::bio::{decode_bio, BioSpan};
use crate::model::output::decoded::SpanOutput;
use crate::model::output::extraction::{ExtractedField, ExtractedValue, ExtractionOutput};
use crate::model::output::relation::{Relation, RelationEntity, RelationOutput};
use crate::model::params::Parameters;
use crate::model::pipeline::multitask::{
    GLiNER2PipelineOutput, GLiNER2PipelineRelation, GLiNER2PipelineSchema,
};
use crate::model::structure::{
    assemble_structure, hierarchy_prompt, SchemaNode, StructureSchema, StructureSpan,
};
use crate::text::span::Span;
use crate::text::token::Token;
use crate::text::tokenizer::HFTokenizer;
use crate::util::result::Result;

const ENCODER_FILE: &str = "onnx/encoder.onnx";
const NER_FILE: &str = "onnx/ner.onnx";
const CLASSIFICATION_FILE: &str = "onnx/classification.onnx";
const RELATIONS_FILE: &str = "onnx/relations.onnx";
const STRUCTURING_FILE: &str = "onnx/structuring.onnx";

pub struct GLiFormer {
    tokenizer: HFTokenizer,
    config: GLiFormerConfig,
    params: Parameters,
    encoder: Session,
    ner: Session,
    classification: Session,
    relations: Session,
    structuring: Session,
}

struct StructuringScores {
    anchor_count: usize,
    membership: Vec<f32>,
    objectness: Vec<f32>,
    anchor_mask: Vec<f32>,
    anchor_relations: Option<Vec<f32>>,
}

struct EncodedSequence {
    embeds: Vec<f32>,
    ids: Vec<i64>,
    words_mask: Vec<i64>,
    words: Vec<Token>,
    sequence_len: usize,
}

impl GLiFormer {
    pub fn from_dir<P: AsRef<Path>>(
        model_dir: P,
        parameters: Parameters,
        runtime_parameters: RuntimeParameters,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("gliner_config.json");
        for (label, path) in [
            ("tokenizer", tokenizer_path.as_path()),
            ("config", config_path.as_path()),
            ("encoder", &model_dir.join(ENCODER_FILE)),
            ("ner head", &model_dir.join(NER_FILE)),
            ("classification head", &model_dir.join(CLASSIFICATION_FILE)),
            ("relation head", &model_dir.join(RELATIONS_FILE)),
            ("structuring head", &model_dir.join(STRUCTURING_FILE)),
        ] {
            if !path.is_file() {
                return Err(format!("missing GLiFormer {label}: {}", path.display()).into());
            }
        }

        let providers: Vec<_> = runtime_parameters.execution_providers().to_vec();
        let open = |path: PathBuf| -> Result<Session> {
            Ok(Session::builder()?
                .with_intra_threads(runtime_parameters.threads())?
                .with_execution_providers(providers.clone())?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(path)?)
        };

        Ok(Self {
            tokenizer: HFTokenizer::from_file(tokenizer_path)?,
            config: GLiFormerConfig::from_file(&config_path)?,
            params: parameters,
            encoder: open(model_dir.join(ENCODER_FILE))?,
            ner: open(model_dir.join(NER_FILE))?,
            classification: open(model_dir.join(CLASSIFICATION_FILE))?,
            relations: open(model_dir.join(RELATIONS_FILE))?,
            structuring: open(model_dir.join(STRUCTURING_FILE))?,
        })
    }

    pub fn inference(&self, text: &str, labels: &[String]) -> Result<SpanOutput> {
        let spans = self.predict_entities(text, labels)?;
        Ok(SpanOutput::new(
            vec![text.to_string()],
            labels.to_vec(),
            vec![spans],
        ))
    }

    pub fn predict_entities(&self, text: &str, labels: &[String]) -> Result<Vec<Span>> {
        if labels.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.encode(text, &GLiFormerPrompt::Entities(labels.to_vec()))?;
        let words = self.word_embeddings(&encoded)?;
        let children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.class_token_index,
        )?;
        if children.shape()[1] == 0 {
            return Ok(Vec::new());
        }
        let parent = self.parent_embedding(&encoded)?;
        let logits = self.run_ner(&words, &children, &parent)?;
        let word_count = words.shape()[1];
        let class_count = children.shape()[1];
        let decoded = decode_bio(&logits, word_count, class_count, self.params.threshold);
        Ok(spans_from_bio(text, &encoded.words, &decoded, labels))
    }

    pub fn classify(&self, text: &str, labels: &[String]) -> Result<ClassificationOutput> {
        if labels.is_empty() {
            return Err("classification requires at least one label".into());
        }
        if text.trim().is_empty() {
            return Err("invalid input: text contains no tokenizable words".into());
        }
        let encoded = self.encode(text, &GLiFormerPrompt::Classes(labels.to_vec()))?;
        let words = self.word_embeddings(&encoded)?;
        let children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.cat_token_index,
        )?;
        let parent = self.parent_embedding(&encoded)?;
        let cls_embed = row_matrix(&encoded.embeds, self.config.hidden_size, 0)?;
        let logits = self.run_classification(&words, &children, &parent, &cls_embed)?;
        let mut scores = labels
            .iter()
            .enumerate()
            .map(|(index, label)| ClassificationScore {
                label: label.clone(),
                score: sigmoid(*logits.get(index).unwrap_or(&0.0)),
            })
            .collect::<Vec<_>>();
        scores.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ClassificationOutput {
            text: text.to_string(),
            scores,
        })
    }

    pub fn extract_relations(
        &self,
        text: &str,
        labels: &[String],
        schema: &RelationSchema,
    ) -> Result<RelationOutput> {
        let relations = self.relation_triples(text, labels, schema)?;
        Ok(RelationOutput {
            texts: vec![text.to_string()],
            entities: labels.to_vec(),
            relations: vec![relations],
        })
    }

    pub fn create_schema(&self) -> GLiNER2PipelineSchema {
        GLiNER2PipelineSchema::new()
    }

    pub fn extract_with_schema(
        &self,
        text: &str,
        schema: &GLiNER2PipelineSchema,
    ) -> Result<GLiNER2PipelineOutput> {
        let mut output = GLiNER2PipelineOutput::default();
        if !schema.entity_labels.is_empty() {
            output.entities = self.predict_entities(text, &schema.entity_labels)?;
        }
        for classification in &schema.classifications {
            output.classifications.insert(
                classification.name.clone(),
                self.classify(text, &classification.labels)?,
            );
        }
        if !schema.relations.is_empty() {
            let relation_schema = relation_schema_from_pipeline(&schema.relations);
            let labels = if schema.entity_labels.is_empty() {
                endpoint_labels(&schema.relations)
            } else {
                schema.entity_labels.clone()
            };
            output.relations = self.relation_triples(text, &labels, &relation_schema)?;
        }
        for structure in &schema.structures {
            output.structures.insert(
                structure.name.clone(),
                self.extract_fields(text, &structure.fields)?,
            );
        }
        Ok(output)
    }

    pub fn extract(&self, text: &str, schema: &ExtractionSchema) -> Result<ExtractionOutput> {
        let flattened = schema.flatten_labels()?;
        let raw = self.extract_fields(text, &flattened.labels)?;
        let mut grouped = vec![Vec::new(); flattened.field_names.len()];
        for field in raw.fields {
            for value in field.values {
                if let Some(index) = flattened
                    .labels
                    .iter()
                    .position(|label| label == &value.label)
                {
                    grouped[flattened.label_to_field[index]].push(value);
                }
            }
        }
        Ok(ExtractionOutput {
            text: text.to_string(),
            fields: flattened
                .field_names
                .into_iter()
                .zip(grouped)
                .map(|(name, values)| ExtractedField { name, values })
                .collect(),
        })
    }

    pub fn extract_json(
        &self,
        text: &str,
        schema: &HashMap<String, Vec<String>>,
    ) -> Result<serde_json::Value> {
        if schema.is_empty() {
            return Err("invalid JSON schema: must contain at least one object".into());
        }
        let mut objects = serde_json::Map::new();
        for (object_name, specs) in schema {
            if object_name.trim().is_empty() {
                return Err("invalid JSON schema: object name cannot be empty".into());
            }
            if specs.is_empty() {
                return Err(format!(
                    "invalid JSON schema: object `{object_name}` must contain at least one field spec"
                )
                .into());
            }
            let mut field_names = Vec::new();
            let mut modes = Vec::new();
            for spec in specs {
                let (name, single) = parse_field_spec(spec)?;
                field_names.push(name);
                modes.push(single);
            }
            let extracted = self.extract_fields(text, &field_names)?;
            let mut object = serde_json::Map::new();
            for (field, single) in extracted.fields.iter().zip(modes) {
                let texts = field
                    .values
                    .iter()
                    .map(|value| serde_json::Value::String(value.text.clone()))
                    .collect::<Vec<_>>();
                let value = if single {
                    texts.into_iter().next().unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Array(texts)
                };
                object.insert(field.name.clone(), value);
            }
            objects.insert(
                object_name.clone(),
                serde_json::Value::Array(vec![serde_json::Value::Object(object)]),
            );
        }
        Ok(serde_json::Value::Object(objects))
    }

    /// Extracts nested records described by `schema`.
    ///
    /// Each top-level entry is one structuring pass. The result uses the same
    /// names, and each value is a list of records. A nested object schema
    /// requires a multi-level checkpoint and anchor-relation scores.
    pub fn structure(&self, text: &str, schema: &StructureSchema) -> Result<serde_json::Value> {
        if schema.fields.is_empty() {
            return Err("structure schema must contain at least one record".into());
        }
        let mut objects = serde_json::Map::new();
        for (name, node) in &schema.fields {
            if node.has_nested_objects() && !self.config.multi_level {
                return Err(format!(
                    "structure `{name}` is nested, but this checkpoint structuring head is not multi-level"
                )
                .into());
            }
            objects.insert(
                name.clone(),
                serde_json::Value::Array(self.structure_object(text, name, node)?),
            );
        }
        Ok(serde_json::Value::Object(objects))
    }

    fn hierarchy_tokens(&self) -> Result<(String, String)> {
        match (&self.config.nest_token, &self.config.end_token) {
            (Some(child), Some(end)) => Ok((child.clone(), end.clone())),
            _ => Err(
                "nested structure requires structuring_child_token and structuring_end_token in gliner_config.json"
                    .into(),
            ),
        }
    }

    fn structure_object(
        &self,
        text: &str,
        name: &str,
        schema: &SchemaNode,
    ) -> Result<Vec<serde_json::Value>> {
        let fields = schema.scalar_fields()?;
        if fields.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let labels = fields
            .iter()
            .map(|field| field.label.clone())
            .collect::<Vec<_>>();
        let prompt = if schema.has_nested_objects() {
            let (child_token, end_token) = self.hierarchy_tokens()?;
            GLiFormerPrompt::Hierarchy {
                name: name.to_string(),
                pieces: hierarchy_prompt(
                    schema,
                    &self.config.field_token,
                    &child_token,
                    &end_token,
                )?,
            }
        } else {
            GLiFormerPrompt::Fields(labels.clone())
        };
        let encoded = self.encode(text, &prompt)?;
        let words = self.word_embeddings(&encoded)?;
        let children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.child_token_index,
        )?;
        let parent = self.parent_embedding(&encoded)?;
        let logits = self.run_ner(&words, &children, &parent)?;
        let decoded = decode_bio(
            &logits,
            words.shape()[1],
            children.shape()[1],
            self.params.threshold,
        );
        if decoded.is_empty() {
            return Ok(Vec::new());
        }
        let rendered = spans_from_bio(text, &encoded.words, &decoded, &labels);
        let predictions = rendered
            .into_iter()
            .zip(&decoded)
            .map(|(span, bio)| {
                let (start, _) = span.offsets();
                StructureSpan {
                    label_index: bio.class_index,
                    text: span.text().to_string(),
                    score: span.probability(),
                    start,
                }
            })
            .collect::<Vec<_>>();
        let head = self.run_structuring(&words, &parent, &decoded)?;
        assemble_structure(
            schema,
            &fields,
            &predictions,
            head.anchor_count,
            &head.membership,
            &head.objectness,
            &head.anchor_mask,
            head.anchor_relations.as_deref(),
            self.params.threshold,
        )
    }

    fn extract_fields(&self, text: &str, fields: &[String]) -> Result<ExtractionOutput> {
        if fields.is_empty() || text.trim().is_empty() {
            return Ok(ExtractionOutput {
                text: text.to_string(),
                fields: fields
                    .iter()
                    .map(|name| ExtractedField {
                        name: name.clone(),
                        values: Vec::new(),
                    })
                    .collect(),
            });
        }
        let encoded = self.encode(text, &GLiFormerPrompt::Fields(fields.to_vec()))?;
        let words = self.word_embeddings(&encoded)?;
        let children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.child_token_index,
        )?;
        let parent = self.parent_embedding(&encoded)?;
        let logits = self.run_ner(&words, &children, &parent)?;
        let decoded = decode_bio(
            &logits,
            words.shape()[1],
            children.shape()[1],
            self.params.threshold,
        );
        let kept = self.filter_structure_spans(&words, &parent, &decoded)?;
        let spans = spans_from_bio(text, &encoded.words, &kept, fields);
        let mut grouped: Vec<Vec<ExtractedValue>> = vec![Vec::new(); fields.len()];
        for (span, bio) in spans.into_iter().zip(kept) {
            if let Some(bucket) = grouped.get_mut(bio.class_index) {
                let (start, end) = span.offsets();
                bucket.push(ExtractedValue {
                    text: span.text().to_string(),
                    label: span.class().to_string(),
                    start,
                    end,
                    score: span.probability(),
                });
            }
        }
        Ok(ExtractionOutput {
            text: text.to_string(),
            fields: fields
                .iter()
                .zip(grouped)
                .map(|(name, values)| ExtractedField {
                    name: name.clone(),
                    values,
                })
                .collect(),
        })
    }

    fn filter_structure_spans(
        &self,
        words: &Array3<f32>,
        parent: &Array2<f32>,
        spans: &[BioSpan],
    ) -> Result<Vec<BioSpan>> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }
        let entity_count = spans.len();
        let head = self.run_structuring(words, parent, spans)?;
        let membership = head.membership;
        let objectness = head.objectness;
        let anchor_mask = head.anchor_mask;
        let anchors = head.anchor_count;
        let mut kept = Vec::new();
        for (span_index, span) in spans.iter().enumerate() {
            let assigned = (0..anchors).any(|anchor| {
                let active = anchor_mask.get(anchor).copied().unwrap_or(0.0) > 0.5
                    && sigmoid(objectness.get(anchor).copied().unwrap_or(f32::NEG_INFINITY))
                        > self.params.threshold;
                let score_index = anchor * entity_count + span_index;
                active
                    && sigmoid(
                        membership
                            .get(score_index)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY),
                    ) > self.params.threshold
            });
            if assigned {
                kept.push(span.clone());
            }
        }
        Ok(kept)
    }

    fn run_structuring(
        &self,
        words: &Array3<f32>,
        parent: &Array2<f32>,
        spans: &[BioSpan],
    ) -> Result<StructuringScores> {
        let entity_count = spans.len().max(1);
        let mut span_idx = Array3::<i64>::zeros((1, entity_count, 2));
        let span_mask = Array2::<f32>::ones((1, entity_count));
        for (index, span) in spans.iter().enumerate() {
            span_idx[[0, index, 0]] = span.start as i64;
            span_idx[[0, index, 1]] = span.end as i64;
        }
        let word_mask = Array2::<f32>::ones((1, words.shape()[1]));
        let outputs = self.structuring.run(ort::inputs![
            "words" => words.view(),
            "word_mask" => word_mask.view(),
            "span_idx" => span_idx.view(),
            "span_mask" => span_mask.view(),
            "parent" => parent.view(),
        ]?)?;
        let membership = tensor_f32(
            outputs
                .get("membership")
                .ok_or("structuring membership missing")?,
        )?;
        let objectness = tensor_f32(
            outputs
                .get("objectness")
                .ok_or("structuring objectness missing")?,
        )?;
        let anchor_mask = tensor_f32(
            outputs
                .get("anchor_mask")
                .ok_or("structuring anchor mask missing")?,
        )?;
        let anchor_relations = match outputs.get("anchor_relations") {
            Some(value) => Some(tensor_f32(value)?),
            None => None,
        };
        Ok(StructuringScores {
            anchor_count: objectness.len(),
            membership,
            objectness,
            anchor_mask,
            anchor_relations,
        })
    }

    fn relation_triples(
        &self,
        text: &str,
        labels: &[String],
        schema: &RelationSchema,
    ) -> Result<Vec<Relation>> {
        let relation_labels = schema.relations().keys().cloned().collect::<Vec<_>>();
        if relation_labels.is_empty() || labels.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.encode(
            text,
            &GLiFormerPrompt::Relations {
                entities: labels.to_vec(),
                relations: relation_labels.clone(),
            },
        )?;
        let words = self.word_embeddings(&encoded)?;
        let entity_children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.class_token_index,
        )?;
        let relation_children = gather_token(
            &encoded.embeds,
            encoded.sequence_len,
            self.config.hidden_size,
            &encoded.ids,
            self.config.rel_token_index,
        )?;
        let parent = self.parent_embedding(&encoded)?;
        let ner_logits = self.run_ner(&words, &entity_children, &parent)?;
        let entities = decode_bio(
            &ner_logits,
            words.shape()[1],
            entity_children.shape()[1],
            self.params.threshold,
        );
        if entities.len() < 2 || relation_children.shape()[1] == 0 {
            return Ok(Vec::new());
        }
        let entity_count = entities.len();
        let mut span_idx = Array3::<i64>::zeros((1, entity_count, 2));
        let span_mask = Array2::<f32>::ones((1, entity_count));
        for (index, span) in entities.iter().enumerate() {
            span_idx[[0, index, 0]] = span.start as i64;
            span_idx[[0, index, 1]] = span.end as i64;
        }
        let word_mask = Array2::<f32>::ones((1, words.shape()[1]));
        let outputs = self.relations.run(ort::inputs![
            "words" => words.view(),
            "word_mask" => word_mask.view(),
            "span_idx" => span_idx.view(),
            "span_mask" => span_mask.view(),
            "relations" => relation_children.view(),
        ]?)?;
        let scores = tensor_f32(
            outputs
                .get("relation_logits")
                .ok_or("relation logits missing")?,
        )?;
        let entity_spans = spans_from_bio(text, &encoded.words, &entities, labels);
        let mut triples = Vec::new();
        let relation_count = relation_children.shape()[1];
        for subject_index in 0..entity_count {
            for object_index in 0..entity_count {
                if subject_index == object_index {
                    continue;
                }
                let Some(subject) = entity_spans.get(subject_index) else {
                    continue;
                };
                let Some(object) = entity_spans.get(object_index) else {
                    continue;
                };
                for (relation_index, relation_name) in relation_labels.iter().enumerate() {
                    let Some(spec) = schema.relations().get(relation_name) else {
                        continue;
                    };
                    if !spec.allows_subject(subject.class()) || !spec.allows_object(object.class())
                    {
                        continue;
                    }
                    let offset = ((subject_index * entity_count + object_index) * relation_count)
                        + relation_index;
                    let probability =
                        sigmoid(scores.get(offset).copied().unwrap_or(f32::NEG_INFINITY));
                    if probability <= self.params.threshold {
                        continue;
                    }
                    let (subject_start, subject_end) = subject.offsets();
                    let (object_start, object_end) = object.offsets();
                    triples.push(Relation::from_parts(
                        relation_name.clone(),
                        RelationEntity::new(
                            subject.text().to_string(),
                            subject.class().to_string(),
                            subject_start,
                            subject_end,
                            subject.probability(),
                        ),
                        RelationEntity::new(
                            object.text().to_string(),
                            object.class().to_string(),
                            object_start,
                            object_end,
                            object.probability(),
                        ),
                        probability,
                    ));
                }
            }
        }
        Ok(triples)
    }

    fn encode(&self, text: &str, prompt: &GLiFormerPrompt) -> Result<EncodedSequence> {
        let prepared = prepare_prompt(&self.tokenizer, &self.config, text, prompt)?;
        self.embed(prepared)
    }

    fn embed(&self, prepared: PreparedPrompt) -> Result<EncodedSequence> {
        let sequence_len = prepared.input_ids.len();
        let ids = Array2::from_shape_vec((1, sequence_len), prepared.input_ids.clone())
            .map_err(|err| err.to_string())?;
        let attention = Array2::from_shape_vec((1, sequence_len), prepared.attention_mask.clone())
            .map_err(|err| err.to_string())?;
        let outputs = self.encoder.run(ort::inputs![
            "input_ids" => ids.view(),
            "attention_mask" => attention.view(),
        ]?)?;
        let (shape, embeds) = tensor_with_shape(
            outputs
                .get("token_embeds")
                .ok_or("encoder output token_embeds is missing")?,
        )?;
        if shape.len() != 3 || shape[2] != self.config.hidden_size {
            return Err("unexpected encoder embedding shape".into());
        }
        Ok(EncodedSequence {
            embeds,
            ids: prepared.input_ids,
            words_mask: prepared.words_mask,
            words: prepared.words,
            sequence_len: shape[1],
        })
    }

    fn word_embeddings(&self, encoded: &EncodedSequence) -> Result<Array3<f32>> {
        let hidden = self.config.hidden_size;
        let word_count = encoded.words_mask.iter().copied().max().unwrap_or(0).max(0) as usize;
        let mut data = vec![0.0_f32; word_count * hidden];
        for (position, word_id) in encoded.words_mask.iter().copied().enumerate() {
            if word_id <= 0 {
                continue;
            }
            let slot = (word_id as usize) - 1;
            if slot >= word_count || position >= encoded.sequence_len {
                continue;
            }
            let source = position * hidden;
            let target = slot * hidden;
            data[target..target + hidden].copy_from_slice(&encoded.embeds[source..source + hidden]);
        }
        Array3::from_shape_vec((1, word_count, hidden), data).map_err(|err| err.to_string().into())
    }

    fn parent_embedding(&self, encoded: &EncodedSequence) -> Result<Array2<f32>> {
        let hidden = self.config.hidden_size;
        let position = encoded
            .ids
            .iter()
            .position(|id| *id == self.config.parent_token_index)
            .ok_or("prompt is missing the schema token")?;
        row_matrix(&encoded.embeds, hidden, position)
    }

    fn run_ner(
        &self,
        words: &Array3<f32>,
        children: &Array3<f32>,
        parent: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let word_count = words.shape()[1];
        let class_count = children.shape()[1];
        let word_mask = Array2::<f32>::ones((1, word_count));
        let child_mask = Array2::<f32>::ones((1, class_count));
        let outputs = self.ner.run(ort::inputs![
            "words" => words.view(),
            "word_mask" => word_mask.view(),
            "children" => children.view(),
            "child_mask" => child_mask.view(),
            "parent" => parent.view(),
        ]?)?;
        tensor_f32(outputs.get("ner_logits").ok_or("ner_logits missing")?)
    }

    fn run_classification(
        &self,
        words: &Array3<f32>,
        children: &Array3<f32>,
        parent: &Array2<f32>,
        cls_embed: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let class_count = children.shape()[1];
        let child_mask = Array2::<f32>::ones((1, class_count));
        let uses_words = self
            .classification
            .inputs
            .iter()
            .any(|input| input.name == "words");
        let outputs = if uses_words {
            let word_mask = Array2::<f32>::ones((1, words.shape()[1]));
            self.classification.run(ort::inputs![
                "words" => words.view(),
                "word_mask" => word_mask.view(),
                "children" => children.view(),
                "child_mask" => child_mask.view(),
                "parent" => parent.view(),
                "cls_embed" => cls_embed.view(),
            ]?)?
        } else {
            self.classification.run(ort::inputs![
                "children" => children.view(),
                "child_mask" => child_mask.view(),
                "parent" => parent.view(),
                "cls_embed" => cls_embed.view(),
            ]?)?
        };
        tensor_f32(outputs.get("class_logits").ok_or("class_logits missing")?)
    }
}

fn spans_from_bio(text: &str, words: &[Token], spans: &[BioSpan], labels: &[String]) -> Vec<Span> {
    spans
        .iter()
        .filter_map(|span| {
            let label = labels.get(span.class_index)?;
            let start = words.get(span.start)?.start();
            let end = words.get(span.end)?.end();
            if end <= start || end > text.len() {
                return None;
            }
            Some(Span::new(
                0,
                start,
                end,
                text[start..end].to_string(),
                label.clone(),
                span.score,
            ))
        })
        .collect()
}

fn gather_token(
    embeds: &[f32],
    sequence_len: usize,
    hidden: usize,
    ids: &[i64],
    token_id: i64,
) -> Result<Array3<f32>> {
    let positions = ids
        .iter()
        .take(sequence_len)
        .enumerate()
        .filter_map(|(index, id)| (*id == token_id).then_some(index))
        .collect::<Vec<_>>();
    let mut data = vec![0.0_f32; positions.len() * hidden];
    for (slot, position) in positions.iter().copied().enumerate() {
        let source = position * hidden;
        let target = slot * hidden;
        data[target..target + hidden].copy_from_slice(&embeds[source..source + hidden]);
    }
    Array3::from_shape_vec((1, positions.len(), hidden), data).map_err(|err| err.to_string().into())
}

fn row_matrix(embeds: &[f32], hidden: usize, position: usize) -> Result<Array2<f32>> {
    let source = position * hidden;
    let row = embeds
        .get(source..source + hidden)
        .ok_or("token embedding is out of range")?
        .to_vec();
    Array2::from_shape_vec((1, hidden), row).map_err(|err| err.to_string().into())
}

fn tensor_f32(value: &ort::value::Value) -> Result<Vec<f32>> {
    Ok(tensor_with_shape(value)?.1)
}

fn tensor_with_shape(value: &ort::value::Value) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = value.try_extract_tensor::<f32>()?;
    Ok((view.shape().to_vec(), view.iter().copied().collect()))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn relation_schema_from_pipeline(relations: &[GLiNER2PipelineRelation]) -> RelationSchema {
    let mut schema = RelationSchema::new();
    for relation in relations {
        match (&relation.subject_labels, &relation.object_labels) {
            (Some(subjects), Some(objects)) => {
                let subjects = subjects.iter().map(String::as_str).collect::<Vec<_>>();
                let objects = objects.iter().map(String::as_str).collect::<Vec<_>>();
                schema.push_with_allowed_labels(&relation.name, &subjects, &objects);
            }
            _ => schema.push(&relation.name),
        }
    }
    schema
}

fn endpoint_labels(relations: &[GLiNER2PipelineRelation]) -> Vec<String> {
    let mut labels = Vec::new();
    for relation in relations {
        for bucket in [&relation.subject_labels, &relation.object_labels] {
            if let Some(values) = bucket {
                for label in values {
                    if !labels.iter().any(|existing| existing == label) {
                        labels.push(label.clone());
                    }
                }
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use orp::params::RuntimeParameters;

    #[test]
    fn text_tasks_match_the_local_export() {
        let model_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/gliformer-base-v1");
        if !model_dir.join(ENCODER_FILE).is_file() {
            return;
        }
        let model = GLiFormer::from_dir(
            &model_dir,
            Parameters::default(),
            RuntimeParameters::default(),
        )
        .expect("load export");
        let text = "Alice works at Acme in London.";
        let spans = model
            .predict_entities(
                text,
                &[
                    "person".to_string(),
                    "organization".to_string(),
                    "location".to_string(),
                ],
            )
            .expect("ner");
        let rendered = spans
            .iter()
            .map(|span| format!("{}:{}", span.class(), span.text()))
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(rendered, "person:Alice,organization:Acme,location:London");

        let scores = model
            .classify(
                text,
                &[
                    "person".to_string(),
                    "company".to_string(),
                    "location".to_string(),
                ],
            )
            .expect("classify");
        assert_eq!(scores.scores[0].label, "company");
        assert!(scores.scores[0].score > 0.99, "{}", scores.scores[0].score);

        let mut schema = RelationSchema::new();
        schema.push_with_allowed_labels("works_at", &["person"], &["organization"]);
        let relations = model
            .relation_triples(
                text,
                &["person".to_string(), "organization".to_string()],
                &schema,
            )
            .expect("relations");
        let rendered = relations
            .iter()
            .map(|relation| {
                format!(
                    "{}:{}:{}",
                    relation.class(),
                    relation.subject().text,
                    relation.object().text
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        assert!(rendered.contains("works_at:Alice:Acme"), "{rendered}");

        let structure = model
            .extract_fields(text, &["person".to_string(), "organization".to_string()])
            .expect("structure");
        let person = &structure.fields[0].values;
        let organization = &structure.fields[1].values;
        assert!(
            person.iter().any(|value| value.text == "Alice"),
            "{person:?}"
        );
        assert!(
            organization.iter().any(|value| value.text == "Acme"),
            "{organization:?}"
        );
    }
}

fn parse_field_spec(spec: &str) -> Result<(String, bool)> {
    let (name, single) = if let Some((name, suffix)) = spec.split_once("::") {
        let single = match suffix.trim() {
            "str" => true,
            _ => false,
        };
        (name.trim(), single)
    } else {
        (spec.trim(), false)
    };
    if name.is_empty() {
        return Err(format!("invalid JSON schema field spec `{spec}`: empty field name").into());
    }
    Ok((name.to_string(), single))
}
