use std::collections::{HashMap, HashSet};

use composable::Composable;

use crate::model::gliner2::classification::ClassificationOutput;
use crate::model::gliner2::extraction::{ExtractedField, ExtractionFieldSchema, ExtractionOutput};
use crate::model::gliner2::model::GLiNER2;
use crate::model::gliner2::relations::OutputsToRelations;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::text::TextInput;
use crate::model::output::decoded::SpanOutput;
use crate::model::output::relation::Relation;
use crate::model::pipeline::context::RelationContext;
use crate::text::span::Span;
use crate::util::result::Result;

const ENTITIES_FIELD: &str = "__pipeline_entities";
const CLASSIFICATION_FIELD_PREFIX: &str = "__pipeline_classification::";
const RELATION_FIELD_PREFIX: &str = "__pipeline_relation::";
const STRUCTURE_FIELD_PREFIX: &str = "__pipeline_structure::";

#[derive(Debug, Clone)]
pub struct GLiNER2PipelineClassification {
    pub name: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GLiNER2PipelineStructure {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GLiNER2PipelineRelation {
    pub name: String,
    pub subject_labels: Option<Vec<String>>,
    pub object_labels: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct GLiNER2PipelineSchema {
    pub classifications: Vec<GLiNER2PipelineClassification>,
    pub entity_labels: Vec<String>,
    pub relations: Vec<GLiNER2PipelineRelation>,
    pub structures: Vec<GLiNER2PipelineStructure>,
    current_structure_index: Option<usize>,
    builder_error: Option<String>,
}

impl GLiNER2PipelineSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn classification<S>(
        mut self,
        name: impl Into<String>,
        labels: impl IntoIterator<Item = S>,
    ) -> Self
    where
        S: Into<String>,
    {
        self.add_classification(name, labels);
        self
    }

    pub fn add_classification<S>(
        &mut self,
        name: impl Into<String>,
        labels: impl IntoIterator<Item = S>,
    ) -> &mut Self
    where
        S: Into<String>,
    {
        self.classifications.push(GLiNER2PipelineClassification {
            name: name.into(),
            labels: labels.into_iter().map(Into::into).collect(),
        });
        self
    }

    pub fn entities<S>(mut self, labels: impl IntoIterator<Item = S>) -> Self
    where
        S: Into<String>,
    {
        self.add_entities(labels);
        self
    }

    pub fn add_entities<S>(&mut self, labels: impl IntoIterator<Item = S>) -> &mut Self
    where
        S: Into<String>,
    {
        self.entity_labels
            .extend(labels.into_iter().map(Into::into));
        self
    }

    pub fn relations<S>(mut self, labels: impl IntoIterator<Item = S>) -> Self
    where
        S: Into<String>,
    {
        self.add_relations(labels);
        self
    }

    pub fn add_relations<S>(&mut self, labels: impl IntoIterator<Item = S>) -> &mut Self
    where
        S: Into<String>,
    {
        self.relations
            .extend(labels.into_iter().map(|name| GLiNER2PipelineRelation {
                name: name.into(),
                subject_labels: None,
                object_labels: None,
            }));
        self
    }

    pub fn relation_with_labels<S1, S2, S3>(
        mut self,
        name: S1,
        subject_labels: impl IntoIterator<Item = S2>,
        object_labels: impl IntoIterator<Item = S3>,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        self.add_relation_with_labels(name, subject_labels, object_labels);
        self
    }

    pub fn relation<S1, S2, S3>(
        mut self,
        name: S1,
        subject_labels: impl IntoIterator<Item = S2>,
        object_labels: impl IntoIterator<Item = S3>,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        self.add_relation(name, subject_labels, object_labels);
        self
    }

    pub fn add_relation_with_labels<S1, S2, S3>(
        &mut self,
        name: S1,
        subject_labels: impl IntoIterator<Item = S2>,
        object_labels: impl IntoIterator<Item = S3>,
    ) -> &mut Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        self.relations.push(GLiNER2PipelineRelation {
            name: name.into(),
            subject_labels: Some(subject_labels.into_iter().map(Into::into).collect()),
            object_labels: Some(object_labels.into_iter().map(Into::into).collect()),
        });
        self
    }

    pub fn add_relation<S1, S2, S3>(
        &mut self,
        name: S1,
        subject_labels: impl IntoIterator<Item = S2>,
        object_labels: impl IntoIterator<Item = S3>,
    ) -> &mut Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        self.add_relation_with_labels(name, subject_labels, object_labels)
    }

    pub fn structure(mut self, name: impl Into<String>) -> Self {
        self.add_structure(name);
        self
    }

    pub fn add_structure(&mut self, name: impl Into<String>) -> &mut Self {
        self.structures.push(GLiNER2PipelineStructure {
            name: name.into(),
            fields: Vec::new(),
        });
        self.current_structure_index = Some(self.structures.len() - 1);
        self
    }

    pub fn field(mut self, name: impl Into<String>) -> Self {
        self.add_field(name);
        self
    }

    pub fn add_field(&mut self, name: impl Into<String>) -> &mut Self {
        let field_name = name.into();
        if let Some(index) = self.current_structure_index {
            self.structures[index].fields.push(field_name);
        } else if self.builder_error.is_none() {
            self.builder_error = Some(
                "invalid schema builder usage: `field()` requires a preceding `structure()` call"
                    .to_string(),
            );
        }
        self
    }

    fn validate(&self) -> Result<()> {
        if let Some(err) = &self.builder_error {
            return Err(err.clone().into());
        }
        if self.classifications.is_empty()
            && self.entity_labels.is_empty()
            && self.relations.is_empty()
            && self.structures.is_empty()
        {
            return Err("invalid pipeline schema: at least one task must be defined".into());
        }

        for classification in &self.classifications {
            if classification.name.trim().is_empty() {
                return Err("invalid pipeline schema: classification name cannot be empty".into());
            }
            if classification.labels.is_empty() {
                return Err(format!(
                    "invalid pipeline schema: classification `{}` has no labels",
                    classification.name
                )
                .into());
            }
            for label in &classification.labels {
                if label.trim().is_empty() {
                    return Err(format!(
                        "invalid pipeline schema: classification `{}` contains an empty label",
                        classification.name
                    )
                    .into());
                }
            }
        }

        for label in &self.entity_labels {
            if label.trim().is_empty() {
                return Err("invalid pipeline schema: entities contain an empty label".into());
            }
        }

        for relation in &self.relations {
            if relation.name.trim().is_empty() {
                return Err("invalid pipeline schema: relations contain an empty label".into());
            }

            if let Some(subject_labels) = &relation.subject_labels {
                if subject_labels.is_empty() {
                    return Err(format!(
                        "invalid pipeline schema: relation `{}` has empty subject label constraints",
                        relation.name
                    )
                    .into());
                }
                for label in subject_labels {
                    if label.trim().is_empty() {
                        return Err(format!(
                            "invalid pipeline schema: relation `{}` has an empty subject label",
                            relation.name
                        )
                        .into());
                    }
                }
            }

            if let Some(object_labels) = &relation.object_labels {
                if object_labels.is_empty() {
                    return Err(format!(
                        "invalid pipeline schema: relation `{}` has empty object label constraints",
                        relation.name
                    )
                    .into());
                }
                for label in object_labels {
                    if label.trim().is_empty() {
                        return Err(format!(
                            "invalid pipeline schema: relation `{}` has an empty object label",
                            relation.name
                        )
                        .into());
                    }
                }
            }
        }

        let mut global_field_names = std::collections::HashSet::new();
        for structure in &self.structures {
            if structure.name.trim().is_empty() {
                return Err("invalid pipeline schema: structure name cannot be empty".into());
            }
            if structure.fields.is_empty() {
                return Err(format!(
                    "invalid pipeline schema: structure `{}` has no fields",
                    structure.name
                )
                .into());
            }

            for field in &structure.fields {
                if field.trim().is_empty() {
                    return Err(format!(
                        "invalid pipeline schema: structure `{}` contains an empty field name",
                        structure.name
                    )
                    .into());
                }

                if !global_field_names.insert(field.clone()) {
                    return Err(format!(
                        "invalid pipeline schema: field name `{field}` appears more than once across structures"
                    )
                    .into());
                }
            }
        }

        if !self.relations.is_empty() && self.entity_labels.is_empty() {
            return Err(
                "invalid pipeline schema: relation extraction requires entity labels in `entities()`"
                    .into(),
            );
        }

        Ok(())
    }
}

#[derive(Default)]
pub struct GLiNER2PipelineOutput {
    pub classifications: HashMap<String, ClassificationOutput>,
    pub entities: Vec<Span>,
    pub relations: Vec<Relation>,
    pub structures: HashMap<String, ExtractionOutput>,
}

pub struct GLiNER2Pipeline<'a> {
    model: &'a GLiNER2,
}

impl<'a> GLiNER2Pipeline<'a> {
    pub fn new(model: &'a GLiNER2) -> Self {
        Self { model }
    }

    pub fn extract(
        &self,
        text: &str,
        schema: &GLiNER2PipelineSchema,
    ) -> Result<GLiNER2PipelineOutput> {
        schema.validate()?;

        if has_entity_only_schema(schema) {
            return Ok(GLiNER2PipelineOutput {
                entities: self.extract_entities(text, &schema.entity_labels)?,
                ..GLiNER2PipelineOutput::default()
            });
        }

        let combined_schema = build_combined_extraction_schema(schema, true);
        let extraction_output = self.model.extract(text, &combined_schema)?;
        let mut output = decode_combined_output(text, schema, extraction_output)?;

        for classification in &schema.classifications {
            let classification_output = self.model.classify(text, &classification.labels)?;
            output
                .classifications
                .insert(classification.name.clone(), classification_output);
        }

        Ok(output)
    }

    fn extract_entities(&self, text: &str, labels: &[String]) -> Result<Vec<Span>> {
        let label_refs = as_str_refs(labels);
        let input = TextInput::from_str(&[text], &label_refs)?;
        let output = self.model.inference(input)?;
        Ok(output.spans.into_iter().next().unwrap_or_default())
    }
}

fn has_entity_only_schema(schema: &GLiNER2PipelineSchema) -> bool {
    !schema.entity_labels.is_empty()
        && schema.classifications.is_empty()
        && schema.relations.is_empty()
        && schema.structures.is_empty()
}

fn filter_self_relations(relations: impl Iterator<Item = Relation>) -> Vec<Relation> {
    relations
        .filter(|relation| {
            let subject = relation.subject();
            let object = relation.object();
            !(subject.text == object.text
                && subject.start == object.start
                && subject.end == object.end)
        })
        .collect()
}

fn decode_combined_output(
    text: &str,
    schema: &GLiNER2PipelineSchema,
    extraction_output: ExtractionOutput,
) -> Result<GLiNER2PipelineOutput> {
    let mut field_map = HashMap::with_capacity(extraction_output.fields.len());
    for field in extraction_output.fields {
        field_map.insert(field.name.clone(), field);
    }

    let entities = decode_entities(&field_map);
    let relations = decode_relations(text, schema, &field_map, &entities)?;
    let structures = decode_structures(text, schema, &mut field_map);

    Ok(GLiNER2PipelineOutput {
        classifications: HashMap::new(),
        entities,
        relations,
        structures,
    })
}

fn decode_entities(field_map: &HashMap<String, ExtractedField>) -> Vec<Span> {
    field_map
        .get(ENTITIES_FIELD)
        .map(|entity_field| {
            entity_field
                .values
                .iter()
                .map(|value| {
                    Span::new(
                        0,
                        value.start,
                        value.end,
                        value.text.clone(),
                        value.label.clone(),
                        value.score,
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn decode_relations(
    text: &str,
    schema: &GLiNER2PipelineSchema,
    field_map: &HashMap<String, ExtractedField>,
    entities: &[Span],
) -> Result<Vec<Relation>> {
    if schema.relations.is_empty() || entities.is_empty() {
        return Ok(Vec::new());
    }

    let relation_schema = build_relation_schema(&schema.relations);
    let context = build_relation_context(entities);
    let mut candidate_spans = Vec::new();

    for relation in &schema.relations {
        let field_name = relation_field_name(&relation.name);
        let values = field_map
            .get(&field_name)
            .map(|field| field.values.clone())
            .unwrap_or_default();
        let values = if values.is_empty() {
            fallback_relation_values(text, relation)
        } else {
            values
        };

        candidate_spans.extend(build_relation_candidates(text, relation, &values, entities));
    }

    if candidate_spans.is_empty() {
        return Ok(Vec::new());
    }

    let relation_output = OutputsToRelations::new(&relation_schema).apply((
        SpanOutput::new(vec![text.to_string()], Vec::new(), vec![candidate_spans]),
        context,
    ))?;

    Ok(filter_self_relations(
        relation_output
            .relations
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter(),
    ))
}

fn decode_structures(
    text: &str,
    schema: &GLiNER2PipelineSchema,
    field_map: &mut HashMap<String, ExtractedField>,
) -> HashMap<String, ExtractionOutput> {
    let mut structures = HashMap::with_capacity(schema.structures.len());

    for structure in &schema.structures {
        let mut fields = Vec::with_capacity(structure.fields.len());

        for field_name in &structure.fields {
            let key = structure_field_name(&structure.name, field_name);
            let values = field_map
                .remove(&key)
                .map(|field| field.values)
                .unwrap_or_default();
            fields.push(ExtractedField {
                name: field_name.clone(),
                values,
            });
        }

        structures.insert(
            structure.name.clone(),
            ExtractionOutput {
                text: text.to_string(),
                fields,
            },
        );
    }

    structures
}

fn build_combined_extraction_schema(
    schema: &GLiNER2PipelineSchema,
    include_entities: bool,
) -> super::extraction::ExtractionSchema {
    let mut fields = Vec::new();

    for classification in &schema.classifications {
        fields.push(ExtractionFieldSchema::new(
            classification_field_name(&classification.name),
            classification.labels.clone(),
        ));
    }

    if include_entities && !schema.entity_labels.is_empty() {
        fields.push(ExtractionFieldSchema::new(
            ENTITIES_FIELD,
            schema.entity_labels.clone(),
        ));
    }

    for relation in &schema.relations {
        fields.push(ExtractionFieldSchema::new(
            relation_field_name(&relation.name),
            vec![relation_prompt_label(relation)],
        ));
    }

    for structure in &schema.structures {
        for field in &structure.fields {
            fields.push(ExtractionFieldSchema::new(
                structure_field_name(&structure.name, field),
                vec![field.clone()],
            ));
        }
    }

    super::extraction::ExtractionSchema::from_fields(fields)
}

fn build_relation_schema(relations: &[GLiNER2PipelineRelation]) -> RelationSchema {
    let mut schema = RelationSchema::new();

    for relation in relations {
        if let (Some(subject_labels), Some(object_labels)) =
            (&relation.subject_labels, &relation.object_labels)
        {
            schema.push_with_allowed_labels(
                &relation.name,
                &as_str_refs(subject_labels),
                &as_str_refs(object_labels),
            );
        } else {
            schema.push(&relation.name);
        }
    }

    schema
}

fn as_str_refs(items: &[String]) -> Vec<&str> {
    items.iter().map(|item| item.as_str()).collect()
}

fn classification_field_name(name: &str) -> String {
    format!("{CLASSIFICATION_FIELD_PREFIX}{name}")
}

fn relation_field_name(name: &str) -> String {
    format!("{RELATION_FIELD_PREFIX}{name}")
}

fn relation_prompt_label(relation: &GLiNER2PipelineRelation) -> String {
    let relation_name = relation.name.replace('_', " ");
    match relation.object_labels.as_ref() {
        Some(object_labels) if !object_labels.is_empty() => {
            format!("{relation_name} {}", object_labels.join(" "))
        }
        _ => relation_name,
    }
}

fn structure_field_name(structure_name: &str, field_name: &str) -> String {
    format!("{STRUCTURE_FIELD_PREFIX}{structure_name}::{field_name}")
}

fn build_relation_context(entities: &[Span]) -> RelationContext {
    let mut entity_labels = HashMap::<String, HashSet<String>>::new();
    let mut entity_offsets = HashMap::<String, (usize, usize)>::new();

    for entity in entities {
        entity_labels
            .entry(entity.text().to_string())
            .or_default()
            .insert(entity.class().to_string());
        entity_offsets
            .entry(entity.text().to_string())
            .or_insert_with(|| entity.offsets());
    }

    RelationContext {
        entity_labels,
        entity_offsets,
    }
}

fn build_relation_candidates(
    text: &str,
    relation: &GLiNER2PipelineRelation,
    values: &[crate::model::gliner2::extraction::ExtractedValue],
    entities: &[Span],
) -> Vec<Span> {
    let mut candidates = Vec::new();

    for value in values {
        let (sentence_start, sentence_end) = sentence_bounds(text, value.start, value.end);
        let subject = select_subject_entity(
            relation,
            entities,
            value.start,
            sentence_start,
            sentence_end,
        );
        let object = select_object_entity(relation, entities, value, sentence_start, sentence_end);

        let (Some(subject), Some(object)) = (subject, object) else {
            continue;
        };

        if subject.same_offsets(&object) {
            continue;
        }

        let (object_start, object_end) = object.offsets();
        candidates.push(Span::new(
            0,
            object_start,
            object_end,
            object.text().to_string(),
            format!("{} <> {}", subject.text(), relation.name),
            value
                .score
                .min(subject.probability())
                .min(object.probability()),
        ));
    }

    candidates
}

fn select_subject_entity<'a>(
    relation: &GLiNER2PipelineRelation,
    entities: &'a [Span],
    trigger_start: usize,
    sentence_start: usize,
    sentence_end: usize,
) -> Option<&'a Span> {
    entities
        .iter()
        .filter(|entity| {
            let (start, end) = entity.offsets();
            start >= sentence_start
                && end <= sentence_end
                && end <= trigger_start
                && relation_allows_label(relation.subject_labels.as_ref(), entity.class())
        })
        .min_by_key(|entity| {
            let (_, end) = entity.offsets();
            trigger_start.saturating_sub(end)
        })
}

fn select_object_entity<'a>(
    relation: &GLiNER2PipelineRelation,
    entities: &'a [Span],
    value: &crate::model::gliner2::extraction::ExtractedValue,
    sentence_start: usize,
    sentence_end: usize,
) -> Option<&'a Span> {
    entities
        .iter()
        .filter(|entity| {
            let (start, end) = entity.offsets();
            start >= sentence_start
                && end <= sentence_end
                && relation_allows_label(relation.object_labels.as_ref(), entity.class())
        })
        .min_by_key(|entity| relation_object_distance(entity, value))
}

fn relation_object_distance(
    entity: &Span,
    value: &crate::model::gliner2::extraction::ExtractedValue,
) -> usize {
    let (start, end) = entity.offsets();
    if start == value.start && end == value.end {
        0
    } else if start >= value.end {
        start.saturating_sub(value.end)
    } else if end <= value.start {
        value.start.saturating_sub(end)
    } else {
        1
    }
}

fn relation_allows_label(allowed_labels: Option<&Vec<String>>, label: &str) -> bool {
    allowed_labels
        .map(|labels| labels.iter().any(|allowed| allowed == label))
        .unwrap_or(true)
}

fn sentence_bounds(text: &str, start: usize, end: usize) -> (usize, usize) {
    let sentence_start = text[..start]
        .rfind(['.', '!', '?', '\n'])
        .map(|index| index + 1)
        .unwrap_or(0);
    let sentence_end = text[end..]
        .find(['.', '!', '?', '\n'])
        .map(|index| end + index)
        .unwrap_or(text.len());

    (sentence_start, sentence_end)
}

fn fallback_relation_values(
    text: &str,
    relation: &GLiNER2PipelineRelation,
) -> Vec<crate::model::gliner2::extraction::ExtractedValue> {
    let phrase = relation.name.replace('_', " ");
    let lowercase_text = text.to_lowercase();
    let lowercase_phrase = phrase.to_lowercase();
    let mut offset = 0usize;
    let mut values = Vec::new();

    while let Some(index) = lowercase_text[offset..].find(&lowercase_phrase) {
        let start = offset + index;
        let end = start + lowercase_phrase.len();
        values.push(crate::model::gliner2::extraction::ExtractedValue {
            text: text[start..end].to_string(),
            label: phrase.clone(),
            start,
            end,
            score: 1.0,
        });
        offset = end;
    }

    values
}
