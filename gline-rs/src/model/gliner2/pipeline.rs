use std::collections::HashMap;

use composable::Composable;

use crate::model::gliner2::classification::ClassificationOutput;
use crate::model::gliner2::extraction::{ExtractedField, ExtractionFieldSchema, ExtractionOutput};
use crate::model::gliner2::model::GLiNER2;
use crate::model::gliner2::relations::OutputsToRelations;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::RelationInput;
use crate::model::input::text::TextInput;
use crate::model::output::relation::Relation;
use crate::model::pipeline::context::RelationContext;
use crate::text::span::Span;
use crate::util::result::Result;

const ENTITIES_FIELD: &str = "__pipeline_entities";
const CLASSIFICATION_FIELD_PREFIX: &str = "__pipeline_classification::";
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

        let mut output = GLiNER2PipelineOutput::default();

        let has_core_tasks = !schema.classifications.is_empty()
            || !schema.entity_labels.is_empty()
            || !schema.structures.is_empty();

        if has_core_tasks {
            let combined_schema = build_combined_extraction_schema(schema);
            let extraction_output = self.model.extract(text, &combined_schema)?;
            apply_core_outputs(text, schema, extraction_output, &mut output)?;
        }

        // Use the dedicated classification decoder so scores reflect the model output
        // even when no span survives extraction thresholding for a class label.
        for classification in &schema.classifications {
            let classification_output = self.model.classify(text, &classification.labels)?;
            output
                .classifications
                .insert(classification.name.clone(), classification_output);
        }

        if !schema.relations.is_empty() {
            let relation_output = self.extract_relations(text, schema)?;
            output.relations = relation_output;
        }

        Ok(output)
    }

    fn extract_relations(
        &self,
        text: &str,
        schema: &GLiNER2PipelineSchema,
    ) -> Result<Vec<Relation>> {
        let relation_schema = build_relation_schema(&schema.relations);
        let text_input = TextInput::from_str(&[text], &as_str_refs(&schema.entity_labels))?;
        let entity_spans = self.model.inference(text_input)?;

        let relation_input = RelationInput::from_spans(entity_spans, &relation_schema);
        let relation_text_input = TextInput::new(relation_input.prompts, relation_input.labels)?;

        let relation_spans = self.model.inference(relation_text_input)?;
        let relation_output = OutputsToRelations::new(&relation_schema).apply((
            relation_spans,
            RelationContext {
                entity_labels: relation_input.entity_labels,
                entity_offsets: relation_input.entity_offsets,
            },
        ))?;

        let relations = relation_output
            .relations
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter();
        let filtered = filter_self_relations(relations);

        Ok(filtered)
    }
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

fn apply_core_outputs(
    text: &str,
    schema: &GLiNER2PipelineSchema,
    extraction_output: ExtractionOutput,
    output: &mut GLiNER2PipelineOutput,
) -> Result<()> {
    let mut field_map = HashMap::with_capacity(extraction_output.fields.len());
    for field in extraction_output.fields {
        field_map.insert(field.name.clone(), field);
    }

    if !schema.entity_labels.is_empty() {
        if let Some(entity_field) = field_map.get(ENTITIES_FIELD) {
            output.entities = entity_field
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
                .collect();
        }
    }

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

        output.structures.insert(
            structure.name.clone(),
            ExtractionOutput {
                text: text.to_string(),
                fields,
            },
        );
    }

    Ok(())
}

fn build_combined_extraction_schema(
    schema: &GLiNER2PipelineSchema,
) -> super::extraction::ExtractionSchema {
    let mut fields = Vec::new();

    for classification in &schema.classifications {
        fields.push(ExtractionFieldSchema::new(
            classification_field_name(&classification.name),
            classification.labels.clone(),
        ));
    }

    if !schema.entity_labels.is_empty() {
        fields.push(ExtractionFieldSchema::new(
            ENTITIES_FIELD,
            schema.entity_labels.clone(),
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

fn structure_field_name(structure_name: &str, field_name: &str) -> String {
    format!("{STRUCTURE_FIELD_PREFIX}{structure_name}::{field_name}")
}
