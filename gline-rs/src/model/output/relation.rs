use composable::Composable;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::pipeline::context::RelationContext;
use crate::util::result::Result;
use crate::text::span::Span;
use super::decoded::SpanOutput;

pub struct RelationOutput {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
    pub relations: Vec<Vec<Relation>>,    
}

#[derive(Debug, Clone)]
pub struct RelationEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub probability: f32,
}

impl RelationEntity {
    pub fn new(text: String, label: String, start: usize, end: usize, probability: f32) -> Self {
        RelationEntity {
            text,
            label,
            start,
            end,
            probability,
        }
    }
}

pub struct Relation {
    class: String,
    subject: RelationEntity,
    object: RelationEntity,
    sequence: usize,
    start: usize,
    end: usize,
    probability: f32,
}

impl Relation {
    pub fn from(span: Span, context: &RelationContext) -> Result<Self> {
        let (subject_text, class) = Self::decode(span.class())?;
        let object_text = span.text().to_string();
        let probability = span.probability();

        let subject_label = context
            .entity_labels
            .get(&subject_text)
            .and_then(|labels| labels.iter().next().cloned())
            .unwrap_or_default();

        let (subject_start, subject_end) = context
            .entity_offsets
            .get(&subject_text)
            .copied()
            .unwrap_or((0, 0));

        let object_label = context
            .entity_labels
            .get(&object_text)
            .and_then(|labels| labels.iter().next().cloned())
            .unwrap_or_default();

        let (object_start, object_end) = context
            .entity_offsets
            .get(&object_text)
            .copied()
            .unwrap_or((0, 0));

        let subject = RelationEntity::new(subject_text, subject_label, subject_start, subject_end, probability);
        let object = RelationEntity::new(object_text, object_label, object_start, object_end, probability);

        let (start, end) = span.offsets();

        Ok(Self {
            class,
            subject,
            object,
            sequence: span.sequence(),
            start,
            end,
            probability,
        })
    }

    pub fn class(&self) -> &str {
        &self.class
    }

    pub fn subject(&self) -> &RelationEntity {
        &self.subject
    }

    pub fn object(&self) -> &RelationEntity {
        &self.object
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    pub fn offsets(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    pub fn probability(&self) -> f32 {
        self.probability
    }

    fn decode(rel_class: &str) -> Result<(String, String)> {
        let split: Vec<&str> = rel_class.split(" <> ").collect();
        if split.len() != 2 {
            RelationFormatError::new(&format!(
                "invalid class format: expected 'subject_label <> relation_class', got '{}'",
                rel_class
            ))
            .err()
        } else {
            Ok((split[0].to_string(), split[1].to_string()))
        }
    }
}

impl std::fmt::Display for RelationOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for relations in &self.relations {
            for relation in relations {
                writeln!(
                    f,
                    "{:3} | {:15} | {:10} | {:15} | {:.1}%",
                    relation.sequence(),
                    relation.subject().text,
                    relation.class(),
                    relation.object().text,
                    relation.probability() * 100.0
                )?;
            }
        }
        Ok(())
    }
}

pub struct SpanOutputToRelationOutput<'a> {
    schema: &'a RelationSchema,
}

impl<'a> SpanOutputToRelationOutput<'a> {
    pub fn new(schema: &'a RelationSchema) -> Self {
        Self { schema }
    }

    fn is_valid(&self, relation: &Relation, context: &RelationContext) -> Result<bool> {
        let spec = self
            .schema
            .relations()
            .get(relation.class())
            .ok_or_else(|| {
                RelationFormatError::new(&format!(
                    "unexpected relation class: '{}'",
                    relation.class()
                ))
            })?;
        
        // try to get the object labels from context
        if let Some(object_labels) = context.entity_labels.get(&relation.object.text) {
            return Ok(spec.allows_one_of_objects(object_labels));
        }

        // fall back to the label in the predicted object
        Ok(spec.allows_object(&relation.object.label))
    }
}

impl Composable<(SpanOutput, RelationContext), RelationOutput> for SpanOutputToRelationOutput<'_> {
    fn apply(&self, input: (SpanOutput, RelationContext)) -> Result<RelationOutput> {
        let (input, context) = input;
        let mut result = Vec::new();

        for seq in input.spans {
            let mut relations = Vec::new();
            for span in seq {
                let relation = Relation::from(span, &context)?;
                match self.is_valid(&relation, &context) {
                    Ok(true) => relations.push(relation),
                    Ok(false) => {
                        if std::env::var("GLINER_DEBUG").is_ok() {
                            eprintln!(
                                "[relation rejected] '{}'\n  Subject: '{}' [{}] ({}..{})\n  Object:  '{}' [{}] ({}..{})\n  Score:   {:.4}\n  Reason:  schema mismatch",
                                relation.class(),
                                relation.subject.text,
                                relation.subject.label,
                                relation.subject.start,
                                relation.subject.end,
                                relation.object.text,
                                relation.object.label,
                                relation.object.start,
                                relation.object.end,
                                relation.probability
                            );
                        }
                    }
                    Err(err) => {
                        if std::env::var("GLINER_DEBUG").is_ok() {
                                eprintln!(
                                "relation parsing failed: {}\n  text: '{}'",
                                err, relation.object.text
                            );
                        }
                    }
                }
            }
            result.push(relations);
        }

        Ok(RelationOutput {
            texts: input.texts,
            entities: input.entities,
            relations: result,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RelationFormatError {
    message: String,
}

impl RelationFormatError {
    pub fn new(span_label: &str) -> Self {
        Self {
            message: format!("unexpected relation label format: {span_label}"),
        }
    }

    pub fn err<T>(self) -> Result<T> {
        Err(Box::new(self))
    }
}

impl std::error::Error for RelationFormatError {}

impl std::fmt::Display for RelationFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
