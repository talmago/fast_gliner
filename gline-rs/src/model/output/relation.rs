use composable::Composable;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::pipeline::context::RelationContext;
use crate::util::result::Result;
use crate::text::span::Span;
use super::decoded::SpanOutput;

/// Defines the final output of the relation extraction pipeline
pub struct RelationOutput {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
    pub relations: Vec<Vec<Relation>>,    
}

/// Represents a subject or object in a relation
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

/// Defines an individual relation
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
        let (start, end) = span.offsets();
        let (subject_text, class) = Self::decode(span.class())?;
        let object_text = span.text().to_string();

        let subject_label = context
            .entity_labels
            .get(&subject_text)
            .and_then(|labels| labels.iter().next().cloned())
            .unwrap_or_else(|| "".to_string());

        let object_label = context
            .entity_labels
            .get(&object_text)
            .and_then(|labels| labels.iter().next().cloned())
            .unwrap_or_else(|| "".to_string());

        let probability = span.probability();

        let subject = RelationEntity::new(subject_text, subject_label, start, end, probability);
        let object = RelationEntity::new(object_text, object_label, start, end, probability);

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
            RelationFormatError::new(rel_class).err()
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

/// SpanOutput -> RelationOutput
pub struct SpanOutputToRelationOutput<'a> {
    schema: &'a RelationSchema,
}

impl<'a> SpanOutputToRelationOutput<'a> {
    pub fn new(schema: &'a RelationSchema) -> Self {
        Self { schema }
    }

    fn is_valid(&self, relation: &Relation, context: &RelationContext) -> Result<bool> {
        let potential_classes = context
            .entity_labels
            .get(&relation.object.text)
            .ok_or(RelationFormatError::new("unexpected entity found as object"))?;

        let spec = self
            .schema
            .relations()
            .get(relation.class())
            .ok_or(RelationFormatError::new("unexpected relation class"))?;

        Ok(spec.allows_one_of_objects(potential_classes))
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
                if self.is_valid(&relation, &context)? {
                    relations.push(relation);
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
/// Defines an error caused by an incorrect format of the span label
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
