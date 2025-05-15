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

/// Defines an individual relation
pub struct Relation {
    /// Relation label
    class: String,
    /// Text of the subject
    subject: String,
    /// Text of the object
    object: String,
    /// Input index in the batch
    sequence: usize,    
    /// Start offset
    start: usize,
    /// End offset
    end: usize,
    /// Probability 
    probability: f32,
}


impl Relation {
    
    pub fn from(span: Span) -> Result<Self> {
        let (start, end) = span.offsets();
        let (subject, class) = Self::decode(span.class())?;
        Ok(Self {
            class,
            subject,
            object: span.text().to_string(),
            sequence: span.sequence(),
            start,
            end,
            probability: span.probability(),
        })
    }
    
    pub fn class(&self) -> &str {
        &self.class
    }
    
    pub fn subject(&self) -> &str {
        &self.subject
    }
    
    pub fn object(&self) -> &str {
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
        }
        else {
            Ok((split.get(0).unwrap().to_string(), split.get(1).unwrap().to_string()))
        }        
    }
}


impl std::fmt::Display for RelationOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for relations in &self.relations {
            for relation in relations {
                writeln!(f, "{:3} | {:15} | {:10} | {:15} | {:.1}%", relation.sequence(), relation.subject(), relation.class(), relation.object(), relation.probability() * 100.0)?;
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
        // check that the class of the object of the given relation if allowed by the relation schema
        let potential_classes = context.entity_labels.get(relation.object()).ok_or(RelationFormatError::new("unexpected entity found as object"))?;
        let spec = self.schema.relations().get(relation.class()).ok_or(RelationFormatError::new("unexpected relation class"))?;
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
                let relation = Relation::from(span)?;
                if self.is_valid(&relation, &context)? {
                    relations.push(relation);
                }
            }
            result.push(relations);
        }
        Ok(RelationOutput { 
            texts: input.texts,
            entities: input.entities,
            relations: result 
        })
    }
}



#[derive(Debug, Clone)]
/// Defines an error caused by an incorrect format of the span label
/// obtained by the relation extraction pipeline. This is likely to be 
/// an internal error, unless the pipeline was not used correctly.
pub struct RelationFormatError {
    message: String,
}

impl RelationFormatError {
    pub fn new(span_label: &str) -> Self {
        Self { message: format!("unexpected relation label format: {span_label}") }
    }

    pub fn err<T>(self) -> Result<T> {
        Err(Box::new(self))
    }
}

impl std::error::Error for RelationFormatError { }

impl std::fmt::Display for RelationFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}