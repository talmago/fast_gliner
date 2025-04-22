pub mod schema;

use composable::*;
use std::collections::{HashMap, HashSet};
use crate::model::pipeline::context::RelationContext;
use crate::util::result::Result;
use crate::model::output::decoded::SpanOutput;
use schema::RelationSchema;


const PROMPT_PREFIX: &str = "Extract relationships between entities from the text: \n";

/// Input data for Relation Extraction
pub struct RelationInput {
    pub prompts: Vec<String>,
    pub labels: Vec<String>,
    pub entity_labels: HashMap<String, HashSet<String>>,
}

impl RelationInput {

    /// Builds a relation input from a span output and a relation schema
    pub fn from_spans(spans: SpanOutput, schema: &RelationSchema) -> Self {
        Self {
            prompts: Self::make_prompts(&spans, PROMPT_PREFIX),
            labels: Self::make_labels(&spans, schema),
            entity_labels: Self::make_entity_labels(&spans),
        }
    }
    
    /// Prepare the prompts basing on the provided prefix
    fn make_prompts(spans: &SpanOutput, prefix: &str) -> Vec<String> {        
        spans.texts.iter().map(|t| format!("{prefix} {t}")).collect()
    }
    
    /// Prepare the labels basing on extracted entities and the provided schema
    fn make_labels(spans: &SpanOutput, schema: &RelationSchema) -> Vec<String> {
        // List unique (entity, class) entries found in all spans for all sequences.
        // This is sub-optimal because one huge label list will be made for all sequences, 
        // but this is how GLiNER multitask works...
        let mut unique_entities: HashSet<(&str, &str)> = HashSet::new();
        for seq in &spans.spans {
            for span in seq {
                unique_entities.insert((span.text(), span.class()));
            }
        }

        // Actually create the labels. Labels for not allowed entity classes for the subject (according 
        // to the schema) will not be included. The check on the object class has to be made when
        // decoding the result.
        let mut result = Vec::new();
        for (relation, spec) in schema.relations() {
            unique_entities.iter()
                .filter(|(_, class)| spec.allows_subject(class))
                .map(|(text, _)| format!("{} <> {}", text, relation))
                .for_each(|l| result.push(l));            
        }

        result
    }
    
    /// Build entity-text -> entity-labels map (which will be used when decoding, to filter relations basing on allowed objects).
    /// 
    /// Multiple labels for the same entity text is supported, but in this case there is no guarantee that a 
    /// relation actually mentions an entity as having a given label since we just have this information 
    /// (limitation of GLiNER multi). So, as soon as one expected class is found for an entity, it will have
    /// to be accepted without knowing its actual class within the relation (which is probably ok).
    fn make_entity_labels(spans: &SpanOutput) -> HashMap<String, HashSet<String>> {        
        let mut entity_labels = HashMap::<String, HashSet<String>>::new();
        for seq in &spans.spans {
            for span in seq {
                entity_labels.entry(span.text().to_string()).or_default().insert(span.class().to_string());
            }
        }
        entity_labels
    }

}


pub struct SpanOutputToRelationInput<'a> {
    schema: &'a RelationSchema
}

impl<'a> SpanOutputToRelationInput<'a> {
    pub fn new(schema: &'a RelationSchema) -> Self {
        Self { schema }
    }
}

impl Composable<SpanOutput, RelationInput> for SpanOutputToRelationInput<'_> {
    fn apply(&self, input: SpanOutput) -> Result<RelationInput> {
        Ok(RelationInput::from_spans(input, self.schema))
    }
}


#[derive(Default)]
pub struct RelationInputToTextInput {    
}

impl Composable<RelationInput, (super::text::TextInput, RelationContext)> for RelationInputToTextInput {
    fn apply(&self, input: RelationInput) -> Result<(super::text::TextInput, RelationContext)> {
        Ok((super::text::TextInput::new(input.prompts, input.labels)?, RelationContext { entity_labels: input.entity_labels }))
    }
}
