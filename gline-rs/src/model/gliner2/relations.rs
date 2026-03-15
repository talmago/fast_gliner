use composable::Composable;

use crate::model::input::relation::schema::RelationSchema;
use crate::model::output::decoded::SpanOutput;
use crate::model::output::relation::{RelationOutput, SpanOutputToRelationOutput};
use crate::model::pipeline::context::RelationContext;
use crate::util::result::Result;

pub struct OutputsToRelations<'a> {
    schema: &'a RelationSchema,
}

impl<'a> OutputsToRelations<'a> {
    pub fn new(schema: &'a RelationSchema) -> Self {
        Self { schema }
    }
}

impl Composable<(SpanOutput, RelationContext), RelationOutput> for OutputsToRelations<'_> {
    fn apply(&self, input: (SpanOutput, RelationContext)) -> Result<RelationOutput> {
        SpanOutputToRelationOutput::new(self.schema).apply(input)
    }
}
