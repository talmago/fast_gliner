//! Encapsulation of raw tensor outputs

use ort::session::SessionOutputs;
use composable::Composable;
use crate::util::result::Result;
use crate::model::pipeline::context::EntityContext;

/// Represents the raw tensor output of the inference step
pub struct TensorOutput<'a> {
    pub context: EntityContext,
    pub tensors: SessionOutputs<'a, 'a>,
}


impl<'a> TensorOutput<'a> {
    pub fn from(tensors: SessionOutputs<'a, 'a>, context: EntityContext) -> Self {
        Self { 
            context,
            tensors 
        }
    }
}


/// Composable: (SessionOutput, TensorMeta) => TensorOutput
#[derive(Default)]
pub struct SessionOutputToTensors { }


impl<'a> Composable<(SessionOutputs<'a, 'a>, EntityContext), TensorOutput<'a>> for SessionOutputToTensors {
    fn apply(&self, input: (SessionOutputs<'a, 'a>, EntityContext)) -> Result<TensorOutput<'a>> {
        Ok(TensorOutput::from(input.0, input.1))
    }
}
