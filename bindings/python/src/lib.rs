mod output;
mod pipeline;
mod schema;

use pipeline::*;
use pyo3::prelude::*;
use schema::*;

#[pymodule]
fn fast_gliner(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFastGliNER>()?;
    m.add_class::<PyFastGliNER2>()?;
    m.add_class::<PyGLiNER2PipelineSchema>()?;
    m.add_class::<PyRelationSchemaEntry>()?;
    Ok(())
}
