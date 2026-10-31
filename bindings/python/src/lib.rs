#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod aarch64_atomics_shim;
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
    m.add_class::<PyFastGLiClass>()?;
    m.add_class::<PyFastGLiFormer>()?;
    m.add_class::<PyGLiNER2MultiPipelineSchema>()?;
    m.add_class::<PyRelationSchemaEntry>()?;
    Ok(())
}
