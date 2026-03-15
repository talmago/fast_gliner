use gliner::model::gliner2::GLiNER2PipelineSchema;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyGLiNER2PipelineSchema {
    pub(crate) schema: GLiNER2PipelineSchema,
}

#[pymethods]
impl PyGLiNER2PipelineSchema {
    #[new]
    fn new() -> Self {
        Self {
            schema: GLiNER2PipelineSchema::new(),
        }
    }

    fn classification<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        labels: Vec<String>,
    ) -> PyRefMut<'py, Self> {
        slf.schema.add_classification(name, labels);
        slf
    }

    fn entities<'py>(mut slf: PyRefMut<'py, Self>, labels: Vec<String>) -> PyRefMut<'py, Self> {
        slf.schema.add_entities(labels);
        slf
    }

    fn relations<'py>(mut slf: PyRefMut<'py, Self>, labels: Vec<String>) -> PyRefMut<'py, Self> {
        slf.schema.add_relations(labels);
        slf
    }

    #[pyo3(signature = (name, subject_labels=None, object_labels=None))]
    fn relation<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        subject_labels: Option<Vec<String>>,
        object_labels: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        match (subject_labels, object_labels) {
            (Some(subject), Some(object)) => {
                slf.schema.add_relation_with_labels(name, subject, object);
                Ok(slf)
            }
            (None, None) => {
                slf.schema.add_relations(vec![name]);
                Ok(slf)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "relation() requires both subject_labels and object_labels, or neither",
            )),
        }
    }

    fn structure<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.schema.add_structure(name);
        slf
    }

    #[pyo3(signature = (name, dtype=None, choices=None))]
    fn field<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        dtype: Option<String>,
        choices: Option<Vec<String>>,
    ) -> PyRefMut<'py, Self> {
        // These parameters are currently accepted for Python API compatibility,
        // but the Rust schema builder does not use them yet.
        let _dtype = dtype;
        let _choices = choices;
        slf.schema.add_field(name);
        slf
    }
}
