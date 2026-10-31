use gliner::model::GLiNER2MultiPipelineSchema;
use pyo3::prelude::*;

/// Multi-task pipeline schema builder for GLiNER2 and GLiFormer.
///
/// Chain task methods, then pass the builder to ``extract``.
///
/// Examples:
///     schema = (
///         model.create_schema()
///         .entities(["person", "company"])
///         .classification("sentiment", ["positive", "negative"])
///         .relation("works_for", ["person"], ["company"])
///     )
#[pyclass]
#[derive(Clone)]
pub struct PyGLiNER2MultiPipelineSchema {
    pub(crate) schema: GLiNER2MultiPipelineSchema,
}

#[pymethods]
impl PyGLiNER2MultiPipelineSchema {
    /// Create an empty pipeline schema.
    #[new]
    fn new() -> Self {
        Self {
            schema: GLiNER2MultiPipelineSchema::new(),
        }
    }

    /// Add a classification task.
    ///
    /// Args:
    ///     name: Classification group name.
    ///     labels: Candidate class names.
    ///
    /// Returns:
    ///     This builder, for chaining.
    fn classification<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        labels: Vec<String>,
    ) -> PyRefMut<'py, Self> {
        slf.schema.add_classification(name, labels);
        slf
    }

    /// Add an entity-extraction task.
    ///
    /// Args:
    ///     labels: Entity labels to detect.
    ///
    /// Returns:
    ///     This builder, for chaining.
    fn entities<'py>(mut slf: PyRefMut<'py, Self>, labels: Vec<String>) -> PyRefMut<'py, Self> {
        slf.schema.add_entities(labels);
        slf
    }

    /// Add relation labels without subject or object constraints.
    ///
    /// Args:
    ///     labels: Relation names.
    ///
    /// Returns:
    ///     This builder, for chaining.
    fn relations<'py>(mut slf: PyRefMut<'py, Self>, labels: Vec<String>) -> PyRefMut<'py, Self> {
        slf.schema.add_relations(labels);
        slf
    }

    /// Add one relation, optionally constrained by argument labels.
    ///
    /// Args:
    ///     name: Relation name.
    ///     subject_labels: Allowed subject entity labels. Must be passed
    ///         together with ``object_labels``.
    ///     object_labels: Allowed object entity labels. Must be passed
    ///         together with ``subject_labels``.
    ///
    /// Returns:
    ///     This builder, for chaining.
    ///
    /// Raises:
    ///     ValueError: If only one of ``subject_labels`` or ``object_labels``
    ///         is set.
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

    /// Start a flat structured-extraction record.
    ///
    /// Nested records use ``FastGLiFormer.structure``. Follow this call with
    /// ``field`` entries.
    ///
    /// Args:
    ///     name: Record name.
    ///
    /// Returns:
    ///     This builder, for chaining.
    fn structure<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.schema.add_structure(name);
        slf
    }

    /// Add a field to the current structured-extraction record.
    ///
    /// Args:
    ///     name: Field name.
    ///     dtype: Accepted for API compatibility. Not used by the runtime.
    ///     choices: Accepted for API compatibility. Not used by the runtime.
    ///
    /// Returns:
    ///     This builder, for chaining.
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
