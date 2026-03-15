use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::{Py, Python};
use std::collections::HashMap;
use std::path::Path;

use gliner::model::gliner2::{
    ExtractionFieldSchema, ExtractionSchema, GLiNER2, GLiNER2PipelineOutput, GLiNER2PipelineSchema,
};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::output::{decoded::SpanOutput, relation::RelationOutput};
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::model::runtime::InferenceMode;
use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::util::result::Result as GResult;

use orp::params::RuntimeParameters;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

use composable::*;
use orp::model::Model;
use orp::pipeline::*;

#[pyclass]
pub struct PyFastGliNER {
    model: Box<dyn Inferencer + Send + Sync>,
    tokenizer_path: String,
}

#[pyclass]
pub struct PyFastGliNER2 {
    model: GLiNER2,
}

#[pyclass]
#[derive(Clone)]
pub struct PyGLiNER2PipelineSchema {
    schema: GLiNER2PipelineSchema,
}

trait Inferencer: Send + Sync {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput>;
    fn get_orp_model(&self) -> &Model;
}

trait ToPy {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

impl ToPy for SpanOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let results = PyList::empty_bound(py);

        for spans in &self.spans {
            let py_spans = PyList::empty_bound(py);
            for span in spans {
                let span_dict = span_to_dict(py, span)?;
                py_spans.append(span_dict)?;
            }
            results.append(py_spans)?;
        }

        Ok(results.into())
    }
}

impl ToPy for RelationOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_results = PyList::empty_bound(py);

        for relation_list in &self.relations {
            let py_relations = PyList::empty_bound(py);
            for rel in relation_list {
                let rel_dict = relation_to_dict(py, rel)?;
                py_relations.append(rel_dict)?;
            }
            py_results.append(py_relations)?;
        }

        Ok(py_results.into())
    }
}

impl ToPy for gliner::model::gliner2::ExtractionOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = PyDict::new_bound(py);

        for field in &self.fields {
            let values = PyList::empty_bound(py);

            for value in &field.values {
                values.append(extracted_value_to_dict(py, value)?)?;
            }

            result.set_item(&field.name, values)?;
        }

        Ok(result.into())
    }
}

impl ToPy for GLiNER2PipelineOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = PyDict::new_bound(py);

        let classifications = PyDict::new_bound(py);
        for (task_name, classification) in &self.classifications {
            let scores = PyList::empty_bound(py);
            for score in &classification.scores {
                let score_dict = PyDict::new_bound(py);
                score_dict.set_item("label", &score.label)?;
                score_dict.set_item("score", score.score)?;
                scores.append(score_dict)?;
            }
            classifications.set_item(task_name, scores)?;
        }
        result.set_item("classifications", classifications)?;

        let entities = PyList::empty_bound(py);
        for entity in &self.entities {
            entities.append(span_to_dict(py, entity)?)?;
        }
        result.set_item("entities", entities)?;

        let relations = PyList::empty_bound(py);
        for relation in &self.relations {
            relations.append(relation_to_dict(py, relation)?)?;
        }
        result.set_item("relations", relations)?;

        let structures = PyDict::new_bound(py);
        for (structure_name, structure_output) in &self.structures {
            let structure_dict = PyDict::new_bound(py);
            for field in &structure_output.fields {
                let values = PyList::empty_bound(py);
                for value in &field.values {
                    values.append(extracted_value_to_dict(py, value)?)?;
                }
                structure_dict.set_item(&field.name, values)?;
            }
            structures.set_item(structure_name, structure_dict)?;
        }
        result.set_item("structures", structures)?;

        Ok(result.into())
    }
}

impl ToPy for serde_json::Value {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            serde_json::Value::Null => Ok(py.None()),
            serde_json::Value::Bool(flag) => Ok(flag.into_py(py)),
            serde_json::Value::Number(number) => {
                if let Some(v) = number.as_i64() {
                    Ok(v.into_py(py))
                } else if let Some(v) = number.as_u64() {
                    Ok(v.into_py(py))
                } else if let Some(v) = number.as_f64() {
                    Ok(v.into_py(py))
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "unsupported JSON number representation",
                    ))
                }
            }
            serde_json::Value::String(text) => Ok(text.into_py(py)),
            serde_json::Value::Array(items) => {
                let list = PyList::empty_bound(py);
                for item in items {
                    list.append(item.to_py(py)?)?;
                }
                Ok(list.into())
            }
            serde_json::Value::Object(entries) => {
                let dict = PyDict::new_bound(py);
                for (key, value) in entries {
                    dict.set_item(key, value.to_py(py)?)?;
                }
                Ok(dict.into())
            }
        }
    }
}

impl Inferencer for InferenceMode {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput> {
        self.inference(input)
    }
    fn get_orp_model(&self) -> &Model {
        self.get_inner_model()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyRelationSchemaEntry {
    #[pyo3(get, set)]
    pub relation: String,
    #[pyo3(get, set)]
    pub subject_labels: Vec<String>,
    #[pyo3(get, set)]
    pub object_labels: Vec<String>,
}

#[pymethods]
impl PyRelationSchemaEntry {
    #[new]
    fn new(relation: String, subject_labels: Vec<String>, object_labels: Vec<String>) -> Self {
        PyRelationSchemaEntry {
            relation,
            subject_labels,
            object_labels,
        }
    }
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

    fn field<'py>(mut slf: PyRefMut<'py, Self>, name: String) -> PyRefMut<'py, Self> {
        slf.schema.add_field(name);
        slf
    }
}

#[pymethods]
impl PyFastGliNER {
    #[new]
    fn new(
        model_dir: String,
        filename: Option<String>,
        execution_provider: Option<String>,
    ) -> PyResult<Self> {
        let base = Path::new(&model_dir);
        let tokenizer_path = base.join("tokenizer.json");
        let providers = execution_providers_from_arg(execution_provider)?;
        let runtime_params = RuntimeParameters::default().with_execution_providers(providers);

        let model = match filename.as_deref() {
            Some(onnx_path) => GLiNER::from_dir_with(
                &model_dir,
                Parameters::default(),
                runtime_params,
                None,
                Some(onnx_path),
                None,
            ),
            None => GLiNER::from_dir(&model_dir, Parameters::default(), runtime_params),
        }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let model: Box<dyn Inferencer + Send + Sync> = Box::new(model);

        Ok(PyFastGliNER {
            model,
            tokenizer_path: tokenizer_path.to_string_lossy().to_string(),
        })
    }

    fn predict_entities(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        labels: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&texts_ref, &labels_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        let output = py
            .allow_threads(|| self.model.inference(input))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        output.to_py(py)
    }

    fn extract_relations(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        entity_labels: Vec<String>,
        relation_schema_entries: Vec<PyRelationSchemaEntry>,
    ) -> PyResult<Py<PyAny>> {
        let input = text_input_from_strings(&texts, &entity_labels)?;
        let relation_schema = relation_schema_from_entries(relation_schema_entries);

        let orp_model = self.model.get_orp_model();

        let token_pipeline = TokenPipeline::new(&self.tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let relation_pipeline =
            RelationPipeline::default(&self.tokenizer_path, &relation_schema)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let params = Parameters::default();

        let pipeline = composed![
            token_pipeline.to_composable(orp_model, &params),
            relation_pipeline.to_composable(orp_model, &params)
        ];

        let output = py
            .allow_threads(|| pipeline.apply(input))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        output.to_py(py)
    }
}

#[pymethods]
impl PyFastGliNER2 {
    #[new]
    fn new(
        model_dir: String,
        filename: Option<String>,
        execution_provider: Option<String>,
    ) -> PyResult<Self> {
        let providers = execution_providers_from_arg(execution_provider)?;
        let runtime_params = RuntimeParameters::default().with_execution_providers(providers);

        if let Some(path) = filename.as_deref() {
            if path != "onnx/model.onnx" && path != "model.onnx" {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "PyFastGliNER2 loads models via GLiNER2::from_dir and currently supports only the default ONNX layout",
                ));
            }
        }

        let model = GLiNER2::from_dir(&model_dir, Parameters::default(), runtime_params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        Ok(Self { model })
    }

    fn predict_entities(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        labels: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&texts_ref, &labels_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        let output = py
            .allow_threads(|| self.model.inference(input))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        output.to_py(py)
    }

    fn classify(&self, text: String, labels: Vec<String>) -> PyResult<Vec<(String, f32)>> {
        let output = self
            .model
            .classify(&text, &labels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        Ok(output
            .scores
            .into_iter()
            .map(|score| (score.label, score.score))
            .collect())
    }

    fn create_schema(&self) -> PyGLiNER2PipelineSchema {
        PyGLiNER2PipelineSchema {
            schema: self.model.create_schema(),
        }
    }

    fn extract(
        &self,
        py: Python<'_>,
        text: String,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        if let Ok(schema_ref) = schema.extract::<PyRef<'_, PyGLiNER2PipelineSchema>>() {
            let rust_schema = schema_ref.schema.clone();
            let output = py
                .allow_threads(|| self.model.extract_with_schema(&text, &rust_schema))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            output.to_py(py)
        } else {
            let schema = schema
                .extract::<Vec<(String, Vec<String>)>>()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "schema must be a GLiNER2PipelineSchema object or a list of (field_name, labels) tuples",
                    )
                })?;

            let schema = ExtractionSchema::from_fields(
                schema
                    .into_iter()
                    .map(|(name, labels)| ExtractionFieldSchema::new(name, labels))
                    .collect(),
            );

            let output = py
                .allow_threads(|| self.model.extract(&text, &schema))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            output.to_py(py)
        }
    }

    fn extract_json(
        &self,
        py: Python<'_>,
        text: String,
        schema: HashMap<String, Vec<String>>,
    ) -> PyResult<PyObject> {
        let output = py
            .allow_threads(|| self.model.extract_json(&text, &schema))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        output.to_py(py)
    }

    fn extract_relations(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        entity_labels: Vec<String>,
        relation_schema_entries: Vec<PyRelationSchemaEntry>,
    ) -> PyResult<Py<PyAny>> {
        let input = text_input_from_strings(&texts, &entity_labels)?;
        let relation_schema = relation_schema_from_entries(relation_schema_entries);

        let output = py
            .allow_threads(|| self.model.extract_relations(input, &relation_schema))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        output.to_py(py)
    }
}

#[pymodule]
fn fast_gliner(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastGliNER>()?;
    m.add_class::<PyFastGliNER2>()?;
    m.add_class::<PyGLiNER2PipelineSchema>()?;
    m.add_class::<PyRelationSchemaEntry>()?;
    Ok(())
}

fn execution_providers_from_arg(
    execution_provider: Option<String>,
) -> PyResult<Vec<ExecutionProviderDispatch>> {
    match execution_provider.as_deref() {
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                Ok(vec![CUDAExecutionProvider::default().build()])
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "CUDA execution provider requested but 'cuda' feature is not enabled",
                ))
            }
        }
        Some("cpu") => Ok(vec![CPUExecutionProvider::default().build()]),
        None => Ok(vec![]),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported execution provider: '{}'. Use 'cpu' or 'cuda'.",
            other
        ))),
    }
}

fn text_input_from_strings(texts: &[String], labels: &[String]) -> PyResult<TextInput> {
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

    TextInput::from_str(&texts_ref, &labels_ref)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))
}

fn relation_schema_from_entries(entries: Vec<PyRelationSchemaEntry>) -> RelationSchema {
    let mut relation_schema = RelationSchema::new();
    for entry in entries {
        let subj: Vec<&str> = entry.subject_labels.iter().map(|s| s.as_str()).collect();
        let obj: Vec<&str> = entry.object_labels.iter().map(|s| s.as_str()).collect();
        relation_schema.push_with_allowed_labels(&entry.relation, &subj, &obj);
    }
    relation_schema
}

fn span_to_dict<'py>(
    py: Python<'py>,
    span: &gliner::text::span::Span,
) -> PyResult<Bound<'py, PyDict>> {
    let span_dict = PyDict::new_bound(py);
    span_dict.set_item("text", span.text())?;
    span_dict.set_item("label", span.class())?;
    span_dict.set_item("score", span.probability())?;

    let (start, end) = span.offsets();
    span_dict.set_item("start", start)?;
    span_dict.set_item("end", end)?;

    Ok(span_dict)
}

fn relation_to_dict<'py>(
    py: Python<'py>,
    relation: &gliner::model::output::relation::Relation,
) -> PyResult<Bound<'py, PyDict>> {
    let rel_dict = PyDict::new_bound(py);
    rel_dict.set_item("relation", relation.class())?;
    rel_dict.set_item("score", relation.probability())?;

    let subject = relation.subject();
    let subject_dict = PyDict::new_bound(py);
    subject_dict.set_item("text", &subject.text)?;
    subject_dict.set_item("label", &subject.label)?;
    subject_dict.set_item("score", subject.probability)?;
    subject_dict.set_item("start", subject.start)?;
    subject_dict.set_item("end", subject.end)?;
    rel_dict.set_item("subject", subject_dict)?;

    let object = relation.object();
    let object_dict = PyDict::new_bound(py);
    object_dict.set_item("text", &object.text)?;
    object_dict.set_item("label", &object.label)?;
    object_dict.set_item("score", object.probability)?;
    object_dict.set_item("start", object.start)?;
    object_dict.set_item("end", object.end)?;
    rel_dict.set_item("object", object_dict)?;

    Ok(rel_dict)
}

fn extracted_value_to_dict<'py>(
    py: Python<'py>,
    value: &gliner::model::gliner2::ExtractedValue,
) -> PyResult<Bound<'py, PyDict>> {
    let value_dict = PyDict::new_bound(py);
    value_dict.set_item("text", &value.text)?;
    value_dict.set_item("label", &value.label)?;
    value_dict.set_item("score", value.score)?;
    value_dict.set_item("start", value.start)?;
    value_dict.set_item("end", value.end)?;
    Ok(value_dict)
}
