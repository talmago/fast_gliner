use crate::output::ToPy;
use crate::schema::PyGLiNER2PipelineSchema;
use composable::*;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::model::runtime::InferenceMode;
use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::model::{
    nest_gliclass_scores, ExtractionFieldSchema, ExtractionSchema, GLiClass, GLiClassExample,
    GLiClassLabelNode, GLiClassLabels, GLiClassRequest, GLiFormer, GLiNER2, HierarchicalValue,
    StructureSchema,
};
use gliner::util::result::Result as GResult;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::*;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::{Py, Python};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

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
pub struct PyFastGLiClass {
    model: GLiClass,
}

#[pyclass]
pub struct PyFastGLiFormer {
    model: GLiFormer,
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

trait Inferencer: Send + Sync {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput>;
    fn get_orp_model(&self) -> &Model;
}

impl Inferencer for InferenceMode {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput> {
        self.inference(input)
    }

    fn get_orp_model(&self) -> &Model {
        self.get_inner_model()
    }
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
impl PyFastGliNER {
    #[new]
    #[pyo3(signature = (model_dir, filename=None, execution_provider=None))]
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
        let input = text_input_from_strings(&texts, &labels)?;

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
    #[pyo3(signature = (model_dir, filename=None, execution_provider=None))]
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
        let input = text_input_from_strings(&texts, &labels)?;

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

#[pymethods]
impl PyFastGLiClass {
    #[new]
    #[pyo3(signature = (model_dir, filename=None, execution_provider=None))]
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
                    "PyFastGLiClass loads models via GLiClass::from_dir and currently supports only the default ONNX layout",
                ));
            }
        }

        let model = GLiClass::from_dir(&model_dir, Parameters::default(), runtime_params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        Ok(Self { model })
    }

    #[pyo3(signature = (text, labels, *, examples=None, prompt=None, return_hierarchical=false))]
    fn classify(
        &self,
        py: Python<'_>,
        text: String,
        labels: &Bound<'_, PyAny>,
        examples: Option<&Bound<'_, PyAny>>,
        prompt: Option<String>,
        return_hierarchical: bool,
    ) -> PyResult<PyObject> {
        let parsed_labels = gliclass_labels_from_py(labels)?;
        let parsed_examples = gliclass_examples_from_py(examples)?;
        let output = self
            .model
            .classify_with(GLiClassRequest {
                text,
                labels: parsed_labels.clone(),
                examples: parsed_examples,
                prompt,
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        if return_hierarchical {
            let nested = nest_gliclass_scores(&parsed_labels, &output.scores);
            return hierarchical_value_to_py(py, &nested);
        }

        let pairs: Vec<(String, f32)> = output
            .scores
            .into_iter()
            .map(|score| (score.label, score.score))
            .collect();
        Ok(pairs.into_py(py))
    }
}

#[pymethods]
impl PyFastGLiFormer {
    #[new]
    #[pyo3(signature = (model_dir, _filename=None, execution_provider=None))]
    fn new(
        model_dir: String,
        _filename: Option<String>,
        execution_provider: Option<String>,
    ) -> PyResult<Self> {
        let providers = execution_providers_from_arg(execution_provider)?;
        let runtime_params = RuntimeParameters::default().with_execution_providers(providers);
        let model = GLiFormer::from_dir(&model_dir, Parameters::default(), runtime_params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(Self { model })
    }

    fn predict_entities(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        labels: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        if texts.len() > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GLiFormer currently does not support batched inference. Please pass a single input string.",
            ));
        }
        let text = texts.first().map(String::as_str).unwrap_or("");
        let output = py
            .allow_threads(|| self.model.inference(text, &labels))
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
            let schema = schema.extract::<Vec<(String, Vec<String>)>>().map_err(|_| {
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

    fn structure(&self, py: Python<'_>, text: String, schema_json: String) -> PyResult<PyObject> {
        let schema = StructureSchema::from_json(&schema_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let output = py
            .allow_threads(|| self.model.structure(&text, &schema))
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
        if texts.len() > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "GLiFormer currently does not support batched inference. Please pass a single input string.",
            ));
        }
        let text = texts.first().cloned().unwrap_or_default();
        let relation_schema = relation_schema_from_entries(relation_schema_entries);
        let output = py
            .allow_threads(|| {
                self.model
                    .extract_relations(&text, &entity_labels, &relation_schema)
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        output.to_py(py)
    }
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

fn gliclass_labels_from_py(labels: &Bound<'_, PyAny>) -> PyResult<GLiClassLabels> {
    if let Ok(dict) = labels.downcast::<PyDict>() {
        return Ok(GLiClassLabels::Hierarchical(gliclass_node_from_dict(dict)?));
    }
    if let Ok(flat) = labels.extract::<Vec<String>>() {
        return Ok(GLiClassLabels::Flat(flat));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "labels must be a list of strings or a hierarchical dict",
    ))
}

fn gliclass_node_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<GLiClassLabelNode> {
    let mut entries = Vec::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("hierarchical label keys must be strings")
        })?;
        entries.push((key, gliclass_node_from_py(&value)?));
    }
    Ok(GLiClassLabelNode::Group(entries))
}

fn gliclass_node_from_py(value: &Bound<'_, PyAny>) -> PyResult<GLiClassLabelNode> {
    if let Ok(dict) = value.downcast::<PyDict>() {
        return gliclass_node_from_dict(dict);
    }
    if let Ok(leaves) = value.extract::<Vec<String>>() {
        return Ok(GLiClassLabelNode::Leaves(leaves));
    }
    if let Ok(label) = value.extract::<String>() {
        return Ok(GLiClassLabelNode::Leaf(label));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "hierarchical label values must be a string, a list of strings, or a dict",
    ))
}

fn gliclass_examples_from_py(
    examples: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<GLiClassExample>> {
    let Some(examples) = examples else {
        return Ok(Vec::new());
    };
    if examples.is_none() {
        return Ok(Vec::new());
    }

    let list = examples.downcast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "examples must be a list of {\"text\": ..., \"labels\": [...]} dicts",
        )
    })?;

    let mut parsed = Vec::with_capacity(list.len());
    for (index, item) in list.iter().enumerate() {
        let dict = item.downcast::<PyDict>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "example {index} must be a dict with text and labels"
            ))
        })?;
        let text = dict
            .get_item("text")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("example {index} is missing text"))
            })?
            .extract::<String>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "example {index} text must be a string"
                ))
            })?;
        let labels_value = match dict.get_item("labels")? {
            Some(value) => value,
            None => dict.get_item("true_labels")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "example {index} is missing labels"
                ))
            })?,
        };
        let labels = labels_value.extract::<Vec<String>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "example {index} labels must be a list of strings"
            ))
        })?;
        parsed.push(GLiClassExample { text, labels });
    }
    Ok(parsed)
}

fn hierarchical_value_to_py(py: Python<'_>, value: &HierarchicalValue) -> PyResult<PyObject> {
    match value {
        HierarchicalValue::Score(score) => Ok(score.into_py(py)),
        HierarchicalValue::Group(entries) => {
            let dict = PyDict::new_bound(py);
            for (key, child) in entries {
                dict.set_item(key, hierarchical_value_to_py(py, child)?)?;
            }
            Ok(dict.into_py(py))
        }
    }
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
