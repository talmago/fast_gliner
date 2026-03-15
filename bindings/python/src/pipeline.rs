use crate::output::ToPy;
use crate::schema::PyGLiNER2PipelineSchema;
use composable::*;
use gliner::model::gliner2::{ExtractionFieldSchema, ExtractionSchema, GLiNER2};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::model::runtime::InferenceMode;
use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::util::result::Result as GResult;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::*;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use pyo3::prelude::*;
use pyo3::types::PyAny;
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
