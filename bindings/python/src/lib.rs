use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{Py, Python};
use std::path::Path;

use gliner::model::gliner2::GLiNER2;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::output::decoded::SpanOutput;
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

        span_output_to_py(py, output)
    }

    fn extract_relations(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        entity_labels: Vec<String>,
        relation_schema_entries: Vec<PyRelationSchemaEntry>,
    ) -> PyResult<Py<PyAny>> {
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let entity_labels_ref: Vec<&str> = entity_labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&texts_ref, &entity_labels_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        let mut relation_schema = RelationSchema::new();
        for entry in relation_schema_entries {
            let subj: Vec<&str> = entry.subject_labels.iter().map(|s| s.as_str()).collect();
            let obj: Vec<&str> = entry.object_labels.iter().map(|s| s.as_str()).collect();
            relation_schema.push_with_allowed_labels(&entry.relation, &subj, &obj);
        }

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

        let relation_output = output;
        let py_results = PyList::empty_bound(py);

        for relation_list in &relation_output.relations {
            let py_relations = PyList::empty_bound(py);
            for rel in relation_list {
                let rel_dict = PyDict::new_bound(py);
                rel_dict.set_item("relation", rel.class())?;
                rel_dict.set_item("score", rel.probability())?;

                let subject = rel.subject();
                let object = rel.object();

                let subject_dict = PyDict::new_bound(py);
                subject_dict.set_item("text", &subject.text)?;
                subject_dict.set_item("label", &subject.label)?;
                subject_dict.set_item("score", subject.probability)?;
                subject_dict.set_item("start", subject.start)?;
                subject_dict.set_item("end", subject.end)?;
                rel_dict.set_item("subject", subject_dict)?;

                let object_dict = PyDict::new_bound(py);
                object_dict.set_item("text", &object.text)?;
                object_dict.set_item("label", &object.label)?;
                object_dict.set_item("score", object.probability)?;
                object_dict.set_item("start", object.start)?;
                object_dict.set_item("end", object.end)?;
                rel_dict.set_item("object", object_dict)?;

                py_relations.append(rel_dict)?;
            }
            py_results.append(py_relations)?;
        }

        Ok(py_results.into())
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

        span_output_to_py(py, output)
    }

    fn extract_relations(
        &self,
        _py: Python<'_>,
        _texts: Vec<String>,
        _entity_labels: Vec<String>,
        _relation_schema_entries: Vec<PyRelationSchemaEntry>,
    ) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GLiNER2 Python bindings currently expose NER only; relation extraction is not wired for GLiNER2 yet",
        ))
    }
}

#[pymodule]
fn fast_gliner(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastGliNER>()?;
    m.add_class::<PyFastGliNER2>()?;
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

fn span_output_to_py(py: Python<'_>, output: SpanOutput) -> PyResult<Py<PyAny>> {
    let results = PyList::empty_bound(py);

    for spans in output.spans {
        let py_spans = PyList::empty_bound(py);
        for span in spans {
            let span_dict = PyDict::new_bound(py);
            span_dict.set_item("text", span.text())?;
            span_dict.set_item("label", span.class())?;
            span_dict.set_item("score", span.probability())?;

            let (start, end) = span.offsets();
            span_dict.set_item("start", start)?;
            span_dict.set_item("end", end)?;
            py_spans.append(span_dict)?;
        }
        results.append(py_spans)?;
    }

    Ok(results.into())
}
