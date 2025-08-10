use std::fs;
use std::path::Path;
use pyo3::prelude::*;
use pyo3::{Py, Python};
use pyo3::types::{PyList, PyDict};
use serde::Deserialize;

use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::pipeline::{span::SpanMode, token::TokenMode};
use gliner::model::pipeline::{token::TokenPipeline, relation::RelationPipeline};
use gliner::model::output::decoded::SpanOutput;
use gliner::util::result::Result as GResult;

use orp::params::RuntimeParameters;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

use composable::*;
use orp::model::Model;
use orp::pipeline::*;


#[derive(Deserialize)]
struct PyFastGliNERConfig {
    span_mode: Option<String>,
}

#[pyclass]
pub struct PyFastGliNER {
    model: Box<dyn Inferencer + Send + Sync>,
    tokenizer_path: String,
}

trait Inferencer: Send + Sync {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput>;
    fn get_orp_model(&self) -> &Model;
}

impl Inferencer for GLiNER<SpanMode> {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput> {
        self.inference(input)
    }
    fn get_orp_model(&self) -> &Model {
        self.get_inner_model()
    }
}

impl Inferencer for GLiNER<TokenMode> {
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
    fn new(
        relation: String,
        subject_labels: Vec<String>,
        object_labels: Vec<String>,
    ) -> Self {
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
    fn new(model_dir: String, filename: Option<String>, execution_provider: Option<String>) -> PyResult<Self> {
        let base = Path::new(&model_dir);

        let config_path = base.join("gliner_config.json");
        let tokenizer_path = base.join("tokenizer.json");
        let onnx_path = match filename {
            Some(path) => base.join(path),
            None => base.join("onnx").join("model.onnx"),
        };

        let config_data = fs::read_to_string(config_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Could not read config: {}", e)))?;

        let parsed: PyFastGliNERConfig = serde_json::from_str(&config_data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let providers: Vec<ExecutionProviderDispatch> = match execution_provider.as_deref() {
            Some("cuda") => {
                #[cfg(feature = "cuda")]
                {
                    vec![CUDAExecutionProvider::default().build()]
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "CUDA execution provider requested but 'cuda' feature is not enabled",
                    ));
                }
            },
            Some("cpu") => vec![CPUExecutionProvider::default().build()],
            None => vec![],
            Some(other) => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported execution provider: '{}'. Use 'cpu' or 'cuda'.", other
            ))),
        };

        let runtime_params = RuntimeParameters::default().with_execution_providers(providers);

        let model: Box<dyn Inferencer + Send> = match parsed.span_mode.as_deref() {
            Some("token_level") => Box::new(
                GLiNER::<TokenMode>::new(
                    Parameters::default(),
                    runtime_params,
                    tokenizer_path.to_str().unwrap(),
                    onnx_path.to_str().unwrap(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?
            ),
            _ => Box::new(
                GLiNER::<SpanMode>::new(
                    Parameters::default(),
                    runtime_params,
                    tokenizer_path.to_str().unwrap(),
                    onnx_path.to_str().unwrap(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?
            ),
        };

        Ok(PyFastGliNER {
            model,
            tokenizer_path: tokenizer_path.to_string_lossy().to_string()
        })
    }

    fn predict_entities(&self, py: Python<'_>, texts: Vec<String>, labels: Vec<String>) -> PyResult<Py<PyAny>> {
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&texts_ref, &labels_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        let output = py.allow_threads(|| {
            self.model.inference(input)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

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

        let relation_pipeline = RelationPipeline::default(&self.tokenizer_path, &relation_schema)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let params = Parameters::default();

        let pipeline = composed![
            token_pipeline.to_composable(orp_model, &params),
            relation_pipeline.to_composable(orp_model, &params)
        ];

        let output = py.allow_threads(|| pipeline.apply(input))
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

#[pymodule]
fn fast_gliner(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastGliNER>()?;
    m.add_class::<PyRelationSchemaEntry>()?;
    Ok(())
}