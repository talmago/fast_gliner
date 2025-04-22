use std::fs;
use std::path::Path;
use pyo3::prelude::*;
use pyo3::{Py, Python};
use pyo3::types::{PyList, PyDict};
use serde::Deserialize;

use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::{span::SpanMode, token::TokenMode};
use gliner::model::output::decoded::SpanOutput;
use gliner::util::result::Result as GResult;
use orp::params::RuntimeParameters;


#[derive(Deserialize)]
struct PyFastGliNERConfig {
    span_mode: Option<String>,
}

#[pyclass]
pub struct PyFastGliNER {
    model: Box<dyn Inferencer + Send>,
}

trait Inferencer {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput>;
}

impl Inferencer for GLiNER<SpanMode> {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput> {
        self.inference(input)
    }
}

impl Inferencer for GLiNER<TokenMode> {
    fn inference(&self, input: TextInput) -> GResult<SpanOutput> {
        self.inference(input)
    }
}

#[pymethods]
impl PyFastGliNER {
    #[new]
    fn new(model_dir: String, filename: Option<String>) -> PyResult<Self> {
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

        let model: Box<dyn Inferencer + Send> = match parsed.span_mode.as_deref() {
            Some("token_level") => Box::new(
                GLiNER::<TokenMode>::new(
                    Parameters::default(),
                    RuntimeParameters::default(),
                    tokenizer_path.to_str().unwrap(),
                    onnx_path.to_str().unwrap(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?
            ),
            _ => Box::new(
                GLiNER::<SpanMode>::new(
                    Parameters::default(),
                    RuntimeParameters::default(),
                    tokenizer_path.to_str().unwrap(),
                    onnx_path.to_str().unwrap(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?
            ),
        };

        Ok(PyFastGliNER { model })
    }

    fn predict_entities(&self, py: Python<'_>, texts: Vec<String>, labels: Vec<String>) -> PyResult<Py<PyAny>> {
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&texts_ref, &labels_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        let output = self.model.inference(input)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        
        let results = PyList::empty(py);
        
        for spans in output.spans {
            let py_spans = PyList::empty(py);
            for span in spans {
                let span_dict = PyDict::new(py);
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
}

#[pymodule]
fn fast_gliner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFastGliNER>()?;
    Ok(())
}
