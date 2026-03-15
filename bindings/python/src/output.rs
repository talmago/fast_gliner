use gliner::model::gliner2::{ExtractedValue, ExtractionOutput, GLiNER2PipelineOutput};
use gliner::model::output::{decoded::SpanOutput, relation::Relation, relation::RelationOutput};
use gliner::text::span::Span;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

pub(crate) trait ToPy {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

impl ToPy for SpanOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let results = PyList::empty_bound(py);

        for spans in &self.spans {
            let py_spans = PyList::empty_bound(py);
            for span in spans {
                let span_dict = pipeline_output_to_py(py, span)?;
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
                let rel_dict = extraction_output_to_py(py, rel)?;
                py_relations.append(rel_dict)?;
            }
            py_results.append(py_relations)?;
        }

        Ok(py_results.into())
    }
}

impl ToPy for ExtractionOutput {
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = PyDict::new_bound(py);

        for field in &self.fields {
            let values = PyList::empty_bound(py);

            for value in &field.values {
                values.append(extracted_value_to_py(py, value)?)?;
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
            entities.append(pipeline_output_to_py(py, entity)?)?;
        }
        result.set_item("entities", entities)?;

        let relations = PyList::empty_bound(py);
        for relation in &self.relations {
            relations.append(extraction_output_to_py(py, relation)?)?;
        }
        result.set_item("relations", relations)?;

        let structures = PyDict::new_bound(py);
        for (structure_name, structure_output) in &self.structures {
            let structure_dict = PyDict::new_bound(py);
            for field in &structure_output.fields {
                let values = PyList::empty_bound(py);
                for value in &field.values {
                    values.append(&value.text)?;
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
        json_value_to_py(py, self)
    }
}

fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
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
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        serde_json::Value::Object(entries) => {
            let dict = PyDict::new_bound(py);
            for (key, value) in entries {
                dict.set_item(key, json_value_to_py(py, value)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn pipeline_output_to_py<'py>(
    py: Python<'py>,
    span: &Span,
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

fn extraction_output_to_py<'py>(
    py: Python<'py>,
    relation: &Relation,
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

fn extracted_value_to_py<'py>(
    py: Python<'py>,
    value: &ExtractedValue,
) -> PyResult<Bound<'py, PyDict>> {
    let value_dict = PyDict::new_bound(py);
    value_dict.set_item("text", &value.text)?;
    value_dict.set_item("label", &value.label)?;
    value_dict.set_item("score", value.score)?;
    value_dict.set_item("start", value.start)?;
    value_dict.set_item("end", value.end)?;
    Ok(value_dict)
}
