use serde_json::Value;

use crate::util::result::Result;

pub fn json_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| missing(key))
}

pub fn json_string_opt(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

pub fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| missing(key))
}

pub fn json_bool_opt(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

pub fn json_usize(value: &Value, key: &str) -> Result<usize> {
    json_usize_opt(value, key).ok_or_else(|| missing(key))
}

pub fn json_usize_opt(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|number| number as usize)
}

pub fn json_f32_opt(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|number| number as f32)
}

pub fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| missing(key))
}

fn missing(key: &str) -> Box<dyn std::error::Error + Send + Sync> {
    format!("config is missing {key}").into()
}
