/// Wall-clock stages for one single-text ONNX call, in milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageMillis {
    pub pre_ms: f64,
    pub onnx_ms: f64,
    pub post_ms: f64,
}
