# TASK_MODEL_LOADING_REFACTOR.md

# Objective

Refactor the Rust inference engine (`gline-rs`) so that models are loaded from a **directory** rather than individual file paths.

This change prepares the runtime for GLiNER2 support and simplifies model loading.

This task must be completed **before implementing GLiNER2 runtime support**.

---

# Motivation

The current API requires explicit file paths:

GLiNER::<SpanMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "models/gliner_small-v2.1/tokenizer.json",
    "models/gliner_small-v2.1/onnx/model.onnx",
)

This approach is fragile and difficult to extend once models include additional components such as:

- configuration files
- multiple ONNX files
- tokenizer metadata
- task configuration

The runtime should instead load models from a **directory**.

---

# Target API

Introduce new constructors.

### Default loader

GLiNER::from_dir(
    model_dir,
    parameters,
    runtime_parameters
)

### Loader with optional overrides

GLiNER::from_dir_with(
    model_dir,
    parameters,
    runtime_parameters,
    tokenizer_path,
    onnx_model_path,
    config_path
)

---

# Default Model Layout

Models must follow this structure:

model_dir/
├── tokenizer.json
├── gliner_config.json
└── onnx/
    └── model.onnx

Default paths:

| Component | Default Path |
|----------|--------------|
| tokenizer | tokenizer.json |
| config | gliner_config.json |
| ONNX | onnx/model.onnx |

Paths are resolved relative to `model_dir`.

---

# New Configuration File

Introduce `gliner_config.json`.

Example:

{
  "mode": "span",
  "max_width": 8
}

Fields:

| Field | Description |
|------|-------------|
| mode | "span" or "token" |
| max_width | maximum span width |

---

# Automatic Mode Detection

The runtime must automatically determine whether to use:

SpanMode
TokenMode

based on `gliner_config.json`.

This eliminates the need for explicit generic parameters when loading models.

Example:

```rs
let model = GLiNER::from_dir(
    "models/gliner_small-v2.1",
    Parameters::default(),
    RuntimeParameters::default(),
)?;
```

The runtime reads `mode` from `gliner_config.json` and initializes the correct pipeline.

---

# Required Rust Changes

## 1. Configuration Loader

Create module:

gline-rs/src/model/config.rs

Responsibilities:

- parse `gliner_config.json`
- provide configuration defaults
- expose runtime configuration

Example struct:

pub struct ModelConfig {
    pub mode: ModelMode,
    pub max_width: usize,
}

---

## 2. Loader Implementation

Add methods:

```
GLiNER::from_dir()
GLiNER::from_dir_with()
```

Responsibilities:

- resolve default paths
- validate required files
- load tokenizer
- load ONNX model
- parse config
- select pipeline mode

---

## 3. Mode Selection

Pseudo-logic:

match config.mode {
    ModelMode::Span => initialize_span_pipeline(),
    ModelMode::Token => initialize_token_pipeline(),
}

---

# Milestone Validation

Add example:

gline-rs/examples/load_model.rs

Example:

```rs
use gliner_rs::GLiNER;
use gliner_rs::model::Parameters;
use orp::params::RuntimeParameters;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let model = GLiNER::from_dir(
        "models/gliner_small-v2.1",
        Parameters::default(),
        RuntimeParameters::default(),
    )?;

    println!("Model loaded successfully");

    Ok(())
}
```

---

# Test Command

```sh
cargo run --example load_model
```

---

# Python Validation

Ensure Python API continues working.

```python
from fast_gliner import FastGLiNER

model = FastGLiNER.from_pretrained(
    model_id="onnx-community/gliner_multi-v2.1",
    onnx_path="onnx/model.onnx"
)
```

Internally the Python binding should call the new Rust `from_dir` implementation.

---

# Acceptance Criteria

The task is complete when:

- directory-based model loading works
- mode detection works automatically
- existing examples still run
- Python bindings continue functioning
