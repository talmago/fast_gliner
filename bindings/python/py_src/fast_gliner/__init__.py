"""Python bindings for the Rust GLiNER inference engine.

`fast_gliner` keeps the Python API small. Inference runs in Rust through ONNX
Runtime for GLiNER, GLiNER2, GLiClass, and GLiFormer.

Examples:
    from fast_gliner import FastGLiNER2

    model = FastGLiNER2.from_pretrained("lion-ai/gliner2-multi-v1-onnx")
    model.predict_entities("I am James Bond", ["person"])
"""

from .gliclass import FastGLiClass
from .gliformer import FastGLiFormer
from .gliner import FastGLiNER
from .gliner2 import FastGLiNER2
from .system_one import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneOutput,
    Usage,
)

__version__ = "0.3.2"

__all__ = [
    "FastGLiNER",
    "FastGLiNER2",
    "FastGLiClass",
    "FastGLiFormer",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "SystemOneOutput",
    "Usage",
]
