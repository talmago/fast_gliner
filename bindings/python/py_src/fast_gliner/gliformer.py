from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from huggingface_hub import snapshot_download

from .fast_gliner import PyFastGLiFormer, PyGLiNER2PipelineSchema
from .pretrained_model import _FastGLiNERBase
from .structure import compile_structure_schema, materialize_structure


class FastGLiFormer(_FastGLiNERBase):
    """
    Python wrapper around the GLiFormer text runtime.

    NER, classification, relations, and the flat multi-task schema match
    `FastGLiNER2`. Nested records use `structure`, which accepts a dict or a
    Pydantic model. `classify` returns `(label, score)` pairs sorted from
    highest to lowest, the same contract as `FastGLiClass`.
    """

    _backend = PyFastGLiFormer

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = None,
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        del onnx_path
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    allow_patterns=["*.json", "onnx/*.onnx"],
                    **kwargs,
                )
            )
        encoder = model_dir / "onnx" / "encoder.onnx"
        if not encoder.is_file():
            raise FileNotFoundError(
                f"Missing {encoder}. GLiFormer checkpoints need the split ONNX graphs under onnx/."
            )
        return cls(str(model_dir.resolve()), None, execution_provider)

    def predict_entities(
        self, input_text: Union[str, List[str]], labels: List[str]
    ) -> Union[List[dict], List[List[dict]]]:
        if isinstance(input_text, list) and len(input_text) > 1:
            raise ValueError(
                "GLiFormer currently does not support batched inference. Please pass a single input string."
            )
        return super().predict_entities(input_text, labels)

    def extract_relations(
        self,
        input_text: Union[str, List[str]],
        labels: List[str],
        schema: List[dict],
    ) -> Union[List[dict], List[List[dict]]]:
        return self._extract_relations_common(input_text, labels, schema)

    def classify(self, text: str, labels: List[str]):
        return self.model.classify(text, labels)

    def create_schema(self) -> PyGLiNER2PipelineSchema:
        return self.model.create_schema()

    def extract(
        self,
        text: str,
        schema: Union["PyGLiNER2PipelineSchema", List[Tuple[str, List[str]]]],
    ):
        return self.model.extract(text, schema)

    def structure(self, text: str, schema):
        """
        Extract nested records from ``text``.

        ``schema`` is a mapping of record name to a schema. A schema value may
        be a nested dict, a list of field names, or a Pydantic model class.
        Both forms are compiled to the same tree. A Pydantic schema returns
        instances of that model. A dict schema returns dicts.

        Example
        -------
        ```python
        model.structure(text, {
            "company": {
                "name": "str",
                "departments": [{
                    "name": "str",
                    "employees": [{"name": "str", "role": "str"}],
                }],
            }
        })

        model.structure(text, {"company": Company})
        ```
        """
        compiled = compile_structure_schema(schema)
        return materialize_structure(schema, self.model.structure(text, compiled))
