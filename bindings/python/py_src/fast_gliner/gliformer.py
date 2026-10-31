"""GLiFormer multi-task text runtime with nested structure extraction."""

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from .entities import predict_entities
from .fast_gliner import PyFastGLiFormer, PyGLiNER2MultiPipelineSchema
from .pretrained import _FastGLiNERBase
from .relations import extract_relations
from .structure import compile_structure_schema, materialize_structure
from .system_one import Choice, Noul, Score, SystemOneOutput, run_system_one


class FastGLiFormer(_FastGLiNERBase):
    """Python wrapper around the GLiFormer text runtime.

    NER, classification, relations, and the flat multi-task schema match
    `FastGLiNER2`. Nested records use `structure`, which accepts a dict or a
    Pydantic model. `classify` returns `(label, score)` pairs sorted from
    highest to lowest, the same contract as `FastGLiClass`. `system_one`
    matches `FastGLiClass.system_one`.

    Checkpoints are split ONNX graphs under `onnx/`, including `encoder.onnx`.
    """

    _backend = PyFastGLiFormer
    _snapshot_allow_patterns = ("*.json", "onnx/*.onnx")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = None,
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        """Load a GLiFormer checkpoint from the Hub or a local directory.

        Args:
            model_id: Hugging Face repository id or local directory path.
            onnx_path: Unused. GLiFormer always loads the split graphs under
                `onnx/`.
            execution_provider: ONNX Runtime execution provider.
            **kwargs: Extra arguments forwarded to
                `huggingface_hub.snapshot_download`.

        Returns:
            A loaded `FastGLiFormer` instance.

        Raises:
            FileNotFoundError: If a local path is missing or `onnx/encoder.onnx`
                is not present.
        """
        del onnx_path
        model_dir = cls._load_pretrained(model_id, **kwargs)
        encoder = model_dir / "onnx" / "encoder.onnx"
        if not encoder.is_file():
            raise FileNotFoundError(
                f"Missing {encoder}. GLiFormer checkpoints need the split ONNX graphs under onnx/."
            )
        return cls(str(model_dir), None, execution_provider)

    def predict_entities(
        self, input_text: Union[str, List[str]], labels: List[str]
    ) -> Union[List[dict], List[List[dict]]]:
        """Run named entity recognition.

        Args:
            input_text: A single string. A one-element list is accepted.
            labels: Entity labels to detect.

        Returns:
            Entity dicts with `text`, `label`, `score`, `start`, and `end`.

        Raises:
            ValueError: If more than one text is passed.
        """
        return predict_entities(
            self.model,
            input_text,
            labels,
            batched=False,
            runtime_name="GLiFormer",
        )

    def extract_relations(
        self,
        input_text: Union[str, List[str]],
        labels: List[str],
        schema: List[dict],
    ) -> Union[List[dict], List[List[dict]]]:
        """Extract relations between entities from a user-defined schema.

        Args:
            input_text: Input text or a one-element batch.
            labels: Entity labels to detect.
            schema: Relation definitions. Each dict needs `relation`,
                `subject_labels`, and `object_labels`.

        Returns:
            Relation dicts with `relation`, `score`, `subject`, and `object`.
        """
        return extract_relations(self.model, input_text, labels, schema)

    def classify(self, text: str, labels: List[str]):
        """Score `text` against `labels`.

        Args:
            text: Input text.
            labels: Candidate class names.

        Returns:
            `(label, score)` pairs sorted from highest to lowest.
        """
        return self.model.classify(text, labels)

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Union[Choice, Noul, Score, Dict[str, Any]]],
        *,
        max_length: Optional[int] = None,
    ) -> SystemOneOutput:
        """Answer typed System One questions about a state.

        Each question is one single-label GLiFormer classification pass.
        `choice`, `noul`, and `score` map onto labels the same way as
        `FastGLiClass.system_one`. Inputs that exceed `max_length` tokens
        are refused.

        Args:
            state: Text, a JSON object, a JSON list, or null.
            questions: Named `Choice`, `Noul`, or `Score` objects. A dict
                with a `type` field is also accepted.
            max_length: Token limit. The default is the checkpoint `max_len`.

        Returns:
            A `SystemOneOutput` with typed `answers` and `usage`.
        """
        return run_system_one(self.model, state, questions, max_length=max_length)

    def create_schema(self) -> PyGLiNER2MultiPipelineSchema:
        """Create a multi-task pipeline schema builder.

        The builder matches `FastGLiNER2.create_schema`. Nested records still
        go through `structure`; schema-builder `structure` fields stay flat.

        Returns:
            An empty schema builder.
        """
        return self.model.create_schema()

    def extract(
        self,
        text: str,
        schema: Union["PyGLiNER2MultiPipelineSchema", List[Tuple[str, List[str]]]],
    ):
        """Run multi-task extraction.

        Args:
            text: Input text.
            schema: Pipeline schema from `create_schema()`, or legacy
                structured-extraction tuples of `(name, fields)`.

        Returns:
            For a pipeline schema, a dict with `classifications`, `entities`,
            `relations`, and `structures`. For legacy tuples, a structured
            field dictionary.
        """
        return self.model.extract(text, schema)

    def structure(self, text: str, schema):
        """Extract nested records from `text`.

        `schema` is a mapping of record name to a schema. A schema value may
        be a nested dict, a list of field names, or a Pydantic model class.
        Both forms compile to the same tree. A Pydantic schema returns
        instances of that model. A dict schema returns dicts.

        Args:
            text: Input text.
            schema: Mapping of record name to a dict, field-name list, or
                Pydantic model.

        Returns:
            Extracted records keyed by name. Values are lists of dicts or
            Pydantic instances.

        Examples:
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
        """
        compiled = compile_structure_schema(schema)
        return materialize_structure(schema, self.model.structure(text, compiled))
