"""GLiNER2 multi-task named-entity, classification, and extraction runtime."""

from typing import List, Literal, Optional, Tuple, Union

from .entities import predict_entities
from .fast_gliner import PyFastGliNER2, PyGLiNER2MultiPipelineSchema
from .pretrained import _FastGLiNERBase, resolve_onnx_filename
from .relations import extract_relations


class FastGLiNER2(_FastGLiNERBase):
    """Python wrapper around the GLiNER2 runtime.

    GLiNER2 supports NER, classification, structured extraction, and relation
    extraction. Batched NER is not supported.

    Examples:
        from fast_gliner import FastGLiNER2

        model = FastGLiNER2.from_pretrained(
            model_id="lion-ai/gliner2-multi-v1-onnx"
        )
        model.predict_entities("I am James Bond", ["person"])
    """

    _backend = PyFastGliNER2

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = "onnx/model.onnx",
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        """Load a GLiNER2 checkpoint from the Hub or a local directory.

        Args:
            model_id: Hugging Face repository id or local directory path.
            onnx_path: Preferred ONNX path inside the model directory.
            execution_provider: ONNX Runtime execution provider.
            **kwargs: Extra arguments forwarded to
                `huggingface_hub.snapshot_download`.

        Returns:
            A loaded `FastGLiNER2` instance.
        """
        model_dir = cls._load_pretrained(model_id, **kwargs)
        return cls(
            str(model_dir),
            resolve_onnx_filename(model_dir, onnx_path),
            execution_provider,
        )

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
            runtime_name="GLiNER2",
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

    def create_schema(self) -> PyGLiNER2MultiPipelineSchema:
        """Create a GLiNER2 pipeline schema builder.

        The schema selects which tasks run during `extract`: classification,
        entities, relations, and structured fields.

        Returns:
            An empty schema builder.

        Examples:
            schema = (
                model.create_schema()
                .classification("document_type", ["news", "report"])
                .entities(["person", "company"])
                .relation("works_for", ["person"], ["company"])
            )
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

    def extract_json(self, text: str, schema: dict):
        """Run structured extraction from a JSON-style schema.

        Args:
            text: Input text.
            schema: Mapping of record name to field specs. A field may be a
                plain name or `name::type`, where type is `str` or `list`.

        Returns:
            Extracted records keyed by schema name.

        Examples:
            model.extract_json(
                text,
                {
                    "contact": [
                        "name::str",
                        "email::str",
                        "phone::list",
                        "address",
                    ]
                },
            )
        """
        return self.model.extract_json(text, schema)
