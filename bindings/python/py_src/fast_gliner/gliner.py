"""GLiNER span and token named-entity runtime."""

from typing import List, Literal, Optional, Union

from .entities import predict_entities
from .fast_gliner import PyFastGliNER
from .pretrained import _FastGLiNERBase, resolve_onnx_filename
from .relations import extract_relations


class FastGLiNER(_FastGLiNERBase):
    """Python wrapper for the GLiNER runtime.

    Loads a span-mode or token-mode checkpoint and runs named entity
    recognition. Token-mode checkpoints can also extract relations.

    Examples:
        from fast_gliner import FastGLiNER

        model = FastGLiNER.from_pretrained(
            model_id="juampahc/gliner_multi-v2.1-onnx"
        )
        model.predict_entities("I am James Bond", ["person"])
        # [{'text': 'James Bond', 'label': 'person', 'score': 0.90,
        #   'start': 5, 'end': 15}]
    """

    _backend = PyFastGliNER

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = "onnx/model.onnx",
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        """Load a GLiNER checkpoint from the Hub or a local directory.

        Args:
            model_id: Hugging Face repository id or local directory path.
            onnx_path: Preferred ONNX path inside the model directory.
            execution_provider: ONNX Runtime execution provider.
            **kwargs: Extra arguments forwarded to
                `huggingface_hub.snapshot_download`.

        Returns:
            A loaded `FastGLiNER` instance.
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
        """Predict entities in the given text.

        Args:
            input_text: Input text or batch of texts.
            labels: Entity labels to detect.

        Returns:
            A list of entity dicts for a single string, or a list of those
            lists for a batch. Each entity has `text`, `label`, `score`,
            `start`, and `end`.
        """
        return predict_entities(self.model, input_text, labels)

    def extract_relations(
        self,
        input_text: Union[str, List[str]],
        labels: List[str],
        schema: List[dict],
    ) -> Union[List[dict], List[List[dict]]]:
        """Extract relations between entities from a user-defined schema.

        Requires a token-mode GLiNER checkpoint such as
        `onnx-community/gliner-multitask-large-v0.5`.

        Args:
            input_text: Input text or batch of texts.
            labels: Entity labels to detect.
            schema: Relation definitions. Each dict needs `relation`,
                `subject_labels`, and `object_labels`.

        Returns:
            A list of relation dicts for a single string, or a list of those
            lists for a batch. Each relation has `relation`, `score`,
            `subject`, and `object`.
        """
        return extract_relations(self.model, input_text, labels, schema)
