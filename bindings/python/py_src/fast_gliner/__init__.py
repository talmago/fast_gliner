from pathlib import Path
from typing import List, Literal, Optional, Union
from abc import ABC

from huggingface_hub import snapshot_download

from .fast_gliner import PyFastGliNER, PyFastGliNER2, PyRelationSchemaEntry


class _FastGLiNERBase(ABC):
    """
    Shared functionality for FastGLiNER runtimes.

    This class implements common functionality for loading models,
    normalizing inputs, and running inference. Concrete runtimes
    provide the backend implementation via `_backend`.
    """

    _backend = None

    def __init__(
        self,
        model_path: str,
        onnx_path: Optional[str] = "onnx/model.onnx",
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
    ):
        self.model = self._backend(model_path, onnx_path, execution_provider)

    @staticmethod
    def _normalize_input(input_text):
        """
        Normalize user input to a list of texts.

        Returns
        -------
        Tuple[List[str], bool]
            Normalized texts and whether the original input was a single string.
        """
        if isinstance(input_text, str):
            return [input_text], True
        return input_text, False

    def predict_entities(
        self, input_text: Union[str, List[str]], labels: List[str]
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Predict entities in the given text(s).

        Parameters
        ----------
        input_text : str or List[str]
            Input text or batch of texts.
        labels : List[str]
            Entity labels to detect.

        Returns
        -------
        List[dict] or List[List[dict]]
            Predicted entities.
        """

        texts, single = self._normalize_input(input_text)

        results = self.model.predict_entities(texts, labels)

        return results[0] if single else results

    def extract_relations(
        self,
        input_text: Union[str, List[str]],
        labels: List[str],
        schema: List[dict],
    ):
        """
        Relation extraction is runtime-dependent.

        This method must be implemented by runtimes that support relation extraction.
        """
        raise NotImplementedError("Relation extraction is not supported for this GLiNER runtime.")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = "onnx/model.onnx",
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        """
        Load a pretrained model from the Hugging Face Model Hub or a local directory.

        Parameters
        ----------
        model_id : str
            Hugging Face repository ID or local directory path.
        onnx_path : str, optional
            Path to the ONNX model inside the model directory.
        execution_provider : {"cpu", "cuda"}, optional
            ONNX Runtime execution provider.

        Returns
        -------
        FastGLiNER or FastGLiNER2
            Loaded model instance.

        Raises
        ------
        FileNotFoundError
            If the ONNX model cannot be located.
        """

        model_dir = Path(model_id)

        if not model_dir.exists():
            model_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    allow_patterns=["*.json", "*.model", "onnx/*.onnx"],
                    **kwargs,
                )
            )

        if not (model_dir / onnx_path).exists():
            onnx_files = sorted(model_dir.rglob("*.onnx"))

            if len(onnx_files) == 1:
                onnx_path = onnx_files[0].relative_to(model_dir).as_posix()
            else:
                raise FileNotFoundError(f"Could not resolve ONNX model inside {model_dir}")

        return cls(str(model_dir.resolve()), onnx_path, execution_provider)


class FastGLiNER(_FastGLiNERBase):
    """
    Python wrapper for the GLiNER runtime.

    Example
    -------
    ```python
    from fast_gliner import FastGLiNER

    model = FastGLiNER.from_pretrained(
        model_id="juampahc/gliner_multi-v2.1-onnx"
    )

    model.predict_entities("I am James Bond", ["person"])
    ```

    Output
    ------
    ```python
    [
        {
            "text": "James Bond",
            "label": "person",
            "score": 0.90,
            "start": 5,
            "end": 15
        }
    ]
    ```
    """

    _backend = PyFastGliNER

    def extract_relations(
        self,
        input_text: Union[str, List[str]],
        labels: List[str],
        schema: List[dict],
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Extract relations between entities based on a user-defined schema.

        Parameters
        ----------
        input_text : str or List[str]
            Input text or batch of texts.
        labels : List[str]
            Entity labels to detect.
        schema : List[dict]
            Relation definitions with:
            - relation
            - subject_labels
            - object_labels

        Returns
        -------
        List[dict] or List[List[dict]]
            Extracted relations.
        """

        texts, single = self._normalize_input(input_text)

        schema_entries = [
            PyRelationSchemaEntry(
                relation=entry["relation"],
                subject_labels=entry["subject_labels"],
                object_labels=entry["object_labels"],
            )
            for entry in schema
        ]

        results = self.model.extract_relations(texts, labels, schema_entries)

        return results[0] if single else results


class FastGLiNER2(_FastGLiNERBase):
    """
    Python wrapper around the GLiNER2 runtime.

    GLiNER2 currently supports NER and classification inference.

    Example
    -------
    ```python
    from fast_gliner import FastGLiNER2

    model = FastGLiNER2.from_pretrained(
        model_id="lion-ai/gliner2-multi-v1-onnx"
    )

    model.predict_entities("I am James Bond", ["person"])
    ```
    """

    _backend = PyFastGliNER2

    def predict_entities(
        self, input_text: Union[str, List[str]], labels: List[str]
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Run NER inference using GLiNER2.

        Note
        ----
        GLiNER2 currently does **not support batched inference**.
        """

        if isinstance(input_text, list) and len(input_text) > 1:
            raise ValueError(
                "GLiNER2 currently does not support batched inference. Please pass a single input string."
            )

        return super().predict_entities(input_text, labels)

    def extract_relations(self, *args, **kwargs):
        raise NotImplementedError("GLiNER2 relation extraction is not implemented yet.")

    def classify(self, text: str, labels: List[str]):
        return self.model.classify(text, labels)


__version__ = "0.1.12"

__all__ = ["FastGLiNER", "FastGLiNER2"]
