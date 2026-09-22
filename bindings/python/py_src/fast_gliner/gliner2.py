from typing import List, Tuple, Union

from .fast_gliner import PyFastGliNER2, PyGLiNER2PipelineSchema
from .pretrained_model import _FastGLiNERBase


class FastGLiNER2(_FastGLiNERBase):
    """
    Python wrapper around the GLiNER2 runtime.

    GLiNER2 supports NER, classification, structured extraction, and relation extraction.

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
        """
        Create a GLiNER2 pipeline schema builder.

        The schema defines which tasks will be executed during inference,
        such as classification, entity extraction, relations, and structured fields.

        Example
        -------
        ```python
        schema = (
            model.create_schema()
                .classification("document_type", ["news", "report"])
                .entities(["person", "company"])
                .relations(["works_for"])
        )
        ```
        """
        return self.model.create_schema()

    def extract(
        self,
        text: str,
        schema: Union["PyGLiNER2PipelineSchema", List[Tuple[str, List[str]]]],
    ):
        """
        Run GLiNER2 extraction with either a pipeline schema builder or legacy schema tuples.

        Parameters
        ----------
        text : str
            Input text.
        schema : PyGLiNER2PipelineSchema or List[Tuple[str, List[str]]]
            Pipeline schema builder created by `create_schema()` (recommended),
            or legacy structured extraction schema tuples.

        Returns
        -------
        dict
            For pipeline schemas, returns:
            {
                "classifications": {...},
                "entities": [...],
                "relations": [...],
                "structures": {...}
            }
            For legacy tuple schemas, returns a structured extraction field dictionary.
        """
        return self.model.extract(text, schema)

    def extract_json(self, text: str, schema: dict):
        """
        Run GLiNER2 structured extraction using the original JSON schema format.

        Example schema:
        {
            "contact": [
                "name::str",
                "email::str",
                "phone::list",
                "address"
            ]
        }
        """
        return self.model.extract_json(text, schema)
