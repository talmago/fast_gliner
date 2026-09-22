from typing import List, Union

from .fast_gliner import PyFastGliNER
from .pretrained_model import _FastGLiNERBase


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

        return self._extract_relations_common(input_text, labels, schema)
