from typing import List, Tuple

from .fast_gliner import PyFastGLiClass
from .pretrained_model import _FastGLiNERBase


class FastGLiClass(_FastGLiNERBase):
    """
    Python wrapper around the GLiClass runtime.

    GLiClass scores a text against caller-supplied labels in one forward pass.
    `prompt_first` is read from the checkpoint `config.json`.

    Example
    -------
    ```python
    from fast_gliner import FastGLiClass

    model = FastGLiClass.from_pretrained(
        "knowledgator/gliclass-small-v1.0"
    )

    model.classify(
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ["computing", "science", "programming", "travel", "food", "politics"],
    )
    ```
    """

    _backend = PyFastGLiClass

    def predict_entities(self, input_text, labels):
        raise NotImplementedError("GLiClass supports classification, not entity extraction.")

    def classify(self, text: str, labels: List[str]):
        """
        Score `text` against `labels`.

        Returns
        -------
        List[Tuple[str, float]]
            Label scores sorted from highest to lowest.
        """

        return self.model.classify(text, labels)
