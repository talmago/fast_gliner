from typing import Any, Dict, List, Optional, Tuple, Union

from .fast_gliner import PyFastGLiClass
from .pretrained_model import _FastGLiNERBase


class FastGLiClass(_FastGLiNERBase):
    """
    Python wrapper around the GLiClass runtime.

    GLiClass scores a text against caller-supplied labels in one forward pass.
    `prompt_first` is read from the checkpoint `config.json`.

    Labels may be a flat list or a nested dict. A dict is flattened to dotted
    names such as `sentiment.positive` before scoring. Optional `examples` and
    `prompt` are inserted into that same prompt.

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

    def classify(
        self,
        text: str,
        labels: Union[List[str], Dict[str, Any]],
        *,
        examples: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        return_hierarchical: bool = False,
        classification_type: Optional[str] = None,
    ) -> Union[List[Tuple[str, float]], Dict[str, Any]]:
        """
        Score `text` against `labels`.

        Parameters
        ----------
        text:
            The sequence to classify.
        labels:
            A list of label strings, or a nested dict of groups and leaves.
            Dict leaves are scored as dotted names.
        examples:
            Few-shot examples, each with `text` and `labels` (or `true_labels`).
            These guide the model and do not add scored labels.
        prompt:
            A task description inserted after the label separator.
        return_hierarchical:
            When true, return a dict in the shape of `labels` instead of a
            sorted list. A flat label list becomes `{label: score}` in input
            order.
        classification_type:
            `multi-label` (default) applies an independent sigmoid per label.
            `single-label` applies a softmax over the label set.

        Returns
        -------
        List[Tuple[str, float]] or Dict[str, Any]
            Sorted label scores, or the nested score dict.
        """

        return self.model.classify(
            text,
            labels,
            examples=examples,
            prompt=prompt,
            return_hierarchical=return_hierarchical,
            classification_type=classification_type,
        )

    def system_one(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        *,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer typed System One questions about a state.

        Each question is one single-label GLiClass pass. `choice`, `noul`, and
        `score` map onto labels the same way as the GLiClass v4 reference
        server. Inputs that exceed `max_length` tokens are refused.

        Parameters
        ----------
        state:
            Text, a JSON object, a JSON list, or null.
        questions:
            Named questions. Each value needs `type` and, for `choice` and
            `score`, `criteria`. Optional `instructions` become the task prompt.
        max_length:
            Token limit. The default is the checkpoint cap (16384 for v4).

        Returns
        -------
        dict
            `answers` and `usage` with `input_tokens` and `output_tokens`.
        """

        return self.model.system_one(state, questions, max_length=max_length)
