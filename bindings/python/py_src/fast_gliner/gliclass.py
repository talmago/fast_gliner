"""GLiClass text classification runtime."""

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from .fast_gliner import PyFastGLiClass
from .pretrained import _FastGLiNERBase, resolve_onnx_filename
from .system_one import Choice, Noul, Score, SystemOneOutput, run_system_one


class FastGLiClass(_FastGLiNERBase):
    """Python wrapper around the GLiClass runtime.

    GLiClass scores a text against caller-supplied labels in one forward pass.
    `prompt_first` is read from the checkpoint `config.json`.

    Labels may be a flat list or a nested dict. A dict is flattened to dotted
    names such as `sentiment.positive` before scoring. Optional `examples` and
    `prompt` are inserted into that same prompt.

    Examples:
        from fast_gliner import FastGLiClass

        model = FastGLiClass.from_pretrained("knowledgator/gliclass-small-v1.0")
        model.classify(
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
            ["computing", "science", "programming", "travel", "food", "politics"],
        )
    """

    _backend = PyFastGLiClass

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        onnx_path: Optional[str] = "onnx/model.onnx",
        execution_provider: Optional[Literal["cpu", "cuda"]] = None,
        **kwargs,
    ):
        """Load a GLiClass checkpoint from the Hub or a local directory.

        Args:
            model_id: Hugging Face repository id or local directory path.
            onnx_path: Preferred ONNX path inside the model directory.
            execution_provider: ONNX Runtime execution provider.
            **kwargs: Extra arguments forwarded to
                `huggingface_hub.snapshot_download`.

        Returns:
            A loaded `FastGLiClass` instance.
        """
        model_dir = cls._load_pretrained(model_id, **kwargs)
        return cls(
            str(model_dir),
            resolve_onnx_filename(model_dir, onnx_path),
            execution_provider,
        )

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
        """Score `text` against `labels`.

        Args:
            text: Sequence to classify.
            labels: Label strings, or a nested dict of groups and leaves.
                Dict leaves are scored as dotted names.
            examples: Few-shot examples, each with `text` and `labels` (or
                `true_labels`). These guide the model and are not scored.
                The tokenizer must contain `<<EXAMPLE>>`.
            prompt: Task description inserted after the label separator.
            return_hierarchical: If True, return a dict in the shape of
                `labels`. A flat label list becomes `{label: score}` in input
                order.
            classification_type: `multi-label` (default) applies an
                independent sigmoid per label. `single-label` applies a
                softmax over the label set.

        Returns:
            Sorted `(label, score)` pairs, or the nested score dict when
            `return_hierarchical` is True.
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
        questions: Mapping[str, Union[Choice, Noul, Score, Dict[str, Any]]],
        *,
        max_length: Optional[int] = None,
    ) -> SystemOneOutput:
        """Answer typed System One questions about a state.

        Each question is one single-label GLiClass pass. `choice`, `noul`, and
        `score` map onto labels the same way as the GLiClass v4 reference
        server. Inputs that exceed `max_length` tokens are refused.

        Args:
            state: Text, a JSON object, a JSON list, or null.
            questions: Named `Choice`, `Noul`, or `Score` objects. A dict
                with a `type` field is also accepted.
            max_length: Token limit. The default is the checkpoint cap
                (16384 for v4).

        Returns:
            A `SystemOneOutput` with typed `answers` and `usage`.
        """
        return run_system_one(self.model, state, questions, max_length=max_length)
