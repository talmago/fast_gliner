"""Named-entity helpers shared by GLiNER, GLiNER2, and GLiFormer."""

from typing import List, Tuple, Union


def as_text_batch(input_text: Union[str, List[str]]) -> Tuple[List[str], bool]:
    """Normalize user input to a list of texts.

    Args:
        input_text: A single string or a list of strings.

    Returns:
        A pair `(texts, single)` where `texts` is a list of strings and
        `single` is True when the original input was one string.
    """
    if isinstance(input_text, str):
        return [input_text], True
    return input_text, False


def require_single_text(input_text: Union[str, List[str]], runtime_name: str) -> None:
    """Refuse a multi-text batch for runtimes that only accept one string.

    Args:
        input_text: A single string or a list of strings.
        runtime_name: Name used in the error message.

    Raises:
        ValueError: If more than one text is passed.
    """
    if isinstance(input_text, list) and len(input_text) > 1:
        raise ValueError(
            f"{runtime_name} currently does not support batched inference. Please pass a single input string."
        )


def predict_entities(
    model,
    input_text: Union[str, List[str]],
    labels: List[str],
    *,
    batched: bool = True,
    runtime_name: str = "This runtime",
) -> Union[List[dict], List[List[dict]]]:
    """Predict entities and unwrap a single-text result.

    Args:
        model: Bound Rust runtime with `predict_entities`.
        input_text: Input text or batch of texts.
        labels: Entity labels to detect.
        batched: If False, more than one text raises `ValueError`.
        runtime_name: Name used in the batch error message.

    Returns:
        A list of entity dicts for a single string, or a list of those lists
        for a batch. Each entity has `text`, `label`, `score`, `start`, and
        `end`.
    """
    if not batched:
        require_single_text(input_text, runtime_name)
    texts, single = as_text_batch(input_text)
    results = model.predict_entities(texts, labels)
    return results[0] if single else results
