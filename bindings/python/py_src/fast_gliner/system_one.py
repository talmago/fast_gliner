"""Pydantic models for System One questions and answers.

Question objects (`Choice`, `Noul`, `Score`) are the Python request API.
`run_system_one` serializes questions, calls the Rust runtime, and hydrates
`SystemOneOutput`.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


class Noul(BaseModel):
    """A yes/no question. The answer is P(true).

    Attributes:
        type: Always `noul`.
        instructions: Decision rule used as the classification prompt.
        criteria: Optional `true` / `false` descriptions.
    """

    type: Literal["noul"] = "noul"
    instructions: Optional[str] = None
    criteria: Optional[Dict[str, Optional[str]]] = None


class Choice(BaseModel):
    """A question that picks one named option.

    Attributes:
        type: Always `choice`.
        instructions: Decision rule used as the classification prompt.
        criteria: Option keys mapped to descriptions. A `None` value uses
            the key as the label.
    """

    type: Literal["choice"] = "choice"
    instructions: Optional[str] = None
    criteria: Dict[str, Optional[str]]


class Score(BaseModel):
    """A question that rates a state on an ordered rubric.

    Attributes:
        type: Always `score`.
        instructions: Decision rule used as the classification prompt.
        criteria: Ordered level descriptions, lowest first.
    """

    type: Literal["score"] = "score"
    instructions: Optional[str] = None
    criteria: List[Any]


Question = Annotated[
    Union[Choice, Noul, Score],
    Field(discriminator="type"),
]

_QUESTION_ADAPTER: TypeAdapter[Question] = TypeAdapter(Question)


class Usage(BaseModel):
    """Token accounting for a System One call.

    Attributes:
        input_tokens: Sum of encoded prompt lengths across questions.
        output_tokens: Always 0. System One does not generate tokens.
    """

    input_tokens: int
    output_tokens: int


class ChoiceAnswer(BaseModel):
    """A `choice` answer: one key from the question criteria.

    Attributes:
        type: Always `choice`.
        choice: Selected criterion key.
        confidence: Probability of `choice`.
        probabilities: Softmax over criterion keys.
    """

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]


class NoulAnswer(BaseModel):
    """A `noul` answer: probability of the true option.

    Attributes:
        type: Always `noul`.
        noul: P(true).
    """

    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(BaseModel):
    """A `score` answer: expected level over ordered criteria.

    Attributes:
        type: Always `score`.
        score: Expected level index, weighted by softmax probabilities.
        confidence: Probability of the most likely level.
        probabilities: Softmax over `"0"` .. `"n-1"`.
    """

    type: Literal["score"] = "score"
    score: float
    confidence: float
    probabilities: Dict[str, float]


Answer = Annotated[
    Union[ChoiceAnswer, NoulAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class SystemOneOutput(BaseModel):
    """Result of `FastGLiClass.system_one` or `FastGLiFormer.system_one`.

    Attributes:
        answers: Named answers, one per question.
        usage: Token accounting.
    """

    answers: Dict[str, Answer]
    usage: Usage


def serialize_questions(questions: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Turn question objects or dicts into the Rust System One payload.

    Args:
        questions: Mapping of question name to a `Choice`, `Noul`, `Score`,
            or a dict with a `type` field.

    Returns:
        JSON-ready question dicts.

    Raises:
        TypeError: If `questions` is not a mapping or a value is neither a
            question object nor a mapping.
        pydantic.ValidationError: If a dict does not match a question type.
    """
    if not isinstance(questions, Mapping):
        raise TypeError("questions must be a mapping of name to Choice, Noul, Score, or dict")

    payload = {}
    for name, question in questions.items():
        if isinstance(question, (Choice, Noul, Score)):
            parsed = question
        elif isinstance(question, Mapping):
            parsed = _QUESTION_ADAPTER.validate_python(question)
        else:
            raise TypeError(f"question {name!r} must be a Choice, Noul, Score, or dict, got {type(question).__name__}")
        payload[name] = _question_payload(parsed)
    return payload


def _question_payload(question: Union[Choice, Noul, Score]) -> Dict[str, Any]:
    """Dump a question, keeping `type` and None criterion descriptions.

    Args:
        question: A parsed question object.

    Returns:
        JSON-ready dict for the Rust runtime.
    """
    data = question.model_dump()
    data["type"] = question.type
    if data.get("instructions") is None:
        data.pop("instructions", None)
    if data.get("criteria") is None:
        data.pop("criteria", None)
    return data


def run_system_one(
    model,
    state: Any,
    questions: Mapping[str, Any],
    *,
    max_length: Optional[int] = None,
) -> SystemOneOutput:
    """Serialize questions, run the Rust runtime, and hydrate the result.

    Args:
        model: Bound Rust runtime with `system_one`.
        state: Text, a JSON object, a JSON list, or null.
        questions: Named `Choice`, `Noul`, or `Score` objects, or dicts.
        max_length: Optional token limit forwarded to the runtime.

    Returns:
        A `SystemOneOutput` with typed `answers` and `usage`.
    """
    return materialize_system_one(
        model.system_one(
            state,
            serialize_questions(questions),
            max_length=max_length,
        )
    )


def materialize_system_one(result: Any) -> SystemOneOutput:
    """Turn a raw System One dict into Pydantic models.

    Args:
        result: Mapping with `answers` and `usage`, as returned by the Rust
            runtime, or an existing `SystemOneOutput`.

    Returns:
        A `SystemOneOutput` instance.

    Raises:
        TypeError: If `result` is not a mapping or `SystemOneOutput`.
        pydantic.ValidationError: If the mapping does not match the schema.
    """
    if isinstance(result, SystemOneOutput):
        return result
    if not isinstance(result, Mapping):
        raise TypeError("system_one result must be a mapping with answers and usage")
    return SystemOneOutput.model_validate(result)
