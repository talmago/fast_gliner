"""Relation-extraction helpers shared by GLiNER, GLiNER2, and GLiFormer."""

from typing import List, Union

from .entities import as_text_batch, require_single_text
from .fast_gliner import PyRelationSchemaEntry


def normalize_relation_schema(schema: List[dict]) -> List[PyRelationSchemaEntry]:
    """Convert relation dicts into the PyO3 schema entries.

    Args:
        schema: Relation definitions with `relation`, `subject_labels`, and
            `object_labels`.

    Returns:
        Entries accepted by the Rust relation pipeline.
    """
    return [
        PyRelationSchemaEntry(
            relation=entry["relation"],
            subject_labels=entry["subject_labels"],
            object_labels=entry["object_labels"],
        )
        for entry in schema
    ]


def extract_relations(
    model,
    input_text: Union[str, List[str]],
    labels: List[str],
    schema: List[dict],
    *,
    batched: bool = True,
    runtime_name: str = "This runtime",
) -> Union[List[dict], List[List[dict]]]:
    """Extract relations and unwrap a single-text result.

    Args:
        model: Bound Rust runtime with `extract_relations`.
        input_text: Input text or batch of texts.
        labels: Entity labels used to propose relation arguments.
        schema: Relation definitions with `relation`, `subject_labels`, and
            `object_labels`.
        batched: If False, more than one text raises `ValueError`.
        runtime_name: Name used in the batch error message.

    Returns:
        A list of relation dicts for a single string, or a list of those
        lists for a batch. Each relation has `relation`, `score`, `subject`,
        and `object`.
    """
    if not batched:
        require_single_text(input_text, runtime_name)
    texts, single = as_text_batch(input_text)
    results = model.extract_relations(texts, labels, normalize_relation_schema(schema))
    return results[0] if single else results
