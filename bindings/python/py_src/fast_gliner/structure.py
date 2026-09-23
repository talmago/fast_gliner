"""Compile a GLiFormer structure schema into the tree the Rust runtime reads.

A schema may be a nested dict or a Pydantic model class. Both become one
object, array, and scalar tree. Pydantic is not imported: a model class is
detected by ``model_fields``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, List, Union, get_args, get_origin

_TYPE_NAMES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
}

_ANNOTATION_TYPES = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}


def materialize_structure(schema: Mapping[str, Any], result: Any) -> Any:
    """Turn extracted records into Pydantic models when the schema uses them."""

    if not isinstance(result, Mapping):
        return result
    materialized = {}
    for name, value in result.items():
        model = _record_model(schema.get(name))
        if model is None or not isinstance(value, list):
            materialized[name] = value
            continue
        materialized[name] = [_hydrate(item, model) for item in value]
    return materialized


def compile_structure_schema(schema: Mapping[str, Any]) -> str:
    """Return the JSON schema tree for ``FastGLiFormer.structure``."""

    if not isinstance(schema, Mapping):
        raise TypeError("structure schema must be a mapping of record name to schema")
    if not schema:
        raise ValueError("structure schema must contain at least one record")

    fields = []
    for name, value in schema.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("structure record name must be a non-empty string")
        fields.append({"name": name, "schema": _compile_record(value)})
    return json.dumps({"fields": fields})


def _compile_record(value: Any) -> dict:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if not (len(value) == 1 and _is_type_name(value[0])):
            return {
                "kind": "object",
                "fields": [{"name": item, "schema": _scalar("str")} for item in value],
            }
    # A top-level list of one object is still one record schema. The runtime
    # returns a list of those records, so the extra list is not kept.
    if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], str):
        value = value[0]
    node = _compile_node(value)
    if node.get("kind") != "object":
        raise ValueError("each structure must be an object schema")
    return node


def _compile_node(value: Any) -> dict:
    if _is_pydantic_model(value):
        return _compile_pydantic(value)
    if isinstance(value, str):
        return _scalar(_normalize_type_name(value))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("an array schema must contain exactly one item schema")
        return {"kind": "array", "item": _compile_node(value[0])}
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("an object schema must contain at least one field")
        return {
            "kind": "object",
            "fields": [{"name": str(name), "schema": _compile_node(child)} for name, child in value.items()],
        }
    raise TypeError(f"unsupported structure schema value {value!r}")


def _compile_pydantic(model: type) -> dict:
    fields = []
    for name, field in model.model_fields.items():
        annotation = getattr(field, "annotation", str)
        fields.append({"name": str(name), "schema": _compile_annotation(annotation)})
    if not fields:
        raise ValueError(f"{model.__name__} has no fields")
    return {"kind": "object", "fields": fields}


def _compile_annotation(annotation: Any) -> dict:
    if _is_pydantic_model(annotation):
        return _compile_pydantic(annotation)

    origin = get_origin(annotation)
    if origin in (list, List):
        args = get_args(annotation)
        item = args[0] if args else str
        return {"kind": "array", "item": _compile_annotation(item)}
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _compile_annotation(args[0])

    if isinstance(annotation, str):
        return _scalar(_normalize_type_name(annotation))
    if annotation in _ANNOTATION_TYPES:
        return _scalar(_ANNOTATION_TYPES[annotation])
    raise TypeError(f"unsupported structure annotation {annotation!r}")


def _record_model(value: Any) -> Any:
    if _is_pydantic_model(value):
        return value
    if isinstance(value, list) and len(value) == 1 and _is_pydantic_model(value[0]):
        return value[0]
    return None


def _hydrate(value: Any, annotation: Any) -> Any:
    if _is_pydantic_model(annotation):
        if not isinstance(value, Mapping):
            return value
        data = {}
        for name, field in annotation.model_fields.items():
            if name not in value:
                continue
            data[name] = _hydrate(value[name], getattr(field, "annotation", None))
        construct = getattr(annotation, "model_construct", None)
        if construct is not None:
            return construct(**data)
        return annotation(**data)

    origin = get_origin(annotation)
    if origin in (list, List):
        args = get_args(annotation)
        item = args[0] if args else None
        if not isinstance(value, list):
            return value
        return [_hydrate(item_value, item) for item_value in value]
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _hydrate(value, args[0])
    return value


def _is_pydantic_model(value: Any) -> bool:
    return isinstance(value, type) and hasattr(value, "model_fields")


def _is_type_name(value: str) -> bool:
    return value.strip().lower() in _TYPE_NAMES


def _normalize_type_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _TYPE_NAMES:
        raise ValueError(f"unsupported structure field type `{value}`")
    return _TYPE_NAMES[normalized]


def _scalar(type_name: str) -> dict:
    return {"kind": "scalar", "type": type_name}
