"""Carry an author-declared JSON body under the function parameter's name.

A bare pydantic parameter is published as form fields. ``Body()`` opts that
parameter into ``application/json``. The harvested contract has to use the
same name the worker function expects. The schema title does not always match
that name (``order: MoreParams`` would otherwise be stored as ``more_params``).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Callable

from fastapi.params import Body, Form

from apipod.engine.signatures.bind import _plain_pydantic_model

JSON_BODY_NAME_EXTENSION = "x-apipod-body-name"


def explicit_json_body_name(func: Callable) -> str | None:
    """Return the parameter name when the signature has one JSON body model.

    ``Form`` and ``File`` are ``Body`` subclasses and stay on the form path.
    Zero or several JSON body models leave naming to the schema title.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    names: list[str] = []
    for name, param in signature.parameters.items():
        default = param.default
        if isinstance(default, Form) or not isinstance(default, Body):
            continue
        if _plain_pydantic_model(param.annotation) is None:
            continue
        names.append(name)
    if len(names) == 1:
        return names[0]
    return None


def stamp_json_body_names(schema: dict[str, Any], routes: Iterable[Any]) -> None:
    """Write ``x-apipod-body-name`` on operations whose body is JSON."""
    paths = schema.get("paths") or {}
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or []
        if endpoint is None or not path:
            continue
        name = explicit_json_body_name(endpoint)
        if name is None:
            continue
        path_item = paths.get(path)
        if not isinstance(path_item, dict):
            continue
        for method in methods:
            operation = path_item.get(str(method).lower())
            if not isinstance(operation, dict):
                continue
            content = (operation.get("requestBody") or {}).get("content") or {}
            if "application/json" not in content:
                continue
            operation[JSON_BODY_NAME_EXTENSION] = name
