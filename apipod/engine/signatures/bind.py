"""Bind incoming job kwargs onto an endpoint signature.

FastAPI inlines pydantic model fields as form properties in OpenAPI. The gate
remounts those flattened fields and forwards them as RunPod ``input``. The
worker still calls the original function (``def execute(request: ExecuteRequest)``),
so this helper rebuilds the model from the flattened keys.
"""

from __future__ import annotations

import inspect
from types import UnionType
from typing import Any, Callable, Optional, Union, get_args, get_origin

from pydantic import BaseModel
from socaity_schemas import FileModel

from apipod.engine.signatures.analysis import is_injected_progress_param


def _plain_pydantic_model(annotation: Any) -> Optional[type[BaseModel]]:
    """Return a pydantic model from *annotation*, excluding FileModel (media)."""
    candidates = [annotation]
    if get_origin(annotation) in (Union, UnionType):
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]

    for candidate in candidates:
        if (
            inspect.isclass(candidate)
            and issubclass(candidate, BaseModel)
            and not issubclass(candidate, FileModel)
        ):
            return candidate
    return None


def bind_call_kwargs(func: Callable, kwargs: dict) -> dict:
    """Return kwargs that match *func*'s signature.

    Flattened form fields that belong to a missing pydantic parameter are
    assembled with ``model_validate``. Extra keys are dropped so the call
    cannot collide with ``*args, **kwargs`` wrappers.
    """
    sig = inspect.signature(inspect.unwrap(func))
    bound: dict[str, Any] = {}
    param_names = set(sig.parameters)

    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if is_injected_progress_param(param):
            if name in kwargs:
                bound[name] = kwargs[name]
            continue

        if name in kwargs:
            value = kwargs[name]
            model = _plain_pydantic_model(param.annotation)
            if (
                model is not None
                and not isinstance(value, model)
                and isinstance(value, dict)
            ):
                bound[name] = model.model_validate(value)
            else:
                bound[name] = value
            continue

        model = _plain_pydantic_model(param.annotation)
        if model is None:
            continue

        nested = {
            field: kwargs[field]
            for field in model.model_fields
            if field in kwargs and field not in (param_names - {name})
        }
        if not nested:
            continue
        bound[name] = model.model_validate(nested)

    return bound
