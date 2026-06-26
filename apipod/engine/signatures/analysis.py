"""
Backend-neutral analysis of function signatures and return types.
"""

import inspect
from collections.abc import AsyncIterator, Iterator
from types import UnionType
from typing import Callable, Union, get_args, get_origin, get_type_hints


def _return_type_includes_iterator(return_type) -> bool:
    """True when *return_type* is or contains ``Iterator`` / ``AsyncIterator``."""
    if return_type is None:
        return False
    origin = get_origin(return_type)
    if origin in (Union, UnionType):
        return any(_return_type_includes_iterator(arg) for arg in get_args(return_type))
    resolved = origin or return_type
    return inspect.isclass(resolved) and issubclass(resolved, (Iterator, AsyncIterator))


def is_streaming_endpoint(func: Callable) -> bool:
    """Backend-neutral: True if *func* is a generator or annotated as an Iterator.

    A generator function (``yield``) or async generator function (``async yield``)
    is always considered streaming.  A regular function whose return annotation
    is ``Iterator[...]`` / ``AsyncIterator[...]`` (including inside a ``Union``)
    is also detected.
    """
    target = inspect.unwrap(func)
    if inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target):
        return True
    try:
        return_type = get_type_hints(target).get("return")
    except Exception:
        return False
    return _return_type_includes_iterator(return_type)


def is_injected_progress_param(param: inspect.Parameter) -> bool:
    """True if the parameter is a framework-injected :class:`JobProgress`."""
    return param.name == "job_progress" or "JobProgress" in str(param.annotation)


def job_progress_param_names(func: Callable) -> list[str]:
    """Names of the parameters of *func* that should receive a :class:`JobProgress`.

    A parameter qualifies when it is literally named ``job_progress`` or when its
    annotation refers to a ``JobProgress`` type. This single detection is shared
    by every injection site (queue worker, RunPod handler, direct FastAPI path)
    so they can never disagree about what counts as a progress parameter.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p.name for p in params if is_injected_progress_param(p)]
