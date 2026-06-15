"""
Backend-neutral analysis of function signatures and return types.
"""

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, get_origin, get_type_hints


def is_streaming_endpoint(func: Callable) -> bool:
    """Backend-neutral: True if *func* is a generator or annotated as an Iterator.

    A generator function (``yield``) or async generator function (``async yield``)
    is always considered streaming.  A regular function whose return annotation
    is ``Iterator[...]`` / ``AsyncIterator[...]`` or any subclass is also detected.
    """
    target = inspect.unwrap(func)
    if inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target):
        return True
    try:
        return_type = get_type_hints(target).get("return")
    except Exception:
        return False
    if return_type is None:
        return False
    origin = get_origin(return_type) or return_type
    return inspect.isclass(origin) and issubclass(origin, (Iterator, AsyncIterator))


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
