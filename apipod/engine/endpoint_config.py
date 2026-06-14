"""
Endpoint planning: backend-neutral analysis of an endpoint function.

``is_streaming_endpoint`` inspects a function's definition to decide whether
it streams its output.  ``build_plan`` combines that with schema detection and
optional backend-specific parameters (queue, upload limits, …) into an
immutable :class:`EndpointExecutionPlan` — the shared contract used by every
backend (FastAPI, RunPod).
"""

import inspect
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, get_origin, get_type_hints

from apipod.engine.backend.schema_resolve import SchemaBinding, get_schema_binding


@dataclass(frozen=True)
class EndpointExecutionPlan:
    """
    Immutable plan describing how an endpoint should be registered and executed.

    The plan separates endpoint configuration decisions from router registration
    mechanics. Fields that only apply to a specific backend (e.g. queue options)
    default to sensible no-ops so every backend can build a plan with what it needs.
    """

    path: str
    methods: list[str] | None = None
    should_use_queue: bool = False
    max_upload_file_size_mb: int | None = None
    queue_size: int = 500
    route_args: tuple[Any, ...] = ()
    route_kwargs: dict[str, Any] = field(default_factory=dict)
    schema_binding: SchemaBinding | None = None
    is_streaming: bool = False

    @property
    def is_schema_endpoint(self) -> bool:
        return self.schema_binding is not None

    @property
    def active_methods(self) -> list[str]:
        return ["POST"] if self.methods is None else self.methods


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


def build_plan(
    func: Callable,
    path: str,
    *,
    methods: list[str] | None = None,
    should_use_queue: bool = False,
    max_upload_file_size_mb: int | None = None,
    queue_size: int = 500,
    route_args: tuple[Any, ...] = (),
    route_kwargs: dict[str, Any] | None = None,
) -> EndpointExecutionPlan:
    """Build a backend-neutral :class:`EndpointExecutionPlan` by inspecting *func*.

    Both the FastAPI and RunPod backends call this.  Backend-specific parameters
    (``methods``, ``queue_*``, ``route_*``) are optional; they default to no-ops
    so RunPod can simply call ``build_plan(func, path=path)``.
    """
    schema_binding = get_schema_binding(func)
    return EndpointExecutionPlan(
        path=path,
        methods=methods,
        should_use_queue=should_use_queue,
        max_upload_file_size_mb=max_upload_file_size_mb,
        queue_size=queue_size,
        route_args=route_args,
        route_kwargs=route_kwargs if route_kwargs is not None else {},
        schema_binding=schema_binding,
        is_streaming=schema_binding is None and is_streaming_endpoint(func),
    )
