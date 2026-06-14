from dataclasses import dataclass
from typing import Any, Callable

from apipod.engine.backend.schema_resolve import SchemaBinding, get_schema_binding


@dataclass(frozen=True)
class EndpointExecutionPlan:
    """
    Immutable plan describing how an endpoint should be registered and executed.

    The plan separates endpoint configuration decisions from router registration
    mechanics. This keeps endpoint decorators compact and easier to reason about.
    """

    path: str
    methods: list[str] | None
    should_use_queue: bool
    max_upload_file_size_mb: int | None
    queue_size: int
    route_args: tuple[Any, ...]
    route_kwargs: dict[str, Any]
    schema_binding: SchemaBinding | None = None
    is_streaming: bool = False

    @property
    def is_schema_endpoint(self) -> bool:
        return self.schema_binding is not None

    @property
    def active_methods(self) -> list[str]:
        return ["POST"] if self.methods is None else self.methods


class FastApiEndpointConfigurator:
    """
    Builds endpoint execution plans for the FastAPI backend.

    This component configures endpoint behavior (schema dispatch, streaming,
    queue) independent from provider mechanics (FastAPI routing).
    """

    def __init__(self, router):
        self._router = router

    def build_plan(
        self,
        *,
        func: Callable,
        path: str,
        methods: list[str] | None,
        max_upload_file_size_mb: int | None,
        queue_size: int,
        should_use_queue: bool,
        route_args: tuple[Any, ...],
        route_kwargs: dict[str, Any],
    ) -> EndpointExecutionPlan:
        schema_binding = get_schema_binding(func)
        is_streaming = bool(schema_binding is None and self._router._is_streaming_endpoint(func))

        return EndpointExecutionPlan(
            path=path,
            methods=methods,
            should_use_queue=should_use_queue,
            max_upload_file_size_mb=max_upload_file_size_mb,
            queue_size=queue_size,
            route_args=route_args,
            route_kwargs=route_kwargs,
            schema_binding=schema_binding,
            is_streaming=is_streaming,
        )
