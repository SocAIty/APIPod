from dataclasses import dataclass
from typing import Any, Callable


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
    request_model: type | None = None
    response_model: type | None = None
    endpoint_type: str | None = None
    is_streaming: bool = False

    @property
    def is_llm(self) -> bool:
        return self.request_model is not None

    @property
    def active_methods(self) -> list[str]:
        return ["POST"] if self.methods is None else self.methods


class FastApiEndpointConfigurator:
    """
    Builds endpoint execution plans for the FastAPI backend.

    This component configures endpoint behavior (LLM, streaming, queue)
    independent from provider mechanics (FastAPI routing).
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
        request_model, response_model, endpoint_type = self._router._get_llm_config(func)
        is_streaming = bool(request_model is None and self._router._determine_generator_fun(func))

        return EndpointExecutionPlan(
            path=path,
            methods=methods,
            should_use_queue=should_use_queue,
            max_upload_file_size_mb=max_upload_file_size_mb,
            queue_size=queue_size,
            route_args=route_args,
            route_kwargs=route_kwargs,
            request_model=request_model,
            response_model=response_model,
            endpoint_type=endpoint_type,
            is_streaming=is_streaming,
        )
