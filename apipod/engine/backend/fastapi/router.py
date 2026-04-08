import functools
import inspect
import threading
import logging
from contextlib import asynccontextmanager
from typing import Union, Callable, get_type_hints, Generator, AsyncGenerator, Iterator, AsyncIterator
from fastapi import APIRouter, FastAPI, Response
from fastapi.responses import JSONResponse

from apipod.common.settings import APIPOD_PORT, APIPOD_HOST, SERVER_DOMAIN
from apipod.common.constants import SERVER_HEALTH
from apipod.engine.jobs.job_result import JobResultFactory, JobResult
from apipod.engine.endpoint_config import FastApiEndpointConfigurator, EndpointExecutionPlan
from apipod.engine.base_backend import _BaseBackend
from apipod.engine.queue.queue_mixin import _QueueMixin
from apipod.engine.backend.fastapi.file_handling_mixin import _fast_api_file_handling_mixin
from apipod.engine.utils import normalize_name
from apipod.engine.backend.fastapi.exception_handling import _FastAPIExceptionHandler
from apipod.engine.backend.fastapi.llm_mixin import _FastApiLlmMixin


class SocaityFastAPIRouter(APIRouter, _BaseBackend, _QueueMixin, _fast_api_file_handling_mixin, _FastAPIExceptionHandler, _FastApiLlmMixin):
    """
    FastAPI router extension that adds support for task endpoints.

    Task endpoints run as jobs in the background and return job information
    that can be polled for status and results.
    """

    def __init__(
            self,
            title: str = "APIPod",
            summary: str = "Create web-APIs for long-running tasks",
            app: Union[FastAPI, None] = None,
            prefix: str = "",  # "/api",
            max_upload_file_size_mb: float = None,
            job_queue=None,
            lifespan=None,
            *args,
            **kwargs):
        """
        Initialize the SocaityFastAPIRouter.

        Args:
            title: The title of the app
            summary: The summary of the app
            app: Existing FastAPI app to use (optional)
            prefix: The API route prefix
            max_upload_file_size_mb: Maximum file size in MB for uploads
            job_queue: Optional custom JobQueue implementation
            lifespan: Optional async context manager for custom startup/shutdown logic
            args: Additional arguments
            kwargs: Additional keyword arguments
        """
        # Extract user-provided lifespan (explicit param or kwarg) before parent init
        user_lifespan = lifespan or kwargs.pop('lifespan', None)

        # Initialize parent classes
        api_router_params = inspect.signature(APIRouter.__init__).parameters
        api_router_kwargs = {k: kwargs.get(k) for k in api_router_params if k in kwargs}
        api_router_kwargs.pop('lifespan', None)  # handled via composed lifespan below

        APIRouter.__init__(self, **api_router_kwargs)
        _BaseBackend.__init__(self, title=title, summary=summary, *args, **kwargs)
        _QueueMixin.__init__(self, job_queue=job_queue, *args, **kwargs)
        _fast_api_file_handling_mixin.__init__(self, max_upload_file_size_mb=max_upload_file_size_mb, *args, **kwargs)
        _FastApiLlmMixin.__init__(self)

        self.status = SERVER_HEALTH.INITIALIZING

        # Registry for functions that workers can execute. Keys are function names.
        self._job_func_registry: dict = {}
        # Stop event and thread handle for in-process worker (dev mode)
        self._worker_stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._logger = logging.getLogger(__name__)

        # Build a composed lifespan that merges internal worker hooks with the user-provided lifespan
        combined_lifespan = self._build_lifespan(user_lifespan)

        # Create or use provided FastAPI app
        if app is None:
            app = FastAPI(
                title=self.title,
                summary=self.summary,
                contact={"name": "SocAIty", "url": "https://www.socaity.ai"},
                lifespan=combined_lifespan,
            )
        else:
            # Existing app: replace its lifespan with our composed version
            app.router.lifespan_context = combined_lifespan

        self.app: FastAPI = app
        self.prefix = prefix
        self.add_standard_routes()

        self._endpoint_configurator = FastApiEndpointConfigurator(self)

        # Exception handling
        _FastAPIExceptionHandler.__init__(self)
        if not getattr(self.app.state, "_socaity_exception_handler_added", False):
            self.app.add_exception_handler(Exception, self.global_exception_handler)
            self.app.state._socaity_exception_handler_added = True

        # Save original OpenAPI function and replace it
        self._orig_openapi_func = self.app.openapi
        self.app.openapi = self.custom_openapi

    # ------------------------------------------------------------------
    # Lifespan & worker lifecycle
    # ------------------------------------------------------------------

    def _build_lifespan(self, user_lifespan=None):
        """
        Build a composed lifespan context manager that runs:
        1. Internal worker startup
        2. User-provided lifespan (if any)
        3. Internal worker shutdown on exit
        """
        router_self = self  # capture for closure

        @asynccontextmanager
        async def _combined_lifespan(app):
            router_self._start_background_worker()
            try:
                if user_lifespan:
                    async with user_lifespan(app):
                        yield
                else:
                    yield
            finally:
                router_self._stop_background_worker()

        return _combined_lifespan

    def _start_background_worker(self):
        """Start the in-process job queue worker in a daemon thread (dev convenience)."""
        try:
            if self.job_queue and hasattr(self.job_queue, "start_worker"):
                def _run():
                    try:
                        self.job_queue.start_worker(
                            func_registry=self._job_func_registry,
                            worker_name="api-worker",
                            stop_event=self._worker_stop_event,
                        )
                    except Exception:
                        self._logger.exception("Worker thread exited with exception")

                thread = threading.Thread(target=_run, daemon=True)
                thread.start()
                self._worker_thread = thread
        except Exception:
            self._logger.exception("Failed to start in-process worker on startup")

    def _stop_background_worker(self):
        """Signal the background worker to stop and shut down the job queue."""
        try:
            self._worker_stop_event.set()
        except Exception:
            pass

        if self.job_queue and hasattr(self.job_queue, "shutdown"):
            try:
                self.job_queue.shutdown()
            except Exception:
                self._logger.exception("Error shutting down job queue")

    # ------------------------------------------------------------------
    # Standard routes
    # ------------------------------------------------------------------

    def add_standard_routes(self):
        """Add standard API routes for status and health checks."""
        if self.job_queue is not None:
            self.api_route(path="/status", methods=["POST"])(self.get_job)
        self.api_route(path="/health", methods=["GET"])(self.get_health)

    def get_health(self) -> Response:
        """
        Get server health status.

        Returns:
            HTTP response with health status
        """
        stat, message = self._health_check.get_health_response()
        return JSONResponse(status_code=stat, content=message)

    def custom_openapi(self):
        """
        Customize OpenAPI schema with APIPod version information.

        Returns:
            Modified OpenAPI schema
        """
        if not self.app.openapi_schema:
            self._orig_openapi_func()

        self.app.openapi_schema["info"]["apipod"] = self.version
        return self.app.openapi_schema

    def get_job(self, job_id: str, return_format: str = 'json') -> JobResult:
        """
        Get the status and result of a job.

        Args:
            job_id: The ID of the job
            return_format: Response format ('json' or 'gzipped')

        Returns:
            JobResult with status and results
        """
        # sometimes job-id is inserted with leading " or other unwanted symbols. Remove those.
        job_id = job_id.strip().strip("\"").strip("\'").strip('?').strip("#")

        if self.job_queue is None:
            return JobResultFactory.job_not_found(job_id)

        base_job = self.job_queue.get_job(job_id)
        if base_job is None:
            return JobResultFactory.job_not_found(job_id)

        ret_job = JobResultFactory.from_base_job(base_job)
        ret_job.refresh_job_url = f"{SERVER_DOMAIN}/status?job_id={ret_job.id}"
        ret_job.cancel_job_url = f"{SERVER_DOMAIN}/cancel?job_id={ret_job.id}"

        if return_format != 'json':
            ret_job = JobResultFactory.gzip_job_result(ret_job)

        return ret_job

    def endpoint(self, path: str, methods: list[str] | None = None, max_upload_file_size_mb: int = None, queue_size: int = 500, use_queue: bool = None, *args, **kwargs):
        """
        Unified endpoint decorator.

        If LLM configuration is detected on the function, it creates an LLM endpoint.

        If job_queue is configured (and use_queue is not False), it creates a task endpoint.
        Otherwise, it creates a standard FastAPI endpoint.

        Args:
            path: API path
            methods: List of HTTP methods (e.g. ["POST"])
            max_upload_file_size_mb: Max upload size for files
            queue_size: Max queue size (only if using queue)
            use_queue: Force enable/disable queue. If None, auto-detect based on job_queue presence.
        """
        normalized_path = self._normalize_endpoint_path(path)
        should_use_queue = self._determine_queue_usage(use_queue, normalized_path)

        def decorator(func: Callable) -> Callable:
            plan = self._create_endpoint_plan(
                func=func,
                normalized_path=normalized_path,
                methods=methods,
                max_upload_file_size_mb=max_upload_file_size_mb,
                queue_size=queue_size,
                should_use_queue=should_use_queue,
                route_args=args,
                route_kwargs=kwargs
            )

            if plan.is_llm:
                return self._register_llm_endpoint(func, plan)
            if plan.is_streaming:
                return self._create_streaming_endpoint_decorator(
                    path=plan.path,
                    methods=plan.methods,
                    max_upload_file_size_mb=plan.max_upload_file_size_mb,
                    args=plan.route_args,
                    kwargs=plan.route_kwargs
                )(func)
            if plan.should_use_queue:
                return self._create_task_endpoint_decorator(
                    path=plan.path,
                    methods=plan.methods,
                    max_upload_file_size_mb=plan.max_upload_file_size_mb,
                    queue_size=plan.queue_size,
                    args=plan.route_args,
                    kwargs=plan.route_kwargs
                )(func)

            return self._create_standard_endpoint_decorator(
                path=plan.path,
                methods=plan.methods,
                max_upload_file_size_mb=plan.max_upload_file_size_mb,
                args=plan.route_args,
                kwargs=plan.route_kwargs
            )(func)

        return decorator

    def _create_endpoint_plan(
        self,
        *,
        func: Callable,
        normalized_path: str,
        methods: list[str] | None,
        max_upload_file_size_mb: int | None,
        queue_size: int,
        should_use_queue: bool,
        route_args: tuple,
        route_kwargs: dict,
    ) -> EndpointExecutionPlan:
        """Build an orchestration plan for endpoint registration and execution."""
        return self._endpoint_configurator.build_plan(
            func=func,
            path=normalized_path,
            methods=methods,
            max_upload_file_size_mb=max_upload_file_size_mb,
            queue_size=queue_size,
            should_use_queue=should_use_queue,
            route_args=route_args,
            route_kwargs=route_kwargs,
        )

    def _register_llm_endpoint(self, func: Callable, plan: EndpointExecutionPlan) -> Callable:
        """Register a FastAPI LLM endpoint from an endpoint execution plan."""
        req_model = plan.request_model
        res_model = plan.response_model
        endpoint_type = plan.endpoint_type

        if req_model is None or res_model is None or endpoint_type is None:
            raise ValueError("Invalid LLM endpoint plan: missing request/response metadata")

        if plan.should_use_queue:
            try:
                self._job_func_registry[func.__name__] = func
            except Exception:
                pass

        @functools.wraps(func)
        async def _unified_worker(*w_args, **w_kwargs):
            payload = next(
                (v for v in w_kwargs.values() if isinstance(v, (dict, req_model))),
                next((a for a in w_args if isinstance(a, (dict, req_model))), None)
            )

            if payload is None:
                raise ValueError(f"Request does not match expected model {req_model}")

            clean_kwargs = {k: v for k, v in w_kwargs.items() if not isinstance(v, req_model)}
            openai_req = self._prepare_llm_payload(req_model=req_model, payload=payload)

            return await self.handle_llm_request(
                func=func,
                openai_req=openai_req,
                should_use_queue=plan.should_use_queue,
                res_model=res_model,
                endpoint_type=endpoint_type,
                **clean_kwargs
            )

        final_handler = self._prepare_func_for_media_file_upload_with_fastapi(
            _unified_worker,
            plan.max_upload_file_size_mb
        )

        self.api_route(
            path=plan.path,
            methods=plan.active_methods,
            response_model=JobResult if plan.should_use_queue else res_model,
            *plan.route_args,
            **plan.route_kwargs
        )(final_handler)

        return final_handler
    
    async def _execute_func(self, func, **kwargs):
        """
        Execute a function, handling both sync and async functions.
        
        Args:
            func: Function to execute
            **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
        """
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            # Sync function - run in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, functools.partial(func, **kwargs))

    async def _stream_generator(self, result):
        """
        Convert a sync or async generator into an async generator for StreamingResponse.
        
        Args:
            result: Generator (sync or async) that yields streaming chunks
            
        Yields:
            Streaming chunks as strings or bytes
        """
        import asyncio
        
        if inspect.isasyncgen(result):
            # Async generator - yield directly
            async for chunk in result:
                if isinstance(chunk, (str, bytes)):
                    yield chunk
                else:
                    yield str(chunk)
        elif inspect.isgenerator(result):
            # Sync generator - run in executor
            loop = asyncio.get_event_loop()
            try:
                while True:
                    chunk = await loop.run_in_executor(None, next, result, StopIteration)
                    if chunk is StopIteration:
                        break
                    if isinstance(chunk, (str, bytes)):
                        yield chunk
                    else:
                        yield str(chunk)
            except StopIteration:
                pass
        else:
            raise TypeError(f"Expected generator, got {type(result)}")

    def _normalize_endpoint_path(self, path: str) -> str:
        """Normalize the endpoint path to ensure it starts with '/'."""
        normalized = normalize_name(path, preserve_paths=True)
        return normalized if normalized.startswith("/") else f"/{normalized}"

    def _determine_queue_usage(self, use_queue: bool = None, path: str = None) -> bool:
        """Determine whether to use the job queue based on configuration and parameters."""
        if use_queue is not None:
            if use_queue and self.job_queue is None:
                print("Warning: Endpoint {path} requested use_queue=True but no job_queue is configured. We ignore it.")
                return False
            return use_queue

        return self.job_queue is not None
    
    def _determine_generator_fun(self, func: Callable) -> bool:
        """
        Determine whether the function is streaming data.
        Uses quick, robust checks for common streaming patterns.
        """
        # 1. Direct generator functions
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            return True

        # 2. Check return type hints
        try:
            hints = get_type_hints(func)
            return_type = hints.get('return')

            if return_type is not None:
                origin = getattr(return_type, '__origin__', return_type)
                streaming_types = (Generator, AsyncGenerator, Iterator, AsyncIterator)

                if origin in streaming_types or return_type in streaming_types:
                    return True
        except Exception:
            pass

        # 3. Check source code for yield keyword
        try:
            source = inspect.getsource(func)
            if 'yield' in source:
                return True
        except Exception:
            pass

        return False

    def _create_task_endpoint_decorator(self, path: str, methods: list[str] | None, max_upload_file_size_mb: int, queue_size: int, args, kwargs):
        """Create a decorator for task endpoints (background job execution)."""
        # FastAPI route decorator (returning JobResult)
        fastapi_route_decorator = self.api_route(
            path=path,
            methods=["POST"] if methods is None else methods,
            response_model=JobResult,
            *args,
            **kwargs
        )

        # Queue decorator
        queue_decorator = super().job_queue_func(
            path=path,
            queue_size=queue_size,
            *args,
            **kwargs
        )

        def decorator(func: Callable) -> Callable:
            # Add job queue functionality and prepare for FastAPI file handling
            queue_decorated = queue_decorator(func)
            # Register the function so workers can execute it (dev mode).
            try:
                self._job_func_registry[func.__name__] = func
            except Exception:
                pass
            upload_enabled = self._prepare_func_for_media_file_upload_with_fastapi(queue_decorated, max_upload_file_size_mb)
            return fastapi_route_decorator(upload_enabled)

        return decorator

    def _create_standard_endpoint_decorator(self, path: str, methods: list[str] | None, max_upload_file_size_mb: int, args, kwargs):
        """Create a decorator for standard endpoints (direct execution)."""
        # FastAPI route decorator
        fastapi_route_decorator = self.api_route(
            path=path,
            methods=["POST"] if methods is None else methods,
            *args,
            **kwargs
        )

        def file_result_modification_decorator(func: Callable) -> Callable:
            """Wrap endpoint result and serialize it."""
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return JobResultFactory._serialize_result(result)
            return sync_wrapper

        def decorator(func: Callable) -> Callable:
            result_modified = file_result_modification_decorator(func)
            with_file_upload_signature = self._prepare_func_for_media_file_upload_with_fastapi(result_modified, max_upload_file_size_mb)
            return fastapi_route_decorator(with_file_upload_signature)

        return decorator

    def _create_streaming_endpoint_decorator(self, path: str, methods: list[str] | None, max_upload_file_size_mb: int, args, kwargs):
        """Create a decorator for streaming endpoints."""
        from fastapi.responses import StreamingResponse

        # Extract custom headers if provided
        custom_headers = kwargs.pop('response_headers', None)

        fastapi_route_decorator = self.api_route(
            path=path,
            methods=["POST"] if methods is None else methods,
            *args,
            **kwargs
        )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def streaming_wrapper(*w_args, **w_kwargs):
                result = await self._execute_func(func, *w_args, **w_kwargs)
                generator = self._stream_generator(result)

                # Merge default and custom headers
                headers = {
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
                if custom_headers:
                    headers.update(custom_headers)

                return StreamingResponse(
                    generator,
                    media_type="text/event-stream",
                    headers=headers
                )

            with_file_upload_signature = self._prepare_func_for_media_file_upload_with_fastapi(streaming_wrapper, max_upload_file_size_mb)
            return fastapi_route_decorator(with_file_upload_signature)

        return decorator

    def get(self, path: str = None, queue_size: int = 100, *args, **kwargs):
        """
        Create a GET endpoint.
        """
        return self.endpoint(path=path, queue_size=queue_size, methods=["GET"], *args, **kwargs)

    def post(self, path: str = None, queue_size: int = 100, *args, **kwargs):
        """
        Create a POST endpoint.
        """
        return self.endpoint(path=path, queue_size=queue_size, methods=["POST"], *args, **kwargs)

    def start(self, port: int = APIPOD_PORT, host: str = APIPOD_HOST, *args, **kwargs):
        """
        Start the FastAPI server.

        Args:
            port: Server port
            host: Server host
            args: Additional arguments
            kwargs: Additional keyword arguments
        """
        # Create app if not provided
        if self.app is None:
            self.app = FastAPI()

        # Include this router in the app
        self.app.include_router(self)

        # Print help information
        print_host = "localhost" if host == "0.0.0.0" or host is None else host
        print(
            f"APIPod {self.app.title} started. Use http://{print_host}:{port}/docs to see the API documentation.")

        # Start server
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
