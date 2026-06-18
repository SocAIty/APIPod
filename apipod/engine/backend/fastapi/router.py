import functools
import inspect
import threading
import logging
from contextlib import asynccontextmanager
from typing import Union, Callable
from fastapi import APIRouter, FastAPI, Response, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from apipod.common.settings import APIPOD_PORT, APIPOD_HOST
from apipod.common.constants import SERVER_HEALTH
from apipod.engine.jobs.job_result import JobResultFactory, JobResult
from apipod.engine.endpoint_config import build_plan, EndpointExecutionPlan
from apipod.engine.streaming.stream_serializer import build_stream_producer
from apipod.engine.base_backend import _BaseBackend
from apipod.engine.queue.queue_mixin import _QueueMixin
from apipod.engine.backend.fastapi.file_handling_mixin import _fast_api_file_handling_mixin
from apipod.engine.backend.fastapi.streaming_mixin import _FastAPIStreamingMixin
from apipod.engine.utils import normalize_name
from apipod.engine.backend.fastapi.exception_handling import _FastAPIExceptionHandler
from apipod.engine.backend.schema_resolve import wrap_schema_response


class SocaityFastAPIRouter(APIRouter, _BaseBackend, _QueueMixin, _fast_api_file_handling_mixin, _FastAPIStreamingMixin, _FastAPIExceptionHandler):
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
            kwargs: May include ``stream_store`` (SSE backend for GET /stream/{job_id}),
                plus additional keyword arguments for parent classes.
        """
        # Extract user-provided lifespan (explicit param or kwarg) before parent init
        user_lifespan = lifespan or kwargs.pop('lifespan', None)
        stream_store = kwargs.pop("stream_store", None)

        # Initialize parent classes
        api_router_params = inspect.signature(APIRouter.__init__).parameters
        api_router_kwargs = {k: kwargs.get(k) for k in api_router_params if k in kwargs}
        api_router_kwargs.pop('lifespan', None)  # handled via composed lifespan below

        APIRouter.__init__(self, **api_router_kwargs)
        _BaseBackend.__init__(self, title=title, summary=summary, *args, **kwargs)
        _QueueMixin.__init__(self, job_queue=job_queue, *args, **kwargs)
        _fast_api_file_handling_mixin.__init__(self, max_upload_file_size_mb=max_upload_file_size_mb, *args, **kwargs)

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
        self.stream_store = stream_store

        # Let the queue produce streaming job output into the stream store so a
        # client can consume it via GET /stream/{job_id} (serverless emulation).
        if self.job_queue is not None and self.stream_store is not None:
            set_store = getattr(self.job_queue, "set_stream_store", None)
            if callable(set_store):
                set_store(self.stream_store)

        self.add_standard_routes()

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
            self.api_route(path="/status/{job_id}", methods=["GET"], response_model_exclude_none=True)(self.get_job)
            self.api_route(path="/status", methods=["POST"], response_model_exclude_none=True)(self.get_job)
            self.api_route(path="/cancel/{job_id}", methods=["POST"])(self.post_cancel_job)
            if self.stream_store is not None:
                self.api_route(path="/stream/{job_id}", methods=["GET"])(self.stream_job_sse)
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

        ret_job = self.job_queue.get_job_result(job_id)
        if ret_job is None:
            return JobResultFactory.job_not_found(job_id)

        if return_format != 'json':
            ret_job = JobResultFactory.gzip_job_result(ret_job)

        return ret_job

    def post_cancel_job(self, job_id: str) -> dict:
        """Cancel a background job (gateway / orchestrator integration)."""
        job_id = job_id.strip().strip('"').strip("'").strip("?").strip("#")
        if self.job_queue is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Job queue not configured.")

        cancel_fn = getattr(self.job_queue, "cancel_gateway_job", None)
        if callable(cancel_fn):
            return cancel_fn(job_id)

        try:
            self.job_queue.cancel_job(job_id)
        except NotImplementedError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Cancellation is not supported for this job queue.",
            ) from None
        return {"id": job_id, "status": "cancelled", "message": "Job cancelled."}

    def endpoint(self, path: str, methods: list[str] | None = None, max_upload_file_size_mb: int = None, queue_size: int = 500, use_queue: bool = None, *args, **kwargs):
        """
        Unified endpoint decorator.

        If a parameter is annotated with a standardized request schema, it creates a schema endpoint.

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
            plan = build_plan(
                func,
                path=normalized_path,
                methods=methods,
                max_upload_file_size_mb=max_upload_file_size_mb,
                queue_size=queue_size,
                should_use_queue=should_use_queue,
                route_args=args,
                route_kwargs=kwargs,
            )

            # Streaming with a queue + stream store mimics a real deployment:
            # the job is queued, the worker produces chunks into the stream store
            # and the client consumes them from GET /stream/{job_id}. Without a
            # queue (plain FastAPI) streaming goes straight to the client.
            if plan.is_streaming and not plan.should_use_queue:
                return self._create_streaming_endpoint_decorator(plan)(func)
            if plan.should_use_queue:
                return self._create_task_endpoint_decorator(plan)(func)

            return self._create_standard_endpoint_decorator(plan)(func)

        return decorator

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

    # ------------------------------------------------------------------
    # Endpoint decorators (task / standard / streaming)
    # ------------------------------------------------------------------
    def _modify_result_decorator(self, func: Callable, plan: EndpointExecutionPlan, *, queued: bool) -> Callable:
        """
        Wraps endpoint responses into their final transport form.

        Streamable results (a token/chunk generator) become a stream:
        - ``queued`` (job queue): a :class:`StreamProducer` is returned so the
          worker relays chunks into the stream store (consumed via /stream) while
          aggregating the full result for /status;
        - direct (no queue): a :class:`StreamingResponse` is returned to the client.

        Non-streaming schema results are wrapped into the response model; all
        results are then serialized (media files -> JSON).
        """
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = self.run_callable(func, *args, **kwargs)

            producer = build_stream_producer(result, plan.schema_binding)
            if producer is not None:
                return producer if queued else self._streaming_response_from_producer(producer)

            binding = plan.schema_binding
            if binding is not None:
                result = wrap_schema_response(result, binding)
            return JobResultFactory._serialize_result(result)

        # Python 3.12+ follows __wrapped__ when evaluating inspect.isgeneratorfunction().
        # sync_wrapper is never a generator (it returns a JobResult or StreamProducer).
        # Without this, FastAPI would mistake a task endpoint wrapping a generator for a
        # streaming route and serve the JobResult as iterated key-value NDJSON chunks.
        if hasattr(sync_wrapper, "__wrapped__"):
            # Pin the signature explicitly so FastAPI can parse parameters
            sync_wrapper.__signature__ = inspect.signature(func)
            del sync_wrapper.__wrapped__

        return sync_wrapper

    def _create_task_endpoint_decorator(self, plan: EndpointExecutionPlan):
        """Create a decorator for task endpoints (background job execution)."""
        # FastAPI route decorator (returning JobResult)
        task_kwargs = dict(plan.route_kwargs)
        task_kwargs["response_model_exclude_none"] = True
        fastapi_route_decorator = self.api_route(
            path=plan.path,
            methods=plan.active_methods,
            response_model=JobResult,
            *plan.route_args,
            **task_kwargs
        )

        # Queue decorator
        queue_decorator = super().job_queue_func(
            path=plan.path,
            queue_size=plan.queue_size,
            *plan.route_args,
            **plan.route_kwargs
        )

        def decorator(func: Callable) -> Callable:
            func = self._modify_result_decorator(func, plan, queued=True)
            # Add job queue functionality and prepare for FastAPI file handling
            queue_decorated = queue_decorator(func)
            # Register the function so workers can execute it (dev mode).
            try:
                self._job_func_registry[func.__name__] = func
            except Exception:
                pass

            upload_enabled = self._prepare_func_for_media_file_upload_with_fastapi(queue_decorated, plan.max_upload_file_size_mb)
            return fastapi_route_decorator(upload_enabled)

        return decorator

    def _create_standard_endpoint_decorator(self, plan: EndpointExecutionPlan):
        """Create a decorator for standard and schema endpoints (direct execution)."""
        fastapi_route_decorator = self.api_route(
            path=plan.path,
            methods=plan.active_methods,
            response_model=plan.schema_binding.response_model if plan.is_schema_endpoint else None,
            *plan.route_args,
            **plan.route_kwargs
        )

        def decorator(func: Callable) -> Callable:
            result_modified = self._modify_result_decorator(func, plan, queued=False)
            with_file_upload_signature = self._prepare_func_for_media_file_upload_with_fastapi(result_modified, plan.max_upload_file_size_mb)
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
