import functools
import inspect
from typing import Union, Callable
from fastapi import APIRouter, FastAPI, Response, Request
from pydantic import BaseModel

from apipod.core.routers.schemas import ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse, EmbeddingRequest, EmbeddingResponse
from apipod.settings import APIPOD_PORT, APIPOD_HOST, SERVER_DOMAIN
from apipod.CONSTS import SERVER_HEALTH
from apipod.core.job.job_result import JobResultFactory, JobResult
from apipod.core.routers._socaity_router import _SocaityRouter
from apipod.core.routers.router_mixins._queue_mixin import _QueueMixin
from apipod.core.routers.router_mixins._fast_api_file_handling_mixin import _fast_api_file_handling_mixin
from apipod.core.utils import normalize_name
from apipod.core.routers.router_mixins._fast_api_exception_handling import _FastAPIExceptionHandler


class SocaityFastAPIRouter(APIRouter, _SocaityRouter, _QueueMixin, _fast_api_file_handling_mixin, _FastAPIExceptionHandler):
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
            args: Additional arguments
            kwargs: Additional keyword arguments
        """
        # Initialize parent classes
        api_router_params = inspect.signature(APIRouter.__init__).parameters
        api_router_kwargs = {k: kwargs.get(k) for k in api_router_params if k in kwargs}

        APIRouter.__init__(self, **api_router_kwargs)
        _SocaityRouter.__init__(self, title=title, summary=summary, *args, **kwargs)
        _QueueMixin.__init__(self, job_queue=job_queue, *args, **kwargs)
        _fast_api_file_handling_mixin.__init__(self, max_upload_file_size_mb=max_upload_file_size_mb, *args, **kwargs)

        self.status = SERVER_HEALTH.INITIALIZING

        # Create or use provided FastAPI app
        if app is None:
            app = FastAPI(
                title=self.title,
                summary=self.summary,
                contact={"name": "SocAIty", "url": "https://www.socaity.ai"}
            )

        self.app: FastAPI = app
        self.prefix = prefix
        self.add_standard_routes()

        # excpetion handling
        _FastAPIExceptionHandler.__init__(self)
        if not getattr(self.app.state, "_socaity_exception_handler_added", False):
            self.app.add_exception_handler(Exception, self.global_exception_handler)
            self.app.state._socaity_exception_handler_added = True

        # Save original OpenAPI function and replace it
        self._orig_openapi_func = self.app.openapi
        self.app.openapi = self.custom_openapi

    def add_standard_routes(self):
        """Add standard API routes for status and health checks."""
        self.api_route(path="/status", methods=["POST"])(self.get_job)
        self.api_route(path="/health", methods=["GET"])(self.get_health)

    def get_health(self) -> Response:
        """
        Get server health status.

        Returns:
            HTTP response with health status
        """
        stat, message = self._health_check.get_health_response()
        return Response(status_code=stat, content=message)

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
        import time
        import uuid
        from fastapi.concurrency import run_in_threadpool
        
        normalized_path = self._normalize_endpoint_path(path)
        should_use_queue = self._determine_queue_usage(use_queue, normalized_path)

        model_map = {
            ChatCompletionRequest: (ChatCompletionResponse, "chat"),
            CompletionRequest: (CompletionResponse, "completion"),
            EmbeddingRequest: (EmbeddingResponse, "embedding"),
        }

        def decorator(func: Callable) -> Callable:
            # 1. Auto-Detection
            sig = inspect.signature(func)
            request_model = None
            response_model = None
            endpoint_type_str = None
            
            for param in sig.parameters.values():
                ann = param.annotation
                if inspect.isclass(ann) and ann in model_map:
                    request_model = ann
                    response_model, endpoint_type_str = model_map[ann]
                    break
            
            # 2. Fallback: Standard Endpoint
            if request_model is None:
                if should_use_queue:
                    return self._create_task_endpoint_decorator(
                        path=normalized_path, methods=methods, max_upload_file_size_mb=max_upload_file_size_mb, 
                        queue_size=queue_size, args=args, kwargs=kwargs
                    )(func)
                else:
                    return self._create_standard_endpoint_decorator(
                        path=normalized_path, methods=methods, max_upload_file_size_mb=max_upload_file_size_mb, 
                        args=args, kwargs=kwargs
                    )(func)
            
            assert response_model is not None
            assert endpoint_type_str is not None

            # 3. LLM Endpoint Logic
            @functools.wraps(func)
            async def _unified_worker(*w_args, **w_kwargs):
                # A. Extract Request Model
                openai_req = None
                for arg in w_args:
                    if isinstance(arg, request_model): openai_req = arg; break
                if not openai_req:
                    for val in w_kwargs.values():
                        if isinstance(val, request_model): openai_req = val; break
                
                request_obj = next((arg for arg in w_args if isinstance(arg, Request)), 
                                 next((val for val in w_kwargs.values() if isinstance(val, Request)), None))
                
                if openai_req is None and request_obj:
                    try:
                        # Pydantic usage: parse raw JSON body
                        body = await request_obj.json()
                        openai_req = request_model.model_validate(body)
                    except Exception: pass
                
                if request_obj and openai_req:
                    request_obj.state.openai_request = openai_req

                # B. Execute User Function
                if inspect.iscoroutinefunction(func):
                    result = await func(*w_args, **w_kwargs)
                else:
                    result = await run_in_threadpool(func, *w_args, **w_kwargs)
                
                # C. Pydantic Response Construction
                model_name = getattr(openai_req, "model", "unknown") if openai_req else "unknown"
                timestamp = int(time.time())

                # 1. Pass through if it is already the correct Pydantic Object
                if isinstance(result, response_model):
                    return result

                # 2. Handle Dictionary Responses (Partial or Full)
                if isinstance(result, dict):
                    # If it's a Chat/Completion dict with "choices"
                    if "choices" in result:
                        return response_model(
                            id=result.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                            object=result.get("object", "chat.completion"),
                            created=result.get("created", timestamp),
                            model=result.get("model", model_name),
                            choices=result["choices"],
                            usage=result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                        )
                    # If it's an Embedding dict with "data"
                    elif "data" in result and endpoint_type_str == "embedding":
                         return response_model(
                            object=result.get("object", "list"),
                            data=result["data"],
                            model=result.get("model", model_name),
                            usage=result.get("usage", {"prompt_tokens": 0, "total_tokens": 0})
                        )

                # 3. Fallback: Treat result as raw content string (Auto-Wrap)
                if endpoint_type_str == "chat":
                    # Handle diverse return types
                    content = result
                    if isinstance(result, dict):
                        content = result.get("content", result.get("message", str(result)))
                    
                    return response_model(
                        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        object="chat.completion",
                        created=timestamp,
                        model=model_name,
                        choices=[{
                            "index": 0,
                            "message": {"role": "assistant", "content": str(content)},
                            "finish_reason": "stop"
                        }],
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    )

                elif endpoint_type_str == "completion":
                    text = result
                    if isinstance(result, dict):
                        text = result.get("text", str(result))
                        
                    return response_model(
                        id=f"cmpl-{uuid.uuid4().hex[:8]}",
                        object="text_completion",
                        created=timestamp,
                        model=model_name,
                        choices=[{
                            "text": str(text),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }],
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    )

                elif endpoint_type_str == "embedding":
                    embedding = result
                    if isinstance(result, dict):
                        embedding = result.get("embedding")
                    
                    if not isinstance(embedding, list):
                         raise ValueError("Result must be a list or dict with 'embedding'")

                    return response_model(
                        object="list",
                        data=[{
                            "object": "embedding",
                            "embedding": embedding,
                            "index": 0
                        }],
                        model=model_name,
                        usage={"prompt_tokens": 0, "total_tokens": 0}
                    )
                
                return result


            # 4. Route Registration
            active_methods = ["POST"] if methods is None else methods

            if should_use_queue:
                queued_func = self.job_queue_func(
                    path=normalized_path, queue_size=queue_size, *args, **kwargs
                )(_unified_worker)
                
                final_handler = self._prepare_func_for_media_file_upload_with_fastapi(
                    queued_func, max_upload_file_size_mb
                )
                self.api_route(
                    path=normalized_path, methods=active_methods, response_model=JobResult, *args, **kwargs
                )(final_handler)
                return final_handler
            else:
                final_handler = self._prepare_func_for_media_file_upload_with_fastapi(
                    _unified_worker, max_upload_file_size_mb
                )
                self.api_route(
                    path=normalized_path, methods=active_methods, response_model=response_model, *args, **kwargs
                )(final_handler)
                return final_handler

        return decorator

    def _normalize_endpoint_path(self, path: str) -> str:
        """Normalize the endpoint path to ensure it starts with '/'."""
        normalized = normalize_name(path, preserve_paths=True)
        return normalized if normalized.startswith("/") else f"/{normalized}"

    def _determine_queue_usage(self, use_queue: bool = None, path: str = None) -> bool:
        """Determine whether to use the job queue based on configuration and parameters."""
        if use_queue is not None:
            if use_queue and self.job_queue is None:
                raise ValueError(f"Endpoint {path} requested use_queue=True but no job_queue is configured.")
            return use_queue

        return self.job_queue is not None

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
