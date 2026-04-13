import asyncio
import functools
import inspect
import traceback
from datetime import datetime, timezone
from typing import Union, Callable

from apipod.common import constants
from apipod.engine.jobs.base_job import JOB_STATUS
from apipod.engine.jobs.job_progress import JobProgressRunpod, JobProgress
from apipod.engine.jobs.job_result import JobResultFactory, JobResult
from apipod.engine.base_backend import _BaseBackend
from apipod.engine.files.base_file_mixin import _BaseFileHandlingMixin
from apipod.engine.backend.runpod.llm_mixin import _RunPodLLMMixin

from apipod.engine.utils import normalize_name
from apipod.common.settings import APIPOD_PROVIDER, APIPOD_PORT, DEFAULT_DATE_TIME_FORMAT


class SocaityRunpodRouter(_BaseBackend, _BaseFileHandlingMixin, _RunPodLLMMixin):
    """
    Adds routing functionality for the runpod serverless framework.
    Provides enhanced file handling and conversion capabilities.
    """
    def __init__(self, title: str = "APIPod for ", summary: str = None, *args, **kwargs):
        super().__init__(title=title, summary=summary, *args, **kwargs)
        _RunPodLLMMixin.__init__(self)

        self.routes = {}  # routes are organized like {"ROUTE_NAME": "ROUTE_FUNCTION"}

        self.add_standard_routes()

    def add_standard_routes(self):
        self.endpoint(path="openapi.json")(self.get_openapi_schema)

    def endpoint(self, path: str = None, use_queue: bool = None, *args, **kwargs):
        path = normalize_name(path, preserve_paths=True).strip("/")

        def decorator(func: Callable) -> Callable:
            # 1. Auto-Detection
            req_model, res_model, endpoint_type = self._get_llm_config(func)

            @functools.wraps(func)
            def wrapper(*w_args, **w_kwargs):
                self.status = constants.SERVER_HEALTH.BUSY
                
                try:
                    if req_model:
                        payload = w_kwargs.get("payload", None)

                        openai_req = self._prepare_llm_payload(
                            req_model=req_model,
                            payload=payload
                        )

                        w_kwargs["payload"] = openai_req

                        return self.handle_llm_request(
                            func=func,
                            openai_req=openai_req,
                            req_model=req_model,
                            res_model=res_model,
                            endpoint_type=endpoint_type,
                            w_args=w_args,
                            w_kwargs=w_kwargs
                        )

                    # Default execution for standard endpoints
                    return self._execute_sync_or_async(func, w_args, w_kwargs)
                finally:
                    self.status = constants.SERVER_HEALTH.RUNNING

            self.routes[path] = wrapper
            return wrapper
        return decorator

    def _yield_native_stream(self, func, args, kwargs):
        """Bridge for RunPod native generator streaming."""
        from starlette.responses import StreamingResponse
        
        # Execute the function
        result = self._execute_sync_or_async(func, args, kwargs)
        
        # If it's a StreamingResponse (shouldn't happen in RunPod, but handle it)
        if isinstance(result, StreamingResponse):
            body_iterator = result.body_iterator
            
            if inspect.isasyncgen(body_iterator):
                while True:
                    try:
                        chunk = self._run_in_loop(body_iterator.__anext__())
                        yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
                    except StopAsyncIteration:
                        break
            else:
                for chunk in body_iterator:
                    yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
            return
        
        # Handle async generators
        if inspect.isasyncgen(result):
            while True:
                try:
                    chunk = self._run_in_loop(result.__anext__())
                    yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
                except StopAsyncIteration:
                    break
            return
        
        # Handle sync generators
        if inspect.isgenerator(result):
            for chunk in result:
                yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
            return
        
        # Not a generator - shouldn't happen for streaming
        raise TypeError(f"Expected generator for streaming, got {type(result)}")

    def _execute_sync_or_async(self, func, args, kwargs):
        if inspect.iscoroutinefunction(func):
            return self._run_in_loop(func(*args, **kwargs))
        return func(*args, **kwargs)

    def _run_in_loop(self, coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def get(self, path: str = None, *args, **kwargs):
        return self.endpoint(path=path, *args, **kwargs)

    def post(self, path: str = None, *args, **kwargs):
        return self.endpoint(path=path, *args, **kwargs)

    def _add_job_progress_to_kwargs(self, func, job, kwargs):
        """
        Add job_progress parameter to function arguments if necessary.

        Args:
            func: Original function
            job: Runpod job
            kwargs: Current function arguments

        Returns:
            Updated kwargs with job_progress added
        """
        job_progress_params = []
        for param in inspect.signature(func).parameters.values():
            if param.annotation in (JobProgress, JobProgressRunpod) or param.name == "job_progress":
                job_progress_params.append(param.name)

        if job_progress_params:
            jp = JobProgressRunpod(job)
            for job_progress_param in job_progress_params:
                kwargs[job_progress_param] = jp

        return kwargs

    def _router(self, path, job, **kwargs):
        """
        Internal app function that routes the path to the correct function.

        Args:
            path: Route path
            job: Runpod job
            kwargs: Function arguments

        Returns:
            JSON-encoded job result or generator for streaming
        """
        if not isinstance(path, str):
            raise Exception("Path must be a string")

        path = normalize_name(path, preserve_paths=True)
        path = path.strip("/")

        route_function = self.routes.get(path, None)
        if route_function is None:
            raise Exception(f"Route {path} not found")

        # Add job progress to kwargs if necessary
        kwargs = self._add_job_progress_to_kwargs(route_function, job, kwargs)

        # Check for missing arguments
        sig = inspect.signature(route_function)
        missing_args = [arg for arg in sig.parameters if arg not in kwargs]
        if missing_args:
            raise Exception(f"Arguments {missing_args} are missing")

        # Handle file uploads and conversions
        route_function = self._handle_file_uploads(route_function)

        # Prepare result tracking
        start_time = datetime.now(timezone.utc)
        result = JobResult(
            job_id=job["id"],
            created_at=start_time.strftime(DEFAULT_DATE_TIME_FORMAT),
        )

        try:
            # Execute the function (Sync or Async Handling)
            res = self._execute_route_function(route_function, kwargs)

            # Check if result is a generator (streaming response)
            if inspect.isgenerator(res) or inspect.isasyncgen(res):
                # For streaming, return the generator directly
                # RunPod will handle the streaming
                return res

            # Convert result to JSON if it's a MediaFile / MediaList / Pydantic Model
            res = JobResultFactory._serialize_result(res)

            result.result = res
            result.status = JOB_STATUS.FINISHED.value
        except Exception as e:
            result.error = str(e)
            result.status = JOB_STATUS.FAILED.value
            print(f"Job {job['id']} failed: {str(e)}")
            traceback.print_exc()
        finally:
            result.updated_at = datetime.now(timezone.utc).strftime(DEFAULT_DATE_TIME_FORMAT)

        result = result.model_dump_json()
        return result

    def _execute_route_function(self, route_function, kwargs):
        """
        Execute a route function, handling both sync and async functions.
        
        Args:
            route_function: The function to execute
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function execution
        """
        if not inspect.iscoroutinefunction(route_function):
            # Synchronous function - simple execution
            return route_function(**kwargs)
        
        # Async function - need event loop handling
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - shouldn't happen in RunPod handler
                # But if it does, we need to handle it differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        route_function(**kwargs)
                    )
                    return future.result()
            else:
                # Loop exists but not running - use it
                return loop.run_until_complete(route_function(**kwargs))
        except RuntimeError:
            # No event loop exists - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(route_function(**kwargs))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

    def handler(self, job):
        """
        The handler function that is called by the runpod serverless framework.
        """
        inputs = job["input"]
        if "path" not in inputs:
            raise Exception("No path provided")

        route = inputs["path"]
        del inputs["path"]

        result = self._router(route, job, **inputs)
        
        # If it's a generator, return it directly for RunPod to stream
        if inspect.isgenerator(result):
            return result
        
        # Otherwise return the JSON result
        return result

    def start_runpod_serverless_localhost(self, port):
        # add the -rp_serve_api to the command line arguments to allow debugging
        import sys
        sys.argv.append("--rp_serve_api")
        sys.argv.extend(["--rp_api_port", str(port)])

        # overwrite runpod variables. Little hacky but runpod does not expose the variables in a nice way.
        import runpod.serverless
        from runpod.serverless.modules import rp_fastapi
        rp_fastapi.TITLE = self.title + " " + rp_fastapi.TITLE
        rp_fastapi.DESCRIPTION = self.summary + " " + rp_fastapi.DESCRIPTION
        desc = '''\
                        In input declare your path as route for the function. Other parameters follow in the input as usual.
                        The APIPod app will use the path argument to route to the correct function declared with
                        @endpoint(path="your_path").
                        { "input": { "path": "your_path", "your_other_args": "your_other_args" } }
                    '''
        rp_fastapi.RUN_DESCRIPTION = desc + "\n" + rp_fastapi.RUN_DESCRIPTION

        # hack to print version also in runpod
        version = self.version

        class WorkerAPIWithModifiedInfo(rp_fastapi.WorkerAPI):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._orig_openapi_func = self.rp_app.openapi
                self.rp_app.openapi = self.custom_openapi

            def custom_openapi(self):
                if not self.rp_app.openapi_schema:
                    self._orig_openapi_func()
                self.rp_app.openapi_schema["info"]["apipod"] = version
                self.rp_app.openapi_schema["info"]["runpod"] = rp_fastapi.runpod_version
                return self.rp_app.openapi_schema

        rp_fastapi.WorkerAPI = WorkerAPIWithModifiedInfo

        runpod.serverless.start({"handler": self.handler, "return_aggregate_stream": True})

    def _create_openapi_compatible_function(self, func: Callable) -> Callable:
        """
        Create a function compatible with FastAPI OpenAPI generation by applying 
        the same conversion logic as the FastAPI mixin, but without runtime dependencies.

        This generates the rich schema with proper file upload handling.

        Args:
            func: Original function to convert
            max_upload_file_size_mb: Maximum file size in MB

        Returns:
            Function with FastAPI-compatible signature for OpenAPI generation
        """
        # Import FastAPI-specific conversion logic
        from apipod.engine.backend.fastapi.file_handling_mixin import _fast_api_file_handling_mixin
        from apipod.engine.jobs.job_result import JobResult
        import inspect
        from apipod.engine.utils import replace_func_signature
        # Create a temporary instance of the FastAPI mixin to use its conversion methods
        temp_mixin = _fast_api_file_handling_mixin(max_upload_file_size_mb=5)
        # Apply the same preparation logic as FastAPI router
        with_file_upload_signature = temp_mixin._prepare_func_for_media_file_upload_with_fastapi(func, 5)
        # 4. Set proper return type for job-based endpoints

        sig = inspect.signature(with_file_upload_signature)
        job_result_sig = sig.replace(return_annotation=JobResult)
        # Update the signature

        final_func = replace_func_signature(with_file_upload_signature, job_result_sig)
        return final_func
    
    def _create_openapi_safe_function(self, func: Callable) -> Callable:
        """
        Create a minimal function signature for OpenAPI when full conversion fails.
        
        Args:
            func: Original function
            
        Returns:
            A simple function with basic signature for OpenAPI
        """
        import inspect
        from typing import Any, Dict
        
        # Extract basic parameter information
        sig = inspect.signature(func)
        params = []
        
        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name in ('job_progress', 'self', 'cls'):
                continue
                
            # Create a simple parameter with Any type if annotation is complex
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Simplify complex types to Dict or Any
            if hasattr(annotation, '__origin__'):  # Generic types
                annotation = Dict if 'dict' in str(annotation).lower() else Any
                
            params.append(
                inspect.Parameter(
                    param_name,
                    kind=param.kind,
                    default=param.default,
                    annotation=annotation
                )
            )
        
        # Create new signature
        new_sig = inspect.Signature(
            parameters=params,
            return_annotation=Dict[str, Any]
        )
        
        # Create wrapper function with new signature
        def safe_wrapper(**kwargs) -> Dict[str, Any]:
            """Auto-generated safe wrapper for OpenAPI documentation."""
            return {"message": "Execute via RunPod handler"}
        
        # Apply signature
        safe_wrapper.__signature__ = new_sig
        safe_wrapper.__name__ = func.__name__
        safe_wrapper.__doc__ = func.__doc__ or "API endpoint"
        
        return safe_wrapper

    def get_openapi_schema(self):
        from fastapi.openapi.utils import get_openapi
        from fastapi.routing import APIRoute

        fastapi_routes = []
        for path, func in self.routes.items():
            # Create FastAPI-compatible function for rich OpenAPI generation
            try:
                compatible_func = self._create_openapi_compatible_function(func)
                fastapi_routes.append(APIRoute(
                    path=f"/{path.strip('/')}", 
                    endpoint=compatible_func, 
                    methods=["POST"]
                ))
            except Exception as e:
                print(f"Error creating OpenAPI compatible function for {path}: {e}")
                # Fallback to safe function approach
                try:
                    safe_func = self._create_openapi_safe_function(func)
                    fastapi_routes.append(APIRoute(
                        path=f"/{path.strip('/')}",
                        endpoint=safe_func,
                        methods=["POST"],
                        response_model=None
                    ))
                except Exception as e2:
                    print(f"Error creating safe function for {path}: {e2}")
                    # Ultimate fallback - create minimal route

                    def minimal_func():
                        return {"message": "Documentation not available"}

                    fastapi_routes.append(APIRoute(
                        path=f"/{path.strip('/')}",
                        endpoint=minimal_func,
                        methods=["POST"],
                        response_model=None
                    ))

        # Generate the OpenAPI schema dict (similar to FastAPI openapi())
        schema = get_openapi(
            title=self.title,
            version="1.0.0",
            routes=fastapi_routes,
            summary=self.summary,
            description=self.summary,
        )

        # Add APIPod version information like the FastAPI router
        schema["info"]["apipod"] = self.version

        return schema

    def start(self, port: int = APIPOD_PORT, provider: Union[constants.PROVIDER, str, None] = None, *args, **kwargs):
        if provider is None:
            provider = APIPOD_PROVIDER
        if isinstance(provider, str):
            provider = constants.PROVIDER(provider)

        if provider == constants.PROVIDER.LOCALHOST:
            self.start_runpod_serverless_localhost(port=port)
        else:
            import runpod.serverless
            runpod.serverless.start({"handler": self.handler, "return_aggregate_stream": True})
