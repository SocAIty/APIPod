import functools
import inspect
import traceback
from datetime import datetime, timezone
from typing import Union, Callable, Iterator

from apipod.common import constants
from apipod.engine.jobs.base_job import BaseJob, JOB_STATUS
from apipod.engine.jobs.job_progress import JobProgressRunpod
from apipod.engine.jobs.job_result import JobResultFactory
from apipod.engine.base_backend import _BaseBackend
from apipod.engine.endpoint_config import build_plan, EndpointExecutionPlan
from apipod.models import load_declared_models
from apipod.engine.signatures.analysis import job_progress_param_names
from apipod.engine.files.base_file_mixin import _BaseFileHandlingMixin
from apipod.engine.backend.schema_resolve import (
    SchemaBinding,
    SchemaStreamSerializer,
    STREAM_CHUNK_SPECS,
    iter_media_chunks,
    prepare_schema_call,
    wrap_schema_response,
)
from apipod.engine.streaming.stream_serializer import as_sync_iter, encode_chunk, is_streaming_result

from apipod.engine.utils import normalize_name, normalize_mount_prefix
from apipod.common.settings import APIPOD_PORT
from media_toolkit import AudioFile, VideoFile


class SocaityRunpodRouter(_BaseBackend, _BaseFileHandlingMixin):
    """
    Adds routing functionality for the runpod serverless framework.
    Provides enhanced file handling and conversion capabilities.
    """
    def __init__(self, title: str = "APIPod for ", summary: str = None, simulate: bool = False, prefix: str = "", *args, **kwargs):
        super().__init__(title=title, summary=summary, *args, **kwargs)

        # When True, start() runs RunPod's local API emulator instead of the real
        # serverless worker (set by APIPod(simulate="serverless-runpod", direct=True)).
        self.simulate = simulate
        self.prefix = normalize_mount_prefix(prefix)
        self.routes = {}  # routes are organized like {"ROUTE_NAME": "ROUTE_FUNCTION"}
        self._endpoint_plans: dict[str, EndpointExecutionPlan] = {}
        self._endpoint_source_funcs: dict[str, Callable] = {}

        self.add_standard_routes()

    def _apply_mount_prefix(self, mount_prefix: str) -> None:
        mount = normalize_mount_prefix(mount_prefix)
        if mount:
            self.prefix = mount

    def include_router(
        self,
        router: "SocaityRunpodRouter",
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Mount another APIPod RunPod router under *prefix* (path prefix only)."""
        del kwargs
        if not isinstance(router, SocaityRunpodRouter):
            raise TypeError(
                f"APIPod include_router expects a SocaityRunpodRouter instance, got {type(router)!r}"
            )
        mount = normalize_mount_prefix(prefix)
        router._apply_mount_prefix(mount)
        head = mount.strip("/")
        for path, route in router.routes.items():
            if path == "openapi.json":
                continue
            prefixed = f"{head}/{path.strip('/')}" if head else path.strip("/")
            self.routes[prefixed] = route
            if path in router._endpoint_plans:
                self._endpoint_plans[prefixed] = router._endpoint_plans[path]
            if path in router._endpoint_source_funcs:
                self._endpoint_source_funcs[prefixed] = router._endpoint_source_funcs[path]

    def add_standard_routes(self):
        self.endpoint(path="openapi.json")(self.get_openapi_schema)

    def endpoint(self, path: str = None, use_queue: bool = None, *args, **kwargs):
        path = normalize_name(path, preserve_paths=True).strip("/")

        def decorator(func: Callable) -> Callable:
            plan = build_plan(func, path=path)
            self._endpoint_plans[path] = plan
            self._endpoint_source_funcs[path] = func
            route = self._build_route(func, plan)
            self.routes[path] = route
            return route
        return decorator

    def _build_route(self, func: Callable, plan: EndpointExecutionPlan) -> Callable:
        """
        Compose the callable registered for a route from its execution plan.

        Schema endpoints validate + media-parse their request themselves; plain
        endpoints get the generic media file-upload conversion wrapped in.
        """
        @functools.wraps(func)
        def wrapper(*w_args, **w_kwargs):
            self.status = constants.SERVER_HEALTH.BUSY
            try:
                if plan.is_schema_endpoint:
                    return self._handle_schema_request(func, plan.schema_binding, w_args, w_kwargs)
                result = self.run_callable(func, *w_args, **w_kwargs)
                # Plain generator endpoints stream too: turn the generator into a
                # RunPod-native stream of JSON-safe chunks (RunPod aggregates it).
                native_stream = self._as_native_stream(result)
                return native_stream if native_stream is not None else result
            finally:
                self.status = constants.SERVER_HEALTH.RUNNING

        return wrapper if plan.is_schema_endpoint else self._handle_file_uploads(wrapper)

    # ------------------------------------------------------------------
    # Standardized schema endpoints
    # ------------------------------------------------------------------
    def _handle_schema_request(self, func: Callable, binding: SchemaBinding, args: tuple, kwargs: dict):
        """
        Run a standardized schema endpoint: validate + media-parse the request
        (``prepare_schema_call``), then stream or wrap the response into the
        registered response model (``wrap_schema_response``).
        """
        request = prepare_schema_call(binding, kwargs)
        result = self.run_callable(func, *args, **kwargs)

        if getattr(request, "stream", False):
            native_stream = self._as_native_stream(result, binding)
            if native_stream is not None:
                return native_stream

        return wrap_schema_response(result, binding)

    def _as_native_stream(self, result, binding: Union[SchemaBinding, None] = None) -> Union[Iterator, None]:
        """
        Wrap a streamable result into a RunPod-native generator of JSON-safe chunks.

        - ``AudioFile`` / ``VideoFile`` → base64-encoded byte chunks;
        - schema generator with a registered chunk model (e.g. chat) → standardized
          ``ChatCompletionChunk`` stream via :class:`SchemaStreamSerializer`;
        - any other generator → ``encode_chunk`` (base64 for binary, str for text).

        Returns ``None`` when the result is not streamable.
        """
        if isinstance(result, (AudioFile, VideoFile)):
            return (encode_chunk(chunk) for chunk in iter_media_chunks(result))

        if is_streaming_result(result):
            tokens = as_sync_iter(result)
            if binding is not None and binding.tag in STREAM_CHUNK_SPECS:
                return SchemaStreamSerializer(binding).stream(tokens)
            return (encode_chunk(chunk) for chunk in tokens)

        return None

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
        job_progress_params = job_progress_param_names(func)
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

        # Create a BaseJob record so the result can be assembled by the same
        # from_base_job path used by the local queue worker.
        job_record = BaseJob(id=job["id"])
        job_record.metrics.started_at = datetime.now(timezone.utc)

        try:
            res = self.run_callable(route_function, **kwargs)

            # Streaming response: hand the generator straight to RunPod, which
            # aggregates it (no JobResult envelope is produced for streams).
            if inspect.isgenerator(res) or inspect.isasyncgen(res):
                return res

            job_record.result = res
            job_record.status = JOB_STATUS.FINISHED
        except Exception as e:
            job_record.error = str(e)
            job_record.status = JOB_STATUS.FAILED
            print(f"Job {job['id']} failed: {str(e)}")
            traceback.print_exc()

        # Adopt the progress handle the function reported through, so its final
        # progress/message flow into the JobResult.
        for arg in kwargs.values():
            if isinstance(arg, JobProgressRunpod):
                job_record.job_progress = arg
                break

        job_record.metrics.finished_at = datetime.now(timezone.utc)

        return JobResultFactory.from_base_job(job_record).model_dump_json(exclude_none=True)

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
        # Add APIPod manifest like in the FastAPI router
        manifest = {
            "compute": "serverless",
            "version": self.version,
            "simulate": self.simulate,
        }
   
        class WorkerAPIWithModifiedInfo(rp_fastapi.WorkerAPI):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._orig_openapi_func = self.rp_app.openapi
                self.rp_app.openapi = self.custom_openapi

            def custom_openapi(self):
                if not self.rp_app.openapi_schema:
                    self._orig_openapi_func()
                self.rp_app.openapi_schema["info"]["apipod"] = manifest
                self.rp_app.openapi_schema["info"]["runpod"] = rp_fastapi.runpod_version
                return self.rp_app.openapi_schema

        rp_fastapi.WorkerAPI = WorkerAPIWithModifiedInfo

        runpod.serverless.start({"handler": self.handler, "return_aggregate_stream": True})

    def _create_openapi_compatible_function(
        self,
        func: Callable,
        plan: EndpointExecutionPlan | None = None,
    ) -> Callable:
        """
        Create a function compatible with FastAPI OpenAPI generation by applying
        the same conversion logic as the FastAPI mixin, but without runtime dependencies.

        This generates the rich schema with proper file upload handling.

        Args:
            func: Original function to convert
            plan: Endpoint execution plan (controls ``stream`` in request schema)

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
        with_file_upload_signature = temp_mixin._prepare_func_for_media_file_upload_with_fastapi(
            func, 5, plan=plan,
        )
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
            source_func = self._endpoint_source_funcs.get(path, func)
            plan = self._endpoint_plans.get(path)
            # Create FastAPI-compatible function for rich OpenAPI generation
            try:
                compatible_func = self._create_openapi_compatible_function(source_func, plan)
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

        # Add APIPod manifest like in the FastAPI router
        manifest = {
            "compute": "serverless",
            "version": self.version,
            "simulate": self.simulate,
        }
        schema["info"]["apipod"] = manifest

        return schema

    def start(self, port: int = APIPOD_PORT, **kwargs):
        """Start the RunPod worker.

        In simulation (``APIPod(simulate="serverless-runpod", direct=True)``) RunPod's
        local API emulator is used. In a managed deployment the real serverless worker runs.
        """
        # Load declared apipod.Model instances before the worker accepts jobs.
        load_declared_models()

        if self.simulate:
            self.start_runpod_serverless_localhost(port=port)
        else:
            import runpod.serverless
            runpod.serverless.start({"handler": self.handler, "return_aggregate_stream": True})