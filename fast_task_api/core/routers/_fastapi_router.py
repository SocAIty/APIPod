import functools
import inspect
from fast_task_api.core.utils import get_func_signature, replace_func_signature
from typing import Union
from fastapi import APIRouter, FastAPI

from fast_task_api.compatibility.upload import (convert_param_type_to_fast_api_upload_file,
                                                is_param_media_toolkit_file)
from fast_task_api.settings import FTAPI_PORT, FTAPI_HOST
from media_toolkit import media_from_any
from fast_task_api.CONSTS import SERVER_HEALTH
from fast_task_api.core.job_queue import JobQueue
from fast_task_api.core.job.job_result import JobResult, JobResultFactory
from fast_task_api.core.routers._socaity_router import _SocaityRouter
from fast_task_api.core.routers.router_mixins._queue_mixin import _QueueMixin

import importlib.metadata


class SocaityFastAPIRouter(APIRouter, _SocaityRouter, _QueueMixin):
    def __init__(
            self,
            title: str = "FastTaskAPI",
            summary: str = "Create web-APIs for long-running tasks",
            app: Union[FastAPI, None] = None,
            prefix: str = "/api",
            *args, **kwargs):
        """
        :param title: The title of the app. (Like FastAPI(title))
        :param summary: The summary of the app. (Like FastAPI(summary))
        :param app: You can pass an existing fastapi app, if you like to have multiple routers in one app
        :param prefix: The prefix of this app for the paths
        :param args: other fastapi app arguments
        :param kwargs: other fastapi app keyword arguments
        """
        # INIT upper classes
        # inspect the APIRouter params and init with only the ones that are needed
        pams = inspect.signature(APIRouter.__init__).parameters
        api_router_init_kwargs = {
            key: kwargs[key]
            for key in pams.keys()
            if key in kwargs
        }
        # need to do each init separately instead of super().__init__(*args, **kwargs) to avoid conflicts with APIRouter
        APIRouter.__init__(self, **api_router_init_kwargs)
        _SocaityRouter.__init__(self=self, title=title, summary=summary, *args, **kwargs)
        _QueueMixin.__init__(self, *args, **kwargs)

        self.job_queue = JobQueue()
        self.status = SERVER_HEALTH.INITIALIZING

        # Configuring the fastapi app and router
        if app is None:
            app = FastAPI(
                title=self.title,
                summary=self.summary,
                contact={
                    "name": "SocAIty",
                    "url": "https://github.com/SocAIty",
            })

        self.app = app
        self.prefix = prefix
        self.add_standard_routes()
        self._orig_openapi_func = self.app.openapi
        self.app.openapi = self.custom_openapi

    def add_standard_routes(self):
        #self.api_route(path="/status", methods=["GET", "POST"])(self.get_job)
        self.api_route(path="/health", methods=["GET"])(self.get_health)
        #self.api_route(path="/cancel", methods=["POST"])(self.get_status)
        # ToDo: add favicon
        #self.api_route('/favicon.ico', include_in_schema=False)(self.favicon)

    def custom_openapi(self):
        if not self.app.openapi_schema:
            self._orig_openapi_func()
        version = importlib.metadata.version("fast-task-api")
        self.app.openapi_schema["info"]["fast-task-api"] = version
        return self.app.openapi_schema

    def get_job(self, job_id: str, return_format: str = 'json', keep_in_memory: bool = False) -> JobResult:
        """
        Get the job with the given job_id.
        :param job_id: The id of the job.
        :param return_format: json or gzipped
        :param keep_in_memory: If the job should be kept in memory.
            If False, the job is removed after the result is returned.
        """
        base_job = self.job_queue.get_job(job_id, keep_in_memory=keep_in_memory)
        if base_job is None:
            return JobResultFactory.job_not_found(job_id)

        ret_job = JobResultFactory.from_base_job(base_job)
        ret_job.refresh_job_url = f"/status?job_id={ret_job.id}"
        ret_job.cancel_job_url = f"/cancel?job_id={ret_job.id}"

        if return_format != 'json':
            ret_job = JobResultFactory.gzip_job_result(ret_job)

        return ret_job

    def cancel_job(self, job_id: str):
        """
        Cancel the job with the given job_id.
        :param job_id: The id of the job.
        """
        cancelled = self.job_queue.cancel_job(job_id)
        raise Exception("Not implemented yet. Feel free to contribute.")

    @staticmethod
    def _remove_job_progress_from_signature(func: callable) -> callable:
        sig = get_func_signature(func)
        new_sig = sig.replace(parameters=[
            p for p in sig.parameters.values()
            if p.name != "job_progress" and "JobProgress" not in p.annotation.__name__
        ])
        return replace_func_signature(func, new_sig)


    def _handle_file_uploads(self, func: callable) -> callable:
        """
        Modify the function signature for fastapi to handle file uploads.
        Parse/Read the starlette.MediaFile and give it as read socaity MediaFile to the function while execution.
        """

        # original func parameter names: needed multiple times
        original_func_sig = get_func_signature(func)
        original_func_parameters = original_func_sig.parameters.values()
        # create a dict to store the params that are UploadFiles
        # this is used to later map the file while reading
        upload_params = {
            param.name: param.annotation
            for param in original_func_parameters
            if is_param_media_toolkit_file(param)
        }

        def read_file_if_is_upload_file(param_name: str, data):
            # check if we have the file in our list
            my_data_type = upload_params.get(param_name, None)
            if my_data_type is not None:
                return media_from_any(data, my_data_type)
            # if is not a file, return as is
            return data

        @functools.wraps(func)
        def file_upload_wrapper(*args, **kwargs):
            # args, kwargs to _ kwargs
            org_func_names = [param.name for param in original_func_parameters]
            nkwargs = {org_func_names[i]: arg for i, arg in enumerate(args)}
            nkwargs.update(kwargs)
            # convert to socaity MediaFile if it is a file
            n_kwargs = {key: read_file_if_is_upload_file(key, value) for key, value in kwargs.items()}

            return func(**n_kwargs)

        # replace signature with fastapi signature
        new_sig = original_func_sig.replace(parameters=[
            convert_param_type_to_fast_api_upload_file(param)
            if is_param_media_toolkit_file(param) else param
            for param in original_func_parameters
        ])
        file_upload_wrapper.__signature__ = new_sig
        func = replace_func_signature(func, new_sig)

        return file_upload_wrapper

    @functools.wraps(APIRouter.api_route)
    def endpoint(self, path: str, methods: list[str] = None, *args, **kwargs):
        def decorator(func):
            self.api_route(path=path, methods=methods)(func)
        return decorator

    def task_endpoint(
            self,
            path: str,
            queue_size: int = 100,
            methods: list[str] = None,
            *args,
            **kwargs
    ):
        """
        Adds an additional wrapper to the API path to add functionality like:
        - Add api key validation
        - Create a job and add to the job queue
        - Return job
        """
        if len(path) > 0 and path[0] != "/":
            path = "/" + path

        fastapi_route_decorator_func = self.api_route(
            path=path,
            methods=["POST"] if methods is None else methods,
            *args,
            **kwargs
        )

        queue_router_decorator_func = super().job_queue_func(
            path=path,
            queue_size=queue_size,
            *args,
            **kwargs
        )

        def decorator(func):
            # add the queue to the job queue
            queue_decorated = queue_router_decorator_func(func)
            # remove job_progress from the function signature to display nice for fastapi
            job_progress_removed = self._remove_job_progress_from_signature(queue_decorated)
            # modify file uploads for compatibility reasons
            file_upload_modified = self._handle_file_uploads(job_progress_removed)
            # modify file responses so that functions can return multimodal files.
            # file_response_modified = self._handle_file_responses(file_upload_modified)
            # add the route to fastapi
            return fastapi_route_decorator_func(file_upload_modified)

        return decorator

    def get(self, path: str = None, queue_size: int = 100, *args, **kwargs):
        return self.task_endpoint(path=path, queue_size=queue_size, methods=["GET"], *args, **kwargs)

    def post(self, path: str = None, queue_size: int = 100, *args, **kwargs):
        return self.task_endpoint(path=path, queue_size=queue_size, methods=["POST"], *args, **kwargs)

    def start(self, port: int = FTAPI_PORT, host: str = FTAPI_HOST, *args, **kwargs):
        """
        Start the FastAPI server and add this app.
        """
        # fast API start
        if self.app is None:
            self.app = FastAPI()

        self.app.include_router(self)

        # print helping statement
        print_host = "localhost" if host == "0.0.0.0" or host is None else host
        print(f"FastTaskAPI {self.app.title} started. Use http://{print_host}:{port}/docs to see the API documentation.")
        # start uvicorn
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

