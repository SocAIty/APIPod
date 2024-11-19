from typing import Union

from fast_task_api.CONSTS import SERVER_HEALTH, FTAPI_DEPLOYMENTS
from fast_task_api.settings import FTAPI_DEPLOYMENT, FTAPI_PORT


class _SocaityRouter:
    """
    Base class for all routers.
    """
    def __init__(
            self, title: str = "FastTaskAPI", summary: str = "Create web-APIs for long-running tasks", *args, **kwargs
    ):
        if title is None:
            title = "FastTaskAPI"
        if summary is None:
            summary = "Create web-APIs for long-running tasks"

        self.title = title
        self.summary = summary
        self.status = SERVER_HEALTH.INITIALIZING

    def get_health(self) -> SERVER_HEALTH:
        return self.status

    def get_job(self, job_id: str):
        """
        Get the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job by requesting a task_endpoint.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def cancel_job(self, job_id: str):
        """
        Cancel the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job by requesting a task_endpoint.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def start(self, deployment: Union[FTAPI_DEPLOYMENTS, str] = FTAPI_DEPLOYMENT, port: int = FTAPI_PORT, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def endpoint(self, path: str = None, *args, **kwargs):
        """
        Add a non-task route to the app. This means the method is called directly; no job thread is created.
        :param path:
            In case of fastapi will be resolved as url in form http://{host:port}/{prefix}/{path}
            In case of runpod will be resolved as url in form http://{host:port}?route={path}
        :param args: any other arguments to configure the app
        :param kwargs: any other keyword arguments to configure the app
        :return:
        """
        raise NotImplementedError("Implement in subclass. Use a decorator for that.")

    def task_endpoint(
            self,
            path: str = None,
            queue_size: int = 100,
            *args,
            **kwargs
    ):
        """
        This adds a task-route to the app. This means a job thread is created for each request.
        Then the method returns an JobResult object with the job_id.
        :param path: will be resolved as url in form http://{host:port}/{prefix}/{path}
        :param queue_size: The maximum number of jobs that can be queued. If exceeded the job is rejected.
        """
        raise NotImplementedError("Implement in subclass")

    def get(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")

    def post(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")

    @staticmethod
    def _handle_file_uploads(func: callable):
        """
        Modify the function signature to handle file uploads.
        :param func: the route function
        """
        raise NotImplementedError("Implement in subclass")