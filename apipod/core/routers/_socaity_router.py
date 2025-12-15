from abc import abstractmethod
from typing import Union
import importlib.metadata
from apipod import CONSTS
from apipod.compatibility.HealthCheck import HealthCheck
from apipod.settings import APIPOD_DEPLOYMENT, APIPOD_PORT


class _SocaityRouter:
    """
    Base class for all routers.
    """
    def __init__(
            self, title: str = "APIPod", summary: str = "Create web-APIs for long-running tasks", *args, **kwargs
    ):
        if title is None:
            title = "APIPod"
        if summary is None:
            summary = "Create web-APIs for long-running tasks"

        self.title = title
        self.summary = summary
        self._health_check = HealthCheck()
        self.version = importlib.metadata.version("apipod")

    @property
    def status(self) -> CONSTS.SERVER_HEALTH:
        return self._health_check.status

    @status.setter
    def status(self, value: CONSTS.SERVER_HEALTH):
        self._health_check.status = value

    def get_health(self) -> Union[dict, str]:
        stat, message = self._health_check.get_health_response()
        return message

    @abstractmethod
    def get_job(self, job_id: str):
        """
        Get the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def cancel_job(self, job_id: str):
        """
        Cancel the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def start(self, deployment: Union[CONSTS.APIPOD_DEPLOYMENT, str] = APIPOD_DEPLOYMENT, port: int = APIPOD_PORT, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def endpoint(self, path: str = None, *args, **kwargs):
        """
        Add a route to the app. 
        Can be a task route (async job) or standard route depending on configuration.
        
        :param path:
            In case of fastapi will be resolved as url in form http://{host:port}/{prefix}/{path}
            In case of runpod will be resolved as url in form http://{host:port}?route={path}
        :param args: any other arguments to configure the app
        :param kwargs: any other keyword arguments to configure the app
        :return:
        """
        raise NotImplementedError("Implement in subclass. Use a decorator for that.")


    def get(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")

    def post(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")
