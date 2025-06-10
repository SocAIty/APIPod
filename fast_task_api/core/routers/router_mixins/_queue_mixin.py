import functools

from fast_task_api.CONSTS import SERVER_HEALTH
from fast_task_api.core.routers.router_mixins.job_queue import JobQueue
from fast_task_api.core.job.job_result import JobResultFactory, JobResult
from fast_task_api.settings import SERVER_DOMAIN


class _QueueMixin:
    """
    Adds a job queue to a app.
    Then instead of returning the result of the function, it returns a job object.
    Jobs are executed in threads. The user can check the status of the job and get the result.
    """
    def __init__(self, *args, **kwargs):
        self.job_queue = JobQueue()
        self.status = SERVER_HEALTH.INITIALIZING

        # add the get_status function to the routes
        # self.add_api_route(path="/status")

    def job_queue_func(
            self,
            queue_size: int = 500,
            *args,
            **kwargs
    ):
        """
        Adds an additional wrapper to the API path to add functionality like:
        - Create a job and add to the job queue
        - Return job
        """
        # add the queue to the job queue
        def decorator(func):
            self.job_queue.set_queue_size(func, queue_size)

            @functools.wraps(func)
            def job_creation_func_wrapper(*wrapped_func_args, **wrapped_func_kwargs) -> JobResult:
                # combine args and kwargs
                wrapped_func_kwargs.update(wrapped_func_args)
                # create a job and add to the job queue
                base_job = self.job_queue.add_job(
                    job_function=func,
                    job_params=wrapped_func_kwargs
                )
                ret_job = JobResultFactory.from_base_job(base_job)
                ret_job.refresh_job_url = f"{SERVER_DOMAIN}/status?job_id={ret_job.id}"
                return ret_job

            return job_creation_func_wrapper

        return decorator

