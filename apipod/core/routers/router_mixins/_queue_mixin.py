import functools
from typing import Callable, Optional

from apipod.CONSTS import SERVER_HEALTH
from apipod.core.job_queues.job_queue import JobQueue
from apipod.core.job.job_result import JobResultFactory, JobResult
from apipod.settings import SERVER_DOMAIN


class _QueueMixin:
    """
    Adds a job queue to a app.
    Then instead of returning the result of the function, it returns a job object.
    Jobs are executed in threads. The user can check the status of the job and get the result.
    """
    def __init__(self, job_queue=None, *args, **kwargs):
        # job_queue is optional. If None, no queue is used.
        self.job_queue = job_queue
        self.status = SERVER_HEALTH.INITIALIZING

    def add_job(self, func: Callable, job_params: dict) -> JobResult:
        """
        Use for creating jobs internally without using the task_decorator / job_queue_func decorator.
        """
        if self.job_queue is None:
             raise ValueError("Job Queue is not initialized. Cannot add job.")

        # create a job and add to the job queue
        base_job = self.job_queue.add_job(
            job_function=func,
            job_params=job_params
        )
        # add the get_status function to the routes so the user can check the status of the job
        ret_job = JobResultFactory.from_base_job(base_job)
        ret_job.refresh_job_url = f"{SERVER_DOMAIN}/status?job_id={ret_job.id}"
        return ret_job

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
            if self.job_queue:
                self.job_queue.set_queue_size(func, queue_size)

            @functools.wraps(func)
            def job_creation_func_wrapper(*wrapped_func_args, **wrapped_func_kwargs) -> JobResult:
                # combine args and kwargs
                wrapped_func_kwargs.update(wrapped_func_args)
                
                if self.job_queue:
                    # create a job and add to the job queue
                    return self.add_job(func, wrapped_func_kwargs)
                else:
                    # No queue, execute directly? 
                    # This method is specifically for job creation. 
                    # If called without queue, it should probably fail or be handled by caller.
                    # But in the new design, the caller (router) decides whether to use this wrapper.
                    raise ValueError("job_queue_func called but no job_queue is configured.")

            return job_creation_func_wrapper

        return decorator
