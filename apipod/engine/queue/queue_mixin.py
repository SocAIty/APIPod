import functools
from typing import Callable

from media_toolkit import MediaFile, MediaList

from apipod.common.constants import SERVER_HEALTH
from apipod.engine.jobs.job_result import JobResult


def _materialize_media_params(job_params: dict) -> None:
    """Copy lazy media views into owned buffers before enqueueing.

    Uploads are wrapped lazily over the Starlette request spool, which closes
    when the request ends. A queued job runs later in a worker thread, so it
    must own its bytes (a lazy handle in the queue is a use-after-close).
    """
    for value in job_params.values():
        if isinstance(value, MediaFile):
            value.materialize()
        elif isinstance(value, MediaList):
            for item in value:
                if isinstance(item, MediaFile):
                    item.materialize()


class _QueueMixin:
    """Adds job-queue submission to a router.

    When a queue is configured, endpoint handlers are wrapped so they enqueue
    work and immediately return a :class:`JobResult` (status + links) instead
    of blocking for the result.
    """

    def __init__(self, job_queue=None, *args, **kwargs):
        self.job_queue = job_queue
        self.status = SERVER_HEALTH.INITIALIZING

    def add_job(self, func: Callable, job_params: dict) -> JobResult:
        """Enqueue *func* and return the public :class:`JobResult`."""
        if self.job_queue is None:
            raise ValueError("Job Queue is not initialized. Cannot add job.")
        return self.job_queue.add_job(job_function=func, job_params=job_params)

    def job_queue_func(self, queue_size: int = 500, *args, **kwargs):
        """Decorator that wraps an endpoint handler with job-queue submission.

        The wrapper accepts only ``**kwargs`` because FastAPI always passes
        request parameters as keyword arguments (never positional).
        ``@functools.wraps`` copies the original function's ``__signature__``
        so FastAPI still sees the correct parameter schema for OpenAPI docs.
        """

        def decorator(func):
            if self.job_queue:
                self.job_queue.set_queue_size(func, queue_size)

            @functools.wraps(func)
            def job_creation_func_wrapper(**func_kwargs) -> JobResult:
                if self.job_queue:
                    _materialize_media_params(func_kwargs)
                    return self.add_job(func, func_kwargs)
                raise ValueError("job_queue_func called but no job_queue is configured.")

            return job_creation_func_wrapper

        return decorator
