from abc import ABC, abstractmethod
from typing import Optional, Callable, TypeVar, Generic
from apipod.engine.jobs.base_job import BaseJob

T = TypeVar('T', bound=BaseJob)


class JobQueueInterface(Generic[T], ABC):
    """
    Abstract interface for JobQueue implementations.
    Allows swapping between local (Thread/Process) and broker (celery) backends.
    """

    @abstractmethod
    def set_queue_size(self, job_function: Callable, queue_size: int = 500) -> None:
        """
        Set the maximum queue size for a specific job function.
        """
        pass

    @abstractmethod
    def add_job(self, job_function: Callable, job_params: Optional[dict] = None) -> T:
        """
        Add a job to the queue.
        Returns the created job object (with ID and status).
        """
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[T]:
        """
        Retrieve a job by its ID.
        Returns None if job not found.
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a running or queued job.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Cleanup resources, stop workers, close connections.
        """
        pass

