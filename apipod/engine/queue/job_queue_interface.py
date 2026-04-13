from abc import ABC, abstractmethod
from typing import Optional, Callable, TypeVar, Generic

from apipod.engine.jobs.base_job import BaseJob
from apipod.engine.jobs.job_result import JobResult, JobResultFactory

T = TypeVar('T', bound=BaseJob)


class JobQueueInterface(Generic[T], ABC):
    """Abstract interface for JobQueue implementations.

    Both :meth:`add_job` (submission) and :meth:`get_job_result` (status
    polling) return a :class:`JobResult`.  The default implementations convert
    via ``JobResultFactory.from_base_job``; queue backends that store richer
    job types (e.g. Redis ``ServiceJob``) override to add extra fields.
    """

    def add_job(self, job_function: Callable, job_params: Optional[dict] = None) -> JobResult:
        """Create a job, enqueue it, and return the public :class:`JobResult`.

        Subclasses **must** override :meth:`_add_job` (raw enqueue logic) and
        may override this method to customise the ``JobResult`` conversion.
        """
        job = self._add_job(job_function, job_params)
        return JobResultFactory.from_base_job(job)

    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Resolve job status for API responses."""
        job = self.get_job(job_id)
        return JobResultFactory.from_base_job(job) if job is not None else None

    @abstractmethod
    def _add_job(self, job_function: Callable, job_params: Optional[dict] = None) -> T:
        """Internal: create and enqueue a job.  Returns the raw job object."""
        ...

    @abstractmethod
    def set_queue_size(self, job_function: Callable, queue_size: int = 500) -> None:
        ...

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[T]:
        """Retrieve a job by its ID. Returns None if not found."""
        ...

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

