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
    job types override to add extra fields.

    The transport layer (router) passes presentation context through this
    port — ``supports_streaming`` (may the job's output be consumed via
    ``GET /stream/{job_id}``) and ``link_prefix`` (mount prefix for hypermedia
    links) — so it never needs to know which queue implementation it holds.
    """

    def add_job(
        self,
        job_function: Callable,
        job_params: Optional[dict] = None,
        *,
        supports_streaming: bool = False,
        link_prefix: str = "",
    ) -> JobResult:
        """Create a job, enqueue it, and return the public :class:`JobResult`.

        Subclasses **must** override :meth:`_add_job` (raw enqueue logic) and
        may override this method to customise the ``JobResult`` conversion.
        """
        job = self._add_job(job_function, job_params)
        job.supports_streaming = supports_streaming
        return JobResultFactory.from_base_job(
            job,
            include_stream_link=supports_streaming,
            link_prefix=link_prefix,
        )

    def get_job_result(self, job_id: str, *, link_prefix: str = "") -> Optional[JobResult]:
        """Resolve job status for API responses."""
        job = self.get_job(job_id)
        if job is None:
            return None
        return JobResultFactory.from_base_job(
            job,
            include_stream_link=bool(getattr(job, "supports_streaming", False)),
            link_prefix=link_prefix,
        )

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
    def cancel_job(self, job_id: str) -> Optional[dict]:
        """Cancel a job.

        Returns an optional cancellation summary dict (``id`` / ``status`` /
        ``message``) for the HTTP response; ``None`` means "cancelled, use the
        default summary".  Raise :class:`NotImplementedError` when the queue
        does not support cancellation.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

