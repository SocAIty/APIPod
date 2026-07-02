from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, FrozenSet, Optional
from uuid import uuid4

from apipod.engine.jobs.job_progress import JobProgress


class JOB_STATUS(Enum):
    """Internal lifecycle states for APIPod's in-process job queue.

    Values are lowercase strings suitable for :class:`JobResult.status` when
    APIPod runs standalone.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    STREAMING = "streaming"
    FINISHED = "finished"
    FAILED = "failed"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JOB_STATUS.FINISHED,
            JOB_STATUS.FAILED,
            JOB_STATUS.TIMEOUT,
        }


# Statuses where GET /stream may attach before the local stream store opens.
STREAM_WAIT_STATUSES: FrozenSet[str] = frozenset(
    {
        JOB_STATUS.QUEUED.value,
        JOB_STATUS.PROCESSING.value,
        JOB_STATUS.STREAMING.value,
    }
)


class JobMetrics:
    """Storage for all job timing and performance data."""

    def __init__(self):
        self.created_at: datetime = datetime.now(timezone.utc)
        self.queued_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.time_out_at: Optional[datetime] = None

    @property
    def execution_time_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            delta = (self.finished_at - self.started_at).total_seconds()
            return round(delta, 2) if delta >= 0 else 0.0
        return None


class BaseJob:
    """
    Essential job record shared by all job types (local thread, remote service, etc.).
    Subclass to add domain-specific fields like service_id, endpoint, etc.
    """

    def __init__(self, id: Optional[str] = None):
        self.id: str = id or str(uuid4())
        self.status = JOB_STATUS.QUEUED
        self.result: Any = None
        self.error: Optional[str] = None
        self.job_progress = JobProgress()
        self.metrics = JobMetrics()


class LocalJob(BaseJob):
    """In-process job for the local :class:`~apipod.engine.queue.job_queue.JobQueue`
    (thread pool, ``job_function``, :class:`JobProgress`).
    """

    def __init__(self, job_function: callable, job_params: Optional[dict] = None, timeout_seconds: int = 3600):
        super().__init__()
        self.job_function = job_function
        self.job_params: dict = job_params or {}
        self.metrics.time_out_at = self.metrics.created_at + timedelta(seconds=timeout_seconds)

    @property
    def is_timed_out(self) -> bool:
        if not self.metrics.time_out_at:
            return False
        return datetime.now(timezone.utc) > self.metrics.time_out_at
