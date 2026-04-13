from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4
from enum import Enum

from apipod.engine.jobs.job_progress import JobProgress


class JOB_STATUS(Enum):
    QUEUED = "Queued"
    PROCESSING = "Processing"
    FINISHED = "Finished"
    FAILED = "Failed"
    TIMEOUT = "Timeout"


class PROVIDERS(Enum):
    RUNPOD = "runpod"
    OPENAI = "openai"
    REPLICATE = "replicate"


class BaseJob:
    """Essential job record shared by all job types (local thread, remote service, etc.).

    Subclasses add domain-specific fields:

    * :class:`LocalJob` — thread-pool execution (``job_function``, ``job_progress``).
    * Platform-specific subclasses (e.g. gateway ``ServiceJob``) — remote/Redis
      (``service_id``, ``api_url``, ``endpoint``, ``input_data``).

    :class:`~apipod.engine.jobs.job_result.JobResultFactory.from_base_job` maps any
    ``BaseJob`` subclass to the public :class:`~apipod.engine.jobs.job_result.JobResult`.
    """

    def __init__(self, id=None, status="pending", created_at=None, updated_at=None):
        self.id: str = id or str(uuid4())
        self.status = status
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.progress: Optional[float] = None
        self.message: Optional[str] = None


class LocalJob(BaseJob):
    """In-process job for the local :class:`~apipod.engine.queue.job_queue.JobQueue`
    (thread pool, ``job_function``, :class:`JobProgress`).
    """

    def __init__(self, job_function: callable, job_params: Optional[dict] = None, timeout_seconds: int = 3600):
        super().__init__(status=JOB_STATUS.QUEUED)
        self.job_function = job_function
        self.job_params: dict = job_params or {}
        self.job_progress = JobProgress()

        self.queued_at: Optional[datetime] = None
        self.execution_started_at: Optional[datetime] = None
        self.execution_finished_at: Optional[datetime] = None
        self.time_out_at = self.created_at + timedelta(seconds=timeout_seconds)

    @property
    def is_timed_out(self) -> bool:
        return datetime.now(timezone.utc) > self.time_out_at

    @property
    def execution_duration_ms(self) -> int:
        if not self.execution_started_at:
            return 0
        end_time = self.execution_finished_at or datetime.now(timezone.utc)
        return int((end_time - self.execution_started_at).total_seconds() * 1000)

    @property
    def delay_time_ms(self) -> int:
        if not self.queued_at:
            return int((datetime.now(timezone.utc) - self.created_at).total_seconds() * 1000)
        if not self.execution_started_at:
            return int((datetime.now(timezone.utc) - self.queued_at).total_seconds() * 1000)
        return int((self.execution_started_at - self.created_at).total_seconds() * 1000)
