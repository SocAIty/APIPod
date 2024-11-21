from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4
from enum import Enum

from fast_task_api.core.job.job_progress import JobProgress


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
    """Base class for all job types with common functionality"""

    def __init__(self, job_function: callable, job_params: Optional[dict] = None, timeout_seconds: int = 3600):
        self.id: str = str(uuid4())
        self.job_function = job_function
        self.job_params: dict = job_params or {}
        self.status: JOB_STATUS = JOB_STATUS.QUEUED
        self.job_progress = JobProgress()
        self.result = None

        # Timing information
        self.created_at = datetime.utcnow()
        self.queued_at: Optional[datetime] = None
        self.execution_started_at: Optional[datetime] = None
        self.execution_finished_at: Optional[datetime] = None
        self.time_out_at = self.created_at + timedelta(seconds=timeout_seconds)

    @property
    def is_timed_out(self) -> bool:
        return datetime.utcnow() > self.time_out_at

    @property
    def execution_duration_ms(self) -> int:
        if not self.execution_started_at:
            return 0
        end_time = self.execution_finished_at or datetime.utcnow()
        return int((end_time - self.execution_started_at).total_seconds() * 1000)

    @property
    def delay_time_ms(self) -> int:
        """
        Is the time of the job being created + queued until it was started.
        If not started the time since creation
        """
        if not self.queued_at:
            return int((datetime.utcnow() - self.queued_at).total_seconds() * 1000)

        if not self.execution_started_at:
            return int((datetime.utcnow() - self.execution_started_at).total_seconds() * 1000)

        return int((self.execution_started_at - self.created_at).total_seconds() * 1000)