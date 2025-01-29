import threading
from collections import deque
from typing import Optional, Set, Dict, TypeVar, Generic, List

from fast_task_api.core.job.base_job import BaseJob

T = TypeVar('T', bound=BaseJob)


class JobStore(Generic[T]):
    """Thread-safe storage for jobs with efficient lookups"""

    def __init__(self):
        self._lock = threading.Lock()
        # this includes all the actual jobs
        self._jobs: Dict[str, T] = {}
        # the following three lists are just references to the jobs in _jobs
        self._queue: deque[str] = deque()
        self._in_progress: Set[str] = set()
        self._completed: deque[str] = deque()

    def add_to_queue(self, job: T) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)

    def move_to_in_progress(self, job_id: str) -> Optional[T]:
        with self._lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
                self._in_progress.add(job_id)
                return self._jobs.get(job_id)
        return None

    def complete_job(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._in_progress:
                self._in_progress.remove(job_id)
                self._completed.append(job_id)

    def get_job(self, job_id: str) -> Optional[T]:
        return self._jobs.get(job_id)

    def remove_completed_job(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._completed:
                self._completed.remove(job_id)
                self._jobs.pop(job_id, None)

    @property
    def queued_jobs(self) -> List[T]:
        return [self._jobs[jid] for jid in self._queue]

    @property
    def in_progress_jobs(self) -> List[T]:
        return [self._jobs[jid] for jid in self._in_progress]

    @property
    def completed_jobs(self) -> List[T]:
        return [self._jobs[jid] for jid in self._completed]
