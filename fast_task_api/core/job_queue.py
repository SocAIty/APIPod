import threading
import traceback
from datetime import datetime
import time
from typing import Generic, Dict, Optional, TypeVar

from fast_task_api.core.job_store import JobStore
from fast_task_api.core.job.base_job import BaseJob, JOB_STATUS
from fast_task_api.core.utils import get_func_signature

T = TypeVar('T', bound=BaseJob)


class JobQueue(Generic[T]):
    """
    The JobQueue provides a simple interface to queue and process jobs in the background.
    Features:
    - Queue size limits for each job function
    - Threading of jobs
    - Job progress tracking
    - Job timeouts
    - Job result storage
    - Job status tracking
    """
    def __init__(self, delete_orphan_jobs_after_s: int = 60 * 30):
        """
        :param delete_orphan_jobs_after_s: If a completed job is not collected within this time, it will be deleted.
            This is useful to prevent memory leaks when jobs are not collected.
        """
        self.job_store = JobStore[T]()
        self.queue_sizes: Dict[str, int] = {}

        self.worker_thread = threading.Thread(target=self._process_jobs_in_background, daemon=True)
        self._shutdown = threading.Event()
        self._job_threads: Dict[str, threading.Thread] = {}
        self._delete_orphan_jobs_after_seconds = delete_orphan_jobs_after_s

    def set_queue_size(self, job_function: callable, queue_size: int = 500) -> None:
        self.queue_sizes[job_function.__name__] = queue_size

    def _validate_queue_size(self, job: BaseJob) -> (bool, str):
        queue_size = self.queue_sizes.get(job.job_function.__name__, 100)
        queued_count = sum(1 for qjob in self.job_store.queued_jobs
                           if qjob.job_function == job.job_function)

        queue_size_exceeded = queued_count >= queue_size
        if queue_size_exceeded:
            return False, f"Queue size limit reached for {job.job_function.__name__}"
        return True, None

    def _validate_job_before_add(self, job: BaseJob) -> (bool, str):
        valid, message = self._validate_queue_size(job)
        if not valid:
            return True, message

        return True, None

    def add_job(self, job_function: callable, job_params: Optional[dict] = None) -> T:
        job = self._create_job(job_function, job_params)

        valid, message = self._validate_job_before_add(job)

        if not valid:
            job.status = JOB_STATUS.FAILED
            job.error = message
            job.job_progress.set_status(1.0, message)
            return job

        job.status = JOB_STATUS.QUEUED
        job.queued_at = datetime.utcnow()
        self.job_store.add_to_queue(job)

        if not self.worker_thread.is_alive():
            self.worker_thread.start()

        return job

    def _create_job(self, job_function: callable, job_params: Optional[dict] = None) -> T:
        """Override this method in subclasses to create specific job types"""
        return BaseJob(job_function=job_function, job_params=job_params)

    def _inject_job_progress(self, job: T) -> T:
        sig = get_func_signature(job.job_function)

        job_progress_params = [
            p for p in sig.parameters.values()
            if p.name == "job_progress" or "JobProgress" in str(p.annotation)
        ]
        for job_progress_param in job_progress_params:
            job.job_params[job_progress_param.name] = job.job_progress

        return job

    def _job_completed_event(self, job: T) -> T:
        """
        Override this method to add custom event handling when a job is completed.
        Will be called when finished, failed, timed out or cancelled.
        """
        return job

    def _process_job(self, job: T) -> None:
        try:
            job.execution_started_at = datetime.utcnow()
            job.status = JOB_STATUS.PROCESSING

            self._inject_job_progress(job)

            job.result = job.job_function(**job.job_params)
            job.job_progress.set_status(1.0)
            job.status = JOB_STATUS.FINISHED
        except Exception as e:
            job.result = None
            job.job_progress.set_status(1.0, str(e))
            job.error = str(e)
            job.status = JOB_STATUS.FAILED
            # Print the full stack trace to standard error
            print(f"Job {job.id} failed: {str(e)}")
            traceback.print_exc()  # Writes full traceback to stderr
        finally:
            job.execution_finished_at = datetime.utcnow()
            job = self._job_completed_event(job)
            self.job_store.complete_job(job.id)

    def _process_jobs_in_background(self) -> None:
        while not self._shutdown.is_set():
            self._check_job_cancel_criteria()   # Timeouts and other cancel check
            self._cleanup()  # Remove completed jobs with living threads. Delete data and more.
            self._start_queued_jobs()  # move queued jobs to in_progress

            if not (self.job_store.queued_jobs or self.job_store.in_progress_jobs):
                time.sleep(0.1)
                continue

            time.sleep(0.01)

    def _check_job_cancel_criteria(self) -> None:
        """
        Check if any job has timed out or other cancel criteria.
        Override this method to add custom cancel criteria.
        """
        self._check_timeouts()

    def _check_timeouts(self) -> None:
        for job in self.job_store.in_progress_jobs:
            if job.is_timed_out:
                job.status = JOB_STATUS.TIMEOUT
                job.execution_finished_at = datetime.utcnow()
                self.job_store.complete_job(job.id)

    def _cleanup(self) -> None:
        """
        Override this method to add custom cleanup logic. For example cleanup of temporary files.
        """
        self._remove_completed_jobs_with_living_threads()
        self._clean_up_orphan_jobs()

    def _remove_completed_jobs_with_living_threads(self) -> None:
        for job_id in self.job_store.completed_jobs:
            if thread := self._job_threads.pop(job_id, None):
                try:
                    thread.join(timeout=0.1)
                except Exception as e:
                    print(f"Error joining thread for job {job_id}: {str(e)}")

    def _clean_up_orphan_jobs(self) -> None:
        """
        Cleanup the job queue. This method is called after the worker thread has finished.
        We want to remove jobs which are not collected orphans.
        """
        if not self._delete_orphan_jobs_after_seconds:
            return

        for job in self.job_store.completed_jobs:
            if (datetime.utcnow() - job.execution_finished_at).total_seconds() > self._delete_orphan_jobs_after_seconds:
                self.job_store.remove_completed_job(job.id)

    def _start_queued_jobs(self) -> None:
        for job in self.job_store.queued_jobs:
            if job := self.job_store.move_to_in_progress(job.id):
                thread = threading.Thread(target=self._process_job, args=(job,), daemon=True)
                self._job_threads[job.id] = thread
                thread.start()

    def get_job(self, job_id: str, keep_alive: bool = False) -> Optional[T]:
        job = self.job_store.get_job(job_id)
        if job and job.status in {JOB_STATUS.FINISHED, JOB_STATUS.FAILED, JOB_STATUS.TIMEOUT}:
            if not keep_alive:
                self.job_store.remove_completed_job(job_id)
        return job

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError("Job cancellation is not implemented yet.")
        #if job := self.get_job(job_id):
        #    job.status = JOB_STATUS.FAILED
        #    job.job_progress.set_status(1.0, "Job cancelled")
        #    todo: sent event to thread, make a cancel request...
        #    self.job_store.complete_job(job_id)

    def shutdown(self) -> None:
        self._shutdown.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)