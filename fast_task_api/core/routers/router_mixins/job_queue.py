import threading
import traceback
from datetime import datetime
import time
from typing import Generic, Dict, Optional, TypeVar, Tuple

from fast_task_api.core.job_store import JobStore
from fast_task_api.core.job.base_job import BaseJob, JOB_STATUS
import inspect

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

    def _validate_queue_size(self, job: BaseJob) -> Tuple[bool, str]:
        queue_size = self.queue_sizes.get(job.job_function.__name__, 100)
        queued_count = sum(1 for qjob in self.job_store.queued_jobs
                           if qjob.job_function == job.job_function)

        queue_size_exceeded = queued_count >= queue_size
        if queue_size_exceeded:
            return False, f"Queue size limit reached for {job.job_function.__name__}"
        return True, None

    def _validate_job_before_add(self, job: BaseJob) -> Tuple[bool, str]:
        valid, message = self._validate_queue_size(job)
        if not valid:
            return False, message

        return True, None

    def add_job(self, job_function: callable, job_params: Optional[dict] = None) -> T:
        job = self._create_job(job_function, job_params)

        # pre add validation
        valid, message = self._validate_job_before_add(job)
        if not valid:
            job.status = JOB_STATUS.FAILED
            job.error = message
            job.job_progress.set_status(1.0, message)
            # if not added to completed job, some clients fail with job not found on polling.
            self.job_store._add_job(job)
            self._complete_job(job=job, final_state=JOB_STATUS.FAILED) 
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

    def _process_job(self, job: T) -> None:
        try:
            job.execution_started_at = datetime.utcnow()
            job.status = JOB_STATUS.PROCESSING

            self._inject_job_progress(job)

            job.result = job.job_function(**job.job_params)
            job.job_progress.set_status(1.0)
            self._complete_job(job=job, final_state=JOB_STATUS.FINISHED)

        except Exception as e:
            job.result = None
            job.job_progress.set_status(1.0, str(e))
            job.error = str(e)
            self._complete_job(job, JOB_STATUS.FAILED)
            # Print the full stack trace to standard error
            print(f"Job {job.id} failed: {str(e)}")
            traceback.print_exc()  # Writes full traceback to stderr

    def _complete_job(self, job: T, final_state: JOB_STATUS) -> T:
        self.job_store.complete_job(job.id)
        # setting status here, because if this is done earlier, race conditions in get_job are the problem
        job.execution_finished_at = datetime.utcnow()
        job.status = final_state
        return job

    def _remove_job(self, job: T) -> None:
        """ Override this method to add custom job removal logic """
        self.job_store.remove_completed_job(job.id)

    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError("Job cancellation is not implemented yet.")
        # if job := self.get_job(job_id):
        #    job.status = JOB_STATUS.FAILED
        #    job.job_progress.set_status(1.0, "Job cancelled")
        #    todo: sent event to thread, make a cancel request...
        #    self._complete_job(job_id)

    def _inject_job_progress(self, job: T) -> T:
        sig = inspect.signature(job.job_function)

        job_progress_params = [
            p for p in sig.parameters.values()
            if p.name == "job_progress" or "FastJobProgress" in str(p.annotation)
        ]
        for job_progress_param in job_progress_params:
            job.job_params[job_progress_param.name] = job.job_progress

        return job

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
                self._complete_job(job, JOB_STATUS.TIMEOUT)

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
                try:
                    self._remove_job(self.job_store.get_job(job_id))
                except Exception as e:
                    print(f"Error removing job {job_id}: {str(e)}")

    def _clean_up_orphan_jobs(self) -> None:
        """
        Cleanup the job queue. This method is called after the worker thread has finished.
        We want to remove jobs which are not collected orphans.
        """
        if not self._delete_orphan_jobs_after_seconds:
            return

        for job in self.job_store.completed_jobs:
            if job.execution_finished_at is None:
                self._remove_job(job)
            elif (datetime.utcnow() - job.execution_finished_at).total_seconds() > self._delete_orphan_jobs_after_seconds:
                self._remove_job(job)

    def _start_queued_jobs(self) -> None:
        for job in self.job_store.queued_jobs:
            if job := self.job_store.move_to_in_progress(job.id):
                thread = threading.Thread(target=self._process_job, args=(job,), daemon=True)
                self._job_threads[job.id] = thread
                thread.start()

    def get_job(self, job_id: str, keep_alive: bool = False) -> Optional[T]:
        job = self.job_store.get_job(job_id)
        if not job:
            return None

        if job and self.job_store.is_completed(job.id):  # job.status in {JOB_STATUS.FINISHED, JOB_STATUS.FAILED, JOB_STATUS.TIMEOUT}:
            if not keep_alive:
                self._remove_job(job)
        return job

    def shutdown(self) -> None:
        self._shutdown.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)