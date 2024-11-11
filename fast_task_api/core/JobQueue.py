import inspect
import traceback
from datetime import datetime
import time
import threading
from typing import Union

from fast_task_api.core.job.InternalJob import InternalJob, JOB_STATUS


class JobQueue:
    def __init__(self):
        self.queue = []
        self.in_progress = []  # a list of {"job_id": job.id, "thread": t_job, "job": job}
        self.results = []
        self.worker_thread = threading.Thread(target=self.process_jobs_in_background, daemon=True)
        # used to store the queue size for each function.
        # Limits the number of jobs that can be created for a specific path / function
        self.queue_sizes = {}  # a dictionary of {path: queue_size}

    def set_queue_size(self, job_function: callable, queue_size: int):
        self.queue_sizes[job_function.__name__] = queue_size

    def _add_job_to_queue(self, job: InternalJob):
        """
        :param job:
        :return:
        """
        # check if queue size is reached
        if self.queue_sizes.get(job.job_function.__name__, 1) <= len(
                [j for j in self.queue if j.job_function == job.job_function]):
            job.status = JOB_STATUS.FAILED
            job.result = f"Queue size for function {job.job_function.__name__} reached."
            self.results.append(job)
            return job

        # add job to queue
        job.status = JOB_STATUS.QUEUED
        job.queued_at = datetime.utcnow()
        self.queue.append(job)

        # start worker thread if not already done so
        if not self.worker_thread.is_alive():
            self.worker_thread.start()

        return job

    def add_job(
        self,
        job_function: callable,
        job_params: dict = None
    ):
        job = InternalJob(
            job_function=job_function,
            job_params=job_params
        )
        return self._add_job_to_queue(job)

    def process_job(self, job: InternalJob):
        job.execution_started_at = datetime.utcnow()
        job.status = JOB_STATUS.PROCESSING

        # if function has a param with the type JobProgress in the function signature, pass the job_progress object
        for p in inspect.signature(job.job_function).parameters.values():
            if p.name == "job_progress" or "JobProgress" in p.annotation.__name__:
                job.job_params[p.name] = job.job_progress

        try:
            job.result = job.job_function(**job.job_params)
            # if execution was successful set _progress to 1.0 and status to finished
            job.job_progress.set_status(1.0, None)
            job.status = JOB_STATUS.FINISHED
        except Exception as e:
            job.result = None
            job.job_progress.set_status(1.0, str(e))
            job.status = JOB_STATUS.FAILED
            print(traceback.format_exc())

        job.execution_finished_at = datetime.utcnow()
        # store result in results. Necessary in threading because thread itself cannot easily return values
        self.results.append(job)

    def check_in_progress_jobs(self):
        """
        Checks over all jobs in the in_progress list.
        Removes finished jobs, timed-out jobs.
        :return:
        """
        # check if jobs are finished
        for job_thread in self.in_progress:
            # remove finished jobs
            if not job_thread["thread"].is_alive():
                self.in_progress.remove(job_thread)

            # remove timeout jobs
            if job_thread["job"].time_out_at < datetime.utcnow():
                # set status to failed
                j = job_thread["job"]
                j.status = JOB_STATUS.TIMEOUT
                # todo: implement method e.g with multiprocessing to kill thread
                self.in_progress.remove(job_thread)

                self.results.append(j)

    def check_queuing_up_jobs(self):
        """
        Check if jobs are in queue.
        Adds queued jobs to in_progress
        """
        # create new jobs from queue
        for job in self.queue:
            t_job = threading.Thread(target=self.process_job, args=(job,), daemon=True)
            t_job.start()

            self.in_progress.append({"job_id": job.id, "thread": t_job, "job": job})
            self.queue.remove(job)

    def process_jobs_in_background(self):
        while True:
            if len(self.queue) == 0 and len(self.in_progress) == 0:
                time.sleep(1)  # server is idle
                continue

            # add jobs from queue to in_progress
            self.check_queuing_up_jobs()
            self.check_in_progress_jobs()

            # give threads time to breath
            time.sleep(0.01)
            # ToDo: remove jobs from memory which are long finished and results not retrieved

    def get_job(self, job_id: str, keep_in_memory: bool = False) -> Union[InternalJob, None]:
        """
        Get a job by its id. Returns None if the job does not exist.
        :param job_id: the id of the job
        :param keep_in_memory: if True, the job will be kept in memory even when finished and retrieved.
            - By default jobs are removed from result memory when finished and get_job is called.
        """
        # check if in results
        job = next((job for job in self.results if job.id == job_id), None)
        if job:
            return job

        # check if in _progress
        job = next((th['job'] for th in self.in_progress if th["job_id"] == job_id), None)
        if job:
            return job

        # return if in queue
        job = next((job for job in self.results if job.id == job_id), None)
        if not keep_in_memory:
            self.queue.remove(job)
