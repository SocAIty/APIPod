import logging

logger = logging.getLogger(__name__)


class JobProgress:
    """Live progress handle for a running job.

    Holds the single source of truth for a job's ``progress`` and ``message``:
    the endpoint function receives this object (via a ``job_progress`` parameter)
    and reports updates through :meth:`set_status`, while the job record exposes
    the same object so the public ``JobResult`` can read the current values.
    """

    def __init__(self, progress: float = 0.0, message: str = None):
        """:param progress: value between 0 and 1.0. :param message: message to deliver to the client."""
        self.progress = progress
        self.message = message
        logger.setLevel(logging.INFO)

    def set_status(self, progress: float = None, message: str = None):
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        logger.info(f"Progress: {self.progress} Message: {self.message}")
        return self


class JobProgressRunpod(JobProgress):
    def __init__(self, runpod_job, progress: float = 0.0, message: str = None):
        super().__init__(progress=progress, message=message)
        self.runpod_job = runpod_job

    def set_status(self, progress: float = None, message: str = None):
        super().set_status(progress=progress, message=message)

        try:
            import runpod
            runpod.serverless.progress_update(
                self.runpod_job,
                f"Progress: {int(self.progress)} Message: {self.message}"
            )
        except Exception as e:
            print(f"Problem in progress update: {e}")
