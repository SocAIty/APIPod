import inspect
import logging
from typing import Callable, List

logger = logging.getLogger(__name__)


def job_progress_param_names(func: Callable) -> List[str]:
    """Names of the parameters of *func* that should receive a :class:`JobProgress`.

    A parameter qualifies when it is literally named ``job_progress`` or when its
    annotation refers to a ``JobProgress`` type. This single detection is shared
    by every injection site (queue worker, RunPod handler, direct FastAPI path)
    so they can never disagree about what counts as a progress parameter.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return []
    return [
        p.name for p in params
        if p.name == "job_progress" or "JobProgress" in str(p.annotation)
    ]


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
