from datetime import datetime, timedelta
from typing import Union
from uuid import uuid4
from enum import Enum

from fast_task_api.core.job.JobProgress import JobProgress


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


class InternalJob:
    def __init__(
            self,
            job_function: callable,
            job_params: Union[dict, None],
            timeout: int = 3600
    ):
        """
        Internal Job object to keep track of the job status and relevant information.
        :job_function (callable): The function to execute
        :job_params (dict): Parameters for the request
        :timeout (int): Timeout in seconds. If none timeout is set to one year.
        """

        self.id = str(uuid4())
        self.job_function = job_function
        self.job_params: Union[dict, None] = job_params
        self.status: JOB_STATUS = JOB_STATUS.QUEUED  # Use job.status to modify in other places NOT _status
        self.job_progress = JobProgress()
        self.result = None

        # timeout used to kill long running jobs in the queue
        if timeout is not None:
            self.time_out_at = datetime.utcnow() + timedelta(seconds=timeout)
        else:
            # set timeout to one year avoids other none checks
            self.time_out_at = datetime.utcnow() + timedelta(days=365)

        # statistics
        self.created_at = datetime.utcnow()
        self.queued_at = None
        self.execution_started_at = None
        self.execution_finished_at = None

