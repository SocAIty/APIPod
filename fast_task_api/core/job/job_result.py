import gzip
from io import BytesIO
from typing import Optional, Union, Any

from pydantic import BaseModel

from connectors.settings import DEFAULT_DATE_TIME_FORMAT
from fast_task_api.compatibility.upload import is_param_media_toolkit_file
from fast_task_api.core.job.base_job import JOB_STATUS, BaseJob


class FileResult(BaseModel):
    file_name: str
    content_type: str
    content: str  # base64 encoded


class JobResult(BaseModel):
    """
    When the user (client) sends a request to an Endpoint, a ClientJob is created.
    This job contains the information about the request and the response.
    """
    id: str
    status: Optional[str] = None
    progress: Optional[float] = 0.0
    message: Optional[str] = None
    result: Union[FileResult, Any, str, None] = None
    refresh_job_url: Optional[str] = None

    created_at: Optional[str] = None
    queued_at: Optional[str] = None
    execution_started_at: Optional[str] = None
    execution_finished_at: Optional[str] = None

    endpoint_protocol: Optional[str] = "socaity"


class JobResultFactory:

    @staticmethod
    def from_base_job(ij: BaseJob) -> JobResult:
        format_date = lambda date: date.strftime(DEFAULT_DATE_TIME_FORMAT) if date else None
        created_at = format_date(ij.created_at)
        queued_at = format_date(ij.queued_at)
        execution_started_at = format_date(ij.execution_started_at)
        execution_finished_at = format_date(ij.execution_finished_at)

        # if the internal job returned a media-toolkit file, convert it to a json serializable FileResult
        result = ij.result
        if is_param_media_toolkit_file(ij.result):
            result = FileResult(**result.to_json())

        # Job_status is an Enum, convert it to a string to return it as json
        status = ij.status
        if isinstance(status, JOB_STATUS):
            status = status.value

        return JobResult(
            id=ij.id,
            status=status,
            progress=ij.job_progress._progress,
            message=ij.job_progress._message,
            result=result,
            created_at=created_at,
            queued_at=queued_at,
            execution_started_at=execution_started_at,
            execution_finished_at=execution_finished_at
        )

    @staticmethod
    def gzip_job_result(job_result: JobResult) -> bytes:
        job_result_bytes = job_result.json().encode('utf-8')
        # Compress the serialized bytes with gzip
        gzip_buffer = BytesIO()
        with gzip.GzipFile(fileobj=gzip_buffer, mode='wb') as gzip_file:
            gzip_file.write(job_result_bytes)

        # Retrieve the gzipped data
        return gzip_buffer.getvalue()


    @staticmethod
    def job_not_found(job_id: str) -> JobResult:
        return JobResult(
            id=job_id,
            status=JOB_STATUS.FAILED,
            message="Job not found.",
        )

