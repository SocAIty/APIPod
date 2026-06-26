import gzip
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional, Union

from pydantic import BaseModel

from apipod.engine.jobs.base_job import JOB_STATUS, BaseJob
from apipod.engine.signatures.upload import is_param_media_toolkit_file
from media_toolkit import IMediaContainer
from media_toolkit.utils.data_type_utils import is_file_model_dict
from apipod.common.schemas.media_files import FileModel


def _public_status(status: Any) -> Optional[str]:
    """Public status string: a JOB_STATUS carries its own value; pass strings through."""
    if status is None:
        return None
    return status.value if isinstance(status, JOB_STATUS) else str(status)


class JobLinks(BaseModel):
    """Hypermedia links for job status polling, cancellation, and streaming."""
    status: Optional[str] = None
    cancel: Optional[str] = None
    stream: Optional[str] = None


class JobMetrics(BaseModel):
    """Execution metrics APIPod measures for a job."""
    created_at: Optional[datetime] = None
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    execution_time_s: Optional[float] = None


class JobResult(BaseModel):
    """Public job snapshot returned by GET /status and job submissions.

    Unified response: same shape whether the client just submitted a job
    (``status="queued"``) or is polling for completion. The ``status`` values
    mirror :class:`~apipod.engine.jobs.base_job.JOB_STATUS`
    (``queued``/``processing``/``streaming``/``finished``/``failed``/``timeout``).

    Null fields are excluded from the serialized response so the client
    only receives relevant information for the current job state.
    """

    job_id: str
    status: Optional[str] = None
    result: Union[FileModel, List[FileModel], list, str, Any, None] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None

    metrics: Optional[JobMetrics] = None
    links: Optional[JobLinks] = None


class JobResultFactory:
    @staticmethod
    def _serialize_result(data: Any) -> Union[FileModel, List[FileModel], list, str, None]:
        if isinstance(data, IMediaContainer):
            return data.to_json()

        if is_param_media_toolkit_file(data):
            return FileModel(**data.to_json())

        if isinstance(data, FileModel):
            return data

        if is_file_model_dict(data):
            try:
                return FileModel(**data)
            except Exception:
                pass

        # Pydantic schema responses (e.g. SpeechResponse): dump to a dict and
        # recurse so nested media files / FileModels serialize correctly too.
        if isinstance(data, BaseModel):
            return JobResultFactory._serialize_result(data.model_dump())

        if isinstance(data, list):
            return [JobResultFactory._serialize_result(item) for item in data]

        if isinstance(data, dict):
            return {
                key: JobResultFactory._serialize_result(value)
                for key, value in data.items()
            }

        return data

    @staticmethod
    def from_base_job(job: BaseJob, *, include_stream_link: bool = False) -> JobResult:
        """Map a :class:`BaseJob` to the public :class:`JobResult`."""
        m = job.metrics
        metrics = JobMetrics(
            created_at=m.created_at,
            queued_at=m.queued_at,
            started_at=m.started_at,
            finished_at=m.finished_at,
            execution_time_s=m.execution_time_s,
        )

        return JobResult(
            job_id=job.id,
            status=_public_status(job.status),
            result=JobResultFactory._serialize_result(job.result),
            error=job.error,
            progress=job.job_progress.progress,
            message=job.job_progress.message,
            metrics=metrics,
            links=JobLinks(
                status=f"/status/{job.id}",
                cancel=f"/cancel/{job.id}",
                stream=f"/stream/{job.id}" if include_stream_link else None,
            ),
        )

    @staticmethod
    def _job_result_to_json_bytes(job_result: JobResult) -> bytes:
        if hasattr(job_result, "model_dump_json"):
            return job_result.model_dump_json(exclude_none=True).encode("utf-8")
        return job_result.json(exclude_none=True).encode("utf-8")

    @staticmethod
    def gzip_job_result(job_result: JobResult) -> bytes:
        raw = JobResultFactory._job_result_to_json_bytes(job_result)
        gzip_buffer = BytesIO()
        with gzip.GzipFile(fileobj=gzip_buffer, mode="wb") as gzip_file:
            gzip_file.write(raw)
        return gzip_buffer.getvalue()

    @staticmethod
    def job_not_found(job_id: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            status="not_found",
            error="Job not found.",
        )
