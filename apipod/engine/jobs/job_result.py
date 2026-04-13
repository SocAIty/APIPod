import gzip
from io import BytesIO
from typing import Any, List, Optional, Union

from pydantic import BaseModel, AnyUrl

from apipod.common.settings import DEFAULT_DATE_TIME_FORMAT
from apipod.engine.jobs.base_job import JOB_STATUS, BaseJob
from apipod.engine.signatures.upload import is_param_media_toolkit_file
from media_toolkit import IMediaContainer
from media_toolkit.utils.data_type_utils import is_file_model_dict


class FileModel(BaseModel):
    file_name: str
    content_type: str
    content: Union[str, AnyUrl]  # base64 encoded or url
    max_size_mb: Optional[float] = 4000

    class Config:
        json_schema_extra = {
            "x-media-type": "MediaFile",
            "example": {
                "file_name": "example.csv",
                "content_type": "text/csv",
                "content": "https://example.com/example.csv",
            }
        }


class ImageFileModel(FileModel):
    class Config:
        json_schema_extra = {
            "x-media-type": "ImageFile",
            "example": {
                "file_name": "example.png",
                "content_type": "image/png",
                "content": "base64 encoded image data",
            }
        }


class AudioFileModel(FileModel):
    class Config:
        json_schema_extra = {
            "x-media-type": "AudioFile",
            "example": {
                "file_name": "example.mp3",
                "content_type": "audio/mpeg",
                "content": "base64 encoded audio data",
            }
        }


class VideoFileModel(FileModel):
    class Config:
        json_schema_extra = {
            "x-media-type": "VideoFile",
            "example": {
                "file_name": "example.mp4",
                "content_type": "video/mp4",
                "content": "base64 encoded video data",
            }
        }


def _job_status_to_public(status: Any) -> Optional[str]:
    """Map internal JOB_STATUS (or legacy string) to public API strings (gateway-aligned)."""
    if status is None:
        return None
    if isinstance(status, JOB_STATUS):
        return {
            JOB_STATUS.QUEUED: "pending",
            JOB_STATUS.PROCESSING: "processing",
            JOB_STATUS.FINISHED: "completed",
            JOB_STATUS.FAILED: "failed",
            JOB_STATUS.TIMEOUT: "failed",
        }.get(status, status.value.lower())
    if isinstance(status, str):
        lowered = status.lower()
        legacy = {
            "queued": "pending",
            "pending": "pending",
            "processing": "processing",
            "finished": "completed",
            "completed": "completed",
            "failed": "failed",
            "timeout": "failed",
            "rejected": "failed",
        }
        return legacy.get(lowered, lowered)
    return str(status)


def _format_date(date: Any) -> Optional[str]:
    """Format a datetime or ISO string for the public API."""
    if date is None:
        return None
    if isinstance(date, str):
        return date
    try:
        return date.strftime(DEFAULT_DATE_TIME_FORMAT)
    except Exception:
        return str(date)


def _parse_iso(value: Any):
    """Best-effort parse an ISO 8601 string or datetime into a datetime."""
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return value
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _compute_duration_s(start: Any, end: Any) -> Optional[float]:
    """Compute seconds between two timestamps, returning None if either is missing."""
    s, e = _parse_iso(start), _parse_iso(end)
    if s is None or e is None:
        return None
    delta = (e - s).total_seconds()
    return round(delta, 2) if delta >= 0 else None


def _opt_float(value: Any) -> Optional[float]:
    """Coerce to positive float, else None."""
    if value is None:
        return None
    try:
        f = float(value)
        return round(f, 3) if f > 0 else None
    except (ValueError, TypeError):
        return None


class JobLinks(BaseModel):
    """Hypermedia links for job status polling, cancellation, and streaming."""

    status: Optional[str] = None
    cancel: Optional[str] = None
    stream: Optional[str] = None


class JobMetrics(BaseModel):
    """Performance metrics populated as a job progresses through the platform.

    Segments (chronological):
        upload_time_s          – file upload duration (gateway)
        platform_queue_time_s  – our validation + dispatch + Celery routing
        provider_queue_time_s  – provider-side GPU / resource wait
        inference_time_s       – actual model execution
        execution_time_s       – orchestrator end-to-end (queue + inference, excludes upload)
    """

    execution_time_s: Optional[float] = None
    inference_time_s: Optional[float] = None
    platform_queue_time_s: Optional[float] = None
    provider_queue_time_s: Optional[float] = None
    upload_time_s: Optional[float] = None


class JobResult(BaseModel):
    """Public job snapshot returned by GET /status and job submissions.

    Unified response: same shape whether the client just submitted a job
    (``status="pending"``) or is polling for completion.

    Null fields are excluded from the serialized response so the client
    only receives relevant information for the current job state.
    """

    job_id: str
    status: Optional[str] = None
    result: Union[FileModel, List[FileModel], list, str, Any, None] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    message: Optional[str] = None

    service: Optional[str] = None
    endpoint: Optional[str] = None

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

        if isinstance(data, list):
            return [JobResultFactory._serialize_result(item) for item in data]

        if isinstance(data, dict):
            return {
                key: JobResultFactory._serialize_result(value)
                for key, value in data.items()
            }

        return data

    @staticmethod
    def from_base_job(job: BaseJob) -> JobResult:
        """Map any :class:`BaseJob` subclass to the public :class:`JobResult`.

        Works for :class:`~apipod.engine.jobs.base_job.LocalJob` (thread queue)
        and any platform ``BaseJob`` subclass (e.g. gateway ``ServiceJob``).
        """
        status = _job_status_to_public(job.status)
        result = JobResultFactory._serialize_result(job.result)

        progress = job.progress
        message = job.message

        job_progress = getattr(job, "job_progress", None)
        if job_progress is not None:
            try:
                progress = float(job_progress._progress)
                message = job_progress._message
            except Exception:
                pass

        service = getattr(job, "service_id", None)
        endpoint = getattr(job, "endpoint", None)

        metrics = JobResultFactory._build_metrics(job)

        links = JobLinks(
            status=f"/status/{job.id}",
            cancel=f"/cancel/{job.id}",
            stream=f"/stream/{job.id}",
        )

        return JobResult(
            job_id=job.id,
            status=status,
            error=job.error,
            result=result,
            progress=progress,
            message=message,
            service=service,
            endpoint=endpoint,
            metrics=metrics,
            links=links,
        )

    @staticmethod
    def _build_metrics(job: BaseJob) -> Optional[JobMetrics]:
        """Derive timing metrics from orchestrator-provided values or timestamps."""
        execution_time_s = _opt_float(getattr(job, "execution_time_s", None)) or _compute_duration_s(
            getattr(job, "created_at", None),
            getattr(job, "completed_at", None) or getattr(job, "failed_at", None),
        )
        upload_time_s = _compute_duration_s(
            getattr(job, "upload_started_at", None),
            getattr(job, "upload_finished_at", None),
        )
        inference_time_s = _opt_float(getattr(job, "inference_time_s", None))
        platform_queue_time_s = _opt_float(getattr(job, "platform_queue_time_s", None))
        provider_queue_time_s = _opt_float(getattr(job, "provider_queue_time_s", None))

        values = (execution_time_s, upload_time_s, inference_time_s,
                  platform_queue_time_s, provider_queue_time_s)
        if all(v is None for v in values):
            return None

        return JobMetrics(
            execution_time_s=execution_time_s,
            inference_time_s=inference_time_s,
            platform_queue_time_s=platform_queue_time_s,
            provider_queue_time_s=provider_queue_time_s,
            upload_time_s=upload_time_s,
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
