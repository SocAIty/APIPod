import gzip
from io import BytesIO
from typing import Optional, Union, List, Any

from pydantic import BaseModel, AnyUrl

from fast_task_api.compatibility.upload import is_param_media_toolkit_file
from fast_task_api.core.job.base_job import JOB_STATUS, BaseJob
from fast_task_api.settings import DEFAULT_DATE_TIME_FORMAT
from media_toolkit import MediaList


class FileModel(BaseModel):
    file_name: str
    content_type: str
    content: Union[str, AnyUrl]  # base64 encoded or url
    max_size_mb: Optional[float] = 4000

    class Config:
        schema_extra = {
            "x-media-type": "MediaFile",
            "example": {
                "file_name": "example.csv",
                "content_type": "text/csv",
                "content": "https://example.com/example.csv"   # bytes, base64 encoded or url
            }
        }


class ImageFileModel(FileModel):
    class Config:
        schema_extra = {
            "x-media-type": "ImageFile",
            "example": {
                "file_name": "example.png",
                "content_type": "image/png",
                "content": "base64 encoded image data"
            }
        }


class AudioFileModel(FileModel):
    class Config:
        schema_extra = {
            "x-media-type": "AudioFile",
            "example": {
                "file_name": "example.mp3",
                "content_type": "audio/mpeg",
                "content": "base64 encoded audio data"
            }
        }


class VideoFileModel(FileModel):
    class Config:
        schema_extra = {
            "x-media-type": "VideoFile",
            "example": {
                "file_name": "example.mp4",
                "content_type": "video/mp4",
                "content": "base64 encoded video data"
            }
        }


class JobProgress(BaseModel):
    progress: float = 0.0
    message: Optional[str] = None


class JobResult(BaseModel):
    """
    When the user (client) sends a request to an Endpoint, a ClientJob is created.
    This job contains the information about the request and the response.
    """
    id: str
    status: Optional[str] = None
    progress: Optional[JobProgress] = None
    error: Optional[str] = None
    result: Union[FileModel, List[FileModel], List, str, Any, None] = None
    refresh_job_url: Optional[str] = None
    cancel_job_url: Optional[str] = None

    created_at: Optional[str] = None
    queued_at: Optional[str] = None
    execution_started_at: Optional[str] = None
    execution_finished_at: Optional[str] = None

    endpoint_protocol: Optional[str] = "socaity"


class JobResultFactory:

    @staticmethod
    def _serialize_result(data: Any) -> Union[FileModel, List[FileModel], List, str, None]:
        def convert_media_list(media_list: MediaList) -> List:
            """Convert a MediaList to a list of FileModels or original items"""
            json_list = media_list.to_json()
            return [
                FileModel(**item) if isinstance(item, dict) else item
                for item in json_list
            ]

        # Handle single MediaList or MediaFile
        if is_param_media_toolkit_file(data):
            if isinstance(data, MediaList):
                return convert_media_list(data)
            return FileModel(**data.to_json())

        # Handle list of MediaLists/MediaFiles/other items
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, MediaList):
                    result.append(convert_media_list(item))
                elif is_param_media_toolkit_file(item):
                    result.append(FileModel(**item.to_json()))
                else:
                    result.append(item)
            return result

        return data

    @staticmethod
    def from_base_job(ij: BaseJob) -> JobResult:
        def format_date(date):
            return date.strftime(DEFAULT_DATE_TIME_FORMAT) if date else None

        created_at = format_date(ij.created_at)
        queued_at = format_date(ij.queued_at)
        execution_started_at = format_date(ij.execution_started_at)
        execution_finished_at = format_date(ij.execution_finished_at)

        # if the internal job returned a media-toolkit file, convert it to a json serializable FileModel
        result = ij.result
        result = JobResultFactory._serialize_result(result)

        # Job_status is an Enum, convert it to a string to return it as json
        status = ij.status
        if isinstance(status, JOB_STATUS):
            status = status.value

        try:
            jp = JobProgress(progress=ij.job_progress._progress, message=ij.job_progress._message)
        except Exception:
            jp = JobProgress(progress=0.0, message='')

        return JobResult(
            id=ij.id,
            status=status,
            progress=jp,
            error=ij.error,
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
            error="Job not found."
        )

