from fast_task_api.fast_task_api import FastTaskAPI
from fast_task_api.core.job.job_progress import JobProgress
from fast_task_api.core.job.job_result import FileModel, JobResult
from media_toolkit import MediaFile, ImageFile, AudioFile, VideoFile, MediaList, MediaDict

try:
    import importlib.metadata as metadata
except ImportError:
    # For Python < 3.8
    import importlib_metadata as metadata

try:
    __version__ = metadata.version("fast-task-api")
except Exception:
    __version__ = "0.0.0"

__all__ = ["FastTaskAPI", "JobProgress", "FileModel", "JobResult", "MediaFile", "ImageFile", "AudioFile", "VideoFile", "MediaList", "MediaDict"]
