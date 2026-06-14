from pydantic import BaseModel, AnyUrl, model_validator
from typing import Union, Optional


class FileModel(BaseModel):
    """
    Wire-format mirror of a media-toolkit file: name, content type and the
    content itself as base64 or URL. Plain strings (URL / base64) and
    media-toolkit file objects are coerced automatically so the model can be
    used directly as a field type in request and response schemas.
    """

    file_name: str
    content_type: str
    content: Union[str, bytes, AnyUrl]  # base64 encoded or url
    max_size_mb: Optional[float] = 4000

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, value):
        if isinstance(value, (str, bytes)):
            return {"file_name": "file", "content_type": "application/octet-stream", "content": value}
        if isinstance(value, FileModel):
            # Re-validate a (sibling/parent) FileModel against the concrete field type.
            return value if isinstance(value, cls) else value.model_dump()
        if not isinstance(value, (dict, BaseModel)) and hasattr(value, "to_json"):
            # media-toolkit files serialize to {file_name, content_type, content}.
            return value.to_json()
        return value

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


class ThreeDFileModel(FileModel):
    class Config:
        json_schema_extra = {
            "x-media-type": "3DFile",
            "example": {
                "file_name": "example.glb",
                "content_type": "model/gltf-binary",
                "content": "base64 encoded 3D model data",
            }
        }
