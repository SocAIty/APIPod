"""
Standard request / response schemas for APIPod services.

The shape of every schema in this file mirrors the OpenAI API so that
clients written against the OpenAI SDK (or any OpenAI-compatible tool)
can talk to an APIPod service without translation. That choice is about
the wire format; it does NOT imply the schemas are tied to OpenAI's own
models. Any provider (Flux, Stable Diffusion, ElevenLabs, Whisper, Suno,
DeepSeek, etc.) plugs into the same schemas and the routing layer
dispatches to whatever runs behind it.
"""

from types import UnionType
from typing import Any, List, Literal, Optional, Union, get_args, get_origin
from pydantic import BaseModel, model_validator

from media_toolkit import AudioFile, ImageFile, MediaFile, VideoFile, media_from_any

_MEDIA_FIELD_TYPES = (ImageFile, AudioFile, VideoFile, MediaFile)


def _media_field_type(annotation: Any) -> Optional[type]:
    """Media class if annotation is media-typed (bare, Optional, or Union), else None."""
    if annotation in _MEDIA_FIELD_TYPES:
        return annotation
    if get_origin(annotation) in (Union, UnionType):
        for arg in get_args(annotation):
            if arg in _MEDIA_FIELD_TYPES:
                return arg
    return None


# =====================================================
# Base schema
# =====================================================

class APIPodSchemaBase(BaseModel):
    """
    Base for every APIPod request / response model.

    Shape follows the OpenAI API spec so OpenAI-compatible clients work
    out of the box, but the schemas are provider-agnostic: anything that
    matches the shape (Flux, SD, ElevenLabs, Whisper, in-house models...)
    is a valid backend.
    """

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
        # ImageFile / AudioFile / VideoFile / MediaFile are non-BaseModel types
        # used as field annotations; the pre-validator below converts URL /
        # base64 strings into instances before per-field validation.
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="before")
    @classmethod
    def _convert_media_strings(cls, data: Any) -> Any:
        """URL / data URI through media_from_any, bare base64 through from_base64."""
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            value = data[field_name]
            if not isinstance(value, str):
                continue
            media_type = _media_field_type(field_info.annotation)
            if media_type is None:
                continue
            if value.startswith(("http://", "https://", "data:")):
                data[field_name] = media_from_any(
                    data=value,
                    type_hint=media_type,
                    use_temp_file=True,
                    temp_dir=None,
                    allow_reads_from_disk=False,
                )
            else:
                data[field_name] = media_type().from_base64(value)
        return data


# =====================================================
# Chat Completions - Input schemas
# =====================================================

class ChatMessage(APIPodSchemaBase):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(APIPodSchemaBase):
    model: str
    messages: List[ChatMessage]

    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None


# =====================================================
# Text Completion - Input schemas
# =====================================================

class CompletionRequest(APIPodSchemaBase):
    model: str
    prompt: Union[str, List[str]]

    temperature: float = 0.7
    max_tokens: int = 16
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


# =====================================================
# Embedding - Input schemas
# =====================================================

class EmbeddingRequest(APIPodSchemaBase):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


# =====================================================
# Image Generation - Input schemas
# =====================================================

class ImageGenerationRequest(APIPodSchemaBase):
    model: str
    prompt: str

    negative_prompt: Optional[str] = None
    image: Optional[ImageFile] = None
    mask: Optional[ImageFile] = None
    size: Optional[str] = None
    num_images: int = 1
    seed: Optional[int] = None
    steps: Optional[int] = None


# =====================================================
# Video Generation - Input schemas
# =====================================================

class VideoGenerationRequest(APIPodSchemaBase):
    model: str
    prompt: str

    image: Optional[ImageFile] = None
    duration_s: float = 5.0
    fps: int = 24
    aspect_ratio: Optional[str] = None
    seed: Optional[int] = None


# =====================================================
# Audio - Input schemas (TTS, STT, music)
# =====================================================

class AudioRequest(APIPodSchemaBase):
    model: str

    text: Optional[str] = None
    audio: Optional[AudioFile] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    format: Optional[str] = None
    duration_s: Optional[float] = None


# =====================================================
# 3D Generation - Input schemas
# =====================================================

class Generation3DRequest(APIPodSchemaBase):
    model: str

    prompt: Optional[str] = None
    image: Optional[ImageFile] = None
    output_format: str = "glb"
    seed: Optional[int] = None


# =====================================================
# Vision - Input schemas (classify, detect, OCR)
# =====================================================

class VisionRequest(APIPodSchemaBase):
    model: str
    image: ImageFile

    labels: Optional[List[str]] = None
    threshold: Optional[float] = None
    return_boxes: bool = False


# =====================================================
# Multimodal Embedding - Input schemas
# =====================================================

class MultimodalEmbeddingRequest(APIPodSchemaBase):
    model: str

    input: Optional[Union[str, List[str]]] = None
    image: Optional[ImageFile] = None
    audio: Optional[AudioFile] = None
    user: Optional[str] = None


# Supported request schemas that should be interpreted as JSON bodies
# by router decorators, even when endpoint authors do not specify Body(...).
SUPPORTED_LLM_REQUEST_SCHEMAS = (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
    AudioRequest,
    Generation3DRequest,
    VisionRequest,
    MultimodalEmbeddingRequest,
)


# =====================================================
# Shared output schemas
# =====================================================

class Usage(APIPodSchemaBase):
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int


# =====================================================
# Chat Completions - Output schemas
# =====================================================

class ChatCompletionMessage(APIPodSchemaBase):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(APIPodSchemaBase):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter"]


class ChatCompletionResponse(APIPodSchemaBase):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# =====================================================
# Text Completion - Output schemas
# =====================================================

class CompletionChoice(APIPodSchemaBase):
    text: str
    index: int
    logprobs: None = None
    finish_reason: Literal["stop", "length", "content_filter"]


class CompletionResponse(APIPodSchemaBase):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# =====================================================
# Embedding - Output schemas
# =====================================================

class EmbeddingData(APIPodSchemaBase):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingResponse(APIPodSchemaBase):
    object: Literal["list"]
    data: List[EmbeddingData]
    model: str
    usage: Usage


# =====================================================
# Image Generation - Output schemas
# =====================================================

class ImageGenerationData(APIPodSchemaBase):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    seed: Optional[int] = None


class ImageGenerationResponse(APIPodSchemaBase):
    id: str
    object: Literal["image_generation"]
    created: int
    model: str
    data: List[ImageGenerationData]
    usage: Optional[Usage] = None


# =====================================================
# Video Generation - Output schemas
# =====================================================

class VideoGenerationData(APIPodSchemaBase):
    url: Optional[str] = None
    duration_s: Optional[float] = None
    seed: Optional[int] = None


class VideoGenerationResponse(APIPodSchemaBase):
    id: str
    object: Literal["video_generation"]
    created: int
    model: str
    data: List[VideoGenerationData]
    usage: Optional[Usage] = None


# =====================================================
# Audio - Output schemas
# =====================================================

class AudioData(APIPodSchemaBase):
    audio: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    duration_s: Optional[float] = None


class AudioResponse(APIPodSchemaBase):
    id: str
    object: Literal["audio"]
    created: int
    model: str
    data: List[AudioData]
    usage: Optional[Usage] = None


# =====================================================
# 3D Generation - Output schemas
# =====================================================

class Generation3DData(APIPodSchemaBase):
    url: Optional[str] = None
    output_format: Optional[str] = None
    seed: Optional[int] = None


class Generation3DResponse(APIPodSchemaBase):
    id: str
    object: Literal["generation_3d"]
    created: int
    model: str
    data: List[Generation3DData]
    usage: Optional[Usage] = None


# =====================================================
# Vision - Output schemas (classify, detect, OCR)
# =====================================================

class VisionLabel(APIPodSchemaBase):
    label: str
    score: float
    box: Optional[List[float]] = None


class VisionData(APIPodSchemaBase):
    labels: List[VisionLabel] = []
    text: Optional[str] = None


class VisionResponse(APIPodSchemaBase):
    id: str
    object: Literal["vision"]
    created: int
    model: str
    data: List[VisionData]
    usage: Optional[Usage] = None


# =====================================================
# Multimodal Embedding - Output schemas
# =====================================================

class MultimodalEmbeddingData(APIPodSchemaBase):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int
    modality: Optional[Literal["text", "image", "audio"]] = None


class MultimodalEmbeddingResponse(APIPodSchemaBase):
    object: Literal["list"]
    data: List[MultimodalEmbeddingData]
    model: str
    usage: Optional[Usage] = None


# ======================================================
# Streaming Models
# ======================================================
class ChatDelta(BaseModel):
    content: Optional[str] = None

class ChatStreamChoice(BaseModel):
    index: int
    delta: ChatDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChoice]