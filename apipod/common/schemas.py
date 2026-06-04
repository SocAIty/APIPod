from typing import List, Optional, Union, Literal
from pydantic import BaseModel

from media_toolkit import ImageFile, AudioFile

# =====================================================
# Base schema
# =====================================================

class OpenAIBaseModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
        # Allow media_toolkit file types (ImageFile, AudioFile, VideoFile, MediaFile)
        # as field annotations. APIPod's file_handling_mixin converts URL/base64
        # inputs into these types before request validation.
        "arbitrary_types_allowed": True,
    }


# =====================================================
# Chat Completions - Input schemas
# =====================================================

class ChatMessage(OpenAIBaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(OpenAIBaseModel):
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

class CompletionRequest(OpenAIBaseModel):
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

class EmbeddingRequest(OpenAIBaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


# =====================================================
# Image Generation - Input schemas
# =====================================================

class ImageGenerationRequest(OpenAIBaseModel):
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

class VideoGenerationRequest(OpenAIBaseModel):
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

class AudioRequest(OpenAIBaseModel):
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

class Generation3DRequest(OpenAIBaseModel):
    model: str

    prompt: Optional[str] = None
    image: Optional[ImageFile] = None
    output_format: str = "glb"
    seed: Optional[int] = None


# =====================================================
# Vision - Input schemas (classify, detect, OCR)
# =====================================================

class VisionRequest(OpenAIBaseModel):
    model: str
    image: ImageFile

    labels: Optional[List[str]] = None
    threshold: Optional[float] = None
    return_boxes: bool = False


# =====================================================
# Multimodal Embedding - Input schemas
# =====================================================

class MultimodalEmbeddingRequest(OpenAIBaseModel):
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

class Usage(OpenAIBaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int


# =====================================================
# Chat Completions - Output schemas
# =====================================================

class ChatCompletionMessage(OpenAIBaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(OpenAIBaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter"]


class ChatCompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# =====================================================
# Text Completion - Output schemas
# =====================================================

class CompletionChoice(OpenAIBaseModel):
    text: str
    index: int
    logprobs: None = None
    finish_reason: Literal["stop", "length", "content_filter"]


class CompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# =====================================================
# Embedding - Output schemas
# =====================================================

class EmbeddingData(OpenAIBaseModel):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingResponse(OpenAIBaseModel):
    object: Literal["list"]
    data: List[EmbeddingData]
    model: str
    usage: Usage


# =====================================================
# Image Generation - Output schemas
# =====================================================

class ImageGenerationData(OpenAIBaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    seed: Optional[int] = None


class ImageGenerationResponse(OpenAIBaseModel):
    id: str
    object: Literal["image_generation"]
    created: int
    model: str
    data: List[ImageGenerationData]
    usage: Optional[Usage] = None


# =====================================================
# Video Generation - Output schemas
# =====================================================

class VideoGenerationData(OpenAIBaseModel):
    url: Optional[str] = None
    duration_s: Optional[float] = None
    seed: Optional[int] = None


class VideoGenerationResponse(OpenAIBaseModel):
    id: str
    object: Literal["video_generation"]
    created: int
    model: str
    data: List[VideoGenerationData]
    usage: Optional[Usage] = None


# =====================================================
# Audio - Output schemas
# =====================================================

class AudioData(OpenAIBaseModel):
    audio: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    duration_s: Optional[float] = None


class AudioResponse(OpenAIBaseModel):
    id: str
    object: Literal["audio"]
    created: int
    model: str
    data: List[AudioData]
    usage: Optional[Usage] = None


# =====================================================
# 3D Generation - Output schemas
# =====================================================

class Generation3DData(OpenAIBaseModel):
    url: Optional[str] = None
    output_format: Optional[str] = None
    seed: Optional[int] = None


class Generation3DResponse(OpenAIBaseModel):
    id: str
    object: Literal["generation_3d"]
    created: int
    model: str
    data: List[Generation3DData]
    usage: Optional[Usage] = None


# =====================================================
# Vision - Output schemas (classify, detect, OCR)
# =====================================================

class VisionLabel(OpenAIBaseModel):
    label: str
    score: float
    box: Optional[List[float]] = None


class VisionData(OpenAIBaseModel):
    labels: List[VisionLabel] = []
    text: Optional[str] = None


class VisionResponse(OpenAIBaseModel):
    id: str
    object: Literal["vision"]
    created: int
    model: str
    data: List[VisionData]
    usage: Optional[Usage] = None


# =====================================================
# Multimodal Embedding - Output schemas
# =====================================================

class MultimodalEmbeddingData(OpenAIBaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int
    modality: Optional[Literal["text", "image", "audio"]] = None


class MultimodalEmbeddingResponse(OpenAIBaseModel):
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