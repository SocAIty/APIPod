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
from pydantic import BaseModel, Field, model_validator

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
            data[field_name] = media_from_any(
                data=value,
                type_hint=media_type,
                use_temp_file=True,
                temp_dir=None,
                allow_reads_from_disk=False,
            )
        return data


# =====================================================
# Chat Completions - Input schemas
# =====================================================

class ChatMessage(APIPodSchemaBase):
    role: Literal["system", "user", "assistant"] = Field(description="The role of the message author (system, user, or assistant).")
    content: str = Field(description="The content of the message.")


class ChatCompletionRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for this request.")
    messages: List[ChatMessage] = Field(description="A list of messages comprising the conversation history.")

    temperature: float = Field(default=0.7, description="Sampling temperature between 0.0 and 2.0. Higher values make output more random.")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate in the completion.")
    top_p: float = Field(default=1.0, description="Nucleus sampling threshold. Only tokens with cumulative probability >= top_p are considered.")
    n: int = Field(default=1, description="How many completions to generate for each prompt.")
    stream: bool = Field(default=False, description="If set, partial message deltas will be sent as server-sent events.")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Up to 4 sequences where the API will stop generating further tokens.")
    presence_penalty: float = Field(default=0.0, description="Penalty for new tokens based on whether they appear in the text so far.")
    frequency_penalty: float = Field(default=0.0, description="Penalty for new tokens based on their existing frequency in the text so far.")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user.")


# =====================================================
# Text Completion - Input schemas
# =====================================================

class CompletionRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for this request.")
    prompt: Union[str, List[str]] = Field(description="The prompt(s) to generate completions for.")

    temperature: float = Field(default=0.7, description="Sampling temperature between 0.0 and 2.0.")
    max_tokens: int = Field(default=16, description="The maximum number of tokens to generate.")
    top_p: float = Field(default=1.0, description="Nucleus sampling threshold.")
    n: int = Field(default=1, description="How many completions to generate.")
    stream: bool = Field(default=False, description="Whether to stream partial progress.")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Sequences where the API will stop generating.")


# =====================================================
# Embedding - Input schemas
# =====================================================

class EmbeddingRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for generating embeddings.")
    input: Union[str, List[str]] = Field(description="The input text to embed.")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user.")


# =====================================================
# Image Generation - Input schemas
# =====================================================

class ImageGenerationRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for image generation.")
    prompt: str = Field(description="A text description of the desired image(s).")

    negative_prompt: Optional[str] = Field(default=None, description="A text description of what to exclude from the generated image.")
    image: Optional[ImageFile] = Field(default=None, description="An optional reference image for image-to-image or inpainting tasks.")
    mask: Optional[ImageFile] = Field(default=None, description="An optional mask image for inpainting, where white pixels indicate areas to edit.")
    size: Optional[str] = Field(default=None, description="The size of the generated images, e.g. 1024x1024.")
    num_images: int = Field(default=1, description="The number of images to generate.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    steps: Optional[int] = Field(default=None, description="The number of inference steps to perform.")


# =====================================================
# Video Generation - Input schemas
# =====================================================

class VideoGenerationRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for video generation.")
    prompt: str = Field(description="A text description of the desired video.")

    image: Optional[ImageFile] = Field(default=None, description="An optional reference image (frame0) to start the video from.")
    duration_s: float = Field(default=5.0, description="The desired duration of the video in seconds.")
    fps: int = Field(default=24, description="Frames per second for the generated video.")
    aspect_ratio: Optional[str] = Field(default=None, description="The aspect ratio of the generated video, e.g. 16:9.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


# =====================================================
# Audio - Input schemas (TTS, STT, music)
# =====================================================

class AudioRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for audio tasks (TTS, STT, or music).")

    text: Optional[str] = Field(default=None, description="The input text for text-to-speech or music generation.")
    audio: Optional[AudioFile] = Field(default=None, description="The input audio file for speech-to-text tasks.")
    voice: Optional[str] = Field(default=None, description="The voice ID or style to use for audio generation.")
    language: Optional[str] = Field(default=None, description="The language of the input audio (for STT) or target language (for TTS).")
    format: Optional[str] = Field(default=None, description="The desired output audio format, e.g. mp3, wav, flac.")
    duration_s: Optional[float] = Field(default=None, description="The desired duration of the generated audio in seconds.")


# =====================================================
# 3D Generation - Input schemas
# =====================================================

class Generation3DRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for 3D generation.")

    prompt: Optional[str] = Field(default=None, description="A text description of the 3D object to generate.")
    image: Optional[ImageFile] = Field(default=None, description="An optional reference image to generate the 3D object from.")
    output_format: str = Field(default="glb", description="The desired 3D output format (glb, obj, fbx, etc.).")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


# =====================================================
# Vision - Input schemas (classify, detect, OCR)
# =====================================================

class VisionRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for vision tasks (OCR, detection, etc.).")
    image: ImageFile = Field(description="The image to process for vision tasks.")

    labels: Optional[List[str]] = Field(default=None, description="Optional list of labels to look for (e.g. for object detection).")
    threshold: Optional[float] = Field(default=None, description="Confidence threshold for detection or classification.")
    return_boxes: bool = Field(default=False, description="Whether to return bounding box coordinates for detected objects.")


# =====================================================
# Multimodal Embedding - Input schemas
# =====================================================

class MultimodalEmbeddingRequest(APIPodSchemaBase):
    model: str = Field(description="ID of the model to use for generating multimodal embeddings.")

    input: Optional[Union[str, List[str]]] = Field(default=None, description="The input text to embed.")
    image: Optional[ImageFile] = Field(default=None, description="The input image to embed.")
    audio: Optional[AudioFile] = Field(default=None, description="The input audio to embed.")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user.")


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
    prompt_tokens: int = Field(description="Number of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in the generated completion.")
    total_tokens: int = Field(description="Total number of tokens used in the request (prompt + completion).")


# =====================================================
# Chat Completions - Output schemas
# =====================================================

class ChatCompletionMessage(APIPodSchemaBase):
    role: Literal["assistant"] = Field(description="The role of the message author, always 'assistant'.")
    content: str = Field(description="The content of the message.")


class ChatCompletionChoice(APIPodSchemaBase):
    index: int = Field(description="The index of the choice in the list of choices.")
    message: ChatCompletionMessage = Field(description="A chat completion message generated by the model.")
    finish_reason: Literal["stop", "length", "content_filter"] = Field(description="The reason the model stopped generating tokens.")


class ChatCompletionResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the chat completion.")
    object: Literal["chat.completion"] = Field(description="The object type, always 'chat.completion'.")
    created: int = Field(description="The Unix timestamp when the chat completion was created.")
    model: str = Field(description="The model used for the chat completion.")
    choices: List[ChatCompletionChoice] = Field(description="A list of chat completion choices.")
    usage: Usage = Field(description="Usage statistics for the completion request.")


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
    url: Optional[str] = Field(default=None, description="The URL of the generated image.")
    b64_json: Optional[str] = Field(default=None, description="The base64-encoded JSON of the generated image.")
    revised_prompt: Optional[str] = Field(default=None, description="The prompt as revised by the model, if applicable.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class ImageGenerationResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the generation request.")
    object: Literal["image_generation"] = Field(description="The object type, always 'image_generation'.")
    created: int = Field(description="The Unix timestamp when the generation was created.")
    model: str = Field(description="The model used for generation.")
    data: List[ImageGenerationData] = Field(description="The generated image data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Video Generation - Output schemas
# =====================================================

class VideoGenerationData(APIPodSchemaBase):
    url: Optional[str] = Field(default=None, description="The URL of the generated video.")
    duration_s: Optional[float] = Field(default=None, description="The duration of the generated video in seconds.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class VideoGenerationResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the generation request.")
    object: Literal["video_generation"] = Field(description="The object type, always 'video_generation'.")
    created: int = Field(description="The Unix timestamp when the generation was created.")
    model: str = Field(description="The model used for generation.")
    data: List[VideoGenerationData] = Field(description="The generated video data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Audio - Output schemas
# =====================================================

class AudioData(APIPodSchemaBase):
    audio: Optional[str] = Field(default=None, description="The URL or base64 data of the generated audio.")
    text: Optional[str] = Field(default=None, description="The transcribed text, if applicable.")
    language: Optional[str] = Field(default=None, description="The detected language, if applicable.")
    duration_s: Optional[float] = Field(default=None, description="The duration of the audio in seconds.")


class AudioResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the request.")
    object: Literal["audio"] = Field(description="The object type, always 'audio'.")
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: str = Field(description="The model used for the task.")
    data: List[AudioData] = Field(description="The resulting audio data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# 3D Generation - Output schemas
# =====================================================

class Generation3DData(APIPodSchemaBase):
    url: Optional[str] = Field(default=None, description="The URL of the generated 3D asset.")
    output_format: Optional[str] = Field(default=None, description="The format of the generated 3D asset.")
    seed: Optional[int] = Field(default=None, description="The seed used for generation.")


class Generation3DResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the request.")
    object: Literal["generation_3d"] = Field(description="The object type, always 'generation_3d'.")
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: str = Field(description="The model used for generation.")
    data: List[Generation3DData] = Field(description="The generated 3D data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Vision - Output schemas (classify, detect, OCR)
# =====================================================

class VisionLabel(APIPodSchemaBase):
    label: str = Field(description="The label of the detected object or classification.")
    score: float = Field(description="The confidence score of the detection or classification.")
    box: Optional[List[float]] = Field(default=None, description="The bounding box coordinates [ymin, xmin, ymax, xmax], if applicable.")


class VisionData(APIPodSchemaBase):
    labels: List[VisionLabel] = Field(default_factory=list, description="A list of detected labels and scores.")
    text: Optional[str] = Field(default=None, description="The extracted text from OCR, if applicable.")


class VisionResponse(APIPodSchemaBase):
    id: str = Field(description="Unique identifier for the request.")
    object: Literal["vision"] = Field(description="The object type, always 'vision'.")
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: str = Field(description="The model used for the vision task.")
    data: List[VisionData] = Field(description="The resulting vision data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Multimodal Embedding - Output schemas
# =====================================================

class MultimodalEmbeddingData(APIPodSchemaBase):
    object: Literal["embedding"] = Field(default="embedding", description="The object type, always 'embedding'.")
    embedding: List[float] = Field(description="The embedding vector.")
    index: int = Field(description="The index of the embedding in the list of inputs.")
    modality: Optional[Literal["text", "image", "audio"]] = Field(default=None, description="The modality of the input that generated this embedding.")


class MultimodalEmbeddingResponse(APIPodSchemaBase):
    object: Literal["list"] = Field(description="The object type, always 'list'.")
    data: List[MultimodalEmbeddingData] = Field(description="The list of embedding data objects.")
    model: str = Field(description="The model used for generating embeddings.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


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