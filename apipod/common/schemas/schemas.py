"""
Standard request / response schemas for APIPod services.

The shape of every schema in this file mirrors the OpenAI API so that
clients written against the OpenAI SDK (or any OpenAI-compatible tool)
can talk to an APIPod service without translation. That choice is about
the wire format; it does NOT imply the schemas are tied to OpenAI's own
models. Any provider (Flux, Stable Diffusion, ElevenLabs, Whisper, Suno,
DeepSeek, etc.) plugs into the same schemas and the routing layer
dispatches to whatever runs behind it.

`model` is optional on every request: an APIPod service typically serves
exactly one model, so forcing clients to repeat its name adds nothing.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from .media_files import FileModel, ImageFileModel, AudioFileModel, VideoFileModel, ThreeDFileModel


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

    Media fields are declared with the FileModel variants (ImageFileModel,
    AudioFileModel, ...) which accept uploads, FileModel JSON objects, URLs
    and base64 strings. At runtime the file-handling layer replaces them
    with parsed media-toolkit objects before the endpoint function runs.

    Serialization omits optional fields that are ``None`` so wire JSON matches
    OpenAI-style responses (no ``"model": null`` or ``"usage": null`` keys).
    Pass ``exclude_none=False`` to :meth:`model_dump` / :meth:`model_dump_json`
    when you need explicit nulls.
    """

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


# =====================================================
# Chat Completions - Input schemas
# =====================================================


class ChatMessage(APIPodSchemaBase):
    role: Literal["system", "user", "assistant"] = Field(description="The role of the message author (system, user, or assistant).")
    content: str = Field(description="The content of the message.")


class ChatCompletionRequest(APIPodSchemaBase):
    messages: List[ChatMessage] = Field(description="A list of messages comprising the conversation history.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

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
    prompt: Union[str, List[str]] = Field(description="The prompt(s) to generate completions for.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

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
    input: Union[str, List[str]] = Field(description="The input text to embed.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user.")


# =====================================================
# Image Generation - Input schemas
# =====================================================

class ImageGenerationRequest(APIPodSchemaBase):
    prompt: str = Field(description="A text description of the desired image(s).")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    negative_prompt: Optional[str] = Field(default=None, description="A text description of what to exclude from the generated image.")
    image: Optional[ImageFileModel] = Field(default=None, description="An optional reference image for image-to-image or inpainting tasks.")
    mask: Optional[ImageFileModel] = Field(default=None, description="An optional mask image for inpainting, where white pixels indicate areas to edit.")
    size: Optional[str] = Field(default=None, description="The size of the generated images, e.g. 1024x1024.")
    num_images: int = Field(default=1, description="The number of images to generate.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    steps: Optional[int] = Field(default=None, description="The number of inference steps to perform.")


# =====================================================
# Video Generation - Input schemas
# =====================================================

class VideoGenerationRequest(APIPodSchemaBase):
    prompt: str = Field(description="A text description of the desired video.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    image: Optional[ImageFileModel] = Field(default=None, description="An optional reference image (frame0) to start the video from.")
    duration_s: float = Field(default=5.0, description="The desired duration of the video in seconds.")
    fps: int = Field(default=24, description="Frames per second for the generated video.")
    aspect_ratio: Optional[str] = Field(default=None, description="The aspect ratio of the generated video, e.g. 16:9.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")
    stream: bool = Field(default=False, description="If set, the generated video is streamed back as raw bytes instead of a JSON response.")


# =====================================================
# Audio - Input schemas (transcription, speech, voices)
# =====================================================

class TranscriptionRequest(APIPodSchemaBase):
    """Speech-to-text. Mirrors OpenAI POST /audio/transcriptions ('file' is called 'audio' here)."""

    audio: AudioFileModel = Field(description="The audio file to transcribe. Accepts uploads, FileModel objects, URLs or base64.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: only specify if you serve multiple models.")

    language: Optional[str] = Field(default=None, description="The language of the input audio in ISO-639-1 format (e.g. 'en'). Improves accuracy and latency.")
    prompt: Optional[str] = Field(default=None, description="Optional text to guide the model's style or to continue a previous audio segment.")
    response_format: Literal["json", "verbose_json"] = Field(default="json", description="Format of the transcript output.")
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(default=None, description="Timestamp detail to include; requires response_format='verbose_json'.")
    stream: bool = Field(default=False, description="If set, partial transcript deltas will be sent as server-sent events.")


class SpeechRequest(APIPodSchemaBase):
    """Text-to-speech. Mirrors OpenAI POST /audio/speech."""

    input: str = Field(description="The text to generate audio for.")
    voice: Optional[Union[str, AudioFileModel]] = Field(default=None, description="A named voice of the service OR a reference audio / voice embedding file for voice cloning.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    instructions: Optional[str] = Field(default=None, description="Additional instructions to control voice, emotion or style.")
    response_format: Literal["mp3", "wav", "opus", "flac", "pcm"] = Field(default="mp3", description="The desired output audio format.")
    speed: float = Field(default=1.0, description="Playback speed of the generated audio (0.25 to 4.0).")
    stream: bool = Field(default=False, description="If set, the generated audio is streamed back as raw audio chunks instead of a JSON response.")


class CreateVoiceRequest(APIPodSchemaBase):
    """Voice cloning: create a reusable voice (embedding) from an audio sample. Mirrors OpenAI POST /audio/voices."""

    name: str = Field(description="A name for the new voice.")
    audio_sample: AudioFileModel = Field(description="A clean speech sample (~5-20s) of the voice to clone.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")


class VoiceConversionRequest(APIPodSchemaBase):
    """Voice-to-voice: re-render existing audio with another voice (named, sample or embedding)."""

    audio: AudioFileModel = Field(description="The audio whose content should be kept.")
    voice: Union[str, AudioFileModel] = Field(description="The target voice: a named voice of the service OR a reference audio / voice embedding file.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")


# =====================================================
# 3D Generation - Input schemas
# =====================================================

class Generation3DRequest(APIPodSchemaBase):
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    prompt: Optional[str] = Field(default=None, description="A text description of the 3D object to generate.")
    image: Optional[ImageFileModel] = Field(default=None, description="An optional reference image to generate the 3D object from.")
    output_format: str = Field(default="glb", description="The desired 3D output format (glb, obj, fbx, etc.).")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation.")


# =====================================================
# Vision - Input schemas (classify, detect, OCR)
# =====================================================

class VisionRequest(APIPodSchemaBase):
    image: ImageFileModel = Field(description="The image to process for vision tasks.")
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    labels: Optional[List[str]] = Field(default=None, description="Optional list of labels to look for (e.g. for object detection).")
    threshold: Optional[float] = Field(default=None, description="Confidence threshold for detection or classification.")
    return_boxes: bool = Field(default=False, description="Whether to return bounding box coordinates for detected objects.")


# =====================================================
# Multimodal Embedding - Input schemas
# =====================================================

class MultimodalEmbeddingRequest(APIPodSchemaBase):
    model: Optional[str] = Field(default=None, description="ID of the model to use. Optional: an APIPod service usually serves exactly one model.")

    input: Optional[Union[str, List[str]]] = Field(default=None, description="The input text to embed.")
    image: Optional[ImageFileModel] = Field(default=None, description="The input image to embed.")
    audio: Optional[AudioFileModel] = Field(default=None, description="The input audio to embed.")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user.")


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
    role: Literal["assistant"] = Field(default="assistant", description="The role of the message author, always 'assistant'.")
    content: str = Field(description="The content of the message.")


class ChatCompletionChoice(APIPodSchemaBase):
    index: int = Field(description="The index of the choice in the list of choices.")
    message: ChatCompletionMessage = Field(description="A chat completion message generated by the model.")
    finish_reason: Literal["stop", "length", "content_filter"] = Field(description="The reason the model stopped generating tokens.")


class ChatCompletionResponse(APIPodSchemaBase):
    object: Literal["chat.completion"] = Field(default="chat.completion", description="The object type, always 'chat.completion'.")
    created: int = Field(description="The Unix timestamp when the chat completion was created.")
    model: Optional[str] = Field(default=None, description="The model used for the chat completion.")
    choices: List[ChatCompletionChoice] = Field(description="A list of chat completion choices.")
    usage: Optional[Usage] = Field(default=None, description="Usage statistics for the completion request.")


# =====================================================
# Text Completion - Output schemas
# =====================================================

class CompletionChoice(APIPodSchemaBase):
    text: str = Field(description="The generated completion text.")
    index: int = Field(description="The index of the choice in the list of choices.")
    logprobs: None = None
    finish_reason: Literal["stop", "length", "content_filter"] = Field(description="The reason the model stopped generating tokens.")


class CompletionResponse(APIPodSchemaBase):
    object: Literal["text_completion"] = Field(default="text_completion", description="The object type, always 'text_completion'.")
    created: int = Field(description="The Unix timestamp when the completion was created.")
    model: Optional[str] = Field(default=None, description="The model used for the completion.")
    choices: List[CompletionChoice] = Field(description="A list of completion choices.")
    usage: Optional[Usage] = Field(default=None, description="Usage statistics for the completion request.")


# =====================================================
# Embedding - Output schemas
# =====================================================

class EmbeddingData(APIPodSchemaBase):
    object: Literal["embedding"] = Field(default="embedding", description="The object type, always 'embedding'.")
    embedding: List[float] = Field(description="The embedding vector.")
    index: int = Field(description="The index of the embedding in the list of inputs.")


class EmbeddingResponse(APIPodSchemaBase):
    object: Literal["list"] = Field(default="list", description="The object type, always 'list'.")
    data: List[EmbeddingData] = Field(description="The list of embedding data objects.")
    model: Optional[str] = Field(default=None, description="The model used for generating embeddings.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Image Generation - Output schemas
# =====================================================

class ImageGenerationResponse(APIPodSchemaBase):
    object: Literal["image_generation"] = Field(default="image_generation", description="The object type, always 'image_generation'.")
    created: int = Field(description="The Unix timestamp when the generation was created.")
    model: Optional[str] = Field(default=None, description="The model used for generation.")
    data: List[ImageFileModel] = Field(description="The generated image data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Video Generation - Output schemas
# =====================================================

class VideoGenerationResponse(APIPodSchemaBase):
    object: Literal["video_generation"] = Field(default="video_generation", description="The object type, always 'video_generation'.")
    created: int = Field(description="The Unix timestamp when the generation was created.")
    model: Optional[str] = Field(default=None, description="The model used for generation.")
    data: List[VideoFileModel] = Field(description="The generated video data.")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


# =====================================================
# Audio - Output schemas (transcription, speech, voices)
# =====================================================

class TranscriptionWord(APIPodSchemaBase):
    word: str = Field(description="The transcribed word.")
    start: float = Field(description="Start time of the word in seconds.")
    end: float = Field(description="End time of the word in seconds.")


class TranscriptionSegment(APIPodSchemaBase):
    id: int = Field(default=0, description="Index of the segment.")
    start: float = Field(description="Start time of the segment in seconds.")
    end: float = Field(description="End time of the segment in seconds.")
    text: str = Field(description="The transcribed text of the segment.")


class TranscriptionResponse(APIPodSchemaBase):
    """Mirrors the OpenAI transcription object: plain `text` plus optional verbose details."""
    text: str = Field(description="The transcribed text.")
    language: Optional[str] = Field(default=None, description="The detected or requested language of the audio.")
    duration: Optional[float] = Field(default=None, description="Duration of the input audio in seconds.")
    segments: Optional[List[TranscriptionSegment]] = Field(default=None, description="Segment-level details (verbose_json).")
    words: Optional[List[TranscriptionWord]] = Field(default=None, description="Word-level timestamps (verbose_json).")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


class AudioEnvelopeBase(APIPodSchemaBase):
    """Shared envelope for endpoints that return audio files (speech synthesis, voice conversion)."""
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: Optional[str] = Field(default=None, description="The model used for the task.")
    data: List[AudioFileModel] = Field(description="The resulting audio file(s).")
    usage: Optional[Usage] = Field(default=None, description="Token usage information, if applicable.")


class SpeechResponse(AudioEnvelopeBase):
    object: Literal["audio.speech"] = Field(default="audio.speech", description="The object type, always 'audio.speech'.")


class VoiceConversionResponse(AudioEnvelopeBase):
    object: Literal["audio.conversion"] = Field(default="audio.conversion", description="The object type, always 'audio.conversion'.")


class VoiceResponse(APIPodSchemaBase):
    """A created (cloned) voice, optionally carrying its embedding file (e.g. a SpeechCraft .npz)."""
    id: str = Field(description="Unique identifier of the voice.")
    object: Literal["audio.voice"] = Field(default="audio.voice", description="The object type, always 'audio.voice'.")
    name: str = Field(description="The name of the voice.")
    created: int = Field(description="The Unix timestamp when the voice was created.")
    model: Optional[str] = Field(default=None, description="The model used to create the voice.")
    embedding: Optional[FileModel] = Field(default=None, description="The voice embedding file, if the service exposes it for reuse.")


# =====================================================
# 3D Generation - Output schemas
# =====================================================

class Generation3DResponse(APIPodSchemaBase):
    object: Literal["generation_3d"] = Field(default="generation_3d", description="The object type, always 'generation_3d'.")
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: Optional[str] = Field(default=None, description="The model used for generation.")
    data: List[ThreeDFileModel] = Field(description="The generated 3D data.")
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
    object: Literal["vision"] = Field(default="vision", description="The object type, always 'vision'.")
    created: int = Field(description="The Unix timestamp when the response was created.")
    model: Optional[str] = Field(default=None, description="The model used for the vision task.")
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
    object: Literal["list"] = Field(default="list", description="The object type, always 'list'.")
    data: List[MultimodalEmbeddingData] = Field(description="The list of embedding data objects.")
    model: Optional[str] = Field(default=None, description="The model used for generating embeddings.")
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
    model: Optional[str] = None
    choices: List[ChatStreamChoice]
