"""
Registry and backend-neutral helpers for standardized (OpenAI-compatible) schemas.

``SCHEMA_REGISTRY`` is the single source of truth: it maps every supported
request schema to how it is served (response model + semantic tag). Everything
else derives from it:

- :func:`resolve_request_model` / :func:`get_schema_binding` detect schema
  endpoints by annotation (used by routers and signature policies),
- module-level helpers prepare the incoming request (validation + nested-media
  parsing) and wrap raw endpoint results into response models.

Schemas are a stable part of APIPod, not an optional extension. Routers keep the
normal decorator pipeline; schema helpers are called from the existing standard,
task and streaming decorators where needed.

Note: response ``id`` fields are intentionally NOT populated here — identifiers
belong to the ``JobResult`` envelope produced by the platform, not to the model
schemas themselves.
"""

import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import UnionType
from typing import Any, Callable, Iterable, Iterator, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel
from media_toolkit import MediaFile

from apipod.common.schemas import *
from apipod.engine.files.base_file_mixin import parse_schema_media_fields


@dataclass(frozen=True)
class SchemaEndpointSpec:
    """How a registered request schema is served."""
    response_model: Type
    tag: str


# Single source of truth for all standardized schemas. Adding a schema here
# automatically enables endpoint detection, body-parameter policies and
# response wrapping everywhere.
SCHEMA_REGISTRY: dict[Type, SchemaEndpointSpec] = {
    ChatCompletionRequest:      SchemaEndpointSpec(ChatCompletionResponse,      "chat"),
    CompletionRequest:          SchemaEndpointSpec(CompletionResponse,          "completion"),
    EmbeddingRequest:           SchemaEndpointSpec(EmbeddingResponse,           "embedding"),
    ImageGenerationRequest:     SchemaEndpointSpec(ImageGenerationResponse,     "image_generation"),
    VideoGenerationRequest:     SchemaEndpointSpec(VideoGenerationResponse,     "video_generation"),
    TranscriptionRequest:       SchemaEndpointSpec(TranscriptionResponse,       "transcription"),
    SpeechRequest:              SchemaEndpointSpec(SpeechResponse,              "speech"),
    CreateVoiceRequest:         SchemaEndpointSpec(VoiceResponse,               "voice"),
    VoiceConversionRequest:     SchemaEndpointSpec(VoiceConversionResponse,     "voice_conversion"),
    Generation3DRequest:        SchemaEndpointSpec(Generation3DResponse,        "generation_3d"),
    VisionRequest:              SchemaEndpointSpec(VisionResponse,              "vision"),
    MultimodalEmbeddingRequest: SchemaEndpointSpec(MultimodalEmbeddingResponse, "embedding_multimodal"),
}

# Tags whose streams are token deltas (served as server-sent events). Media
# tags stream raw encoded bytes instead, matching OpenAI's stream_format="audio".
SSE_STREAM_TAGS = {"chat", "completion", "transcription"}


@dataclass(frozen=True)
class SchemaBinding:
    """Resolved binding between an endpoint function parameter and a registered schema."""
    param_name: str
    request_model: Type
    response_model: Type
    tag: str


def _registry_spec(request_model: Type) -> Optional[SchemaEndpointSpec]:
    """Find the registry spec for a request model, honoring subclasses of registered schemas."""
    if not inspect.isclass(request_model):
        return None
    for cls in request_model.__mro__:
        if cls in SCHEMA_REGISTRY:
            return SCHEMA_REGISTRY[cls]
    return None


def resolve_request_model(annotation: Any) -> Optional[Type]:
    """
    Resolve an annotation to a registered request schema (or a subclass of one).
    Supports direct annotations and Optional/Union annotations.
    """
    candidates = [annotation]
    if get_origin(annotation) in (Union, UnionType):
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]

    for candidate in candidates:
        if _registry_spec(candidate) is not None:
            return candidate
    return None


def get_schema_binding(func: Callable) -> Optional[SchemaBinding]:
    """
    Detect the schema-typed parameter of an endpoint function by annotation — the
    parameter may have any name, mirroring how ``JobProgress`` is detected.

    Returns ``None`` for plain (non-schema) functions. When a schema parameter is
    found, the rest of the signature is validated: a schema endpoint must take the
    request schema as its only user input (``job_progress`` and framework
    dependencies aside), so that the whole request lives in the schema body.
    """
    for param in inspect.signature(func).parameters.values():
        request_model = resolve_request_model(param.annotation)
        if request_model is None:
            continue

        spec = _registry_spec(request_model)
        binding = SchemaBinding(
            param_name=param.name,
            request_model=request_model,
            response_model=spec.response_model,
            tag=spec.tag,
        )
        _validate_schema_endpoint_signature(func, binding)
        return binding
    return None


def _is_injected_param(param: inspect.Parameter) -> bool:
    """True for parameters the framework supplies (job progress, self/cls)."""
    return (
        param.name in ("self", "cls", "job_progress")
        or "JobProgress" in str(param.annotation)
    )


def _validate_schema_endpoint_signature(func: Callable, binding: SchemaBinding) -> None:
    """Reject schema endpoints that declare user parameters besides the request schema."""
    extra = [
        param.name
        for param in inspect.signature(func).parameters.values()
        if param.name != binding.param_name and not _is_injected_param(param)
    ]
    if extra:
        raise TypeError(
            f"Schema endpoint '{func.__name__}' must take the request schema "
            f"'{binding.request_model.__name__}' as its only parameter, but also declares "
            f"{extra}. Move these inputs into the schema (or a subclass of it)."
        )


def prepare_schema_call(binding: SchemaBinding, call_kwargs: dict):
    """
    Replace the raw request body in call kwargs with a validated schema object
    and return that object.
    """
    payload = call_kwargs.get(binding.param_name)
    if payload is None:
        raise ValueError(
            f"Request body for parameter '{binding.param_name}' "
            f"({binding.request_model.__name__}) is missing"
        )
    
    # Prepare the request object
    if isinstance(payload, binding.request_model):
        request = payload
    elif isinstance(payload, dict):
        request = binding.request_model.model_validate(payload)
    else:
        raise ValueError(f"Invalid payload for {binding.request_model.__name__}: {type(payload).__name__}")
    request = parse_schema_media_fields(request)

    call_kwargs[binding.param_name] = request
    return request


def wrap_schema_response(result: Any, binding: SchemaBinding) -> Any:
    """
    Wrap a raw endpoint result into the response model of the binding.

    Response models share a uniform envelope (``created``, ``model``) where
    applicable. Raw results (strings, vectors, media files, dicts) are normalized
    into the target response model.
    """
    response_model = binding.response_model
    if isinstance(result, response_model):
        return result

    payload = _normalize_response_model(result, response_model)
    payload = _apply_envelope(payload, response_model)
    return response_model.model_validate(payload)


def _normalize_response_model(result: Any, response_model: Type) -> dict:
    """
    Normalize raw endpoint results into a dictionary matching the response model.
    Handles convenient raw shapes for chat, completion, embedding and transcription.
    """
    if response_model is ChatCompletionResponse and isinstance(result, str):
        result = {"choices": [{"index": 0, "message": {"content": result}, "finish_reason": "stop"}]}
    elif response_model is CompletionResponse and isinstance(result, str):
        result = {"choices": [{"text": result, "index": 0, "finish_reason": "stop"}]}
    elif response_model is EmbeddingResponse and isinstance(result, list):
        vectors = result if result and isinstance(result[0], (list, tuple)) else [result]
        result = {"data": [{"embedding": list(vector), "index": i} for i, vector in enumerate(vectors)]}
    elif response_model is TranscriptionResponse and isinstance(result, str):
        result = {"text": result}

    return _normalize_result(result, response_model)


def _normalize_result(result: Any, response_model: Type) -> dict:
    """Bring raw results into dict form; media files are lifted into the `data` list."""
    if isinstance(result, MediaFile):
        return {"data": [result]}
    if isinstance(result, (list, tuple)) and result and all(isinstance(item, MediaFile) for item in result):
        return {"data": list(result)}
    if isinstance(result, dict):
        return dict(result)
    raise ValueError(
        f"Cannot wrap a result of type {type(result).__name__} into {response_model.__name__}. "
        f"Return a {response_model.__name__}, a dict matching its fields, or a media-toolkit file."
    )


def _apply_envelope(payload: dict, response_model: Type) -> dict:
    """Fill the shared response envelope (``created`` timestamp, request ``model``) when applicable."""
    fields = response_model.model_fields
    if "created" in fields and payload.get("created") is None:
        payload["created"] = int(datetime.now(timezone.utc).timestamp())

    return payload


def iter_media_chunks(media_file: MediaFile, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    """Yield the encoded bytes of a media file in chunks (for raw media streaming)."""
    buffer = media_file.to_bytes_io()
    while chunk := buffer.read(chunk_size):
        yield chunk


# ----------------------------------------------------------------------------
# Streaming: wrap raw token deltas into standardized chunk SSE events
# ----------------------------------------------------------------------------

SSE_DONE = "data: [DONE]\n\n"


def _to_sse(chunk: BaseModel) -> str:
    """Serialize a chunk model into a single server-sent-event data line."""
    return f"data: {chunk.model_dump_json()}\n\n"


def _build_chat_chunk(chunk_id: str, created: int, content: Optional[str], finish_reason: Optional[str]) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chunk_id,
        created=created,
        choices=[ChatStreamChoice(index=0, delta=ChatDelta(content=content), finish_reason=finish_reason)],
    )


@dataclass(frozen=True)
class StreamChunkSpec:
    """How raw token deltas of a streaming schema are wrapped into chunk events."""
    id_prefix: str
    build: Callable[[str, int, Optional[str], Optional[str]], BaseModel]


# Schema tags whose token stream is wrapped into a standardized chunk SSE stream.
# Endpoints for these tags may simply yield text tokens; APIPod owns the envelope.
STREAM_CHUNK_SPECS: dict[str, StreamChunkSpec] = {
    "chat": StreamChunkSpec(id_prefix="chatcmpl", build=_build_chat_chunk),
}


class SchemaStreamSerializer:
    """
    Turns the raw token deltas yielded by a streaming schema endpoint into its
    standardized chunk SSE stream (e.g. ``ChatCompletionChunk``).

    The endpoint only yields text tokens; APIPod owns the envelope: a stable
    chunk id, the ``created`` timestamp and the ``object`` discriminator are
    generated out of the box, closed by a final delta and the ``[DONE]`` sentinel.
    """

    def __init__(self, binding: SchemaBinding):
        spec = STREAM_CHUNK_SPECS.get(binding.tag)
        if spec is None:
            raise ValueError(f"Schema '{binding.tag}' does not support streaming chunks.")
        self._build = spec.build
        self._chunk_id = f"{spec.id_prefix}-{uuid.uuid4().hex[:8]}"
        self._created = int(datetime.now(timezone.utc).timestamp())

    def delta(self, content: str) -> str:
        """Serialize a content token into a chunk SSE event."""
        return _to_sse(self._build(self._chunk_id, self._created, content, None))

    def finish(self) -> str:
        """Serialize the closing chunk (``finish_reason='stop'``)."""
        return _to_sse(self._build(self._chunk_id, self._created, None, "stop"))

    def stream(self, tokens: Iterable[str]) -> Iterator[str]:
        """Wrap a synchronous token iterable into the full chunk SSE stream (deltas, closing chunk, [DONE])."""
        for token in tokens:
            yield self.delta(token)
        yield self.finish()
        yield SSE_DONE
