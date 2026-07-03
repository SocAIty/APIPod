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

from pydantic import BaseModel, Field, create_model
from pydantic.json_schema import SkipJsonSchema
from media_toolkit import MediaFile

from socaity_schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatStreamChoice,
    CompletionRequest,
    CompletionResponse,
    CreateVoiceRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    Generation3DRequest,
    Generation3DResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VisionRequest,
    VisionResponse,
    VoiceConversionRequest,
    VoiceConversionResponse,
    VoiceResponse,
)
from apipod.engine.files.base_file_mixin import parse_schema_media_fields
from apipod.engine.signatures.analysis import is_injected_progress_param


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


_OPENAPI_WITHOUT_STREAM_CACHE: dict[type, type] = {}
_OPENAPI_DOC_TO_SOURCE: dict[type, type] = {}


def source_request_model(model: type[BaseModel]) -> type[BaseModel]:
    """Return the author-facing schema for an OpenAPI doc model, or *model* itself."""
    return _OPENAPI_DOC_TO_SOURCE.get(model, model)


def openapi_request_model(model: type[BaseModel], *, is_streaming: bool) -> type[BaseModel]:
    """Return *model*, or a cached OpenAPI variant that hides ``stream`` when not streaming.

    When ``is_streaming`` is false and the model declares a ``stream`` field, a sibling
    model is built via :func:`pydantic.create_model` with ``SkipJsonSchema`` on ``stream``.
    The field stays in validation (OpenAI clients may send ``stream: false``) but is
    omitted from the generated OpenAPI/JSON schema.
    """
    if is_streaming or "stream" not in model.model_fields:
        return model
    if model in _OPENAPI_WITHOUT_STREAM_CACHE:
        return _OPENAPI_WITHOUT_STREAM_CACHE[model]

    stream_field = model.model_fields["stream"]
    fields = {}
    for name, field in model.model_fields.items():
        if name == "stream":
            fields[name] = (
                SkipJsonSchema[bool],
                Field(default=False, description=stream_field.description),
            )
        else:
            fields[name] = (field.annotation, field)

    doc_model = create_model(
        f"{model.__name__}OpenAPI",
        __config__=model.model_config,
        **fields,
    )
    _OPENAPI_WITHOUT_STREAM_CACHE[model] = doc_model
    _OPENAPI_DOC_TO_SOURCE[doc_model] = model
    return doc_model


def openapi_schema_annotation(annotation: Any, *, is_streaming: bool) -> Any:
    """Apply :func:`openapi_request_model` to a parameter annotation (incl. Optional)."""
    resolved = resolve_request_model(annotation)
    if resolved is None:
        return annotation

    doc_model = openapi_request_model(resolved, is_streaming=is_streaming)
    if doc_model is resolved:
        return annotation

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = get_args(annotation)
        new_args = tuple(
            doc_model if resolve_request_model(arg) is not None else arg
            for arg in args
        )
        return origin[new_args]
    return doc_model


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
    return param.name in ("self", "cls") or is_injected_progress_param(param)


def _validate_schema_endpoint_signature(func: Callable, binding: SchemaBinding) -> None:
    """Reject schema endpoints that declare user parameters besides the request schema."""
    from apipod.engine.signatures.policies import FastAPISignaturePolicies

    extra = [
        param.name
        for param in inspect.signature(func).parameters.values()
        if param.name != binding.param_name
        and not _is_injected_param(param)
        and not FastAPISignaturePolicies.is_fastapi_dependency(param)
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


# Empty values for required JSON-schema property types (`created` is filled by the envelope).
_SCHEMA_EMPTY = {"string": "", "array": [], "integer": 0, "object": {}}


def _schema_defaults(model: Type[BaseModel]) -> dict:
    """Required-field empties from pydantic's JSON schema."""
    schema = model.model_json_schema()
    properties = schema.get("properties") or {}
    # Only top-level required fields; nested item shapes are validated by pydantic later.
    return {
        name: _SCHEMA_EMPTY[prop["type"]]
        for name in schema.get("required") or []
        if name != "created" and (prop := properties.get(name)) and prop.get("type") in _SCHEMA_EMPTY
    }


def _normalize_response_model(result: Any, response_model: Type) -> dict:
    """
    Normalize raw endpoint results into a dictionary matching the response model.
    Handles convenient raw shapes for chat, completion, embedding and transcription.
    """
    # None → empty text or dict; dict gaps are filled from schema defaults below.
    if result is None:
        result = "" if response_model in (ChatCompletionResponse, CompletionResponse, TranscriptionResponse) else {}

    # Shorthand raw returns authors may use instead of a full response dict.
    if response_model is ChatCompletionResponse and isinstance(result, str):
        result = {"choices": [{"index": 0, "message": {"content": result}, "finish_reason": "stop"}]}
    elif response_model is CompletionResponse and isinstance(result, str):
        result = {"choices": [{"text": result, "index": 0, "finish_reason": "stop"}]}
    elif response_model is EmbeddingResponse and isinstance(result, list):
        if not result:
            result = {"data": []}
        elif isinstance(result[0], (list, tuple)):
            result = {"data": [{"embedding": list(vector), "index": i} for i, vector in enumerate(result)]}
        else:
            result = {"data": [{"embedding": list(result), "index": 0}]}
    elif response_model is TranscriptionResponse and isinstance(result, str):
        result = {"text": result}

    payload = _normalize_result(result, response_model)
    if not isinstance(payload, dict):
        return payload
    # Defaults first, then author payload; drop None so defaults apply.
    return {**_schema_defaults(response_model), **{k: v for k, v in payload.items() if v is not None}}


def _normalize_result(result: Any, response_model: Type) -> dict:
    """Bring raw results into dict form; media files are lifted into the `data` list."""
    # Single file or file list → standard media envelope shape.
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
