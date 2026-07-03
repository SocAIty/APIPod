"""Schema service: endpoints for the standardized (OpenAI-compatible) schemas.

- ``register_all``       one endpoint per registered request schema.
- ``register_extended``  an endpoint whose input extends a schema with a field.
- ``register_mapping``   raw + typed return endpoint for each mapping CASE.

``CASES`` drives the response-model passthrough vs. raw-value normalization
tests (text, embedding and media-envelope schemas).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

from media_toolkit import AudioFile, ImageFile, VideoFile, media_from_any

from apipod.common import schemas
from apipod.engine.backend.schema_resolve import SCHEMA_REGISTRY

_FILES_DIR = Path(__file__).resolve().parent.parent / "files"


def tag_path(request_model: Type) -> str:
    """Public route path for a schema (APIPod normalizes ``_`` to ``-``)."""
    return "/" + SCHEMA_REGISTRY[request_model].tag.replace("_", "-")


def _load_test_media(file_name: str, media_type: Type):
    """Read a fixture from ``test/files`` into a media-toolkit object."""
    return media_from_any(str(_FILES_DIR / file_name), type_hint=media_type)


def _schema_endpoint(
    request_model: Type,
    return_value: Any,
    name: str,
    *,
    media_loader: Callable[[], Any] | None = None,
    typed_response: Type | None = None,
):
    """Build an endpoint whose only parameter is annotated with ``request_model``.

    When ``media_loader`` is set, the return value is produced at request time
    (raw media file, or a typed response model built from that file).
    """
    def endpoint(request):
        if media_loader is not None:
            media = media_loader()
            if typed_response is not None:
                return typed_response(created=0, data=[media])
            return media
        return return_value

    endpoint.__name__ = name
    endpoint.__annotations__ = {"request": request_model}
    return endpoint


def register_all(app):
    for request_model, spec in SCHEMA_REGISTRY.items():
        app.endpoint(tag_path(request_model))(_schema_endpoint(request_model, {}, f"ep_{spec.tag}"))


class ChatRequestPlus(schemas.ChatCompletionRequest):
    persona: str = "pirate"


def register_extended(app):
    @app.endpoint("/chat-extended")
    def chat_extended(request: ChatRequestPlus):
        # The extra `persona` input field is parsed and drives the answer.
        return f"[{request.persona}] {request.messages[-1].content}"


@dataclass
class SchemaCase:
    request_model: Type
    response_model: Type
    payload: dict
    raw: Any = None
    typed: Any = None
    media_file: str | None = None
    media_type: Type | None = None


CASES = [
    SchemaCase(
        schemas.ChatCompletionRequest, schemas.ChatCompletionResponse,
        payload={"messages": [{"role": "user", "content": "hi"}]},
        raw="hello there",
        typed=schemas.ChatCompletionResponse(
            created=0,
            choices=[schemas.ChatCompletionChoice(
                index=0, message=schemas.ChatCompletionMessage(content="hello there"), finish_reason="stop")],
        ),
    ),
    SchemaCase(
        schemas.CompletionRequest, schemas.CompletionResponse,
        payload={"prompt": "hi"},
        raw="completed",
        typed=schemas.CompletionResponse(
            created=0,
            choices=[schemas.CompletionChoice(text="completed", index=0, finish_reason="stop")],
        ),
    ),
    SchemaCase(
        schemas.EmbeddingRequest, schemas.EmbeddingResponse,
        payload={"input": "hi"},
        raw=[0.1, 0.2, 0.3],
        typed=schemas.EmbeddingResponse(data=[schemas.EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)]),
    ),
    SchemaCase(
        schemas.ImageGenerationRequest, schemas.ImageGenerationResponse,
        payload={"prompt": "a cat"},
        media_file="test_image.png",
        media_type=ImageFile,
    ),
    SchemaCase(
        schemas.VideoGenerationRequest, schemas.VideoGenerationResponse,
        payload={"prompt": "a cat"},
        media_file="test_video.mp4",
        media_type=VideoFile,
    ),
    SchemaCase(
        schemas.SpeechRequest, schemas.SpeechResponse,
        payload={"input": "hello"},
        media_file="test_audio.wav",
        media_type=AudioFile,
    ),
    SchemaCase(
        schemas.Generation3DRequest, schemas.Generation3DResponse,
        payload={"prompt": "a chair"}, raw={"data": []},
        typed=schemas.Generation3DResponse(created=0, data=[]),
    ),
    SchemaCase(
        schemas.MultimodalEmbeddingRequest, schemas.MultimodalEmbeddingResponse,
        {"input": "hi"}, raw={"data": []},
        typed=schemas.MultimodalEmbeddingResponse(data=[]),
    ),
]


def register_mapping(app):
    for case in CASES:
        tag = SCHEMA_REGISTRY[case.request_model].tag.replace("_", "-")
        if case.media_file is not None:
            loader = lambda f=case.media_file, t=case.media_type: _load_test_media(f, t)
            app.endpoint(f"/{tag}-raw")(
                _schema_endpoint(case.request_model, None, f"{tag}_raw", media_loader=loader)
            )
            app.endpoint(f"/{tag}-typed")(
                _schema_endpoint(
                    case.request_model, None, f"{tag}_typed",
                    media_loader=loader, typed_response=case.response_model,
                )
            )
        else:
            app.endpoint(f"/{tag}-raw")(_schema_endpoint(case.request_model, case.raw, f"{tag}_raw"))
            app.endpoint(f"/{tag}-typed")(_schema_endpoint(case.request_model, case.typed, f"{tag}_typed"))
        app.endpoint(f"/{tag}-none")(_schema_endpoint(case.request_model, None, f"{tag}_none"))
