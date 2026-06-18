"""Schema service: endpoints for the standardized (OpenAI-compatible) schemas.

- ``register_all``       one endpoint per registered request schema.
- ``register_extended``  an endpoint whose input extends a schema with a field.
- ``register_mapping``   raw + typed return endpoint for each mapping CASE.

``CASES`` drives the response-model passthrough vs. raw-value normalization
tests (text, embedding and media-envelope schemas).
"""

from dataclasses import dataclass
from typing import Any, Type

from apipod.common import schemas
from apipod.engine.backend.schema_resolve import SCHEMA_REGISTRY


def tag_path(request_model: Type) -> str:
    """Public route path for a schema (APIPod normalizes ``_`` to ``-``)."""
    return "/" + SCHEMA_REGISTRY[request_model].tag.replace("_", "-")


def _schema_endpoint(request_model: Type, return_value: Any, name: str):
    """Build an endpoint whose only parameter is annotated with ``request_model``."""

    def endpoint(request):
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
    raw: Any  # raw value the endpoint may return
    typed: Any  # the response_model instance the endpoint may return


CASES = [
    SchemaCase(
        schemas.ChatCompletionRequest, schemas.ChatCompletionResponse,
        {"messages": [{"role": "user", "content": "hi"}]},
        raw="hello there",
        typed=schemas.ChatCompletionResponse(
            created=0,
            choices=[schemas.ChatCompletionChoice(
                index=0, message=schemas.ChatCompletionMessage(content="hello there"), finish_reason="stop")],
        ),
    ),
    SchemaCase(
        schemas.CompletionRequest, schemas.CompletionResponse,
        {"prompt": "hi"},
        raw="completed",
        typed=schemas.CompletionResponse(
            created=0,
            choices=[schemas.CompletionChoice(text="completed", index=0, finish_reason="stop")],
        ),
    ),
    SchemaCase(
        schemas.EmbeddingRequest, schemas.EmbeddingResponse,
        {"input": "hi"},
        raw=[0.1, 0.2, 0.3],
        typed=schemas.EmbeddingResponse(data=[schemas.EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)]),
    ),
    SchemaCase(
        schemas.ImageGenerationRequest, schemas.ImageGenerationResponse,
        {"prompt": "a cat"}, raw={"data": []},
        typed=schemas.ImageGenerationResponse(created=0, data=[]),
    ),
    SchemaCase(
        schemas.VideoGenerationRequest, schemas.VideoGenerationResponse,
        {"prompt": "a cat"}, raw={"data": []},
        typed=schemas.VideoGenerationResponse(created=0, data=[]),
    ),
    SchemaCase(
        schemas.SpeechRequest, schemas.SpeechResponse,
        {"input": "hello"}, raw={"data": []},
        typed=schemas.SpeechResponse(created=0, data=[]),
    ),
    SchemaCase(
        schemas.Generation3DRequest, schemas.Generation3DResponse,
        {"prompt": "a chair"}, raw={"data": []},
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
        app.endpoint(f"/{tag}-raw")(_schema_endpoint(case.request_model, case.raw, f"{tag}_raw"))
        app.endpoint(f"/{tag}-typed")(_schema_endpoint(case.request_model, case.typed, f"{tag}_typed"))
