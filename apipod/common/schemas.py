"""Inference schemas live in socaity-schemas. This module keeps the names the tests use."""

from socaity_schemas.public.inference.generation import (
    Generation3DRequest,
    Generation3DResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
    SpeechRequest,
    SpeechResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from socaity_schemas.public.inference.language import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
)
