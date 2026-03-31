from typing import List, Optional, Union, Literal
from pydantic import BaseModel

# =====================================================
# Base schema
# =====================================================

class OpenAIBaseModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
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


# Supported request schemas that should be interpreted as JSON bodies
# by router decorators, even when endpoint authors do not specify Body(...).
SUPPORTED_LLM_REQUEST_SCHEMAS = (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
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