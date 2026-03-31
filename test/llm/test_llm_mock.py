"""
Mock API Server using APIPod - No heavy dependencies (No Torch/Transformers)
"""

from contextlib import asynccontextmanager
from typing import List, Union
from datetime import datetime
import time
import json
import uuid
import random

from apipod import APIPod
from apipod.common import schemas
from fastapi import FastAPI

# ============================================================================
# Mock Model Wrapper
# ============================================================================

class MockModel:
    """A mock model that returns random strings to mimic an LLM."""
    
    def __init__(self, model_name: str = "mock-model-v1"):
        print(f"Initializing Mock Model: {model_name}...")
        self.model_name = model_name
        self.responses = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "APIPod makes it easy to deploy long-running tasks.",
            "I am a mock model returning random strings for testing.",
            "Python is a versatile programming language.",
            "Deep learning is a subset of machine learning.",
            "FastAPI is high-performance and easy to use.",
        ]
        print("Mock Model initialized successfully!")

    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Simulate non-streaming generation."""
        # Just pick a random response
        return random.choice(self.responses)
    
    def stream_generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 512):
        """Simulate streaming generation by yielding words."""
        response = random.choice(self.responses)
        words = response.split()
        for i, word in enumerate(words):
            # Add a small delay to simulate processing
            time.sleep(0.05)
            # Add space between words except for the first one
            yield word + (" " if i < len(words) - 1 else "")

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Simulate embedding generation with random vectors."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Return random vectors of size 128
        return [[random.uniform(-1, 1) for _ in range(128)] for _ in texts]

# ============================================================================
# Application State
# ============================================================================

class AppState:
    model: MockModel | None = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize mock model
    state.model = MockModel("mock-model-v1")
    print("Performing mock warmup...")
    yield
    print("Shutting down mock server...")

# ============================================================================
# Logic Handlers
# ============================================================================

def chat_logic(payload: schemas.ChatCompletionRequest):
    """Non-streaming chat completion logic."""
    if state.model is None:
        raise RuntimeError("Model not initialized")

    response_text = state.model.generate(
        messages=payload.messages,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens or 512
    )
    
    # Mock token counts
    prompt_tokens = sum(len(m.content.split()) for m in payload.messages) * 2
    completion_tokens = len(response_text.split()) * 2
    
    return schemas.ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=payload.model or state.model.model_name,
        choices=[
            schemas.ChatCompletionChoice(
                index=0,
                message=schemas.ChatCompletionMessage(
                    role="assistant",
                    content=response_text
                ),
                finish_reason="stop"
            )
        ],
        usage=schemas.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

def embeddings_logic(payload: schemas.EmbeddingRequest):
    """Embedding logic."""
    if state.model is None:
        raise RuntimeError("Model not initialized")

    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    embeddings_list = state.model.get_embeddings(texts)
    
    # Mock token counts
    total_tokens = sum(len(t.split()) for t in texts) * 2

    return schemas.EmbeddingResponse(
        object="list",
        model=payload.model or state.model.model_name,
        data=[
            schemas.EmbeddingData(
                object="embedding",
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings_list)
        ],
        usage=schemas.Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens
        )
    )

# ============================================================================
# API Setup
# ============================================================================

app = APIPod(
    backend="fastapi",
    lifespan=lifespan,
)

@app.endpoint(path="/chat")
def chat_endpoint(payload: schemas.ChatCompletionRequest):
    """Chat completion endpoint with streaming support."""
    if payload.stream:
        def sse_generator():
            if state.model is None:
                state.model = MockModel("mock-model-v1")
            
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(datetime.now().timestamp())
            model_name = payload.model or state.model.model_name
            
            for token in state.model.stream_generate(
                messages=payload.messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens
            ):
                chunk = schemas.ChatCompletionChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=created_time,
                    model=model_name,
                    choices=[
                        schemas.ChatStreamChoice(
                            index=0,
                            delta=schemas.ChatDelta(content=token),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Final chunk
            final_chunk = schemas.ChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=created_time,
                model=model_name,
                choices=[
                    schemas.ChatStreamChoice(
                        index=0,
                        delta=schemas.ChatDelta(content=None),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        
        return sse_generator()
    else:
        return chat_logic(payload)

@app.endpoint(path="/embeddings")
def embeddings_endpoint(payload: schemas.EmbeddingRequest):
    return embeddings_logic(payload)

if __name__ == "__main__":
    app.start()
