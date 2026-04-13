"""
Local API Server using APIPod - Optimized for AMD ROCm
"""

from contextlib import asynccontextmanager
from typing import List, Union
from datetime import datetime
import time
import json

from apipod import APIPod
from apipod.common import schemas
from fastapi import FastAPI
import uuid
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ============================================================================
# Model Wrapper (Optimized for ROCm)
# ============================================================================

class LocalSmallModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading {model_name}...")
        
        # Detect AMD GPU with ROCm
        self.device = self._detect_device()
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"ROCm GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Qwen requires setting pad_token for batching if it's missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use float16 for GPU (ROCm supports it well)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def _detect_device(self) -> str:
        """Detect if ROCm/CUDA GPU is available"""
        if torch.cuda.is_available():
            # This works for both NVIDIA CUDA and AMD ROCm
            # PyTorch with ROCm reports as 'cuda'
            return "cuda"
        else:
            print("Warning: No GPU detected. Falling back to CPU.")
            return "cpu"

    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 512) -> str:
        # Standardize messages to list of dicts
        chat = [
            msg.model_dump() if hasattr(msg, 'model_dump') else 
            msg.__dict__ if hasattr(msg, '__dict__') else msg 
            for msg in messages
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        
        safe_max_tokens = max_tokens if max_tokens is not None else 512
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=safe_max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_len = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()
    
    def stream_generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 512):
        """Streaming version of the generation logic."""
        chat = [
            msg.model_dump() if hasattr(msg, 'model_dump') else 
            msg.__dict__ if hasattr(msg, '__dict__') else msg 
            for msg in messages
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        # skip_prompt=True is critical; otherwise, you'll stream back the user's question
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens or 512,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Run generation in a separate thread to prevent blocking the generator
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            # Hugging Face streamer often yields empty strings at the start/end
            if new_text:
                yield new_text

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Batch processing for embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        # Batch tokenize
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Simple Mean Pooling
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
        return embeddings.cpu().numpy().tolist()

# ============================================================================
# Application State
# ============================================================================

class AppState:
    model: LocalSmallModel | None = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model
    state.model = LocalSmallModel("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Warmup inference (prevents first-request lag)
    print("Performing warmup inference...")
    try:
        state.model.generate([{"role": "user", "content": "ping"}], max_tokens=1)
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# Logic Handlers (Pure Python)
# ============================================================================
def chat_logic(payload: schemas.ChatCompletionRequest):
    """Non-streaming chat completion"""
    state.model = LocalSmallModel("Qwen/Qwen2.5-0.5B-Instruct")
    if state.model is None:
        raise RuntimeError("Model not initialized")

    response_text = state.model.generate(
        messages=payload.messages,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens or 512
    )
    
    # Calculate usage
    input_str = "".join([m.content for m in payload.messages])
    prompt_tokens = len(state.model.tokenizer.encode(input_str))
    completion_tokens = len(state.model.tokenizer.encode(response_text))
    
    return schemas.ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=payload.model or "Qwen/Qwen2.5-0.5B-Instruct",
        choices=[
            schemas.ChatCompletionChoice(
                index=0,
                message=schemas.ChatCompletionMessage(  # ✅ Correct type
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
    if state.model is None:
        raise RuntimeError("Model not initialized")

    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    embeddings_list = state.model.get_embeddings(texts)
    total_tokens = sum(len(state.model.tokenizer.encode(t)) for t in texts)

    return schemas.EmbeddingResponse(
        object="list",
        model=payload.model or "Qwen/Qwen2.5-0.5B-Instruct",
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
            completion_tokens=0,  # Embeddings don't have completion tokens
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
            state.model = LocalSmallModel("Qwen/Qwen2.5-0.5B-Instruct")
            if state.model is None:
                raise RuntimeError("Model not initialized")
            
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(datetime.now().timestamp())
            model_name = payload.model or "Qwen/Qwen2.5-0.5B-Instruct"
            
            for token in state.model.stream_generate(
                messages=payload.messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens
            ):
                # Use ChatCompletionChunk schema
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
            
            # Final chunk with finish_reason
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


@app.endpoint(path="/stream")
def stream_endpoint():
    def simple_stream():
        try:
            for i in range(10):
                message = {"index": i, "text": f"Message {i}"}
                yield f"data: {json.dumps(message)}\n\n"
                time.sleep(1)
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            print("Client disconnected")
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return simple_stream()


if __name__ == "__main__":
    app.start()