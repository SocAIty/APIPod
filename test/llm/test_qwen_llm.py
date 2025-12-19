"""
Local API Server using APIPod - Optimized
"""

from contextlib import asynccontextmanager
from typing import List, Union

from apipod import APIPod
from apipod.core.routers import schemas
from fastapi import FastAPI, Body
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Model Wrapper (Optimized)
# ============================================================================

class LocalSmallModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Qwen requires setting pad_token for batching if it's missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
    state.model = LocalSmallModel("Qwen/Qwen2.5-0.5B-Instruct")
    if state.model is None:
        raise RuntimeError("Model not initialized")

    response_text = state.model.generate(
        messages=payload.messages,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens or 512
    )
    
    # Calculate usage (approximation)
    input_str = "".join([getattr(m, 'content', str(m)) for m in payload.messages])
    prompt_tokens = len(state.model.tokenizer.encode(input_str))
    completion_tokens = len(state.model.tokenizer.encode(response_text))
    
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

def embeddings_logic(payload: schemas.EmbeddingRequest):
    if state.model is None:
        raise RuntimeError("Model not initialized")

    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    
    # Use the new batched method
    embeddings_list = state.model.get_embeddings(texts)
    
    # Calculate tokens for usage stats
    total_tokens = sum(len(state.model.tokenizer.encode(t)) for t in texts)
    
    return {
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings_list)
        ],
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

# ============================================================================
# API Setup
# ============================================================================

app = APIPod(
    backend="runpod", 
    lifespan=lifespan, 
    queue_backend="redis", 
    redis_url="redis://localhost:6379/0"
)

@app.endpoint(path="/chat", use_queue=False)
def chat_endpoint(payload: schemas.ChatCompletionRequest = Body(...)):
    return chat_logic(payload)

@app.endpoint(path="/embeddings", use_queue=True)
def embeddings_endpoint(payload: schemas.EmbeddingRequest = Body(...)):
    return embeddings_logic(payload)

if __name__ == "__main__":
    app.start()