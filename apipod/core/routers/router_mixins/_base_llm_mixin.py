from typing import Callable, Type, Any
import uuid
import time
import inspect
from apipod.core.routers import schemas


class _BaseLLMMixin:
    """
    Mixin class for Base LLM functionality
    """

    def __init__(self, *args, **kwargs):
        self._llm_configs = {
            schemas.ChatCompletionRequest: (schemas.ChatCompletionResponse, "chat"),
            schemas.CompletionRequest: (schemas.CompletionResponse, "completion"),
            schemas.EmbeddingRequest: (schemas.EmbeddingResponse, "embedding"),
        }
    
    def _get_llm_config(self, func: Callable):
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.annotation in self._llm_configs:
                res_model, endpoint_type = self._llm_configs[param.annotation]
                return param.annotation, res_model, endpoint_type
        return None, None, None

    def _prepare_llm_payload(self, req_model: Type, payload: Any) -> Any:
        """
        Prepare the LLM request payload.
        """
        if isinstance(payload, req_model):
            return payload
        elif isinstance(payload, dict):
            return req_model.model_validate(payload)
        else:
            raise ValueError(f"Invalid payload type for {req_model}: {type(payload)}")

    def _wrap_llm_response(self, result: Any, response_model: Type, endpoint_type: str, openai_req: Any) -> Any:
        """
        Wrap the raw result into the appropriate LLM response model.
        """
        if isinstance(result, response_model):
            return result
        
        model_name = getattr(openai_req, "model", "unknown-model")
        ts, uid = int(time.time()), uuid.uuid4().hex[:8]

        if endpoint_type == "chat":
            return response_model(
                id=f"chatcmpl-{ts}-{uid}",
                object="chat.completion",
                created=ts,
                model=model_name,
                choices=[schemas.ChatCompletionChoice(**choice) for choice in result["choices"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "completion":
            return response_model(
                id=f"cmpl-{ts}-{uid}",
                object="text.completion",
                created=ts,
                model=model_name,
                choices=[schemas.CompletionChoice(**choice) for choice in result["choices"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "embedding":
            return response_model(
                object="embedding",
                data=[schemas.EmbeddingData(**data) for data in result["data"]],
                model=model_name,
                usage=result.get("usage")
            )
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")