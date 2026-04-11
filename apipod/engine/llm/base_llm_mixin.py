from types import UnionType
from typing import Callable, Type, Any, get_args, get_origin, Union
import uuid
from datetime import datetime, timezone
import inspect
from apipod.common.schemas import ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse, EmbeddingRequest, EmbeddingResponse, ChatCompletionChoice, CompletionChoice, EmbeddingData


class _BaseLLMMixin:
    """
    Mixin class for Base LLM functionality
    """

    def __init__(self, *args, **kwargs):
        self._llm_configs = {
            ChatCompletionRequest: (ChatCompletionResponse, "chat"),
            CompletionRequest: (CompletionResponse, "completion"),
            EmbeddingRequest: (EmbeddingResponse, "embedding"),
        }
        self._supported_llm_request_models = tuple(self._llm_configs.keys())

    def _resolve_supported_llm_request_model(self, annotation: Any):
        """Resolve direct/optional request model annotations to a supported LLM request model."""
        if annotation in self._llm_configs:
            return annotation

        origin = get_origin(annotation)
        if origin in (Union, UnionType):
            for arg in get_args(annotation):
                if arg is type(None):
                    continue
                if arg in self._llm_configs:
                    return arg

        return None

    def _get_llm_config(self, func: Callable):
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            req_model = self._resolve_supported_llm_request_model(param.annotation)
            if req_model is not None:
                res_model, endpoint_type = self._llm_configs[req_model]
                return req_model, res_model, endpoint_type
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
        ts, uid = int(datetime.now(timezone.utc)), uuid.uuid4().hex[:8]

        if endpoint_type == "chat":
            return response_model(
                id=f"chatcmpl-{ts}-{uid}",
                object="chat.completion",
                created=ts,
                model=model_name,
                choices=[ChatCompletionChoice(**choice) for choice in result["choices"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "completion":
            return response_model(
                id=f"cmpl-{ts}-{uid}",
                object="text.completion",
                created=ts,
                model=model_name,
                choices=[CompletionChoice(**choice) for choice in result["choices"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "embedding":
            return response_model(
                object="embedding",
                data=[EmbeddingData(**data) for data in result["data"]],
                model=model_name,
                usage=result.get("usage")
            )
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")