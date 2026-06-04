from types import UnionType
from typing import Callable, Type, Any, get_args, get_origin, Union
import uuid
from datetime import datetime, timezone
import inspect
from apipod.common.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    CompletionRequest, CompletionResponse, CompletionChoice,
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    ImageGenerationRequest, ImageGenerationResponse, ImageGenerationData,
    VideoGenerationRequest, VideoGenerationResponse, VideoGenerationData,
    AudioRequest, AudioResponse, AudioData,
    Generation3DRequest, Generation3DResponse, Generation3DData,
    VisionRequest, VisionResponse, VisionData,
    MultimodalEmbeddingRequest, MultimodalEmbeddingResponse, MultimodalEmbeddingData,
)


class _BaseLLMMixin:
    """
    Mixin class for Base LLM functionality
    """

    def __init__(self, *args, **kwargs):
        self._llm_configs = {
            ChatCompletionRequest:        (ChatCompletionResponse,        "chat"),
            CompletionRequest:            (CompletionResponse,            "completion"),
            EmbeddingRequest:             (EmbeddingResponse,             "embedding"),
            ImageGenerationRequest:       (ImageGenerationResponse,       "image_generation"),
            VideoGenerationRequest:       (VideoGenerationResponse,       "video_generation"),
            AudioRequest:                 (AudioResponse,                 "audio"),
            Generation3DRequest:          (Generation3DResponse,          "generation_3d"),
            VisionRequest:                (VisionResponse,                "vision"),
            MultimodalEmbeddingRequest:   (MultimodalEmbeddingResponse,   "embedding_multimodal"),
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
        ts, uid = int(datetime.now(timezone.utc).timestamp()), uuid.uuid4().hex[:8]

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
            # FRICTION #15 fix: the outer object on EmbeddingResponse is "list"
            # (each item in data carries object="embedding"). The schema declares
            # object: Literal["list"]; passing "embedding" here used to fail
            # Pydantic validation, which is why qwen-models constructs the
            # response by hand. After this fix, the auto-wrap path is safe.
            return response_model(
                object="list",
                data=[EmbeddingData(**data) for data in result["data"]],
                model=model_name,
                usage=result.get("usage")
            )
        elif endpoint_type == "image_generation":
            return response_model(
                id=f"imggen-{ts}-{uid}",
                object="image_generation",
                created=ts,
                model=model_name,
                data=[ImageGenerationData(**item) for item in result["data"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "video_generation":
            return response_model(
                id=f"vidgen-{ts}-{uid}",
                object="video_generation",
                created=ts,
                model=model_name,
                data=[VideoGenerationData(**item) for item in result["data"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "audio":
            return response_model(
                id=f"aud-{ts}-{uid}",
                object="audio",
                created=ts,
                model=model_name,
                data=[AudioData(**item) for item in result["data"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "generation_3d":
            return response_model(
                id=f"gen3d-{ts}-{uid}",
                object="generation_3d",
                created=ts,
                model=model_name,
                data=[Generation3DData(**item) for item in result["data"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "vision":
            return response_model(
                id=f"vis-{ts}-{uid}",
                object="vision",
                created=ts,
                model=model_name,
                data=[VisionData(**item) for item in result["data"]],
                usage=result.get("usage")
            )
        elif endpoint_type == "embedding_multimodal":
            return response_model(
                object="list",
                data=[MultimodalEmbeddingData(**item) for item in result["data"]],
                model=model_name,
                usage=result.get("usage")
            )
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")