"""vLLM-backed presets. Weights are served by a ``vllm serve`` subprocess."""
from apipod.models.vllm.chat import VLLMChat

__all__ = ["VLLMChat"]
