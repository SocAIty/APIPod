"""Transformers-backed model presets.

``Transformers`` holds what every preset shares (HF include handling, load
kwargs, chat normalization, token streaming); ``TransformersLLM`` and
``TransformersVLM`` are the concrete presets for causal chat LLMs and
vision-language models.
"""
from apipod.models.transformers.base import Transformers
from apipod.models.transformers.llm import TransformersLLM
from apipod.models.transformers.vlm import TransformersVLM

__all__ = ["Transformers", "TransformersLLM", "TransformersVLM"]
