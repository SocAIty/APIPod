"""Declarative model loading: include handles + the Model base class.

``include`` / ``include_hf`` declare where bytes live (no download at import);
``Model`` subclasses define how to load and use them. The platform reads the
declarations (``apipod scan``) to pick the fastest shipping strategy per
provider (RunPod HF cache, image baking) and the runtime loads everything at
app start (lazy thread-safe fallback on first use).
"""
from apipod.models.includes import IncludeHandle, include, include_hf, declared_includes
from apipod.models.model import Model, declared_models, load_declared_models
from apipod.models.transformers import Transformers, TransformersLLM, TransformersVLM

__all__ = [
    "IncludeHandle",
    "include",
    "include_hf",
    "Model",
    "Transformers",
    "TransformersLLM",
    "TransformersVLM",
    "declared_includes",
    "declared_models",
    "load_declared_models",
]
