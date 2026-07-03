"""Deployment profiles for scan/build (CPU minimal vs ML/GPU)."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Set

PROFILE_SERVERLESS_MINIMAL = "serverless-minimal"
PROFILE_WEB_API = "web-api"
PROFILE_ML_GPU = "ml-gpu"

PYTORCH_PACKAGES: Set[str] = {
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch-lightning",
    "lightning",
    "accelerate",
    "bitsandbytes",
    "xformers",
    "pytorch3d",
}

TENSORFLOW_PACKAGES: Set[str] = {
    "tensorflow",
    "tensorflow-gpu",
    "tf-nightly",
    "keras",
}

ONNX_PACKAGES: Set[str] = {
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxmltools",
}

TRANSFORMERS_PACKAGES: Set[str] = {
    "transformers",
    "sentence-transformers",
    "optimum",
}

DIFFUSERS_PACKAGES: Set[str] = {
    "diffusers",
}

ML_DIRECT_PACKAGES: FrozenSet[str] = (
    PYTORCH_PACKAGES
    | TENSORFLOW_PACKAGES
    | ONNX_PACKAGES
    | TRANSFORMERS_PACKAGES
    | DIFFUSERS_PACKAGES
)

POETRY_NON_PACKAGE_KEYS: FrozenSet[str] = frozenset(
    {"python", "pip", "setuptools", "wheel"}
)

LIGHTWEIGHT_PACKAGES: Set[str] = {
    "apipod",
    "runpod",
    "uvicorn",
    "fastapi",
    "httpx",
    "pydantic",
    "python-multipart",
    "starlette",
    "requests",
    "click",
    "tqdm",
    "singleton-decorator",
    "fastsdk",
    "apipod-registry",
    "media-toolkit",
}


def direct_ml_dependencies(python_deps: Set[str]) -> Set[str]:
    """Declared dependencies that are ML frameworks (exact package names only)."""
    return {name for name in python_deps if name in ML_DIRECT_PACKAGES}


_ENTRYPOINT_TORCH = frozenset({"torch", "torchvision", "torchaudio"})
_ENTRYPOINT_TF = frozenset({"tensorflow", "keras"})
_ENTRYPOINT_ONNX = frozenset({"onnx", "onnxruntime"})
_ENTRYPOINT_TRANSFORMERS = frozenset({"transformers"})
_ENTRYPOINT_DIFFUSERS = frozenset({"diffusers"})


def reconcile_framework_flags(
    *,
    python_deps: Set[str],
    entrypoint_imports: Set[str],
    model_files: list,
) -> Dict[str, bool]:
    """
    Framework flags are true only for direct ML dependencies, entrypoint imports,
    or on-disk model artifacts — not from scanning the whole repository tree.
    """
    direct_ml = direct_ml_dependencies(python_deps)
    has_weights = bool(model_files)

    pytorch = bool(direct_ml & PYTORCH_PACKAGES) or bool(
        entrypoint_imports & _ENTRYPOINT_TORCH
    ) or has_weights
    tensorflow = bool(direct_ml & TENSORFLOW_PACKAGES) or bool(
        entrypoint_imports & _ENTRYPOINT_TF
    )
    onnx = bool(direct_ml & ONNX_PACKAGES) or bool(entrypoint_imports & _ENTRYPOINT_ONNX)
    transformers = bool(direct_ml & TRANSFORMERS_PACKAGES) or bool(
        entrypoint_imports & _ENTRYPOINT_TRANSFORMERS
    )
    diffusers = bool(direct_ml & DIFFUSERS_PACKAGES) or bool(
        entrypoint_imports & _ENTRYPOINT_DIFFUSERS
    )
    cuda = any("cuda" in name or name.endswith("-gpu") for name in direct_ml)

    return {
        "pytorch": pytorch,
        "tensorflow": tensorflow,
        "onnx": onnx,
        "transformers": transformers,
        "diffusers": diffusers,
        "cuda": cuda,
    }


def infer_profile(
    *,
    pytorch: bool,
    tensorflow: bool,
    onnx: bool,
    transformers: bool,
    diffusers: bool,
    cuda: bool,
    compute: Optional[str],
    provider: Optional[str],
    python_deps: Set[str],
    model_files: Optional[list] = None,
) -> str:
    """Choose a deployment profile from scan results."""
    has_ml = any([pytorch, tensorflow, onnx, transformers, diffusers, cuda])
    model_files = model_files or []
    direct_ml = direct_ml_dependencies(python_deps)

    if compute == "serverless" and provider == "runpod":
        if not has_ml and not model_files and not direct_ml:
            return PROFILE_SERVERLESS_MINIMAL
        if has_ml or model_files or direct_ml:
            return PROFILE_ML_GPU
        return PROFILE_SERVERLESS_MINIMAL

    if compute == "serverless" or provider == "runpod":
        if not has_ml:
            return PROFILE_SERVERLESS_MINIMAL
        return PROFILE_ML_GPU

    if has_ml:
        return PROFILE_ML_GPU

    if "uvicorn" in python_deps or "fastapi" in python_deps:
        return PROFILE_WEB_API

    if python_deps and python_deps.issubset(LIGHTWEIGHT_PACKAGES):
        return PROFILE_SERVERLESS_MINIMAL

    return PROFILE_WEB_API


def recommend_base_image(profile: str, python_version: str, config: Dict[str, Any]) -> str:
    version = str(python_version or "3.12")
    if profile == PROFILE_SERVERLESS_MINIMAL:
        return f"ghcr.io/astral-sh/uv:python{version}-bookworm-slim"
    if profile == PROFILE_ML_GPU:
        if config.get("pytorch"):
            return "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
        return "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
    return f"python:{version}-slim"
