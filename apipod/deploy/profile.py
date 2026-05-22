"""Deployment profiles for scan/build (CPU minimal vs ML/GPU)."""

from __future__ import annotations

from __future__ import annotations

from typing import Any, Dict, Optional, Set

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
) -> str:
    """Choose a deployment profile from scan results."""
    has_ml = any([pytorch, tensorflow, onnx, transformers, diffusers, cuda])

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
