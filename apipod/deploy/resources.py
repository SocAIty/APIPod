"""Local disk / GPU defaults for apipod.json (no backend required)."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_PARAM_SUFFIX = re.compile(r"[-_](\d+(?:\.\d+)?)[Bb](?:[-_.]|$)")
_MOE_PARAMS = re.compile(r"(\d+)x(\d+(?:\.\d+)?)[Bb]")
_GB_PER_BILLION_PARAMS = 2.2
_DISK_WORKSPACE_GB = 20
_MIN_CPU_DISK_GB = 5
_MIN_GPU_DISK_GB = 20
_MAX_DISK_GB = 200
_VRAM_SKUS = (16, 24, 40, 48, 80, 141)
_GPU_PROFILES = frozenset({"ml-gpu", "gpu"})
_GPU_MODEL_CLASS_MARKERS = ("Transformers", "Diffusers", "LLM", "VLM", "StableDiffusion", "VLLM")


def parse_param_billions(text: str) -> Optional[float]:
    """``Qwen/Qwen3.8-27B`` → 27, ``Llama-3.1-8B-Instruct`` → 8."""
    if not text:
        return None
    moe = _MOE_PARAMS.search(text)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    matches = [float(m) for m in _PARAM_SUFFIX.findall(text)]
    return max(matches) if matches else None


_HF_REF_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")
_MODEL_VAR_NAMES = frozenset({
    "DEFAULT_MODEL",
    "APIPOD_MODEL",
    "HF_MODEL",
    "HUGGINGFACE_MODEL",
    "MODEL_ID",
    "MODEL_NAME",
})
_HF_CALL_NAMES = frozenset({
    "include_hf",
    "TransformersLLM",
    "TransformersVLM",
    "Transformers",
    "VLLMChat",
})


def _is_hf_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_HF_REF_PATTERN.match(value.strip()))


def discover_hf_refs(
    project_root: Union[str, Path],
    entrypoint: Optional[str] = None,
) -> List[str]:
    """Find the Hugging Face id the service will actually load.

    Prefers ``APIPOD_MODEL`` / ``HF_MODEL`` in the environment, then
    ``DEFAULT_MODEL`` (and similar) assignments in the entrypoint. Does not
    scrape a full model catalog, which would oversize disk/VRAM.
    """
    for key in ("APIPOD_MODEL", "HF_MODEL", "HUGGINGFACE_MODEL", "DEFAULT_MODEL"):
        raw = os.environ.get(key)
        if _is_hf_ref(raw):
            return [raw.strip()]

    root = Path(project_root)
    files: List[Path] = []
    if entrypoint:
        candidate = root / entrypoint
        if candidate.is_file():
            files.append(candidate)
    if not files:
        skip = {"catalog.py", "test_services.py", "conftest.py"}
        files = [
            path for path in sorted(root.glob("*.py"))
            if path.name not in skip
        ]

    assigned: List[str] = []
    constructed: List[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if any(name in _MODEL_VAR_NAMES for name in names) and isinstance(node.value, ast.Constant):
                    if _is_hf_ref(node.value.value):
                        assigned.append(str(node.value.value).strip())
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in _HF_CALL_NAMES and node.args and isinstance(node.args[0], ast.Constant):
                    if _is_hf_ref(node.args[0].value):
                        constructed.append(str(node.args[0].value).strip())
    if assigned:
        unique = list(dict.fromkeys(assigned))
        if len(files) > 1 and len(unique) > 1:
            return []
        return unique
    return list(dict.fromkeys(constructed))


def extract_hf_refs(config: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for model in config.get("models") or []:
        if not isinstance(model, dict):
            continue
        for key, value in model.items():
            if key == "class" or not isinstance(value, dict):
                continue
            if value.get("kind") == "hf" and value.get("ref"):
                refs.append(str(value["ref"]))
    for include in config.get("includes") or []:
        if isinstance(include, dict) and include.get("kind") == "hf" and include.get("ref"):
            refs.append(str(include["ref"]))
    return list(dict.fromkeys(refs))


def infer_compute_tier(config: Dict[str, Any]) -> str:
    raw = str(config.get("compute_tier") or "").strip().lower()
    if raw in ("cpu", "gpu"):
        return raw
    if config.get("pytorch") or config.get("transformers") or config.get("diffusers"):
        return "gpu"
    if str(config.get("profile") or "").strip().lower() in _GPU_PROFILES:
        return "gpu"
    for model in config.get("models") or []:
        cls = str((model or {}).get("class") or "")
        if any(marker in cls for marker in _GPU_MODEL_CLASS_MARKERS):
            return "gpu"
    if extract_hf_refs(config):
        return "gpu"
    return "cpu"


def estimate_disk_gb(*, hf_refs: Optional[List[str]] = None, gpu: bool = False) -> int:
    known = [
        p for p in (parse_param_billions(ref) for ref in (hf_refs or []))
        if p and p > 0
    ]
    if not known:
        return _MIN_GPU_DISK_GB if gpu else _MIN_CPU_DISK_GB
    weight_gb = max(known) * _GB_PER_BILLION_PARAMS
    disk = int(weight_gb * 1.3) + _DISK_WORKSPACE_GB
    floor = _MIN_GPU_DISK_GB if gpu else _MIN_CPU_DISK_GB
    return max(floor, min(disk, _MAX_DISK_GB))


def estimate_gpu_vram_gb(hf_refs: Optional[List[str]] = None) -> int:
    """Smallest common GPU SKU that can hold a BF16 checkpoint plus activations."""
    known = [
        p for p in (parse_param_billions(ref) for ref in (hf_refs or []))
        if p and p > 0
    ]
    if not known:
        return 16
    needed = int(max(known) * 2.5) + 8
    for sku in _VRAM_SKUS:
        if needed <= sku:
            return sku
    return _VRAM_SKUS[-1]


def _positive_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def apply_resource_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill ``compute_tier``, ``disk_gb``, and ``gpu_vram_gb`` when unset."""
    out = dict(config)
    refs = extract_hf_refs(out)
    tier = infer_compute_tier(out)
    out["compute_tier"] = tier
    gpu = tier == "gpu"

    disk = _positive_int(out.get("disk_gb"))
    out["disk_gb"] = disk if disk is not None else estimate_disk_gb(hf_refs=refs, gpu=gpu)

    if gpu:
        vram = _positive_int(out.get("gpu_vram_gb") or out.get("min_gpu_vram_gb"))
        out["gpu_vram_gb"] = vram if vram is not None else estimate_gpu_vram_gb(refs)
    else:
        out.pop("gpu_vram_gb", None)
        out.pop("min_gpu_vram_gb", None)
    return out
