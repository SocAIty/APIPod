"""Include handles: declarations of external bytes (weights, assets) a service needs.

A handle is a *plan*, not loaded data. Declaring one performs no network or GPU
work; ``handle.path`` resolves it (download / cache lookup) on first access.
``apipod scan`` collects every declared handle into ``apipod.json`` so the
platform can pre-stage the bytes per provider (RunPod HF cache, image baking).
"""
import hashlib
import inspect
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

# RunPod pre-stages cached HF models here (HF cache conventions).
_RUNPOD_HF_CACHE = Path("/runpod-volume/huggingface-cache")
# org/name, both segments word chars, dots and dashes.
_HF_REF_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")

# Idempotent registry: the same ref always returns the same handle instance.
_INCLUDE_REGISTRY: Dict[Tuple[str, str], "IncludeHandle"] = {}


def _scan_mode() -> bool:
    """True while ``apipod scan`` imports the entrypoint (no resolution allowed)."""
    return os.environ.get("APIPOD_SCAN", "") == "1"


class IncludeHandle:
    """One declared external resource. ``kind`` is 'hf', 'path' or 'url'."""

    def __init__(self, kind: str, ref: str, local_root: Optional[Path] = None):
        self.kind = kind
        self.ref = ref
        self._local_root = local_root
        self._resolved: Optional[Path] = None

    def __repr__(self) -> str:
        return f"IncludeHandle(kind={self.kind!r}, ref={self.ref!r})"

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "ref": self.ref}

    @property
    def path(self) -> Path:
        """Filesystem location of the resource, resolving it on first access."""
        return self.resolve()

    def resolve(self) -> Path:
        if self._resolved is not None:
            return self._resolved
        if _scan_mode():
            raise RuntimeError(
                f"Include {self.ref!r} cannot be resolved during `apipod scan` "
                "(declarations must not download or load anything at import)."
            )
        resolver = {"hf": self._resolve_hf, "path": self._resolve_path, "url": self._resolve_url}[self.kind]
        self._resolved = resolver()
        return self._resolved

    # ------------------------------------------------------------------
    # Resolvers per kind
    # ------------------------------------------------------------------

    def _resolve_hf(self) -> Path:
        cached = _runpod_hf_snapshot(self.ref)
        if cached is not None:
            print(f"[apipod] Using RunPod-cached weights for {self.ref}: {cached}")
            return cached

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                f"Resolving the Hugging Face include {self.ref!r} needs 'huggingface_hub'. "
                "Install it with: pip install huggingface_hub"
            ) from None

        token = os.environ.get("HF_TOKEN") or None
        print(f"[apipod] Downloading {self.ref} from Hugging Face (cached after first run)...")
        return Path(snapshot_download(repo_id=self.ref, token=token))

    def _resolve_path(self) -> Path:
        # Existence was validated at declare time; the path may still vanish between
        # declaration and load (e.g. inside a container missing the COPY).
        resolved = _resolve_local(self.ref, self._local_root)
        if resolved is None:
            raise FileNotFoundError(f"Included path {self.ref!r} not found on disk.")
        return resolved

    def _resolve_url(self) -> Path:
        cache_dir = Path.home() / ".cache" / "apipod" / "includes" / hashlib.sha1(self.ref.encode()).hexdigest()
        filename = Path(urllib.parse.urlparse(self.ref).path).name or "download"
        target = cache_dir / filename
        if target.exists():
            return target
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[apipod] Downloading include {self.ref} -> {target}")
        urllib.request.urlretrieve(self.ref, target)  # noqa: S310 - user-declared URL
        return target


def _runpod_hf_snapshot(ref: str) -> Optional[Path]:
    """Locate a RunPod-cached HF model snapshot, or None when not pre-staged."""
    model_dir = _RUNPOD_HF_CACHE / "hub" / f"models--{ref.replace('/', '--')}"
    if not model_dir.exists():
        return None
    refs_main = model_dir / "refs" / "main"
    if refs_main.exists():
        snapshot = model_dir / "snapshots" / refs_main.read_text().strip()
        if snapshot.exists():
            return snapshot
    snapshots = sorted((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").exists() else []
    return snapshots[-1] if snapshots else None


def _resolve_local(ref: str, local_root: Optional[Path]) -> Optional[Path]:
    """Resolve a local ref against the declaring module's directory, then cwd."""
    candidates = [Path(ref)]
    if local_root is not None:
        candidates.insert(0, local_root / ref)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _caller_root() -> Optional[Path]:
    """Directory of the user module that declared the include (for relative paths)."""
    for frame_info in inspect.stack()[2:]:
        module_file = frame_info.frame.f_globals.get("__file__")
        if module_file and "apipod" not in Path(module_file).parts:
            return Path(module_file).resolve().parent
    return None


def _register(kind: str, ref: str, local_root: Optional[Path] = None) -> IncludeHandle:
    key = (kind, ref)
    if key not in _INCLUDE_REGISTRY:
        _INCLUDE_REGISTRY[key] = IncludeHandle(kind, ref, local_root=local_root)
    return _INCLUDE_REGISTRY[key]


def include_hf(ref: str) -> IncludeHandle:
    """Declare a Hugging Face model dependency, e.g. ``include_hf("Qwen/Qwen3.5-7B")``.

    Deployed on RunPod the weights come from the provider's HF cache; locally
    they land in your HF cache on first resolution.
    """
    if not _HF_REF_PATTERN.match(ref or ""):
        raise ValueError(
            f"Malformed Hugging Face model id {ref!r}. Expected 'org/name', e.g. 'Qwen/Qwen3.5-7B'."
        )
    return _register("hf", ref)


def include(ref: Optional[str] = None, hf: Optional[str] = None) -> IncludeHandle:
    """Declare an external resource: HF model, local file/directory, or URL.

    Args:
        ref: Local path (baked into the image at build) or URL (fetched at
            build). The scheme is detected from the string.
        hf: Hugging Face model id, same as :func:`include_hf`.
    """
    if (ref is None) == (hf is None):
        raise ValueError("include() takes exactly one of a positional ref or hf=...")
    if hf is not None:
        return include_hf(hf)
    if ref.startswith(("http://", "https://")):
        return _register("url", ref)

    local_root = _caller_root()
    if _resolve_local(ref, local_root) is None:
        raise FileNotFoundError(
            f"Included path {ref!r} not found (looked in {local_root or Path.cwd()}). "
            "Local includes must exist at declare time."
        )
    return _register("path", ref, local_root=local_root)


def declared_includes() -> Dict[Tuple[str, str], IncludeHandle]:
    """All registered handles (used by scan and by model-owned include lookup)."""
    return dict(_INCLUDE_REGISTRY)
