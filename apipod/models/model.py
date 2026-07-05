"""Model base class: how to load and use declared weights.

Subclasses implement ``load()`` (required), ``warmup()`` (optional) and their
inference methods. Instances register themselves at construction so the app
can load everything at start and ``apipod scan`` can report them without
executing any heavy work (``APIPOD_SCAN=1`` guards resolution).
"""
import threading
from typing import Dict, List

from apipod.models.includes import IncludeHandle, _scan_mode

_MODEL_REGISTRY: List["Model"] = []

_LAZY_LOAD_HINT = (
    "[apipod] Model loaded on first request. Run `apipod start` so weights load at app start."
)


class Model:
    """Base class for user models. Attach includes in ``__init__``, load them in ``load()``.

    Example:
        class QwenLLM(apipod.Model):
            def __init__(self):
                self.weights = apipod.include_hf("Qwen/Qwen3.5-7B")

            def load(self):
                self.net = AutoModelForCausalLM.from_pretrained(self.weights.path)
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        # Set via object.__setattr__-free plain assignment; these exist before
        # user __init__ runs so __getattr__ can rely on them.
        instance._apipod_loaded = False
        instance._apipod_loading = False
        instance._apipod_load_lock = threading.Lock()
        _MODEL_REGISTRY.append(instance)
        return instance

    # ------------------------------------------------------------------
    # User contract
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load weights into memory. Required. Called once at app start (or lazily)."""
        raise NotImplementedError(f"{type(self).__name__} must implement load().")

    def warmup(self) -> None:
        """Optional warm inference pass after load (compile kernels, fill caches)."""

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def includes(self) -> Dict[str, IncludeHandle]:
        """Handles attached to this model instance, keyed by attribute name."""
        return {name: value for name, value in vars(self).items() if isinstance(value, IncludeHandle)}

    def ensure_loaded(self, run_warmup: bool = False) -> None:
        """Resolve includes and run ``load()`` exactly once (thread-safe)."""
        if self._apipod_loaded:
            return
        with self._apipod_load_lock:
            if self._apipod_loaded:
                return
            if _scan_mode():
                raise RuntimeError(
                    f"{type(self).__name__} cannot load during `apipod scan` "
                    "(declarations only, no downloads or GPU work)."
                )
            self._apipod_loading = True
            try:
                for handle in self.includes().values():
                    handle.resolve()
                self.load()
                if run_warmup:
                    self.warmup()
                self._apipod_loaded = True
            finally:
                self._apipod_loading = False

    def __getattr__(self, name: str):
        """Lazy-load fallback: first access to a not-yet-set attribute (e.g.
        ``self.net`` inside an inference method before the app start hook ran)
        triggers a one-time thread-safe load, then retries the lookup."""
        if name.startswith("_apipod") or name.startswith("__"):
            raise AttributeError(name)
        if not self._apipod_loaded and not self._apipod_loading:
            print(_LAZY_LOAD_HINT)
            self.ensure_loaded(run_warmup=False)
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def declared_models() -> List[Model]:
    """All Model instances constructed in this process (used by scan and app start)."""
    return list(_MODEL_REGISTRY)


def load_declared_models(run_warmup: bool = True) -> None:
    """Load every declared model. Called by the backends at app start.

    Fails fast: a load error aborts the start so a broken service never
    reports healthy.
    """
    if _scan_mode():
        return
    for model in _MODEL_REGISTRY:
        if model._apipod_loaded:
            continue
        name = type(model).__name__
        print(f"[apipod] Loading model {name}...")
        model.ensure_loaded(run_warmup=run_warmup)
        print(f"[apipod] Model {name} ready.")
