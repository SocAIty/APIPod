"""vLLM engine options. Read here, not from apipod.common.settings.

Worker images set APIPOD_VLLM_* for this process. Bare VLLM_* names belong
to the vLLM binary and are not APIPod configuration.
"""
from os import environ


def _get(name: str, default: str = "") -> str:
    return environ.get(f"APIPOD_VLLM_{name}", default)


HOST = _get("HOST", "127.0.0.1")
PORT = int(_get("PORT", "18000"))
# Fallback only when config.json cannot be read. Chat.load() prefers the
# checkpoint's max_position_embeddings (and rope scaling).
MAX_MODEL_LEN = _get("MAX_MODEL_LEN", "")
# worker-vllm default. vLLM's own default (1024) exceeds Qwen3.8 Mamba cache.
MAX_NUM_SEQS = _get("MAX_NUM_SEQS", "256")
REASONING_PARSER = _get("REASONING_PARSER", "")
TOOL_CALL_PARSER = _get("TOOL_CALL_PARSER", "")
SPECULATIVE_CONFIG = _get("SPECULATIVE_CONFIG", "")
# Leftover CLI flags that are not first-class (kv cache dtype, batch tokens).
EXTRA_ARGS = _get("EXTRA_ARGS", "")
STARTUP_TIMEOUT = int(_get("STARTUP_TIMEOUT", "1200"))
ENABLE_AUTO_TOOL_CHOICE = _get("ENABLE_AUTO_TOOL_CHOICE", "1").strip().lower() not in (
    "0", "false", "no",
)
