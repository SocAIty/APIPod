"""Pipeline timing logs. Grep ``PROFILE``.

Enabled unless ``APIPOD_PROFILE=0``. ``elapsed_ms`` is from ``start(job_id)``
in this process, else from ``created_at``. Stages that are the first token
or first stream byte set ``ttft=true``.

The same ``event()`` call also appends a hop to Redis ``apipod:profile:{job_id}``
when ``REDIS_URL`` is set. Redis failures are swallowed; logs still emit.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

LOG = logging.getLogger("apipod.profile")
_OFF = os.environ.get("APIPOD_PROFILE", "1").strip().lower() in {"0", "false", "no", "off"}
_T0: dict[str, float] = {}
_CURRENT: ContextVar[Optional[str]] = ContextVar("apipod_profile_job", default=None)

_REDIS_KEY = "apipod:profile:{job_id}"
_HOP_CAP = 40
_HOP_TTL_S = 3600
_TOKEN_KEYS = frozenset({"text", "token", "tokens", "content", "delta", "chunk", "output"})
_redis: Any = None
_redis_lock = threading.Lock()

# Platform TTFT: first SSE byte leaving the gateway.
TTFT_COMPONENT = "gate"
TTFT_STAGE = "first_sse_chunk"


def enabled() -> bool:
    return not _OFF


def start(job_id: Optional[str]) -> None:
    if job_id and not _OFF:
        _T0.setdefault(job_id, time.monotonic())
        _CURRENT.set(job_id)


def bind(job_id: Optional[str]) -> None:
    """Remember ``job_id`` for this task so later hops can omit it."""
    start(job_id)


def elapsed_ms(job_id: Optional[str] = None, created_at: Any = None) -> Optional[int]:
    if job_id and job_id in _T0:
        return int((time.monotonic() - _T0[job_id]) * 1000)
    if created_at is None:
        return None
    try:
        if hasattr(created_at, "timestamp"):
            dt = created_at
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            t0 = dt.timestamp()
        else:
            t0 = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).timestamp()
        return int((time.time() - t0) * 1000)
    except Exception:
        return None


def event(
    component: str,
    stage: str,
    job_id: Optional[str] = None,
    *,
    ttft: bool = False,
    created_at: Any = None,
    **fields: Any,
) -> None:
    if _OFF:
        return
    job_id = job_id or _CURRENT.get()
    parts = [f"PROFILE component={component} stage={stage}"]
    if job_id:
        parts.append(f"job_id={job_id}")
    elapsed = elapsed_ms(job_id, created_at)
    if elapsed is not None:
        parts.append(f"elapsed_ms={elapsed}")
    if ttft:
        parts.append("ttft=true")
    extras: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None or key in _TOKEN_KEYS:
            continue
        parts.append(f"{key}={value}")
        extras[key] = value
    LOG.info(" ".join(parts))
    if job_id:
        _append_hop(
            job_id,
            component=component,
            stage=stage,
            elapsed_ms=elapsed,
            ttft=ttft,
            extras=extras,
        )


def load_hops(job_id: Optional[str]) -> list[dict[str, Any]]:
    """Return hops for ``job_id`` without deleting the Redis list."""
    if not job_id:
        return []
    client = _redis_client()
    if client is None:
        return []
    try:
        raw = client.lrange(_REDIS_KEY.format(job_id=job_id), 0, -1)
    except Exception:
        return []
    return _decode_hops(raw)


def take_hops(job_id: Optional[str]) -> list[dict[str, Any]]:
    """Read hops for ``job_id`` and delete the Redis key."""
    if not job_id:
        return []
    client = _redis_client()
    if client is None:
        return []
    key = _REDIS_KEY.format(job_id=job_id)
    try:
        pipe = client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        raw, _deleted = pipe.execute()
    except Exception:
        return []
    return _decode_hops(raw)


def first_token_at(hops: list[dict[str, Any]]) -> Optional[datetime]:
    """Wall time of platform TTFT (``gate.first_sse_chunk``), else any ``ttft`` hop."""
    for hop in hops:
        if hop.get("component") == TTFT_COMPONENT and hop.get("stage") == TTFT_STAGE:
            return _parse_at(hop.get("at"))
    for hop in hops:
        if hop.get("ttft"):
            return _parse_at(hop.get("at"))
    return None


def ttft_seconds(
    hops: list[dict[str, Any]],
    *,
    t0: Any = None,
) -> Optional[float]:
    """``first_sse_chunk - requested_at`` (or first hop / ``t0``)."""
    t1 = first_token_at(hops)
    if t1 is None:
        return None
    start_at = _parse_at(t0) if t0 is not None else None
    if start_at is None:
        for hop in hops:
            parsed = _parse_at(hop.get("at"))
            if parsed is not None:
                start_at = parsed
                break
    if start_at is None:
        return None
    delta = (t1 - start_at).total_seconds()
    return round(delta, 3) if delta >= 0 else 0.0


def _append_hop(
    job_id: str,
    *,
    component: str,
    stage: str,
    elapsed_ms: Optional[int],
    ttft: bool,
    extras: dict[str, Any],
) -> None:
    client = _redis_client()
    if client is None:
        return
    hop: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "stage": stage,
    }
    if elapsed_ms is not None:
        hop["elapsed_ms"] = elapsed_ms
    if ttft:
        hop["ttft"] = True
    for key, value in extras.items():
        if key in hop or key in _TOKEN_KEYS:
            continue
        hop[key] = value
    key = _REDIS_KEY.format(job_id=job_id)
    try:
        payload = json.dumps(hop, default=str, separators=(",", ":"))
        pipe = client.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -_HOP_CAP, -1)
        pipe.expire(key, _HOP_TTL_S)
        pipe.execute()
    except Exception:
        return


def _redis_client() -> Any:
    global _redis
    if _OFF:
        return None
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    if _redis is not None:
        return _redis
    with _redis_lock:
        if _redis is not None:
            return _redis
        try:
            import redis

            _redis = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        except Exception:
            return None
        return _redis


def _decode_hops(raw: Any) -> list[dict[str, Any]]:
    hops: list[dict[str, Any]] = []
    if not raw:
        return hops
    for item in raw:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str):
            continue
        try:
            hop = json.loads(item)
        except Exception:
            continue
        if isinstance(hop, dict):
            hops.append(hop)
    return hops


def _parse_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        dt = value
        if getattr(dt, "tzinfo", None) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
