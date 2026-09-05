"""vLLM-backed chat engine: spawn ``vllm serve``, proxy OpenAI HTTP.

The worker process never imports vLLM (same shape as RunPod worker-vllm).
``load()`` starts the CLI, polls ``/health``, and inference is HTTP to
localhost so several RunPod jobs can overlap when ``MAX_CONCURRENCY`` > 1.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Iterator, List, Optional, Union

import httpx
from media_toolkit import ImageFile

from apipod.common.chat_parsing import ChatOutputParser, parse_chat_output
from apipod.models.includes import IncludeHandle, include_hf, _runpod_hf_snapshot
from apipod.models.model import Model
from apipod.models.vllm import config as vllm_config

_HEALTH_POLL_S = 2.0
_HTTP_TIMEOUT_S = 3600.0
_VLLM_LOG_PATH = "/tmp/apipod-vllm.log"
_FIRST_CLASS_FLAGS = {
    "--host",
    "--port",
    "--max-model-len",
    "--max-num-seqs",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "--reasoning-parser",
    "--speculative-config",
}


def _as_uint8(arr):
    import numpy as np

    if arr.dtype == np.uint8:
        return arr
    max_val = float(arr.max()) if arr.size else 0.0
    if np.issubdtype(arr.dtype, np.floating) and max_val <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_bgr(image):
    """media-toolkit ImageFile is BGR; raw ndarrays are treated as RGB."""
    import cv2
    import numpy as np

    if isinstance(image, ImageFile) or hasattr(image, "to_np_array"):
        return _as_uint8(np.asarray(image.to_np_array()))
    if isinstance(image, (bytes, bytearray, memoryview)):
        decoded = cv2.imdecode(np.frombuffer(bytes(image), np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Could not decode image bytes.")
        return decoded
    arr = _as_uint8(np.asarray(image))
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr


def _to_data_url(image) -> str:
    """APIPod/cv2/bytes image -> PNG data URI for OpenAI ``image_url`` parts.

    ``ImageFile.to_np_array()`` is BGR. ``cv2.imencode`` expects BGR, so do not
    convert to RGB first.
    """
    import cv2

    ok, buf = cv2.imencode(".png", _to_bgr(image))
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_messages(messages) -> List[dict]:
    return [dict(m) if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in messages]


def _content_parts(content, images=None) -> Any:
    if images:
        parts = [{"type": "image_url", "image_url": {"url": _to_data_url(image)}} for image in images]
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            parts.extend(content or [])
        return parts
    return content


def _extend_without_duplicates(argv: List[str], extra: str) -> None:
    """Append EXTRA_ARGS, skipping flags already set as first-class argv.

    Platform env can still carry an old EXTRA_ARGS string that repeats
    --enable-auto-tool-choice / --tool-call-parser after we moved them out.
    """
    tokens = shlex.split(extra)
    present = {item.split("=")[0] for item in argv if item.startswith("--")}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag = token.split("=")[0] if token.startswith("--") else ""
        takes_value = flag in _FIRST_CLASS_FLAGS and "=" not in token
        skip_value = takes_value and index + 1 < len(tokens) and not tokens[index + 1].startswith("-")
        if flag in present:
            index += 2 if skip_value else 1
            continue
        argv.append(token)
        if flag:
            present.add(flag)
        index += 1


def _read_hf_config(weights: IncludeHandle) -> Optional[dict]:
    """Load ``config.json`` for *weights* without importing the full model."""
    from pathlib import Path

    candidates = []
    resolved = getattr(weights, "_resolved", None)
    if resolved is not None:
        candidates.append(Path(resolved) / "config.json")
    if weights.kind == "hf":
        cached = _runpod_hf_snapshot(weights.ref)
        if cached is not None:
            candidates.append(cached / "config.json")
    elif weights.kind == "path":
        candidates.append(Path(weights.ref) / "config.json")
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    if weights.kind != "hf":
        return None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(weights.ref, "config.json")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _int_field(config: dict, *keys: str) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _context_from_mapping(config: dict) -> Optional[int]:
    """Read one config object (root or ``text_config``). Prefer the largest signal."""
    lengths: list[int] = []
    direct = _int_field(
        config,
        "max_model_len",
        "max_position_embeddings",
        "max_sequence_length",
        "model_max_length",
    )
    if direct:
        lengths.append(direct)
    rope = config.get("rope_scaling") if isinstance(config.get("rope_scaling"), dict) else {}
    original = rope.get("original_max_position_embeddings")
    factor = rope.get("factor") or rope.get("rope_factor")
    if original is not None and factor is not None:
        try:
            scaled = int(int(original) * float(factor))
        except (TypeError, ValueError):
            scaled = 0
        if scaled > 0:
            lengths.append(scaled)
    return max(lengths) if lengths else None


def max_model_len_from_config(weights: IncludeHandle) -> Optional[int]:
    """Context window from the checkpoint, including nested VLM text configs."""
    config = _read_hf_config(weights)
    if not config:
        return None
    lengths = []
    for mapping in (config, config.get("text_config"), config.get("llm_config")):
        if isinstance(mapping, dict):
            value = _context_from_mapping(mapping)
            if value:
                lengths.append(value)
    return max(lengths) if lengths else None


def _speculative_argv(raw: str) -> List[str]:
    speculative_config = raw.strip()
    if not speculative_config:
        return []
    try:
        parsed = json.loads(speculative_config)
    except json.JSONDecodeError as exc:
        raise ValueError("APIPOD_VLLM_SPECULATIVE_CONFIG must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("APIPOD_VLLM_SPECULATIVE_CONFIG must be a JSON object.")
    return ["--speculative-config", json.dumps(parsed, separators=(",", ":"))]


class _SseParseState:
    """Streaming parse: native OpenAI deltas win over Hermes tag scraping."""

    def __init__(self):
        self.parser = ChatOutputParser()
        self.native_tool_calls = False


class VLLMChat(Model):
    """Chat (text + optional images) served by a local ``vllm serve`` process.

    Implements ``generate`` / ``agenerate`` / ``stream`` / ``astream`` so
    ``apipod.serve`` registers ``/chat``. No embeddings: vLLM pooling is not
    the transformers last-token recipe.
    """

    def __init__(self, weights: Union[IncludeHandle, str], *, enable_thinking: bool = False):
        if isinstance(weights, str):
            weights = include_hf(weights)
        if weights.kind != "hf":
            raise ValueError(
                f"{type(self).__name__} supports Hugging Face includes only. "
                "Pass an HF model id string or an include_hf() handle."
            )
        self.weights = weights
        self.enable_thinking = enable_thinking
        self._proc: Optional[subprocess.Popen] = None
        self._log_tail: Deque[str] = deque(maxlen=80)
        self._log_thread: Optional[threading.Thread] = None
        self._argv: List[str] = []
        self._log_path = _VLLM_LOG_PATH
        self._log_file = None
        self._base_url = f"http://{vllm_config.HOST}:{vllm_config.PORT}"

    def load(self) -> None:
        binary = shutil.which("vllm")
        if not binary:
            raise RuntimeError(
                "vllm CLI not found on PATH. Install vLLM in the image "
                "(pip install vllm) or set APIPOD_ENGINE=transformers."
            )
        host = vllm_config.HOST
        port = vllm_config.PORT
        detected = max_model_len_from_config(self.weights)
        max_len = str(detected) if detected else (vllm_config.MAX_MODEL_LEN or "")
        max_num_seqs = vllm_config.MAX_NUM_SEQS or "256"
        extra = vllm_config.EXTRA_ARGS.strip()
        timeout = vllm_config.STARTUP_TIMEOUT
        argv = [
            binary, "serve", str(self.weights.ref),
            "--host", host,
            "--port", str(port),
        ]
        if max_len:
            argv.extend(["--max-model-len", max_len])
        argv.extend(["--max-num-seqs", str(max_num_seqs)])
        if vllm_config.ENABLE_AUTO_TOOL_CHOICE and vllm_config.TOOL_CALL_PARSER:
            argv.extend(["--enable-auto-tool-choice", "--tool-call-parser", vllm_config.TOOL_CALL_PARSER])
        elif vllm_config.TOOL_CALL_PARSER:
            argv.extend(["--tool-call-parser", vllm_config.TOOL_CALL_PARSER])
        if vllm_config.REASONING_PARSER:
            argv.extend(["--reasoning-parser", vllm_config.REASONING_PARSER])
        argv.extend(_speculative_argv(vllm_config.SPECULATIVE_CONFIG))
        if extra:
            _extend_without_duplicates(argv, extra)

        self._argv = argv
        mode = "speculative" if vllm_config.SPECULATIVE_CONFIG.strip() else "standard"
        command = shlex.join(argv)
        print(
            f"[apipod] Starting vLLM model={self.weights.ref} "
            f"endpoint=http://{host}:{port} mode={mode} "
            f"max_model_len={max_len or 'vllm-default'} max_num_seqs={max_num_seqs}",
            flush=True,
        )
        print(f"[apipod] vLLM command: {command}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)
        self._proc = subprocess.Popen(
            argv,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._log_thread = threading.Thread(target=self._pump_logs, daemon=True)
        self._log_thread.start()
        atexit.register(self._stop)
        self._base_url = f"http://{host}:{port}"
        try:
            self._wait_healthy(timeout)
        except BaseException:
            self._stop()
            raise

    def _pump_logs(self) -> None:
        """Follow the vLLM log file so EngineCore output is not stuck in a PIPE."""
        with open(self._log_path, "r", encoding="utf-8", errors="replace") as log_file:
            while True:
                line = log_file.readline()
                if line:
                    text = line.rstrip("\n")
                    self._log_tail.append(text)
                    print(text, flush=True)
                    continue
                if self._proc is not None and self._proc.poll() is not None:
                    leftover = log_file.read()
                    if leftover:
                        for text in leftover.splitlines():
                            self._log_tail.append(text)
                            print(text, flush=True)
                    return
                time.sleep(0.05)

    def _startup_error(self, prefix: str) -> RuntimeError:
        if self._log_file is not None:
            try:
                self._log_file.flush()
            except OSError:
                pass
        if self._log_thread is not None:
            self._log_thread.join(timeout=5)
        parts = [prefix]
        if self._argv:
            parts.append("command: " + shlex.join(self._argv))
        tail = "\n".join(self._log_tail)
        if not tail:
            try:
                with open(self._log_path, "r", encoding="utf-8", errors="replace") as log_file:
                    tail = log_file.read().strip()
            except OSError:
                tail = ""
        if tail:
            parts.append("--- vllm serve log tail ---\n" + tail)
        else:
            parts.append("(no vLLM stdout captured)")
        return RuntimeError("\n".join(parts))

    def _stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_file = self._log_file
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _wait_healthy(self, timeout: Optional[int] = None) -> None:
        if timeout is None:
            timeout = vllm_config.STARTUP_TIMEOUT
        url = f"{self._base_url}/health"
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=10, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise self._startup_error(
                        f"vllm serve exited during startup with code {self._proc.returncode}."
                    )
                try:
                    if client.get(url).status_code == 200:
                        print("[apipod] vLLM is healthy", flush=True)
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(_HEALTH_POLL_S)
        raise self._startup_error(f"vLLM did not become healthy within {timeout}s")

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "ping"}], max_tokens=1)

    def _ensure_started(self) -> None:
        if not self._apipod_loaded and not self._apipod_loading:
            self.ensure_loaded()

    def _openai_messages(self, messages, images=None) -> List[dict]:
        conversation = _normalize_messages(messages)
        if not images:
            return conversation
        attached = False
        out = []
        for message in reversed(conversation):
            item = dict(message)
            if not attached and item.get("role") == "user":
                item["content"] = _content_parts(item.get("content"), images)
                attached = True
            out.append(item)
        out.reverse()
        if not attached:
            out.append({"role": "user", "content": _content_parts("", images)})
        return out

    def _request_body(
        self,
        messages,
        images=None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs=None,
    ) -> dict:
        body: dict[str, Any] = {
            "model": str(self.weights.ref),
            "messages": self._openai_messages(messages, images),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if stop:
            body["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = [t if isinstance(t, dict) else t.model_dump() for t in tools]
            if tool_choice is None:
                body["tool_choice"] = "auto"
            else:
                body["tool_choice"] = (
                    tool_choice
                    if isinstance(tool_choice, (str, dict))
                    else tool_choice.model_dump(exclude_none=True)
                )
            if parallel_tool_calls is None:
                body["parallel_tool_calls"] = True
            else:
                body["parallel_tool_calls"] = parallel_tool_calls
        elif tool_choice is not None:
            body["tool_choice"] = (
                tool_choice
                if isinstance(tool_choice, (str, dict))
                else tool_choice.model_dump(exclude_none=True)
            )
        if logprobs:
            body["logprobs"] = True
            if top_logprobs is not None:
                body["top_logprobs"] = top_logprobs
        return body

    def _completion_url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    def _result_from_openai(self, payload: dict, max_tokens: int):
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content")
        native_tools = message.get("tool_calls")
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        if native_tools:
            assistant = {
                "role": message.get("role") or "assistant",
                "content": text or None,
                "tool_calls": native_tools,
            }
            if reasoning:
                assistant["reasoning_content"] = reasoning
            return {
                "choices": [{
                    "index": 0,
                    "message": assistant,
                    "finish_reason": choice.get("finish_reason") or "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        parsed = parse_chat_output(text)
        if reasoning and not parsed.get("reasoning_content"):
            parsed["reasoning_content"] = reasoning
        is_plain = (
            not parsed.get("tool_calls")
            and not parsed.get("reasoning_content")
            and parsed.get("content") is not None
        )
        if is_plain:
            return parsed["content"]
        if parsed.get("tool_calls"):
            finish_reason = "tool_calls"
        elif completion_tokens >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = choice.get("finish_reason") or "stop"
        return {
            "choices": [{"index": 0, "message": parsed, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.text.strip()
            except httpx.ResponseNotRead:
                detail = ""
            suffix = f": {detail[:2000]}" if detail else ""
            raise RuntimeError(
                f"vLLM returned HTTP {response.status_code}{suffix}"
            ) from exc

    def generate(
        self,
        messages,
        images=None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        logprobs: bool = False,
        top_logprobs=None,
    ):
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, parallel_tool_calls, stream=False,
            logprobs=logprobs, top_logprobs=top_logprobs,
        )
        with httpx.Client(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            response = client.post(self._completion_url(), json=body)
            self._raise_for_status(response)
            return self._result_from_openai(response.json(), max_tokens)

    async def agenerate(
        self,
        messages,
        images=None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        logprobs: bool = False,
        top_logprobs=None,
    ):
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, parallel_tool_calls, stream=False,
            logprobs=logprobs, top_logprobs=top_logprobs,
        )
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            response = await client.post(self._completion_url(), json=body)
            self._raise_for_status(response)
            return self._result_from_openai(response.json(), max_tokens)

    def stream(
        self,
        messages,
        images=None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    ) -> Iterator:
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, parallel_tool_calls, stream=True,
        )
        state = _SseParseState()
        with httpx.Client(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            with client.stream("POST", self._completion_url(), json=body) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    yield from _parse_sse_delta(line, state)
        yield from state.parser.flush()

    async def astream(
        self,
        messages,
        images=None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    ) -> AsyncIterator:
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, parallel_tool_calls, stream=True,
        )
        state = _SseParseState()
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            async with client.stream("POST", self._completion_url(), json=body) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    for delta in _parse_sse_delta(line, state):
                        yield delta
        for delta in state.parser.flush():
            yield delta


def _parse_sse_delta(line: str, state: _SseParseState) -> Iterator:
    """Convert one vLLM SSE event into APIPod chat deltas.

    Native ``delta.tool_calls`` / ``reasoning_content`` are forwarded as-is.
    Content is Hermes-parsed only until native tool_calls appear; after that
    leftover content is yielded as plain text.
    """
    raw = (line or "").strip()
    if not raw.startswith("data:"):
        return
    payload = raw[5:].strip()
    if not payload or payload == "[DONE]":
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return
    choice = (data.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    typed = {
        key: delta[key]
        for key in ("role", "reasoning_content", "tool_calls", "refusal")
        if delta.get(key) is not None
    }
    if typed.get("tool_calls"):
        state.native_tool_calls = True
    if typed:
        yield typed
    content = delta.get("content")
    if not content:
        return
    if state.native_tool_calls:
        yield content
        return
    yield from state.parser.feed(content)
