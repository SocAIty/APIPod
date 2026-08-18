"""vLLM-backed chat preset: spawn ``vllm serve``, proxy OpenAI HTTP.

The worker process never imports vLLM (same shape as RunPod worker-vllm).
``load()`` starts the CLI, polls ``/health``, and inference is HTTP to
localhost so several RunPod jobs can overlap when ``MAX_CONCURRENCY`` > 1.
"""
from __future__ import annotations

import atexit
import base64
import io
import json
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Iterator, List, Optional, Union

import httpx

from apipod.common.chat_parsing import ChatOutputParser, parse_chat_output
from apipod.common.settings import (
    VLLM_EXTRA_ARGS,
    VLLM_HOST,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_SEQS,
    VLLM_PORT,
    VLLM_REASONING_PARSER,
    VLLM_SPECULATIVE_CONFIG,
    VLLM_STARTUP_TIMEOUT,
)
from apipod.models.includes import IncludeHandle, include_hf
from apipod.models.model import Model
from apipod.models.transformers.vlm import to_pil_image

_HEALTH_POLL_S = 2.0
_HTTP_TIMEOUT_S = 3600.0


def _to_data_url(image) -> str:
    """APIPod/PIL/bytes image -> PNG data URI for OpenAI ``image_url`` parts."""
    buffer = io.BytesIO()
    to_pil_image(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
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
        self._base_url = f"http://{VLLM_HOST}:{VLLM_PORT}"

    def load(self) -> None:
        binary = shutil.which("vllm")
        if not binary:
            raise RuntimeError(
                "vllm CLI not found on PATH. Install vLLM in the image "
                "(pip install vllm) or set APIPOD_ENGINE=transformers."
            )
        host = VLLM_HOST
        port = VLLM_PORT
        max_len = VLLM_MAX_MODEL_LEN
        max_num_seqs = VLLM_MAX_NUM_SEQS or "256"
        parser = VLLM_REASONING_PARSER
        speculative_config = VLLM_SPECULATIVE_CONFIG.strip()
        extra = VLLM_EXTRA_ARGS.strip()
        timeout = VLLM_STARTUP_TIMEOUT
        argv = [
            binary, "serve", str(self.weights.ref),
            "--host", host,
            "--port", str(port),
        ]
        if max_len:
            argv.extend(["--max-model-len", max_len])
        argv.extend(["--max-num-seqs", str(max_num_seqs)])
        if parser:
            argv.extend(["--reasoning-parser", parser])
        if speculative_config:
            try:
                parsed_speculative_config = json.loads(speculative_config)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "APIPOD_VLLM_SPECULATIVE_CONFIG must be valid JSON."
                ) from exc
            if not isinstance(parsed_speculative_config, dict):
                raise ValueError(
                    "APIPOD_VLLM_SPECULATIVE_CONFIG must be a JSON object."
                )
            argv.extend([
                "--speculative-config",
                json.dumps(parsed_speculative_config, separators=(",", ":")),
            ])
        if extra:
            argv.extend(shlex.split(extra))

        mode = "speculative" if speculative_config else "standard"
        print(
            f"[apipod] Starting vLLM model={self.weights.ref} "
            f"endpoint=http://{host}:{port} mode={mode} max_num_seqs={max_num_seqs}",
            flush=True,
        )
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
        proc = self._proc
        stdout = getattr(proc, "stdout", None) if proc is not None else None
        if stdout is None:
            return
        for line in stdout:
            text = line.rstrip("\n")
            self._log_tail.append(text)
            print(text, flush=True)

    def _startup_error(self, prefix: str) -> RuntimeError:
        if self._log_thread is not None:
            self._log_thread.join(timeout=2)
        tail = "\n".join(self._log_tail)
        if tail:
            return RuntimeError(f"{prefix}\n--- vllm serve log tail ---\n{tail}")
        return RuntimeError(prefix)

    def _stop(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def _wait_healthy(self, timeout: int = VLLM_STARTUP_TIMEOUT) -> None:
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
        if tool_choice is not None:
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
        parsed = parse_chat_output(text)
        if reasoning and not parsed.get("reasoning_content"):
            parsed["reasoning_content"] = reasoning
        if message.get("tool_calls"):
            parsed["tool_calls"] = message["tool_calls"]
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
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
        logprobs: bool = False,
        top_logprobs=None,
    ):
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, stream=False, logprobs=logprobs, top_logprobs=top_logprobs,
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
        logprobs: bool = False,
        top_logprobs=None,
    ):
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, stream=False, logprobs=logprobs, top_logprobs=top_logprobs,
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
    ) -> Iterator:
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, stream=True,
        )
        parser = ChatOutputParser()
        with httpx.Client(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            with client.stream("POST", self._completion_url(), json=body) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    yield from _parse_sse_delta(line, parser)
        yield from parser.flush()

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
    ) -> AsyncIterator:
        self._ensure_started()
        body = self._request_body(
            messages, images, temperature, max_tokens, top_p, stop, seed,
            tools, tool_choice, stream=True,
        )
        parser = ChatOutputParser()
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, trust_env=False) as client:
            async with client.stream("POST", self._completion_url(), json=body) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    for delta in _parse_sse_delta(line, parser):
                        yield delta
        for delta in parser.flush():
            yield delta


def _parse_sse_delta(line: str, parser: ChatOutputParser) -> Iterator:
    """Convert one vLLM SSE event into APIPod chat deltas."""
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
    if typed:
        yield typed
    content = delta.get("content")
    if content:
        yield from parser.feed(content)
