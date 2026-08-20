"""Capability-based chat preset. Engine is vLLM or transformers, not the public type.

``Chat("Qwen/Qwen3.8-27B-FP8")`` picks vLLM when the CLI is on PATH (or when
``engine="vllm"`` / ``APIPOD_ENGINE=vllm``). Otherwise it uses transformers.
``serve()`` always sees ``generate(..., images=)`` so ``/chat`` accepts images.
"""
from __future__ import annotations

import inspect
import os
import shutil
import threading
from typing import AsyncIterator, Iterator, Optional, Type, Union

from apipod.models.includes import IncludeHandle
from apipod.models.model import Model
from apipod.models.transformers.llm import TransformersLLM
from apipod.models.vllm.chat import VLLMChat

_VALID_ENGINES = ("vllm", "transformers")


def _pick_engine(explicit: Optional[str]) -> str:
    if explicit:
        name = explicit.strip().lower()
    else:
        name = os.environ.get("APIPOD_ENGINE", "").strip().lower()
        if not name:
            name = "vllm" if shutil.which("vllm") else "transformers"
    if name not in _VALID_ENGINES:
        raise ValueError(f"Unknown chat engine {explicit or name!r}. Use 'vllm' or 'transformers'.")
    return name


def _unregistered(cls: Type[Model], *args, **kwargs) -> Model:
    """Construct a Model subclass without appending a second registry entry.

    ``Model.__new__`` registers every instance. ``Chat`` is the public Model;
    the engine instance is an internal delegate.
    """
    instance = object.__new__(cls)
    instance._apipod_loaded = False
    instance._apipod_loading = False
    instance._apipod_load_lock = threading.Lock()
    cls.__init__(instance, *args, **kwargs)
    return instance


class Chat(Model):
    """Chat completion preset. Public surface is generate/stream, not the engine."""

    def __init__(
        self,
        weights: Union[IncludeHandle, str],
        *,
        engine: Optional[str] = None,
        enable_thinking: bool = False,
        transformers_cls: Optional[Type[Model]] = None,
    ):
        self.engine_name = _pick_engine(engine)
        if self.engine_name == "vllm":
            self._engine = _unregistered(VLLMChat, weights, enable_thinking=enable_thinking)
        else:
            cls = transformers_cls or TransformersLLM
            self._engine = _unregistered(cls, weights)
        self.weights = self._engine.weights

    def includes(self):
        return self._engine.includes()

    def load(self) -> None:
        self._engine.load()
        self._engine._apipod_loaded = True

    def warmup(self) -> None:
        self._engine.warmup()

    def _call_engine(self, name: str, messages, images=None, **kwargs):
        self.ensure_loaded()
        method = getattr(self._engine, name)
        params = inspect.signature(method).parameters
        if "images" in params:
            kwargs["images"] = images
        elif images:
            raise ValueError(
                f"{type(self._engine).__name__} does not accept images. "
                "Pass transformers_cls=TransformersVLM (or a VLM subclass) for vision models."
            )
        accepted = {key: value for key, value in kwargs.items() if key in params}
        return method(messages, **accepted)

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
        return self._call_engine(
            "generate",
            messages,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

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
        kwargs = dict(
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        if getattr(type(self._engine), "agenerate", None) is None:
            return self.generate(messages, **kwargs)
        result = self._call_engine("agenerate", messages, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

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
        return self._call_engine(
            "stream",
            messages,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
        )

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
        kwargs = dict(
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
        )
        if getattr(type(self._engine), "astream", None) is None:
            for delta in self.stream(messages, **kwargs):
                yield delta
            return
        stream = self._call_engine("astream", messages, **kwargs)
        async for delta in stream:
            yield delta
