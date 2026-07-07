"""Shared base for models loaded through the Hugging Face transformers library.

``Transformers`` owns everything the concrete presets (``TransformersLLM``,
``TransformersVLM``) have in common: HF include normalization, the
``from_pretrained`` kwargs (dtype, device map, fastest attention backend) and
the threaded token-streaming loop. Subclasses implement ``load()`` plus their
inference surface.
"""
from __future__ import annotations

import threading
from typing import Iterator, List, Union

from apipod.models.includes import IncludeHandle, include_hf
from apipod.models.model import Model


class Transformers(Model):
    """Base for transformers-backed model presets.

    Construction only declares the weights (an ``include_hf`` handle); no
    download or GPU work happens before ``load()``. Subclasses set ``self.net``
    (the torch module) in ``load()``.
    """

    def __init__(self, weights: Union[IncludeHandle, str]):
        if isinstance(weights, str):
            weights = include_hf(weights)
        if weights.kind != "hf":
            raise ValueError(
                f"{type(self).__name__} supports Hugging Face includes only. "
                "Pass an HF model id string or an include_hf() handle."
            )
        self.weights = weights

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    @staticmethod
    def attn_implementation() -> str:
        """Fastest attention backend available on this machine.

        flash-attn 2 gives the largest speed and memory gains (especially for
        multi-image and video prompts) but needs an Ampere+ GPU and the
        compiled ``flash_attn`` package. PyTorch SDPA is the universal
        fallback and the transformers default.
        """
        try:
            import flash_attn  # noqa: F401
            import torch

            if torch.cuda.is_available():
                return "flash_attention_2"
        except ImportError:
            pass
        return "sdpa"

    def _from_pretrained_kwargs(self) -> dict:
        """Standard ``from_pretrained`` kwargs shared by all presets."""
        return {
            "trust_remote_code": True,
            "dtype": "auto",
            "device_map": "auto",
            "attn_implementation": self.attn_implementation(),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(messages) -> List[dict]:
        """Accept pydantic ChatMessage objects or plain dicts."""
        return [m if isinstance(m, dict) else m.model_dump() for m in messages]

    @staticmethod
    def _generation_kwargs(temperature: float, max_tokens: int) -> dict:
        return {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
        }

    def _stream_tokens(self, tokenizer, generate_kwargs: dict) -> Iterator[str]:
        """Run ``net.generate`` on a background thread and yield decoded token deltas."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer
        threading.Thread(target=self.net.generate, kwargs=generate_kwargs, daemon=True).start()
        yield from streamer
