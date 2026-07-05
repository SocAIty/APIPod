"""Convenience factory: one-liner model loading for common architectures.

``TransformersLLM`` is the primary preset for causal chat LLMs. Subclass
:class:`apipod.Model` when you need custom load logic (MoE device rules,
embeddings quirks, non-HF weights).
"""
from __future__ import annotations

import threading
from typing import Union

from socaity_cli import requires

from apipod.models.includes import IncludeHandle, include_hf
from apipod.models.model import Model


class TransformersLLM(Model):
    """Built-in chat LLM preset backed by transformers auto classes."""

    def __init__(self, weights: Union[IncludeHandle, str]):
        if isinstance(weights, str):
            weights = include_hf(weights)
        if weights.kind != "hf":
            raise ValueError(
                "TransformersLLM supports Hugging Face includes only. "
                "Pass an HF model id string or an include_hf() handle."
            )
        self.weights = weights

    @requires("transformers", cli=False)
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(self.weights.path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.net = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype="auto", device_map="auto",
        )

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "ping"}], max_tokens=1)

    def _chat_input_ids(self, messages):
        normalized = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        return self.tokenizer.apply_chat_template(
            normalized, add_generation_prompt=True, return_tensors="pt",
        ).to(self.net.device)

    def generate(self, messages, temperature: float = 0.7, max_tokens: int = 512) -> str:
        input_ids = self._chat_input_ids(messages)
        output = self.net.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

    def stream(self, messages, temperature: float = 0.7, max_tokens: int = 512):
        from transformers import TextIteratorStreamer

        input_ids = self._chat_input_ids(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        threading.Thread(target=self.net.generate, kwargs=kwargs, daemon=True).start()
        yield from streamer

    def embed_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Mean-pooled last-layer hidden states (works for most causal LMs)."""
        single = isinstance(text, str)
        texts = [text] if single else text
        max_length = getattr(self.net.config, "max_position_embeddings", 8192)
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(self.net.device)
        import torch

        with torch.no_grad():
            hidden = self.net(**inputs, output_hidden_states=True).hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vectors = pooled.float().cpu().tolist()
        return vectors[0] if single else vectors


def load_model(weights: Union[IncludeHandle, str]) -> TransformersLLM:
    """Return a :class:`TransformersLLM` for a Hugging Face model id or include handle.

    Example:
        llm = apipod.load_model("Qwen/Qwen3.5-7B")
        llm = apipod.load_model(apipod.include_hf("Qwen/Qwen3.5-7B"))
    """
    return TransformersLLM(weights)
