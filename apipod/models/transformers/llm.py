"""Chat LLM preset backed by transformers auto classes."""
from __future__ import annotations

from socaity_cli import requires

from apipod.models.transformers.base import Transformers


class TransformersLLM(Transformers):
    """Built-in chat LLM preset (``AutoModelForCausalLM`` + ``AutoTokenizer``).

    Covers text chat (``generate``/``stream``) and text embeddings
    (``embed_text``). Subclass :class:`apipod.Model` directly when you need
    custom load logic (MoE device rules, non-HF weights).
    """

    @requires("transformers", cli=False)
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(self.weights.path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.net = AutoModelForCausalLM.from_pretrained(path, **self._from_pretrained_kwargs())

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "ping"}], max_tokens=1)

    def _chat_input_ids(self, messages):
        return self.tokenizer.apply_chat_template(
            self._normalize_messages(messages), add_generation_prompt=True, return_tensors="pt",
        ).to(self.net.device)

    def generate(self, messages, temperature: float = 0.7, max_tokens: int = 512) -> str:
        input_ids = self._chat_input_ids(messages)
        output = self.net.generate(
            input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **self._generation_kwargs(temperature, max_tokens),
        )
        return self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

    def stream(self, messages, temperature: float = 0.7, max_tokens: int = 512):
        input_ids = self._chat_input_ids(messages)
        yield from self._stream_tokens(
            self.tokenizer,
            dict(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **self._generation_kwargs(temperature, max_tokens),
            ),
        )

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
