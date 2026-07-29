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

    def _chat_inputs(self, messages, tools=None):
        inputs = self.tokenizer.apply_chat_template(
            self._text_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.net.device)

    def _text_messages(self, messages) -> list[dict]:
        """Flatten multimodal content parts to plain text (text-only model)."""
        normalized = self._normalize_messages(messages)
        for message in normalized:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
        return normalized

    def generate(
        self,
        messages,
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
        """Chat completion.

        Returns plain text for simple completions. When the model emits
        reasoning or tool calls, or ``logprobs`` is requested, returns a
        ChatCompletionResponse-shaped dict (choices + usage) instead.
        """
        tools = self._prepare_tools(tools, tool_choice)
        if tools:
            self._require_tool_support(self.tokenizer)
        self._apply_seed(seed)
        inputs = self._chat_inputs(messages, tools)
        generation_kwargs = dict(
            pad_token_id=self.tokenizer.eos_token_id,
            **self._generation_kwargs(temperature, max_tokens, top_p, stop, self.tokenizer),
        )
        new_tokens, scores = self._run_generate(inputs, generation_kwargs, with_scores=logprobs)

        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        logprobs_payload = self._token_logprobs(self.tokenizer, new_tokens, scores, top_logprobs) if logprobs else None
        return self._chat_result(
            text,
            prompt_tokens=inputs["input_ids"].shape[-1],
            completion_tokens=len(new_tokens),
            max_tokens=max_tokens,
            logprobs=logprobs_payload,
        )

    def stream(
        self,
        messages,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
    ):
        """Stream typed chat deltas: content as str, reasoning / tool calls as ChatDelta dicts."""
        tools = self._prepare_tools(tools, tool_choice)
        if tools:
            self._require_tool_support(self.tokenizer)
        self._apply_seed(seed)
        inputs = self._chat_inputs(messages, tools)
        yield from self._stream_deltas(
            self.tokenizer,
            dict(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self._generation_kwargs(temperature, max_tokens, top_p, stop, self.tokenizer),
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
