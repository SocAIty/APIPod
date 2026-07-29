"""Shared base for models loaded through the Hugging Face transformers library.

``Transformers`` owns everything the concrete presets (``TransformersLLM``,
``TransformersVLM``) have in common: HF include normalization, the
``from_pretrained`` kwargs (dtype, device map, fastest attention backend) and
the threaded token-streaming loop. Subclasses implement ``load()`` plus their
inference surface.
"""
from __future__ import annotations

import json
import threading
from typing import Iterator, List, Optional, Union

from apipod.common.chat_parsing import ChatOutputParser, Delta, parse_chat_output
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
        """Accept pydantic ChatMessage objects or plain dicts.

        OpenAI wire format carries ``tool_calls[].function.arguments`` as a JSON
        string; HF chat templates iterate it as a mapping, so decode it here.
        """
        normalized = [dict(m) if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in messages]
        for message in normalized:
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                arguments = function.get("arguments") if isinstance(function, dict) else None
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except ValueError:
                        pass
        return normalized

    @staticmethod
    def _normalize_tools(tools) -> Optional[List[dict]]:
        """Accept pydantic Tool objects or plain dicts; None when tools are absent."""
        if not tools:
            return None
        return [t if isinstance(t, dict) else t.model_dump() for t in tools]

    @staticmethod
    def _generation_kwargs(
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        stop=None,
        tokenizer=None,
    ) -> dict:
        kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
        }
        if top_p is not None and top_p < 1.0:
            kwargs["top_p"] = top_p
        if stop:
            # transformers stop_strings needs the tokenizer to detect matches.
            kwargs["stop_strings"] = [stop] if isinstance(stop, str) else list(stop)
            kwargs["tokenizer"] = tokenizer
        return kwargs

    @staticmethod
    def _apply_seed(seed: Optional[int]) -> None:
        if seed is not None:
            import torch

            torch.manual_seed(seed)

    def _template_supports_tools(self, tokenizer) -> bool:
        """A chat template that renders tools references the ``tools`` variable."""
        template = getattr(tokenizer, "chat_template", None) or ""
        return "tools" in template

    def _require_tool_support(self, tokenizer) -> None:
        if not self._template_supports_tools(tokenizer):
            raise ValueError(
                f"{self.weights.ref} does not support tool calls: its chat template "
                "has no 'tools' support. Retry without tools or use a tool-tuned model."
            )

    def _prepare_tools(self, tools, tool_choice) -> Optional[List[dict]]:
        """Apply tool_choice to the tool list.

        'none' disables tools entirely. A named choice narrows the list to that
        one function; 'auto' and 'required' pass all tools to the chat template.
        Forcing a call is best effort (open chat templates cannot constrain
        decoding).
        """
        if tool_choice == "none":
            return None
        tools = self._normalize_tools(tools)
        name = self._tool_choice_name(tool_choice)
        if tools and name:
            tools = [t for t in tools if t.get("function", {}).get("name") == name] or tools
        return tools

    @staticmethod
    def _tool_choice_name(tool_choice) -> Optional[str]:
        """Function name of a NamedToolChoice (dict or pydantic), else None."""
        function = tool_choice.get("function") if isinstance(tool_choice, dict) else getattr(tool_choice, "function", None)
        return function.get("name") if isinstance(function, dict) else None

    def _run_generate(self, inputs, generation_kwargs: dict, with_scores: bool):
        """Run ``net.generate``; return (new_token_ids, per-step scores or None).

        ``inputs`` is an input_ids tensor (LLM) or a processor BatchFeature (VLM).
        """
        if hasattr(inputs, "keys"):
            input_len = inputs["input_ids"].shape[-1]
            run = lambda **kw: self.net.generate(**inputs, **kw)  # noqa: E731
        else:
            input_len = inputs.shape[-1]
            run = lambda **kw: self.net.generate(inputs, **kw)  # noqa: E731

        if with_scores:
            result = run(return_dict_in_generate=True, output_scores=True, **generation_kwargs)
            return result.sequences[0][input_len:], result.scores
        return run(**generation_kwargs)[0][input_len:], None

    def _token_logprobs(self, tokenizer, new_tokens, scores, top_logprobs: Optional[int]) -> dict:
        """Build ChoiceLogprobs payload from per-step generation scores."""
        import torch

        def entry(token_id: int, logp) -> dict:
            token_text = tokenizer.decode([token_id])
            return {"token": token_text, "logprob": float(logp), "bytes": list(token_text.encode("utf-8"))}

        content = []
        for token_id, step_scores in zip(new_tokens.tolist(), scores):
            logp = torch.log_softmax(step_scores[0].float(), dim=-1)
            token_entry = entry(token_id, logp[token_id])
            if top_logprobs:
                top = torch.topk(logp, k=min(top_logprobs, logp.shape[-1]))
                token_entry["top_logprobs"] = [
                    entry(tid, lp) for lp, tid in zip(top.values.tolist(), top.indices.tolist())
                ]
            content.append(token_entry)
        return {"content": content}

    def _chat_result(
        self,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        max_tokens: int,
        logprobs: Optional[dict] = None,
    ):
        """Parse raw output; return plain text or a ChatCompletionResponse-shaped dict.

        Plain text keeps the simple API for simple answers; reasoning, tool
        calls and logprobs need the structured shape.
        """
        message = parse_chat_output(text)
        is_plain = (
            logprobs is None
            and not message.get("tool_calls")
            and not message.get("reasoning_content")
            and message.get("content") is not None
        )
        if is_plain:
            return message["content"]

        if message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif completion_tokens >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        choice = {"index": 0, "message": message, "finish_reason": finish_reason}
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {
            "choices": [choice],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
            },
        }

    def _stream_tokens(self, tokenizer, generate_kwargs: dict) -> Iterator[str]:
        """Run ``net.generate`` on a background thread and yield decoded token deltas."""
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer
        threading.Thread(target=self.net.generate, kwargs=generate_kwargs, daemon=True).start()
        yield from streamer

    def _stream_deltas(self, tokenizer, generate_kwargs: dict) -> Iterator[Delta]:
        """Stream typed chat deltas: plain content as str, reasoning / tool calls as dicts."""
        parser = ChatOutputParser()
        for token in self._stream_tokens(tokenizer, generate_kwargs):
            yield from parser.feed(token)
        yield from parser.flush()
