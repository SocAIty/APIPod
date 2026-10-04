"""vLLM SSE adapter: first_delta kind, usage event, stream_options."""

from __future__ import annotations

from apipod.models.vllm.chat import (
    _SseParseState,
    _parse_sse_delta,
    _prompt_stats,
)


def test_parse_sse_yields_reasoning_then_usage(capsys):
    state = _SseParseState()
    reasoning = list(
        _parse_sse_delta(
            'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}',
            state,
        )
    )
    assert state.first_kind == "reasoning"
    assert reasoning[0]["reasoning_content"] == "hmm"
    logged = capsys.readouterr().out
    assert "first_delta kind=reasoning" in logged
    assert "elapsed_ms=" in logged
    usage = list(
        _parse_sse_delta(
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1}}',
            state,
        )
    )
    assert usage[0]["object"] == "chat.usage"
    assert usage[0]["usage"]["prompt_tokens"] == 12


def test_parse_sse_logs_cached_tokens(capsys):
    state = _SseParseState()
    list(
        _parse_sse_delta(
            'data: {"choices":[],"usage":{"prompt_tokens":100,'
            '"prompt_tokens_details":{"cached_tokens":80},"completion_tokens":1}}',
            state,
        )
    )
    logged = capsys.readouterr().out
    assert "cached_tokens=80" in logged


def test_parse_sse_aliases_delta_reasoning():
    state = _SseParseState()
    events = list(
        _parse_sse_delta(
            'data: {"choices":[{"delta":{"reasoning":"plan"}}]}',
            state,
        )
    )
    assert state.first_kind == "reasoning"
    assert events[0]["reasoning_content"] == "plan"


def test_prompt_stats_counts_tools():
    body = {
        "messages": [{"role": "user", "content": "hello world"}],
        "tools": [{"type": "function", "function": {"name": "ask_user"}}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    stats = _prompt_stats(body)
    assert "messages=1" in stats
    assert "tools=1" in stats
    assert "chars=" in stats
