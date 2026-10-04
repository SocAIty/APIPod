"""vLLM serve argv without spawning the engine."""

from apipod.models.vllm.chat import _cached_tokens, vllm_serve_argv


def test_vllm_serve_argv_recipe_flags():
    argv = vllm_serve_argv(
        "vllm",
        "Qwen/Qwen3.8-27B-FP8",
        host="127.0.0.1",
        port=18000,
        max_num_seqs="8",
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        speculative_config='{"method":"mtp","num_speculative_tokens":3}',
        extra="--kv-cache-dtype fp8 --gpu-memory-utilization 0.95",
    )
    assert "--enable-prefix-caching" not in argv
    assert "--mamba-cache-mode" not in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--max-num-seqs") + 1] == "8"


def test_vllm_serve_argv_skips_duplicate_bool_without_eating_next_flag():
    argv = vllm_serve_argv(
        "vllm",
        "m",
        host="127.0.0.1",
        port=18000,
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_coder",
        extra="--enable-auto-tool-choice --kv-cache-dtype fp8",
    )
    assert argv.count("--enable-auto-tool-choice") == 1
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"


def test_cached_tokens_reads_openai_and_vllm_fields():
    assert _cached_tokens({"prompt_tokens_details": {"cached_tokens": 32000}}) == 32000
    assert _cached_tokens({"cached_tokens": 12}) == 12
    assert _cached_tokens({"num_cached_tokens": 7}) == 7
    assert _cached_tokens({"prompt_tokens": 9}) is None
