# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bisect which config flag causes batch-1 garbage output.

The full failing config is:
  max_model_len=4096, enable_chunked_prefill=False,
  enable_const_eval=True, bfp_bf8, optimization_level=1

The passing baseline is:
  max_model_len=128, enable_chunked_prefill=True (default),
  enable_const_eval=False, no bfp_bf8, no optimization_level

Each test adds one flag at a time to isolate the trigger.

Repro: pytest -svv tests/integrations/vllm_plugin/generative/test_batch1_bisect.py
"""
import pytest
import vllm


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "Tell me a quick story"
MAX_TOKENS = 32
GREEDY_PARAMS = vllm.SamplingParams(temperature=0, max_tokens=MAX_TOKENS)


def run_batch1(extra_llm_args=None, extra_additional_config=None):
    """Run batch-1 greedy generation with given overrides, return text."""
    additional_config = {"min_context_len": 32}
    if extra_additional_config:
        additional_config.update(extra_additional_config)

    llm_args = {
        "model": MODEL,
        "max_num_batched_tokens": 128,
        "max_model_len": 128,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.1,
        "additional_config": additional_config,
    }
    if extra_llm_args:
        llm_args.update(extra_llm_args)

    # Ensure max_num_batched_tokens >= max_model_len
    llm_args["max_num_batched_tokens"] = max(
        llm_args.get("max_num_batched_tokens", 128),
        llm_args["max_model_len"],
    )

    llm = vllm.LLM(**llm_args)
    output = llm.generate([PROMPT], GREEDY_PARAMS)
    text = output[0].outputs[0].text
    llm.llm_engine.engine_core.shutdown()
    del llm
    return text


def assert_coherent(text, label):
    print(f"{label}: {text!r}")
    words = text.lower().split()
    assert len(words) >= 3, f"[{label}] Output too short: {text!r}"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    assert ascii_ratio > 0.8, f"[{label}] Mostly non-ASCII ({ascii_ratio:.0%}): {text!r}"


# --- Baseline: should pass (same as previously passing config) ---

@pytest.mark.single_device
def test_baseline():
    """Baseline: no extra flags, max_model_len=128. Should pass."""
    text = run_batch1()
    assert_coherent(text, "baseline")


# --- Single flag changes ---

@pytest.mark.single_device
def test_max_model_len_4096():
    """Only change: max_model_len=4096."""
    text = run_batch1(extra_llm_args={"max_model_len": 4096})
    assert_coherent(text, "max_model_len=4096")


@pytest.mark.single_device
def test_chunked_prefill_false():
    """Only change: enable_chunked_prefill=False."""
    text = run_batch1(extra_llm_args={"enable_chunked_prefill": False})
    assert_coherent(text, "chunked_prefill=False")


@pytest.mark.single_device
def test_optimization_level_1():
    """Only change: optimization_level=1."""
    text = run_batch1(extra_additional_config={"optimization_level": 1})
    assert_coherent(text, "opt_level=1")


@pytest.mark.single_device
def test_consteval_bfp8():
    """Only change: enable_const_eval + bfp_bf8 (already tested, should pass)."""
    text = run_batch1(extra_additional_config={
        "enable_const_eval": True,
        "experimental_weight_dtype": "bfp_bf8",
    })
    assert_coherent(text, "consteval+bfp8")


# --- Combinations ---

@pytest.mark.single_device
def test_max_model_len_4096_chunked_false():
    """max_model_len=4096 + chunked_prefill=False."""
    text = run_batch1(extra_llm_args={
        "max_model_len": 4096,
        "enable_chunked_prefill": False,
    })
    assert_coherent(text, "4096+chunked_false")


@pytest.mark.single_device
def test_max_model_len_4096_opt_level_1():
    """max_model_len=4096 + optimization_level=1."""
    text = run_batch1(
        extra_llm_args={"max_model_len": 4096},
        extra_additional_config={"optimization_level": 1},
    )
    assert_coherent(text, "4096+opt_level_1")


@pytest.mark.single_device
def test_full_server_config():
    """All server flags together. Expected to FAIL."""
    text = run_batch1(
        extra_llm_args={
            "max_model_len": 4096,
            "enable_chunked_prefill": False,
        },
        extra_additional_config={
            "enable_const_eval": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": 1,
        },
    )
    assert_coherent(text, "full_server_config")
