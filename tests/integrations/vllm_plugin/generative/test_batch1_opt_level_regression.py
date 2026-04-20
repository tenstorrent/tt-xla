# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression test: optimization_level=1 + max_num_seqs=1 produces garbage decode.

Root cause identified via bisect: optimization_level=1 is the sole trigger.
All other flags (bfp_bf8, consteval, chunked_prefill, max_model_len) are innocent.
Batch-2+ works fine — the bug is specific to batch dim=1.

Uses Llama-3.2-1B-Instruct for faster compile (~45s vs ~250s for 8B).

Repro: pytest -svv tests/integrations/vllm_plugin/generative/test_batch1_opt_level_regression.py
"""
import pytest
import vllm


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "Tell me a quick story"
MAX_TOKENS = 32
GREEDY_PARAMS = vllm.SamplingParams(temperature=0, max_tokens=MAX_TOKENS)


def generate_greedy(max_num_seqs, optimization_level=None):
    """Run greedy generation and return (text, token_ids)."""
    additional_config = {
        "enable_const_eval": False,
        "min_context_len": 32,
    }
    if optimization_level is not None:
        additional_config["optimization_level"] = optimization_level

    llm_args = {
        "model": MODEL,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": 128 * max_num_seqs,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": additional_config,
    }
    llm = vllm.LLM(**llm_args)
    output = llm.generate([PROMPT], GREEDY_PARAMS)
    text = output[0].outputs[0].text
    token_ids = output[0].outputs[0].token_ids
    llm.llm_engine.engine_core.shutdown()
    del llm
    return text, token_ids


def assert_coherent(text, label):
    """Assert output is coherent English, not garbage tokens."""
    print(f"\n--- {label} ---")
    print(f"  prompt: {PROMPT!r}")
    print(f"  output: {text!r}")
    words = text.lower().split()
    assert len(words) >= 3, f"[{label}] Output too short to be coherent: {text!r}"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    assert ascii_ratio > 0.8, (
        f"[{label}] Mostly non-ASCII ({ascii_ratio:.0%}), likely garbage: {text!r}"
    )


# --- Controls: no optimization_level (should all pass) ---

@pytest.mark.single_device
def test_batch1_no_opt():
    """Batch-1 without optimization_level. Should pass."""
    text, _ = generate_greedy(max_num_seqs=1)
    assert_coherent(text, "batch1_no_opt")


@pytest.mark.single_device
def test_batch2_no_opt():
    """Batch-2 without optimization_level. Should pass."""
    text, _ = generate_greedy(max_num_seqs=2)
    assert_coherent(text, "batch2_no_opt")


# --- optimization_level=1 ---

@pytest.mark.single_device
def test_batch1_opt1():
    """Batch-1 with optimization_level=1. Triggers the bug on 8B."""
    text, _ = generate_greedy(max_num_seqs=1, optimization_level=1)
    assert_coherent(text, "batch1_opt1")


@pytest.mark.single_device
def test_batch2_opt1():
    """Batch-2 with optimization_level=1. Should pass."""
    text, _ = generate_greedy(max_num_seqs=2, optimization_level=1)
    assert_coherent(text, "batch2_opt1")


# --- Cross-comparison: batch-1 vs batch-2 with opt_level=1 ---

@pytest.mark.single_device
def test_batch1_vs_batch2_opt1():
    """Greedy output must match between batch-1 and batch-2 with opt_level=1.

    Since greedy is deterministic and both process the same single prompt,
    any divergence indicates a compiler bug at batch dim=1.
    """
    text_b1, tokens_b1 = generate_greedy(max_num_seqs=1, optimization_level=1)
    text_b2, tokens_b2 = generate_greedy(max_num_seqs=2, optimization_level=1)

    print(f"\n--- batch1_vs_batch2_opt1 ---")
    print(f"  prompt:       {PROMPT!r}")
    print(f"  batch-1 opt1: {text_b1!r}")
    print(f"  batch-2 opt1: {text_b2!r}")
    print(f"  batch-1 tokens: {tokens_b1}")
    print(f"  batch-2 tokens: {tokens_b2}")

    for i, (t1, t2) in enumerate(zip(tokens_b1, tokens_b2)):
        if t1 != t2:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL)
            pytest.fail(
                f"Greedy output diverged at token {i}: "
                f"batch-1=[{t1}]'{tok.decode([t1])}' vs "
                f"batch-2=[{t2}]'{tok.decode([t2])}'\n"
                f"batch-1 full: {text_b1!r}\n"
                f"batch-2 full: {text_b2!r}"
            )

    assert tokens_b1 == tokens_b2, (
        f"Token sequences differ:\nbatch-1: {tokens_b1}\nbatch-2: {tokens_b2}"
    )
