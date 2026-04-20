# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression test: batch-1 (max_num_seqs=1) decode produces garbage output
while batch-2 (max_num_seqs=2) is correct.

The bug manifests as divergent logits on the very first decode step after
prefill, even though prefill logits are identical between batch-1 and batch-2.
This suggests a KV cache or attention issue specific to max_num_reqs=1.

Uses greedy decoding (temperature=0) so output is deterministic and
comparable across runs.

Repro: pytest -svv tests/integrations/vllm_plugin/generative/test_batch1_decode_correctness.py
"""
import pytest
import torch
import vllm


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "Tell me a quick story"
MAX_TOKENS = 64

# Shared config — only max_num_seqs differs between batch-1 and batch-2
BASE_LLM_ARGS = {
    "model": MODEL,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.1,
    "enable_chunked_prefill": False,
    "additional_config": {
        "enable_const_eval": True,
        "min_context_len": 32,
        "experimental_weight_dtype": "bfp_bf8",
        "optimization_level": 1,
#        "enable_trace": True,
    },
}

GREEDY_PARAMS = vllm.SamplingParams(temperature=0, max_tokens=MAX_TOKENS)


def generate_greedy(max_num_seqs: int) -> str:
    """Run greedy generation with the given max_num_seqs and return output text."""
    llm_args = {
        **BASE_LLM_ARGS,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": BASE_LLM_ARGS["max_model_len"] * max_num_seqs,
    }
    llm = vllm.LLM(**llm_args)
    output = llm.generate([PROMPT], GREEDY_PARAMS)
    text = output[0].outputs[0].text
    token_ids = output[0].outputs[0].token_ids
    # Clean up device resources
    llm.llm_engine.engine_core.shutdown()
    del llm
    return text, token_ids


@pytest.mark.single_device
def test_batch1_greedy_coherent():
    """Batch-1 greedy output should be coherent English, not garbage tokens."""
    text, token_ids = generate_greedy(max_num_seqs=1)
    print(f"batch-1 output: {text!r}")
    print(f"batch-1 tokens: {token_ids}")

    # Basic coherence: output should contain common English words
    # and not be full of garbage like '.log', '.swing', '://', 'зрения'
    words = text.lower().split()
    assert len(words) >= 3, f"Output too short to be coherent: {text!r}"

    # Check that most tokens decode to ASCII (not random unicode)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = max(len(text), 1)
    ascii_ratio = ascii_chars / total_chars
    assert ascii_ratio > 0.8, (
        f"Output is mostly non-ASCII ({ascii_ratio:.0%}), likely garbage: {text!r}"
    )


@pytest.mark.single_device
def test_batch1_matches_batch2_greedy():
    """Batch-1 and batch-2 should produce identical greedy output.

    Since greedy decoding is deterministic and both configs process the
    same single prompt, the output must match. Any divergence indicates
    a bug in the model execution at max_num_reqs=1.
    """
    text_b1, tokens_b1 = generate_greedy(max_num_seqs=1)
    text_b2, tokens_b2 = generate_greedy(max_num_seqs=2)

    print(f"batch-1 output: {text_b1!r}")
    print(f"batch-2 output: {text_b2!r}")
    print(f"batch-1 tokens: {tokens_b1}")
    print(f"batch-2 tokens: {tokens_b2}")

    # Find first divergence point for clear error message
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


@pytest.mark.single_device
def test_batch2_greedy_coherent():
    """Batch-2 greedy output should be coherent (baseline/control)."""
    text, token_ids = generate_greedy(max_num_seqs=2)
    print(f"batch-2 output: {text!r}")
    print(f"batch-2 tokens: {token_ids}")

    words = text.lower().split()
    assert len(words) >= 3, f"Output too short to be coherent: {text!r}"

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = max(len(text), 1)
    ascii_ratio = ascii_chars / total_chars
    assert ascii_ratio > 0.8, (
        f"Output is mostly non-ASCII ({ascii_ratio:.0%}), likely garbage: {text!r}"
    )
