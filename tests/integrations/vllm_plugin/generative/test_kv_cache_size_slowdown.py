# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end regression test: KV cache pool size vs decode throughput.

Runs OPT-125M through vLLM at small and large gpu_memory_utilization
with the same short prompt. Decode tok/s should not degrade when the
cache pool is larger but the active sequence is the same length.

Usage:
    pytest -svv tests/integrations/vllm_plugin/generative/test_kv_cache_size_slowdown.py
"""

import gc
import os
import time

import pytest
import vllm

# Disable XLA compilation cache so each config compiles fresh
os.environ["VLLM_XLA_CACHE_PATH"] = ""

MODEL = "facebook/opt-125m"
PROMPT = "The quick brown fox jumps over the lazy dog. Once upon a time"
MAX_TOKENS = 64


def _bench(gpu_mem_util):
    """Run OPT-125M at a given gpu_memory_utilization, return decode tok/s."""
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_mem_util,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "cpu_sampling": True,
        },
    )

    # Warmup (includes compilation)
    llm.generate([PROMPT], vllm.SamplingParams(temperature=0.0, max_tokens=8))

    # Benchmark
    params = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    start = time.perf_counter()
    outputs = llm.generate([PROMPT], params)
    elapsed = time.perf_counter() - start

    tokens = len(outputs[0].outputs[0].token_ids)
    tok_s = tokens / elapsed

    del llm
    gc.collect()

    return tok_s


@pytest.mark.single_device
def test_kv_cache_size_decode_slowdown():
    """Decode throughput should not degrade with larger KV cache pool.

    Runs OPT-125M at gpu_memory_utilization 0.005 (small cache) and
    0.05 (large cache). Both use the same short prompt so active
    sequence length is identical. Fails if the large config is more
    than 2x slower.
    """
    small_tok_s = _bench(0.005)
    large_tok_s = _bench(0.05)

    ratio = small_tok_s / large_tok_s
    print(f"\n  gpu_mem=0.005: {small_tok_s:.1f} tok/s")
    print(f"  gpu_mem=0.05:  {large_tok_s:.1f} tok/s")
    print(f"  Slowdown: {ratio:.1f}x")

    assert ratio < 2.0, (
        f"Decode throughput regressed {ratio:.1f}x with 10x larger cache pool "
        f"({small_tok_s:.1f} vs {large_tok_s:.1f} tok/s). "
        f"Expected ~1.0x — cache pool size should not affect decode speed."
    )
