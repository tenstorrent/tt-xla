# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the on-device (non-greedy) sampler batch>32 truncation.

The tt::sampling kernel requires exactly batch=32, and the plugin sampler pads
to 32 with ``F.pad(..., (0,0,0, 32 - batch))``. When the decode batch exceeds
32 that pad amount is negative, so ``F.pad`` *removes* rows and the sampler
silently returns only 32 sampled token ids. Downstream, ``sample_tokens``
indexes ``valid_sampled_token_ids[i]`` up to ``num_reqs`` (> 32) and raises
``IndexError`` (model_runner.py). See ISSUE / index_error_sampler_analysis.md.

This reproduces it with the smallest possible config: a single chip, a tiny
model, non-greedy sampling, and > 32 concurrent requests. No DP/TP required --
the truncation is a batch-size bug in ``sampler.py``, not a sharding bug, so it
fires single-device too. ``cpu_sampling=False`` is required to take the
on-device sampler path (the CPU sampler has no batch cap and is unaffected).
"""
import gc

import pytest
import vllm


def _shutdown(llm) -> None:
    """Release the TT device before the process exits (explicit engine-core
    shutdown; a bare ``del`` defers teardown to weakref finalize)."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()


@pytest.mark.nightly
@pytest.mark.single_device
def test_sampler_batch_over_32_single_device():
    """> 32 concurrent non-greedy decodes on the on-device sampler.

    Pre-fix: IndexError in sample_tokens (sampler truncates the batch to 32).
    Post-fix: one sampled token per request for all 64 requests.
    """
    n = 64  # > 32 so the decode batch exceeds the sampler's hard 32-cap
    prompts = ["Continue in English: The weather today is"] * n
    sampling_params = vllm.SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=8  # non-greedy -> on-device sampler path
    )
    llm_args = {
        "model": "facebook/opt-125m",
        # vLLM requires max_num_batched_tokens >= max_model_len * max_num_seqs
        # (64 * 64 = 4096) when chunked prefill is off.
        "max_num_batched_tokens": 4096,
        "max_num_seqs": n,
        "max_model_len": 64,
        "gpu_memory_utilization": 0.2,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "num_hidden_layers": 2,
            "cpu_sampling": False,      # REQUIRED: exercise the on-device sampler
            "optimization_level": 0,    # avoid the separate #4387 trace-insertion crash
        },
    }
    llm = vllm.LLM(**llm_args)
    try:
        outs = llm.generate(prompts, sampling_params)
        assert len(outs) == n, f"expected {n} outputs, got {len(outs)}"
        for i, o in enumerate(outs):
            assert o.outputs[0].token_ids, f"request {i} produced no token"
    finally:
        _shutdown(llm)
