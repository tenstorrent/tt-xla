# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression guard for SamplingParams(logprobs=N) + enable_trace.

The on-device gather_logprobs graph fails trace-insertion at opt_level=1
(tracked in tt-xla#4387). Until that compiler bug is fixed, model_runner
routes logprob post-processing through a CPU fallback whenever
enable_trace=True. This test exercises the path at opt_level=1 to ensure
the fallback stays wired up.

When the compiler bug is fixed, remove the CPU fallback in
model_runner.py and the startup skip of _precompile_gather_logprobs
together; this test should keep passing via the on-device path.
"""
import pytest
import vllm


@pytest.mark.nightly
@pytest.mark.single_device
def test_opt125m_trace_logprobs():
    """Logprobs + trace at opt_level=1: exercised via CPU fallback."""
    llm = vllm.LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.001,
        additional_config={
            "cpu_sampling": False,
            "enable_trace": True,
            "enable_const_eval": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": 1,
        },
    )
    sp = vllm.SamplingParams(temperature=0, max_tokens=8, logprobs=1)
    out = llm.generate(["Hello, my name is"], sp)[0]
    assert out.outputs[0].logprobs is not None
