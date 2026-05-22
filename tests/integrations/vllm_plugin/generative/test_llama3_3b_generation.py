# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


@pytest.mark.nightly
@pytest.mark.single_device
def test_llama3_3b_generation():
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "tt::sampling + enable_trace=True + optimization_level>=1 "
        "triggers a tt-mlir compile-time bug (tt-xla #4570) — rejected by "
        "TTConfig.__post_init__. See test_llama3_3b_generation_trace_opt0 "
        "for the workaround variant. Remove this xfail once the kernel-side "
        "OpModel fix lands and the TTConfig guard is removed."
    ),
)
def test_llama3_3b_generation_trace():
    """Trace variant: greedy + device sampling so the metal-trace path is exercised."""
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "cpu_sampling": False,
            "enable_trace": True,
            "enable_const_eval": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": 1,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Test is failing after tt-mlir uplift on May 20. "
        "Issue: https://github.com/tenstorrent/tt-xla/issues/4878"
    ),
)
def test_llama3_3b_generation_trace_opt0():
    """opt_level=0 workaround variant: preserves 3B trace coverage until #4570 fix lands."""
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "cpu_sampling": False,
            "enable_trace": True,
            "enable_const_eval": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"
