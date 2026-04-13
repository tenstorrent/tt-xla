# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for metal trace mode with greedy, device-side sampling."""
import pytest
import vllm


@pytest.mark.single_device
def test_opt_125m_trace():
    """OPT-125m with enable_trace=True, greedy sampling, device-side sampling."""
    prompts = ["Hello, my name is"]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm = vllm.LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.001,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "cpu_sampling": False,
            "enable_trace": True,
        },
    )

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"


@pytest.mark.single_device
def test_llama_3_2_1b_trace():
    """Llama-3.2-1B with enable_trace=True, greedy sampling, device-side sampling."""
    prompts = ["The capital of France is"]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm = vllm.LLM(
        model="meta-llama/Llama-3.2-1B",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "cpu_sampling": False,
            "enable_trace": True,
        },
    )

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"


@pytest.mark.single_device
def test_llama_3_1_8b_trace():
    """Llama-3.1-8B-Instruct with enable_trace=True, greedy, device sampling, bfp8."""
    prompts = ["The capital of France is"]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm = vllm.LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.05,
        additional_config={
            "enable_const_eval": True,
            "cpu_sampling": False,
            "enable_trace": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": 1,
        },
    )

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"


@pytest.mark.single_device
def test_llama_3_1_8b_notrace():
    """Llama-3.1-8B-Instruct baseline without trace, greedy, device sampling."""
    prompts = ["The capital of France is"]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    llm = vllm.LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.05,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "cpu_sampling": False,
            "enable_trace": False,
        },
    )

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert len(output_text) > 0, "Expected non-empty generation"
