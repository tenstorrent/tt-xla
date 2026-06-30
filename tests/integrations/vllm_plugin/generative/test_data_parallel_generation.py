# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data-parallel (DP-only) generation tests.

enable_data_parallel=True / enable_tensor_parallel=False builds an SPMD mesh
(dp_size, 1): weights replicated, the input batch sharded on the "batch" axis
so each replica sees a disjoint subset of sentences.
"""
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_n300_tight(model_name: str):
    """Tight fit: max_num_seqs == dp_size, so one sentence per DP replica."""
    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 2,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            # Qwen3-0.6B fails to compile some DP configs at opt_level=1.
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {text}")
        assert_output_coherent(text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_n300_wider_batch(model_name: str):
    """Wider batch: per-replica batch > 1.

    Nightly, not push: temp>0 sampling with per-replica batch > 1 produces
    garbage (issue #4440); the tight-fit test escapes it (batch == 1).
    """
    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
        "Continue in English: My favourite season is",
        "Continue in English: The best book I have read is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 4,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            # Qwen3-0.6B fails to compile some DP configs at opt_level=1.
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {text}")
        assert_output_coherent(text)

    check_host_memory(model_name)


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_llmbox_padding(model_name: str):
    """Padding path: dp_size (8) > num_reqs (2); max_num_reqs rounds up to a
    multiple of dp_size and _prepare_inputs pads with zero rows."""
    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 2,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            # Qwen3-0.6B fails to compile some DP configs at opt_level=1.
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {text}")
        assert_output_coherent(text)

    check_host_memory(model_name)


@pytest.mark.data_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_llmbox_tight(model_name: str):
    """Tight fit on llmbox: max_num_seqs == dp_size == 8, one sentence per replica."""
    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
        "Continue in English: My favourite season is",
        "Continue in English: The best book I have read is",
        "Continue in English: The most interesting place I visited is",
        "Continue in English: My favourite food is",
        "Continue in English: The thing I enjoy most about weekends is",
        "Continue in English: The future of technology will",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 8,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            # Qwen3-0.6B fails to compile some DP configs at opt_level=1.
            "optimization_level": 0,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {text}")
        assert_output_coherent(text)

    check_host_memory(model_name)
