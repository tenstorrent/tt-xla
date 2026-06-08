# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data-parallel (DP-only) generation tests.

Background
----------
With `enable_data_parallel=True` and `enable_tensor_parallel=False` the
runner builds an SPMD mesh of shape ``(dp_size, 1)``:

  * On an n300 (2-chip):  ``(2, 1)`` — 2 DP replicas, no TP.
  * On an llmbox (8-chip): ``(8, 1)`` — 8 DP replicas, no TP.

All model weights are replicated across DP replicas. The input batch
(input_ids, position_ids, page_table, cache_position) is sharded along the
``"batch"`` axis so each replica sees a disjoint subset of input sentences.

What this test checks
---------------------
1. The engine builds, the model loads, and the backbone compiles under the
   DP-only ``(dp_size, 1)`` mesh.
2. The batch-padding logic in ``_prepare_inputs`` correctly rounds up to a
   multiple of ``dp_size`` when ``num_reqs < dp_size``.
3. The generated text passes ``assert_output_coherent`` — natural English,
   not token soup.
4. Host RSS stays within the expected envelope.
"""
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_n300_tight(model_name: str):
    """Tight fit: 2 prompts, max_num_seqs=2, one sentence per DP replica.

    On n300 (2 chips) dp_size=2, so max_num_seqs=2 means each replica
    handles exactly 1 sentence per step.
    """
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
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_parallel_generation_n300_wider_batch(model_name: str):
    """Wider batch: 4 prompts, max_num_seqs=4, two sentences per DP replica.

    Exercises the per-replica batching path on n300 (dp_size=2): each
    replica processes 2 sentences per step rather than 1.
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
    """Padding path: 2 prompts on 8 chips (dp_size=8 > num_reqs=2).

    Exercises the ``max_num_reqs`` rounding-up logic: with dp_size=8 the
    runner adjusts max_num_reqs to the next multiple of 8, and
    ``_prepare_inputs`` pads each batch to that multiple with zero rows.
    """
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
    """Tight fit: 8 prompts, max_num_seqs=8, one sentence per DP replica.

    On llmbox (8 chips) dp_size=8, so max_num_seqs=8 means each replica
    handles exactly 1 sentence per step.
    """
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
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B"])
def test_data_parallel_generation_llmbox_large(model_name: str):
    """Nightly: larger model, 8 DP replicas on llmbox."""
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
