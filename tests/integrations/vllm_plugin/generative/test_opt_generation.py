# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


@pytest.mark.push
@pytest.mark.single_device
def test_opt_generation():
    prompts = [
        "Hello, my name is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.push
@pytest.mark.single_device
def test_opt_generation_multibatch():
    prompts = [
        "Hello, my name is",
        "Paris is the capital of",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 256,
        "max_num_seqs": 2,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }

    llm = vllm.LLM(**llm_args)
    output = llm.generate(prompts, sampling_params)
    output_text1 = output[0].outputs[0].text
    output_text2 = output[1].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text1}")
    print(f"prompt: {prompts[1]}, output: {output_text2}")


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size", [8, 16, 32], ids=["batch8", "batch16", "batch32"]
)
def test_opt_generation_large_batch(batch_size):
    prompts = ["The quick brown fox"] * batch_size
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": batch_size * 128,
        "max_num_seqs": batch_size,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == batch_size
    for i, out in enumerate(outputs):
        print(f"[{i}] {out.outputs[0].text}")
