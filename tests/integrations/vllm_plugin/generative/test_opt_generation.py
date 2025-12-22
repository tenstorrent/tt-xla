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
    # output = llm.embed(prompts)
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.push
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
