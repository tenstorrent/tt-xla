# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


@pytest.mark.nightly
@pytest.mark.single_device
def test_llama3_3b_generation():
    prompts = [
        "Tell me a story.",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, max_tokens=64)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_llama3_3b_generation_opt_level_1():
    prompts = [
        "Tell me a story.",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, max_tokens=64)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "optimization_level": 1,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
