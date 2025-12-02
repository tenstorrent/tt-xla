# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm


@pytest.mark.push
def test_opt_generation():
    prompts = [
        "Hello, my name is",
        "Paris is the capital of",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_num_batched_tokens": 128,
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
    # output = llm.embed(prompts)
    print(f"prompt: {prompts[0]}, output: {output_text1}")
    print(f"prompt: {prompts[1]}, output: {output_text2}")

def test_concurrent_requests():
    """동시 요청 시뮬레이션 - 첫 번째 decode 중 두 번째 prefill"""
    
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
    }
    
    llm = vllm.LLM(**llm_args)
    
    # 첫 번째 요청
    print("\n=== Request 1 ===")
    output1 = llm.generate(["Hello, how are you?"], sampling_params)
    print(f"Response 1: {output1[0].outputs[0].text}")
    
    # 두 번째 요청 (첫 번째 완료 후) - 이때 garbage 발생 가능
    print("\n=== Request 2 ===")
    output2 = llm.generate(["Paris is the capital of"], sampling_params)
    print(f"Response 2: {output2[0].outputs[0].text}")
    
