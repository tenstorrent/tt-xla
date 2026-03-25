# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import check_host_memory


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
def test_tensor_parallel_generation_n300(model_name: str):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param("Qwen/Qwen3-0.6B", False, ""),
    ],
)
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param("Qwen/Qwen3-32B", False, ""),
        pytest.param("Qwen/Qwen2.5-32B", False, ""),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp8"),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param("mistralai/Pixtral-Large-Instruct-2411", False, "bfp8")
    ],
)
def test_tensor_parallel_generation_galaxy_wh_6u_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Which of the depicted countries has the best food? Which the second and third and fourth? Name the country, its color on the map and one its city that is visible on the map, but is not the capital. Make absolutely sure to only name a city that can be seen on the map.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
    ]

    inputs = {"model": model_name, "messages": messages}


    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.01,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(inputs, sampling_params)[0].outputs[0].text
    print("output: ", output_text)

    check_host_memory(model_name)
