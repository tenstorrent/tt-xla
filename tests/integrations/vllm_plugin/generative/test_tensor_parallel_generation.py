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
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_tensor_parallel_generation_n300(model_name: str, use_2d_mesh: bool):
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
            "use_2d_mesh": use_2d_mesh,
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
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
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
            "use_2d_mesh": use_2d_mesh,
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
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param("Qwen/Qwen3-32B", False, "", "True"),
        pytest.param("Qwen/Qwen2.5-32B", False, "", "False"),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp_bf8", "True"),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: str,
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
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory(model_name)


# Gemma-4 is instruction-tuned, so the raw seed prompt degenerates to
# repetitive output on both HF and vLLM. Feed it through llm.chat() so vLLM
# wraps the user turn in the model's chat template before generation.
_GEMMA4_SEED_PROMPT = "I like taking walks in the"


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bhqb
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param(True, "", False),
    ],
)
def test_tensor_parallel_generation_bhqb_multimodal_31b(
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    model_name = "google/gemma-4-31B-it"
    messages = [[{"role": "user", "content": _GEMMA4_SEED_PROMPT}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)
    llm_args = {
        "model": model_name,
        # must be >= max_tokens_per_mm_item=2496 (video: _VIDEO_MAX_FRAMES=32 *
        # (_VIDEO_MAX_SOFT_TOKENS=70 + 2 + 6) = 2496 in
        # vllm/model_executor/models/gemma4_mm.py)
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
            "cpu_sampling": False,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"prompt: {_GEMMA4_SEED_PROMPT}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.single_device
def test_generation_single_device_multimodal_e4b():
    model_name = "google/gemma-4-E4B-it"
    messages = [[{"role": "user", "content": _GEMMA4_SEED_PROMPT}]]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    llm_args = {
        "model": model_name,
        # must be >= max_tokens_per_mm_item=2496 (video: _VIDEO_MAX_FRAMES=32 *
        # (_VIDEO_MAX_SOFT_TOKENS=70 + 2 + 6) = 2496 in
        # vllm/model_executor/models/gemma4_mm.py)
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": False,
            "cpu_sampling": False,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"prompt: {_GEMMA4_SEED_PROMPT}, output: {output_text}")

    check_host_memory(model_name)
