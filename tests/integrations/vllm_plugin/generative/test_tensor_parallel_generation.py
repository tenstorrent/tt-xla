# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


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
    assert_output_coherent(output_text)


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name"],
    [
        pytest.param("Qwen/Qwen3-0.6B"),
    ],
)
@pytest.mark.parametrize("mesh_shape", [[2, 4], [1, 8]])
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    mesh_shape: list[int],
):
    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "mesh_shape": mesh_shape,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "mesh_shape"],
    [
        pytest.param("Qwen/Qwen3-32B", False, "", [2, 4]),
        pytest.param("Qwen/Qwen3-8B", False, "", [1, 8]),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp_bf8", [2, 4]),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    mesh_shape: list[int],
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
            "mesh_shape": mesh_shape,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.xfail(
    strict=True,
    reason=(
        "sdpa_decode tree-reduction limit (max 64 cores/head) exceeded after the "
        "QKV sharding axis swap to ('model', 'batch') in this PR. With kv_heads=8 "
        "split 8-way on galaxy_wh_6u's (4, 8) mesh model axis, each device gets "
        "1 KV head and the kernel allocates the full 72-core grid to it. Fixed "
        "in tt-metal by #44617 (L1-safe defaults for paged SDPA decode, "
        "max_cores_per_head_batch=1). Remove this xfail once tt-mlir uplifts "
        "past tt-metal 5cd3b7a (open uplift PRs: tt-mlir #8484, #8597)."
    ),
)
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "mesh_shape"],
    [pytest.param("mistralai/Mistral-Large-Instruct-2411", True, "bfp_bf8", [4, 8])],
)
def test_tensor_parallel_generation_galaxy_wh_6u_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    mesh_shape: list[int],
):
    inputs = ["How many days ago was Mistral founded?"]

    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.02,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 64,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "mesh_shape": mesh_shape,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(inputs, sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bhqb
@pytest.mark.galaxy_wh_6u
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param(True, ""),
    ],
)
def test_tensor_parallel_generation_gemma4_31b(
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    import torch_xla.runtime as xr

    model_name = "google/gemma-4-31B-it"

    # This test runs on both bhqb (4 devices) and galaxy_wh_6u (32 devices).
    # Pick the mesh shape from the device count: galaxy uses a (8, 4) 2D mesh,
    # while bhqb falls back to the default 1D (1, 4) mesh.
    num_devices = xr.global_runtime_device_count()
    mesh_shape = [8, 4] if num_devices == 32 else None

    messages = [[{"role": "user", "content": "Describe Tenstorrent in one sentence."}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        # Gemma-4 mm enforces a floor from MultiModalBudget regardless of
        # limit_mm_per_prompt; 2560 clears the video-frame floor of 2496.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "mesh_shape": mesh_shape,
            "cpu_sampling": False,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)
