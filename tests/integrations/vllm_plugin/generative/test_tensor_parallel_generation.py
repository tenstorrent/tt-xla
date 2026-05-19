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
    assert_output_coherent(output_text)


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
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param("Qwen/Qwen3-32B", False, "", True),
        pytest.param("Qwen/Qwen2.5-32B", False, "", False),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp_bf8", True),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
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
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V2-Lite"])
@pytest.mark.parametrize("use_2d_mesh", [False])
def test_tensor_parallel_mla_prefill_only(model_name: str, use_2d_mesh: bool):
    """Prefill-only smoke test for the MLA attention backend.

    DeepSeek-V2-Lite is the smallest readily-available model that exercises
    MLAAttention. The TT MLA backend implements prefill but raises
    NotImplementedError on paged decode, so this test sets ``max_tokens=1``
    so that the only generated token comes from the prefill logits and the
    decode path is never invoked.

    Coherence is not asserted (one-token output is below the stopword-ratio
    sampling window); we only check that prefill produced a non-empty
    output token without raising.
    """
    prompts = [
        "I like taking walks in the",
    ]
    # max_tokens=1 → single generated token from prefill logits; no decode.
    # temperature=0.0 makes the prefill→sample path deterministic, which
    # makes diagnosing future regressions easier.
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=1)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.02,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text!r}")
    # One-token output is below the coherence helper's _MIN_WORDS gate;
    # the helper still verifies the output is non-empty.
    assert_output_coherent(output_text)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [pytest.param("mistralai/Mistral-Large-Instruct-2411", True, "bfp_bf8", True)],
)
def test_tensor_parallel_generation_galaxy_wh_6u_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
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
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(inputs, sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)
