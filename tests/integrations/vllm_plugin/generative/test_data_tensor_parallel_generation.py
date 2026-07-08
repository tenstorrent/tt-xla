# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Combined data-parallel + tensor-parallel (DP+TP) generation tests.

enable_data_parallel=True / enable_tensor_parallel=True builds an SPMD mesh
(dp_size, tp_size) — e.g. (2, 4) on an 8-chip llmbox. Weights are sharded on
the "model" (TP) axis only (DP replicas hold identical slices); the input
batch is sharded on the "batch" (DP) axis.
"""
import pytest
import vllm
from conftest import (
    GROUNDED_BATCH_CHECKS,
    assert_batch_grounded,
    assert_output_coherent,
    check_host_memory,
)


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_tensor_parallel_generation_push(model_name: str):
    """Smoke test: max_num_seqs == dp_size, one sentence per replica (per-device
    first-dim == 1)."""
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
            "enable_tensor_parallel": True,
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
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_tensor_parallel_generation_wider_batch(model_name: str):
    """Wider batch: per-replica batch == 2 (per-device first-dim > 1).

    Greedy + grounded so per-slot prefill corruption is caught deterministically
    (the stopword-ratio heuristic masked it). cpu_sampling=True isolates the
    prefill path from the #4440 device sampler.
    """
    checks = GROUNDED_BATCH_CHECKS
    prompts = [p for p, _ in checks]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 4,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    for (prompt, _), out in zip(checks, outputs):
        print(f"prompt: {prompt}, output: {out.outputs[0].text}")
    assert_batch_grounded(outputs, checks)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param(True, "bfp_bf8"),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param([4, 8], marks=pytest.mark.bh_galaxy),
    ],
)
def test_data_tensor_parallel_generation_devstral_123b(
    mesh_shape: list[int],
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    """Devstral-2-123B DP+TP on the BH galaxy, mesh (4, 8) — 4 DP replicas of
    8-way TP. Eight distinct prompts -> 2 sentences per replica, so this
    exercises real per-replica batching (not the 1-per-replica tight fit).

    Weights are TP-sharded (shard_weights_on_batch_axis=False) and replicated
    across the 4 DP replicas; the input batch is DP-sharded. cpu_sampling=True
    because the on-device sampler produces token-soup on a 2D mesh when >1
    sample is drawn per device (issue #4440).
    """
    model_name = "mistralai/Devstral-2-123B-Instruct-2512"

    prompts = [
        "Continue in English: The three rules of clean code I follow are",
        "Continue in English: A good way to debug a tricky program is to",
        "Continue in English: The main benefit of writing unit tests is",
        "Continue in English: When I design a new API, the first thing I do is",
        "Continue in English: The difference between a stack and a queue is",
        "Continue in English: To make a slow function faster, you can",
        "Continue in English: The reason developers use version control is",
        "Continue in English: A common cause of bugs in concurrent code is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 8,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            "enable_tensor_parallel": True,
            "shard_weights_on_batch_axis": False,
            "experimental_weight_dtype": experimental_weight_dtype,
            "enable_const_eval": enable_const_eval,
            "mesh_shape": mesh_shape,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        output_text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {output_text}")
        assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param(True, "bfp_bf8"),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param([8, 4], marks=pytest.mark.bh_galaxy),
    ],
)
def test_data_tensor_parallel_generation_qwen3_32b(
    mesh_shape: list[int],
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):
    """Qwen3-32B DP+TP on the BH galaxy, mesh (8, 4) — 8 DP replicas of 4-way
    TP. Sixteen distinct prompts -> 2 sentences per replica (real per-replica
    batching). Same DP+TP scheme as the devstral 4x8 case; cpu_sampling=True
    for coherent multi-sample output on a 2D mesh (issue #4440).
    """
    model_name = "Qwen/Qwen3-32B"

    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
        "Continue in English: My favourite season is",
        "Continue in English: The best book I have read is",
        "Continue in English: The most interesting place I visited is",
        "Continue in English: My favourite food is",
        "Continue in English: The thing I enjoy most about weekends is",
        "Continue in English: The future of technology will",
        "Continue in English: The ocean is full of",
        "Continue in English: In the morning I usually",
        "Continue in English: The best way to learn a new language is",
        "Continue in English: On a rainy day I like to",
        "Continue in English: My favourite kind of music is",
        "Continue in English: The city I would most like to visit is",
        "Continue in English: A healthy breakfast usually includes",
        "Continue in English: The stars in the night sky",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 16,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            "enable_tensor_parallel": True,
            "shard_weights_on_batch_axis": False,
            "experimental_weight_dtype": experimental_weight_dtype,
            "enable_const_eval": enable_const_eval,
            "mesh_shape": mesh_shape,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        output_text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {output_text}")
        assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B"])
def test_data_tensor_parallel_generation_llmbox_large(model_name: str):
    """Larger model (8B) via DP+TP on llmbox.

    An 8B model OOMs per DP replica on a single n300 (~16GB > 12.85GB DRAM), so
    weights are sharded across the TP axis. cpu_sampling=True for the 2D-mesh
    sampler issue (#4440).
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
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "cpu_sampling": True,
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
@pytest.mark.tensor_parallel
@pytest.mark.data_parallel
@pytest.mark.bh_galaxy
@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param([8, 4], marks=pytest.mark.bh_galaxy),
    ],
)
def test_data_tensor_parallel_generation_gemma4_31b(mesh_shape: list[int]):

    model_name = "google/gemma-4-31B-it"

    prompts = [
        "Describe Tenstorrent in one sentence.",
        "Explain what a neural network is in one sentence.",
        "What is the capital of France?",
        "Write one sentence about the ocean.",
        "Summarize the theory of relativity in one sentence.",
        "Give me a one-sentence description of photosynthesis.",
        "What is machine learning, in one sentence?",
        "Describe the sun in one sentence.",
        "Explain gravity in one sentence.",
        "Write a single sentence about mountains.",
        "What does a CPU do, in one sentence?",
        "Describe the internet in one sentence.",
        "Summarize the water cycle in one sentence.",
        "What is a black hole, in one sentence?",
        "Give a one-sentence description of a rainforest.",
        "Explain how a battery works in one sentence.",
        "Describe music in one sentence.",
        "What is democracy, in one sentence?",
        "Write one sentence about the moon.",
        "Explain what DNA is in one sentence.",
        "Describe a thunderstorm in one sentence.",
        "What is a programming language, in one sentence?",
        "Summarize evolution in one sentence.",
        "Describe a desert in one sentence.",
        "What is electricity, in one sentence?",
        "Write a single sentence about the human heart.",
        "Explain what a vaccine is in one sentence.",
        "Describe winter in one sentence.",
        "What is the speed of light, in one sentence?",
        "Give a one-sentence description of a volcano.",
        "Describe the stars in one sentence.",
        "What is a robot, in one sentence?",
    ]

    # repeat prompts 8 times
    prompts = prompts * 8
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        ignore_eos=True,  # TODO(@ddilbaz): Remove when https://github.com/tenstorrent/tt-xla/issues/5778 is fixed.
    )

    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 64,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.3,
        "additional_config": {
            "min_context_len": 32,
            "enable_data_parallel": True,
            "enable_tensor_parallel": True,
            "shard_weights_on_batch_axis": True,
            "mesh_shape": mesh_shape,
            "flat_model_io": True,
        },
    }

    # print llm_args in a pretty format
    print("llm_args:\n")
    for key, value in llm_args.items():
        print(f"  {key}: {value}\n")

    llm = vllm.LLM(**llm_args)

    outputs = llm.chat(messages, sampling_params)
    assert len(outputs) == len(messages)
    for prompt, out in zip(prompts, outputs):
        output_text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {output_text}")
        assert_output_coherent(output_text)

    check_host_memory(model_name)
