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
    assert_slots_agree,
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
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_tensor_parallel_chunked_prefill_llmbox(model_name: str, batch_size: int):
    """Chunked prefill under DP+TP on llmbox (tt-xla #4986/#5691)."""
    prompts = [
        "Continue in English: I like taking walks in the evening after work, "
        "usually along the river path that starts behind the old train station "
        "and follows the water for about two miles before it turns back toward "
        "the main road, and on a clear night you can see the lights of the town "
        "reflected in the water while people cycle past and dogs run ahead of "
        "their owners, and by the time I reach the bridge the shops have closed "
        "and the streets are quiet enough that I can hear my own footsteps, and "
        "it gives me time to think about everything that happened during the day "
        "before I go home and start cooking dinner, which is why the part of the "
        "day I look forward to most is",
        "Continue in English: The weather today is",
        "Continue in English: My favourite season is autumn, mostly because the "
        "air turns cool enough for a jacket but not so cold that you dread going "
        "outside, and the trees along my street change colour over about two "
        "weeks until the whole road is orange and red and the pavement is "
        "covered in leaves that crunch when you step on them, and the evenings "
        "get dark early enough that the windows of the houses all light up by "
        "six o'clock, which makes the walk home feel warmer than it actually is, "
        "and on the weekends there is usually enough sun to sit outside with a "
        "coffee for an hour, and I always tell myself that this year I will take "
        "more photographs before the leaves are gone, so the thing I look "
        "forward to every year is",
        "Continue in English: The best book I have read is",
    ] * (batch_size // 4)
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_seqs": batch_size,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.25,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "cpu_sampling": False,
            "prefill_chunk_size": 128,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    assert_slots_agree(outputs, prompts)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-8B"])
def test_data_tensor_parallel_chunked_prefill_llmbox_large(
    model_name: str, batch_size: int
):
    """Chunked prefill under DP+TP on llmbox with TP-sharded 8B weights."""
    prompts = [
        "Continue in English: I like taking walks in the evening after work, "
        "usually along the river path that starts behind the old train station "
        "and follows the water for about two miles before it turns back toward "
        "the main road, and on a clear night you can see the lights of the town "
        "reflected in the water while people cycle past and dogs run ahead of "
        "their owners, and by the time I reach the bridge the shops have closed "
        "and the streets are quiet enough that I can hear my own footsteps, and "
        "it gives me time to think about everything that happened during the day "
        "before I go home and start cooking dinner, which is why the part of the "
        "day I look forward to most is", # prompt 0
        "Continue in English: The weather today is", # prompt 1
        "Continue in English: My favourite season is autumn, mostly because the "
        "air turns cool enough for a jacket but not so cold that you dread going "
        "outside, and the trees along my street change colour over about two "
        "weeks until the whole road is orange and red and the pavement is "
        "covered in leaves that crunch when you step on them, and the evenings "
        "get dark early enough that the windows of the houses all light up by "
        "six o'clock, which makes the walk home feel warmer than it actually is, "
        "and on the weekends there is usually enough sun to sit outside with a "
        "coffee for an hour, and I always tell myself that this year I will take "
        "more photographs before the leaves are gone, so the thing I look "
        "forward to every year is", # prompt 2
        "Continue in English: The best book I have read is", # prompt 3
    ] * (batch_size // 4)
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_seqs": batch_size,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.25,
        "enable_prefix_caching": False,
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "enable_data_parallel": True,
            "cpu_sampling": False,
            "prefill_chunk_size": 128,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    assert_slots_agree(outputs, prompts)

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
