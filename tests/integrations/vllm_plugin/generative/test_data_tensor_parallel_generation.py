# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Combined data-parallel + tensor-parallel (DP+TP) generation tests.

Background
----------
With both `enable_data_parallel=True` and `enable_tensor_parallel=True` the
runner builds an SPMD mesh of shape ``(dp_size, tp_size)``:

  * On an 8-chip llmbox: ``(2, 4)`` — 2 DP replicas, each running 4-way TP.

The model weights are sharded only along the ``"model"`` (TP) axis, so the
DP replicas hold identical weight slices and never communicate. The input
batch (input_ids, position_ids, page_table, cache_position) is sharded along
the ``"batch"`` (DP) axis, so each replica sees a disjoint subset of the
input sentences.

What this test checks
---------------------
1. The engine builds, the model loads, ``shard_model()`` runs, and the
   backbone compiles under the new ``(dp_size, tp_size)`` mesh.
2. ``_dummy_run`` and ``execute_model`` agree on per-device shapes for
   ``input_ids`` / ``position_ids`` / ``page_table`` / ``cache_position``
   (this is what blocked the original DP-only test before the fix).
3. The generated text passes ``assert_output_coherent`` — i.e. the model
   produces natural English, not token soup. This is a check on numerical
   correctness, not just compile success.
4. Host RSS stays within the expected envelope for the model.
"""
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


@pytest.mark.push
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
def test_data_tensor_parallel_generation_push(model_name: str):
    """Smoke test: 2 prompts, max_num_seqs == dp_size (one sentence per replica).

    With 8 chips → mesh (2, 4), `max_num_seqs=2` means each DP replica handles
    exactly 1 sentence per step. This is the tightest DP+TP shape we expect
    to hit and exercises the per-device first-dim == 1 path (same as DP-only).
    """
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
    """Wider batch: 4 prompts, max_num_seqs=4 (2 sentences per DP replica).

    Exercises the per-replica batching path: per-device first-dim is now
    `max_num_seqs / dp_size == 2` rather than 1, which is a different SPMD
    shape than the tight-fit case above.

    NOTE: forces `cpu_sampling=True` because DP+TP uses a 2D mesh and the
    on-device sampler produces token-soup garbage under 2D meshes when
    multiple samples are drawn per device. See issue #4440. The tight-fit
    test above happens to escape this because per-replica batch == 1
    means only one sample is drawn per device per step.
    """
    prompts = [
        "Continue in English: I like taking walks in the",
        "Continue in English: The weather today is",
        "Continue in English: My favourite season is",
        "Continue in English: The best book I have read is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
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
    assert len(outputs) == len(prompts)
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {text}")
        assert_output_coherent(text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param(True, ""),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param([8, 4], marks=pytest.mark.bh_galaxy),
    ],
)
def test_data_tensor_parallel_generation_gemma4_31b(
    mesh_shape: list[int],
    enable_const_eval: bool,
    experimental_weight_dtype: str,
):

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
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        # Gemma-4 mm enforces a floor from MultiModalBudget regardless of
        # limit_mm_per_prompt; 2560 clears the video-frame floor of 2496.
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 32,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.1,  # Do not increase since shard_weights_on_batch_axis is True.
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_data_parallel": True,
            "enable_tensor_parallel": True,
            "shard_weights_on_batch_axis": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "mesh_shape": mesh_shape,
            "cpu_sampling": False,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.chat(messages, sampling_params)
    assert len(outputs) == len(messages)
    for prompt, out in zip(prompts, outputs):
        output_text = out.outputs[0].text
        print(f"prompt: {prompt}, output: {output_text}")
        assert_output_coherent(output_text)

    check_host_memory(model_name)
