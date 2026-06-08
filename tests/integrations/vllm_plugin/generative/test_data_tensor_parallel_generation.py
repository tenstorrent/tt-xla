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
