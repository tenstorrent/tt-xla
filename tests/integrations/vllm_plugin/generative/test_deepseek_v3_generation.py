# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
vLLM generation tests for DeepSeek models that use MLA (Multi-Latent Attention).

These tests exercise the TTMLAAttentionBackend path, which dispatches to:
  - torch.ops.tt.flash_mla_prefill          (prefill phase)
  - torch.ops.tt.paged_flash_multi_latent_attention_decode  (decode phase)

DeepSeek-V2-Lite is used as a practical stand-in because it shares the same
MLA architecture as DeepSeek-V3/R1 but is small enough (~2.4B params) to
run on a single Tenstorrent device.
"""

import pytest
import vllm


# ---------------------------------------------------------------------------
# Single-device tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.single_device
def test_deepseek_v2_lite_generation():
    """Basic generation sanity check for a DeepSeek MLA model."""
    prompts = [
        "The capital of France is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    llm_args = {
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.9,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]!r}, output: {output_text!r}")
    assert len(output_text) > 0, "Expected non-empty output from DeepSeek-V2-Lite"


@pytest.mark.nightly
@pytest.mark.single_device
def test_deepseek_v2_lite_generation_multibatch():
    """Multi-request generation to exercise batched prefill with MLA attention."""
    prompts = [
        "Once upon a time",
        "The quick brown fox",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)
    llm_args = {
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "max_num_batched_tokens": 256,
        "max_num_seqs": 2,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.9,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    outputs = llm.generate(prompts, sampling_params)
    for prompt, result in zip(prompts, outputs):
        output_text = result.outputs[0].text
        print(f"prompt: {prompt!r}, output: {output_text!r}")
        assert len(output_text) > 0, f"Expected non-empty output for prompt: {prompt!r}"


@pytest.mark.nightly
@pytest.mark.single_device
def test_deepseek_v2_lite_greedy_determinism():
    """Greedy (temperature=0) decoding must be deterministic across two runs."""
    prompts = ["Artificial intelligence is"]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    llm_args = {
        "model": "deepseek-ai/DeepSeek-V2-Lite",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.9,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }

    llm = vllm.LLM(**llm_args)

    out1 = llm.generate(prompts, sampling_params)[0].outputs[0].text
    out2 = llm.generate(prompts, sampling_params)[0].outputs[0].text

    print(f"run1: {out1!r}")
    print(f"run2: {out2!r}")
    assert out1 == out2, (
        f"Greedy decoding must be deterministic: run1={out1!r}, run2={out2!r}"
    )
