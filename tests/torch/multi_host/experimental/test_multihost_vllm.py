# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-host vLLM tensor-parallel inference tests.
"""

import pytest
import torch_xla.runtime as xr
import vllm


def _assert_coherent(text: str) -> None:
    """Cheap sanity check that output is language, not token soup."""
    assert text.strip(), "model produced empty output"
    words = text.lower().split()
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "of",
        "to",
        "and",
        "in",
        "on",
        "at",
        "by",
        "from",
        "or",
        "as",
        "be",
        "that",
        "which",
        "what",
        "who",
    }
    if len(words) >= 5:
        ratio = sum(1 for w in words if w in stopwords) / len(words)
        assert ratio > 0.0, f"output looks incoherent: {text!r}"


@pytest.mark.push
def test_multihost_vllm_inference():
    """
    Test vLLM tensor-parallel inference across multiple hosts.

    Validates:
      - Model loads successfully across all devices
      - Forward pass produces coherent output
      - Multi-host coordination works end-to-end
    """
    xr.set_device_type("TT")

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=32)

    llm = vllm.LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.1,
        additional_config={
            "enable_tensor_parallel": True,
            "use_2d_mesh": xr.global_runtime_device_count() > 8,
            "min_context_len": 32,
            "enable_const_eval": False,
        },
    )

    prompt = "<|user|>\nWhat is machine learning?</s>\n<|assistant|>\n"
    outputs = llm.generate([prompt], sampling_params)

    assert outputs, "No outputs returned"
    output_text = outputs[0].outputs[0].text
    assert output_text, "Output is empty"

    _assert_coherent(output_text)
