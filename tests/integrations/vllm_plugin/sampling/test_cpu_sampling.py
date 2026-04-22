# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the cpu_sampling path.

cpu_sampling=True runs all sampling ops on CPU instead of compiling a
device sampling graph. These tests verify that the CPU path produces
coherent output for the full set of sampling parameters it supports:
greedy, temperature, top-k, top-p, and repetition penalties.
"""
import pytest
import vllm
from conftest import get_or_create_llm

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "Tell me a quick story"

LLM_ARGS = {
    "model": MODEL,
    "max_num_batched_tokens": 128,
    "max_num_seqs": 1,
    "max_model_len": 128,
    "gpu_memory_utilization": 0.002,
    "additional_config": {
        "enable_const_eval": False,
        "min_context_len": 32,
        "cpu_sampling": True,
    },
}


def llm():
    return get_or_create_llm("cpu_sampling_1b", **LLM_ARGS)


def assert_coherent(text: str, label: str) -> None:
    """Assert output is coherent English, not garbage tokens."""
    print(f"\n--- {label} ---")
    print(f"  prompt: {PROMPT!r}")
    print(f"  output: {text!r}")
    words = text.lower().split()
    assert len(words) >= 3, f"[{label}] Output too short: {text!r}"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    assert ascii_ratio > 0.8, (
        f"[{label}] Non-ASCII garbage ({ascii_ratio:.0%}): {text!r}"
    )


@pytest.mark.single_device
@pytest.mark.nightly
def test_cpu_sampling_greedy():
    """Greedy (temperature=0) on CPU sampling path produces coherent output."""
    params = vllm.SamplingParams(temperature=0, max_tokens=32)
    text = llm().generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "greedy")


@pytest.mark.single_device
@pytest.mark.nightly
def test_cpu_sampling_temperature():
    """Non-greedy temperature sampling produces coherent output."""
    params = vllm.SamplingParams(temperature=0.6, max_tokens=32)
    text = llm().generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "temperature=0.6")


@pytest.mark.single_device
@pytest.mark.nightly
def test_cpu_sampling_top_p():
    """top_p filtering produces coherent output."""
    params = vllm.SamplingParams(temperature=0.8, top_p=0.9, max_tokens=32)
    text = llm().generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "top_p=0.9")


@pytest.mark.single_device
@pytest.mark.nightly
def test_cpu_sampling_top_k():
    """top_k filtering produces coherent output."""
    params = vllm.SamplingParams(temperature=0.8, top_k=50, max_tokens=32)
    text = llm().generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "top_k=50")


@pytest.mark.single_device
@pytest.mark.nightly
def test_cpu_sampling_repetition_penalty():
    """Repetition penalty produces coherent output and suppresses loops."""
    params = vllm.SamplingParams(
        temperature=0.6, repetition_penalty=1.1, max_tokens=32
    )
    text = llm().generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "rep_penalty=1.1")
    # With penalty active, a tight repetition loop should not fill the output.
    words = text.lower().split()
    most_common = max(set(words), key=words.count)
    assert words.count(most_common) <= len(words) // 2, (
        f"Output looks like a repetition loop: {text!r}"
    )
