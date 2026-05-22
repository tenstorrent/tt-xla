# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the device sampling path.

Mirror of test_cpu_sampling.py with cpu_sampling=False and
enable_trace=True so each parameter knob exercises the
chunked_topk_candidates → _ttnn_sampling_padded → tt::sampling
fused-kernel path under metal trace. Each test is parametrized
across three hardware targets via for_targets():

  - single_device:  opt-125m on a single chip
  - n300:           TinyLlama on n300 (TP=2; tied-embeddings off
                    exercises the sharding_constraint_tensor logit
                    replication path — see #3590)
  - n300_llmbox:    Qwen3-0.6B on llmbox (TP=4+)

Each test self-checks via assert_coherent (≥3 words, >80% ASCII)
so a CI failure is unambiguous; eyeball the printed prompt → output
lines to triage what tripped the heuristic.
"""

import signal

import pytest
import vllm
from conftest import TEST_TIMEOUT_SECONDS, get_or_create_llm

PROMPT = "Tell me a quick story"


def assert_coherent(text: str, label: str) -> None:
    """Assert output is coherent English, not garbage tokens."""
    print(f"\n--- {label} ---")
    print(f"  prompt: {PROMPT!r}")
    print(f"  output: {text!r}")
    words = text.lower().split()
    assert len(words) >= 3, f"[{label}] Output too short: {text!r}"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    assert (
        ascii_ratio > 0.8
    ), f"[{label}] Non-ASCII garbage ({ascii_ratio:.0%}): {text!r}"


# ---------------------------------------------------------------------------
# Hardware-target parametrize matrix
# ---------------------------------------------------------------------------

_TARGET_MARKS = {
    "single_device": ("vllm_single_device", [pytest.mark.single_device]),
    "n300": ("vllm_n300", [pytest.mark.tensor_parallel, pytest.mark.dual_chip]),
    "n300_llmbox": (
        "vllm_n300_llmbox",
        [pytest.mark.tensor_parallel, pytest.mark.llmbox],
    ),
}


def for_targets(**kwargs):
    """Parametrize a test across hardware targets with per-target CI tier.

    Pass ``target_id="tier"`` or ``target_id=("tier", extra_mark, ...)``
    to xfail/skip individual targets. Mirrors the pattern in
    test_sampling_params.py.
    """
    params = []
    for target_id, tier_or_tuple in kwargs.items():
        if isinstance(tier_or_tuple, tuple):
            tier, *extra_marks = tier_or_tuple
        else:
            tier = tier_or_tuple
            extra_marks = []
        fixture, base_marks = _TARGET_MARKS[target_id]
        all_marks = base_marks + [getattr(pytest.mark, tier)] + extra_marks
        params.append(pytest.param(fixture, id=target_id, marks=all_marks))
    return pytest.mark.parametrize("llm", params, indirect=True)


# Common additional_config: device sampling with metal trace on so each
# test exercises the trace + tt::sampling combination.
_DEVICE_OPTS = {
    "enable_const_eval": False,
    "min_context_len": 32,
    "cpu_sampling": False,
    # "Trace is causing tests to fail after tt-mlir uplift on May 20. "
    # "Issue: https://github.com/tenstorrent/tt-xla/issues/4878"
    "enable_trace": False,
}


@pytest.fixture
def llm(request):
    """Resolve the per-target LLM fixture by name (used with indirect parametrize)."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def vllm_single_device():
    return get_or_create_llm(
        "opt_125m_device_sampling",
        model="facebook/opt-125m",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.001,
        enable_prefix_caching=False,
        disable_log_stats=True,
        enforce_eager=True,
        additional_config=_DEVICE_OPTS,
    )


@pytest.fixture
def vllm_n300():
    return get_or_create_llm(
        "tinyllama_1b_device_sampling",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={**_DEVICE_OPTS, "enable_tensor_parallel": True},
    )


@pytest.fixture
def vllm_n300_llmbox():
    return get_or_create_llm(
        "qwen3_0_6b_device_sampling",
        model="Qwen/Qwen3-0.6B",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={**_DEVICE_OPTS, "enable_tensor_parallel": True},
    )


@pytest.fixture(autouse=True)
def _test_timeout(llm):
    """Kill any test that hangs longer than TEST_TIMEOUT_SECONDS.

    Depends on ``llm`` so the alarm only starts after the LLM is ready.
    """

    def _handler(signum, frame):
        raise TimeoutError(
            f"Test exceeded {TEST_TIMEOUT_SECONDS}s — vLLM engine likely dead"
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TEST_TIMEOUT_SECONDS)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_device_sampling_greedy(llm):
    """Greedy (temperature=0): hits the argmax fast-path in
    sample_from_logits, bypasses Sampler.sample. Validates the
    device-sampling pipeline end-to-end."""
    params = vllm.SamplingParams(temperature=0, max_tokens=32)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "greedy")


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_device_sampling_temperature(llm):
    """Non-greedy temperature: enters Sampler.sample and exercises the
    chunked_topk_candidates + _ttnn_sampling_padded + tt::sampling chain."""
    params = vllm.SamplingParams(temperature=0.6, max_tokens=32)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "temperature=0.6")


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_device_sampling_top_p(llm):
    """top_p filtering: kernel applies top_p mask on the candidate set."""
    params = vllm.SamplingParams(temperature=0.8, top_p=0.9, max_tokens=32)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "top_p=0.9")


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_device_sampling_top_k(llm):
    """top_k filtering: kernel applies top_k mask on the candidate set."""
    params = vllm.SamplingParams(temperature=0.8, top_k=50, max_tokens=32)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "top_k=50")


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_device_sampling_repetition_penalty(llm):
    """Repetition penalty: applied before sampling; should suppress loops."""
    params = vllm.SamplingParams(temperature=0.6, repetition_penalty=1.1, max_tokens=32)
    text = llm.generate([PROMPT], params)[0].outputs[0].text
    assert_coherent(text, "rep_penalty=1.1")
    # With penalty active, a tight repetition loop should not fill the output.
    words = text.lower().split()
    most_common = max(set(words), key=words.count)
    assert (
        words.count(most_common) <= len(words) // 2
    ), f"Output looks like a repetition loop: {text!r}"
