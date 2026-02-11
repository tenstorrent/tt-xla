# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test vLLM sampling parameters across single-device and multi-chip models.

Targets:
  - single_device:  facebook/opt-125m
  - n300 (dual-chip TP):  meta-llama/Llama-3.2-3B
  - n300_llmbox (8-chip TP):  Qwen/Qwen3-0.6B

Usage examples::

    @for_targets(single_device="push", n300="push", n300_llmbox="nightly")
    @for_targets(single_device="push", n300="push")              # skip n300_llmbox
    @for_targets(single_device="push", n300=("push", pytest.mark.xfail(reason="...")))
"""

import pytest
import sampling_helpers as sh
import vllm

# ---------------------------------------------------------------------------
# for_targets: maps target id -> (fixture name, base marks)
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
    for xfail / skip on individual targets.
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

    return pytest.mark.parametrize("target_llm", params, indirect=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target_llm(request):
    """Resolve the LLM fixture by name (used with indirect parametrize)."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def prompt():
    return ["Once upon a time, there was a"]


@pytest.fixture(scope="module")
def vllm_single_device():
    return vllm.LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.001,
        enable_prefix_caching=False,
        disable_log_stats=True,
        enforce_eager=True,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    )


@pytest.fixture(scope="module")
def vllm_n300():
    return vllm.LLM(
        model="meta-llama/Llama-3.2-3B",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    )


@pytest.fixture(scope="module")
def vllm_n300_llmbox():
    return vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_model_len=128,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@for_targets(single_device="push", n300="push", n300_llmbox="push")
def test_greedy_determinism(target_llm, prompt):
    sh.run_greedy_determinism(target_llm, prompt)


@for_targets(single_device="push", n300="push", n300_llmbox="push")
def test_diversity(target_llm, prompt):
    sh.run_diversity_check(target_llm, prompt)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
@pytest.mark.parametrize(
    "param_name,values",
    sh.SAMPLING_PARAM_SWEEPS,
    ids=[s[0] for s in sh.SAMPLING_PARAM_SWEEPS],
)
def test_param_sweep(target_llm, prompt, param_name, values):
    sh.run_sampling_param_sweep(target_llm, prompt, param_name, values)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_combined(target_llm, prompt):
    sh.run_combined_sampling(target_llm, prompt)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_stop_sequences(target_llm, prompt):
    sh.run_stop_sequences(target_llm, prompt)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_logprobs(target_llm, prompt):
    sh.run_logprobs(target_llm, prompt)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_output_length_controls(target_llm, prompt):
    sh.run_output_length_controls(target_llm, prompt)


@for_targets(single_device="nightly", n300="nightly", n300_llmbox="nightly")
def test_boundary_values(target_llm, prompt):
    sh.run_parameter_boundary_values(target_llm, prompt)
