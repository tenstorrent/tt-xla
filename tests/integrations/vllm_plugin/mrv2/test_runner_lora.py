# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 LoRA host mapping (_make_lora_inputs).

``TTModelRunnerV2._make_lora_inputs`` turns the per-slot active-LoRA table into
the (prompt_lora_mapping, token_lora_mapping, lora_requests) triple the LoRA
mixin's _set_active_loras consumes.
"""
import numpy as np
import pytest
from vllm_tt.model_runner_v2 import TTModelRunnerV2


class FakeLoRA:
    """Hashable stand-in for LoRARequest (added to a set in _make_lora_inputs)."""

    def __init__(self, int_id):
        self.lora_int_id = int_id


def runner(slot_to_lora):
    r = object.__new__(TTModelRunnerV2)
    r.lora_requests_by_slot = slot_to_lora
    return r


def lora(int_id):
    return FakeLoRA(int_id)


@pytest.mark.push
@pytest.mark.cpu
def test_make_lora_inputs_repeats_by_token_counts():
    lora_a = lora(1)
    lora_b = lora(2)
    # slot 5 -> lora 1, slot 3 -> lora 2.
    r = runner({5: lora_a, 3: lora_b})
    idx = np.array([5, 3], dtype=np.int32)
    nst = np.array([2, 4], dtype=np.int32)

    prompt_map, token_map, reqs = r._make_lora_inputs(idx, nst)

    # One sampled token per request -> one prompt-map entry per request.
    assert prompt_map == (1, 2)
    # token map repeats each id by its scheduled-token count.
    assert token_map == (1, 1, 2, 2, 2, 2)
    assert reqs == {lora_a, lora_b}


@pytest.mark.push
@pytest.mark.cpu
def test_make_lora_inputs_base_model_is_zero():
    lora_a = lora(7)
    # slot 0 has a LoRA; slot 1 is base model (no entry -> id 0).
    r = runner({0: lora_a})
    idx = np.array([0, 1], dtype=np.int32)
    nst = np.array([1, 3], dtype=np.int32)

    prompt_map, token_map, reqs = r._make_lora_inputs(idx, nst)

    assert prompt_map == (7, 0)
    assert token_map == (7, 0, 0, 0)
    # Only the real LoRA request is active.
    assert reqs == {lora_a}


@pytest.mark.push
@pytest.mark.cpu
def test_make_lora_inputs_all_base_model():
    r = runner({})
    idx = np.array([2, 4], dtype=np.int32)
    nst = np.array([1, 1], dtype=np.int32)

    prompt_map, token_map, reqs = r._make_lora_inputs(idx, nst)

    assert prompt_map == (0, 0)
    assert token_map == (0, 0)
    assert reqs == set()
