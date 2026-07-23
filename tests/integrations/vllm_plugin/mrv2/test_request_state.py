# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 ``TTRequestState`` slot table.

``TTRequestState`` (see vllm_tt/request_state.py) is pure host-side numpy, so
these run on a cpu-only runner with no TT hardware and no model. They pin the
invariants that are TT's own responsibility and stable across the remaining
MRv2 phases: the stable-slot / free-list lifecycle (which replaces v1's
condense-and-shuffle bookkeeping and the slot-corruption bugs it caused),
length accounting, and the numpy substitution for upstream's UVA buffers.

These do NOT exercise the end-to-end runner path -- nothing consumes
``TTRequestState`` until the Phase 3 runner exists.
"""

import numpy as np
import pytest
from vllm_tt.request_state import TTRequestState


def make_state(max_num_reqs=4, max_model_len=32, num_speculative_steps=0):
    return TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=num_speculative_steps,
        vocab_size=1000,
        device="cpu",
    )


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_records_lengths_and_tokens():
    rs = make_state()
    rs.add_request("A", prompt_len=3, all_token_ids=[10, 11, 12], num_computed_tokens=0)

    slot = rs.req_id_to_index["A"]
    assert rs.num_reqs == 1
    assert rs.index_to_req_id[slot] == "A"
    assert rs.prompt_len[slot] == 3
    assert rs.prefill_len[slot] == 3
    assert rs.total_len[slot] == 3
    assert rs.num_computed_tokens[slot] == 0
    assert rs.num_computed_prefill_tokens[slot] == 0
    np.testing.assert_array_equal(rs.all_token_ids[slot, :3], [10, 11, 12])


@pytest.mark.push
@pytest.mark.cpu
def test_free_slot_reused_after_removal_without_condensing():
    """A removed slot returns to the free list and is reused; other requests
    keep their slot and tokens (no v1-style condense/shuffle)."""
    rs = make_state(max_num_reqs=4)
    rs.add_request("A", prompt_len=3, all_token_ids=[10, 11, 12], num_computed_tokens=0)
    rs.add_request("B", prompt_len=2, all_token_ids=[20, 21], num_computed_tokens=0)
    rs.add_request("C", prompt_len=1, all_token_ids=[30], num_computed_tokens=0)

    slot_a, slot_b, slot_c = (rs.req_id_to_index[r] for r in ("A", "B", "C"))

    assert rs.remove_request("B") is True
    assert "B" not in rs.req_id_to_index
    assert slot_b in rs.free_indices

    # A and C are untouched by B's removal.
    assert rs.req_id_to_index["A"] == slot_a
    assert rs.req_id_to_index["C"] == slot_c
    np.testing.assert_array_equal(rs.all_token_ids[slot_a, :3], [10, 11, 12])

    # The next request reuses B's freed slot.
    rs.add_request("D", prompt_len=2, all_token_ids=[40, 41], num_computed_tokens=0)
    assert rs.req_id_to_index["D"] == slot_b
    assert rs.num_reqs == 3


@pytest.mark.push
@pytest.mark.cpu
def test_is_prefilling_mixed_batch():
    rs = make_state()
    # P is mid-prefill: 2 of 5 tokens computed.
    rs.add_request(
        "P", prompt_len=5, all_token_ids=[1, 2, 3, 4, 5], num_computed_tokens=2
    )
    # D has finished prefill (all 3 computed) -> decoding.
    rs.add_request("D", prompt_len=3, all_token_ids=[6, 7, 8], num_computed_tokens=3)

    idx = np.array([rs.req_id_to_index["P"], rs.req_id_to_index["D"]], dtype=np.int32)
    np.testing.assert_array_equal(rs.is_prefilling(idx), [True, False])


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize(
    "num_computed, expected_last",
    [
        (0, 0),  # fresh prefill: slot never seeded (stays zero-initialized)
        (2, 8),  # resumed: seeded with all_token_ids[num_computed - 1]
    ],
)
def test_resumed_request_seeds_last_sampled(num_computed, expected_last):
    rs = make_state()
    rs.add_request(
        "R",
        prompt_len=4,
        all_token_ids=[7, 8, 9, 10],
        num_computed_tokens=num_computed,
    )
    slot = rs.req_id_to_index["R"]
    assert rs.last_sampled_tokens[slot, 0] == expected_last


@pytest.mark.push
@pytest.mark.cpu
def test_remove_request_return_value():
    rs = make_state()
    rs.add_request("A", prompt_len=1, all_token_ids=[10], num_computed_tokens=0)

    assert rs.remove_request("missing") is False
    assert rs.remove_request("A") is True
    assert rs.num_reqs == 0


@pytest.mark.push
@pytest.mark.cpu
def test_apply_staged_writes_is_noop():
    """Locks the numpy substitution: writes land immediately in add_request, so
    apply_staged_writes must not mutate state (upstream flushes UVA here)."""
    rs = make_state()
    rs.add_request("A", prompt_len=2, all_token_ids=[10, 11], num_computed_tokens=0)
    slot = rs.req_id_to_index["A"]
    before = rs.all_token_ids[slot, :2].copy()

    rs.apply_staged_writes()

    assert rs.num_reqs == 1
    np.testing.assert_array_equal(rs.all_token_ids[slot, :2], before)
