# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner attention slot-mapping build.

``TTModelRunnerV2._prepare_attn_tensors`` (see vllm_tt/model_runner_v2.py) builds
the paged-attention tensors. It is pure numpy over ``TTRequestState`` + the runner
block table, so it runs on cpu with no TT hardware: the block table is a fake
returning a fixed numpy block map.

They pin the paged-attention math TT owns: the batch-order gather, the prefix
roll for ``paged_fill_cache``, the zero-scheduled-row redirect, and null padding.
The output feeds ``TTModelState.prepare_attn`` (tested separately).
"""

import numpy as np
import pytest
import torch
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState

VOCAB = 1000
# slot -> block ids (row 3 is the unused/null slot).
BLOCK_MAP = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [0, 0, 0, 0]]


class FakeBlockTable:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, group):  # block_table[0] -> the single kv group
        return self

    def get_cpu_tensor(self):
        return self._arr


def make_runner(block_size=2, max_num_reqs=4, max_model_len=32):
    r = object.__new__(TTModelRunnerV2)
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.block_table = FakeBlockTable(torch.tensor(BLOCK_MAP, dtype=torch.int32))
    r.block_size = block_size
    r._num_kv_cache_groups = 1
    r._group_block_sizes = [block_size]
    r._group_is_sliding = [False]
    r._group_window_blocks = [0]
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_attn_tensors_no_prefix_gathers_in_batch_order():
    r = make_runner()
    # Batch order [slot1, slot0]; no computed prefix.
    idx = np.array([1, 0], dtype=np.int32)
    nst = np.array([5, 3], dtype=np.int32)
    seq_lens = np.array([5, 3, 0, 0], dtype=np.int32)

    page_table, fill_page_table, cache_position = r._prepare_attn_tensors(
        idx, nst, seq_lens, target_num_reqs=4, num_blocks_per_req=4
    )

    assert page_table[0].tolist() == [20, 21, 22, 23]  # slot 1
    assert page_table[1].tolist() == [10, 11, 12, 13]  # slot 0
    assert page_table[2].tolist() == [0, 0, 0, 0]  # padding
    assert cache_position.tolist() == [4, 2, -1, -1]  # seq_lens - 1, pad -1
    # No prefix -> no roll -> fill aliases the read table.
    assert fill_page_table is page_table


@pytest.mark.push
@pytest.mark.cpu
def test_attn_tensors_prefix_roll():
    r = make_runner(block_size=2)
    # slot 0 has 4 computed tokens -> offset 4 // 2 == 2 blocks.
    r.req_states.num_computed_tokens[0] = 4
    idx = np.array([0], dtype=np.int32)
    nst = np.array([3], dtype=np.int32)
    seq_lens = np.array([7, 0], dtype=np.int32)

    page_table, fill_page_table, cache_position = r._prepare_attn_tensors(
        idx, nst, seq_lens, target_num_reqs=2, num_blocks_per_req=4
    )

    assert fill_page_table is not page_table
    # Read path keeps the real order; write path is rolled left by 2.
    assert page_table[0].tolist() == [10, 11, 12, 13]
    assert fill_page_table[0].tolist() == [12, 13, 10, 11]
    assert cache_position.tolist() == [6, -1]


@pytest.mark.push
@pytest.mark.cpu
def test_attn_tensors_zero_scheduled_row_redirects_fill():
    r = make_runner()
    # Row 0 is a re-batched, already-prefilled request: 0 scheduled tokens.
    idx = np.array([0, 1], dtype=np.int32)
    nst = np.array([0, 3], dtype=np.int32)
    seq_lens = np.array([5, 3, 0, 0], dtype=np.int32)

    page_table, fill_page_table, _ = r._prepare_attn_tensors(
        idx, nst, seq_lens, target_num_reqs=4, num_blocks_per_req=4
    )

    assert fill_page_table is not page_table
    # Read path keeps slot 0's real blocks; write path is nulled so it can't
    # clobber the KV written for that request in an earlier step.
    assert page_table[0].tolist() == [10, 11, 12, 13]
    assert fill_page_table[0].tolist() == [0, 0, 0, 0]
    assert fill_page_table[1].tolist() == [20, 21, 22, 23]


def sliding_runner(block_size=2, window_blocks=4, max_num_reqs=4, max_model_len=32):
    """Runner with group 0 full-attention and group 1 a sliding ring."""
    r = make_runner(
        block_size=block_size, max_num_reqs=max_num_reqs, max_model_len=max_model_len
    )
    r.max_num_reqs = max_num_reqs
    r._num_kv_cache_groups = 2
    r._group_block_sizes = [block_size, block_size]
    r._group_is_sliding = [False, True]
    r._group_window_blocks = [0, window_blocks]
    r._req_ring_slot = {}
    r._free_ring_slots = list(range(max_num_reqs))
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_sliding_ring_points_each_row_at_its_own_sub_ring():
    # Row i owns physical blocks [1 + slot_i*wb, 1 + (slot_i+1)*wb); block 0 is
    # the shared null sink. Two concurrent requests must not overlap.
    r = sliding_runner(block_size=2, window_blocks=4)
    r.req_states.add_request(
        "A", prompt_len=2, all_token_ids=[1, 2], num_computed_tokens=0
    )
    r.req_states.add_request(
        "B", prompt_len=2, all_token_ids=[3, 4], num_computed_tokens=0
    )
    sa = r.req_states.req_id_to_index["A"]
    sb = r.req_states.req_id_to_index["B"]

    pt, fill_pt, cache_pos = r._prepare_sliding_attn_tensors(
        np.array([sa, sb], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),  # decode rows
        np.array([2, 2], dtype=np.int32),
        2,
        16,
        1,
    )

    assert pt.shape == (2, 4)  # width capped by window_blocks
    # Disjoint sub-rings, and never the null block for an active row.
    assert set(pt[0]).isdisjoint(set(pt[1]))
    assert 0 not in pt[:2]
    # Context (2) <= window (8 tokens): identity rotation, absolute position.
    assert cache_pos.tolist() == [1, 1]
    assert fill_pt is pt  # no inactive rows, so fill == read


@pytest.mark.push
@pytest.mark.cpu
def test_sliding_ring_slot_is_stable_across_steps():
    # The slot must survive re-batching: a later step for the same req_id has to
    # read the same sub-ring it wrote.
    r = sliding_runner()
    r.req_states.add_request(
        "A", prompt_len=2, all_token_ids=[1, 2], num_computed_tokens=0
    )
    slot = r.req_states.req_id_to_index["A"]
    args = (
        np.array([slot], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([2], dtype=np.int32),
        1,
        16,
        1,
    )
    first, _, _ = r._prepare_sliding_attn_tensors(*args)
    second, _, _ = r._prepare_sliding_attn_tensors(*args)
    assert first.tolist() == second.tolist()


@pytest.mark.push
@pytest.mark.cpu
def test_sliding_ring_refuses_unsupported_prefill():
    # paged_fill_cache matches fill block k to page_table[k] positionally, so a
    # multi-token row continuing a cached prefix would corrupt KV. Same
    # restriction as the v1 runner.
    r = sliding_runner()
    r.req_states.add_request(
        "A", prompt_len=4, all_token_ids=[1, 2, 3, 4], num_computed_tokens=2
    )
    slot = r.req_states.req_id_to_index["A"]
    with pytest.raises(NotImplementedError, match="sliding-window prefill"):
        r._prepare_sliding_attn_tensors(
            np.array([slot], dtype=np.int32),
            np.array([2], dtype=np.int32),  # filling (>1 token)
            np.array([4], dtype=np.int32),
            1,
            16,
            1,
        )


@pytest.mark.push
@pytest.mark.cpu
def test_per_group_builder_uses_the_right_builder_per_group():
    # window_blocks (2) < the pool width (4), so the two groups are told apart by
    # their page-table widths.
    r = sliding_runner(window_blocks=2)
    r.req_states.add_request(
        "A", prompt_len=2, all_token_ids=[1, 2], num_computed_tokens=0
    )
    slot = r.req_states.req_id_to_index["A"]
    per_group = r._prepare_attn_tensors_per_group(
        np.array([slot], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([2], dtype=np.int32),
        1,
        4,
    )
    assert len(per_group) == 2
    # Full group keeps the pool width; the ring is window-width.
    assert per_group[0][0].shape[1] == 4
    assert per_group[1][0].shape[1] == 2
