# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner attention slot-mapping build.

``TTModelRunnerV2._prepare_attn_tensors`` (see vllm_tt/model_runner_v2.py) is the
host substitute for upstream's block-table / slot-mapping Triton kernels. It is
pure numpy over ``TTRequestState`` + the runner block table, so it runs on cpu
with no TT hardware: the runner is allocated without ``__init__`` and the block
table is a fake returning a fixed numpy block map.

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
