# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Block-boundary arithmetic for multi-token rows in ``_prepare_inputs``.

``paged_fill_cache`` takes no write position: it starts at the first block of the
rolled page table, i.e. at ``(num_computed // block_size) * block_size``. The
chunked SDPA read instead offsets by an exact token position, and that offset is
a single value shared by the whole pass. Two consequences, both covered here:

* a row must be extended left to its block boundary so the write and the read
  agree (``_row_lead``), and
* a pass may only carry rows sitting on the same boundary (``_same_block_run``).

Prefill chunks are block aligned by construction so both are no-ops for them. A
speculative decode row starts wherever decoding happens to be, which is what made
these reachable. Device free.
"""

import numpy as np
import pytest
import torch
from vllm_tt.model_runner import (
    _fill_pass_input_ids,
    _fill_pass_positions,
    _row_lead,
    _same_block_run,
)

# Device free, but the vLLM push matrix only runs device-marked jobs; tag so the
# single_device job collects these (they run in milliseconds).
pytestmark = [pytest.mark.push, pytest.mark.single_device]

BLOCK = 32


def test_row_lead_zero_when_already_block_aligned():
    # Prefill chunks always land here, which is why this never bit them.
    computed = np.array([0, 32, 64, 256], dtype=np.int32)
    assert _row_lead(computed, BLOCK).tolist() == [0, 0, 0, 0]


def test_row_lead_is_offset_into_the_block():
    # A decode position mid block: 41 sits 9 past the boundary at 32.
    computed = np.array([41, 44, 63, 65], dtype=np.int32)
    assert _row_lead(computed, BLOCK).tolist() == [9, 12, 31, 1]


def test_row_lead_dtype_is_int32():
    # Fed into numpy index arithmetic and torch.arange offsets.
    assert _row_lead(np.array([41], dtype=np.int64), BLOCK).dtype == np.int32


def test_same_block_run_covers_whole_pass_when_boundaries_match():
    # 33..63 all sit in block 1, so the pass need not be split.
    computed = np.array([33, 40, 63], dtype=np.int32)
    assert _same_block_run(computed, BLOCK) == 3


def test_same_block_run_truncates_at_first_different_boundary():
    # 41 is block 1, 20 is block 0: one shared read offset cannot describe both,
    # so the pass stops before the second request.
    computed = np.array([41, 20], dtype=np.int32)
    assert _same_block_run(computed, BLOCK) == 1


def test_same_block_run_keeps_leading_matching_rows():
    computed = np.array([33, 40, 80, 41], dtype=np.int32)
    assert _same_block_run(computed, BLOCK) == 2


def test_same_block_run_single_request():
    assert _same_block_run(np.array([41], dtype=np.int32), BLOCK) == 1


def test_same_block_run_empty():
    assert _same_block_run(np.array([], dtype=np.int32), BLOCK) == 0


def test_same_block_run_never_returns_zero_for_a_nonempty_pass():
    # A zero-length pass would stall the multi-pass loop, so the leading run must
    # always include the request it starts from.
    for computed in ([0, 1], [41, 20], [63, 64], [7]):
        arr = np.array(computed, dtype=np.int32)
        assert _same_block_run(arr, BLOCK) >= 1


def test_lead_and_run_agree_on_a_shared_boundary():
    # Every row in the trimmed run must reduce to the same block start, which is
    # what makes one shared chunk_start_idx correct.
    computed = np.array([33, 40, 63, 80], dtype=np.int32)
    run = _same_block_run(computed, BLOCK)
    lead = _row_lead(computed, BLOCK)
    starts = (computed - lead)[:run]
    assert len(set(starts.tolist())) == 1


class TestPassRelativeIndexing:
    """Rows must read the request they belong to, not the one at the same offset.

    ``_prepare_inputs`` handles one pass at a time over global requests
    ``[start_index, start_index + n)`` and builds 0-based per-pass lists, while
    ``input_batch`` stays globally indexed. Every ``input_batch`` read therefore
    has to offset by ``start_index``. It did not, so on any pass after the first a
    row was built from another request's tokens, positions and page table. Latent
    because a single request never leaves pass 0 and the row caps rarely split a
    batch, so only multi-pass steps expose it.
    """

    # 3 requests; token ids chosen so each request's are unmistakable.
    TOKENS = np.array(
        [
            [10, 11, 12, 13, 14, 15, 0, 0],
            [20, 21, 22, 23, 24, 25, 0, 0],
            [30, 31, 32, 33, 34, 35, 0, 0],
        ],
        dtype=np.int32,
    )
    COMPUTED = np.array([0, 3, 5], dtype=np.int32)

    def _rows(self, start_index, num_scheduled, row_lead=None, rows=None, width=8):
        n = len(num_scheduled)
        lead = np.zeros(n, dtype=np.int32) if row_lead is None else np.asarray(row_lead)
        ids = torch.zeros((rows or n, width), dtype=torch.int32)
        pos = torch.zeros((rows or n, width), dtype=torch.int32)
        _fill_pass_input_ids(
            ids,
            torch.from_numpy(self.TOKENS),
            self.COMPUTED,
            np.asarray(num_scheduled, dtype=np.int32),
            lead,
            start_index,
        )
        _fill_pass_positions(
            pos,
            self.COMPUTED,
            np.asarray(num_scheduled, dtype=np.int32),
            lead,
            start_index,
            False,
        )
        return ids, pos

    def test_first_pass_reads_first_requests(self):
        ids, pos = self._rows(start_index=0, num_scheduled=[2, 2])
        # Request 0 computed 0 -> tokens 10,11 at positions 0,1.
        assert ids[0, :2].tolist() == [10, 11]
        assert pos[0, :2].tolist() == [0, 1]
        # Request 1 computed 3 -> tokens 23,24 at positions 3,4.
        assert ids[1, :2].tolist() == [23, 24]
        assert pos[1, :2].tolist() == [3, 4]

    def test_second_pass_reads_its_own_requests(self):
        # The regression: row 0 of this pass is request 1, not request 0.
        ids, pos = self._rows(start_index=1, num_scheduled=[2, 2])
        assert ids[0, :2].tolist() == [23, 24], "row 0 must be request 1"
        assert pos[0, :2].tolist() == [3, 4]
        assert ids[1, :2].tolist() == [35, 0], "row 1 must be request 2"
        assert pos[1, :2].tolist() == [5, 6]

    def test_last_request_alone_in_its_own_pass(self):
        ids, pos = self._rows(start_index=2, num_scheduled=[1])
        assert ids[0, 0].item() == 35
        assert pos[0, 0].item() == 5

    def test_row_lead_extends_backwards_from_the_right_request(self):
        # Request 2 computed 5, lead 2 -> row starts at token index 3.
        ids, pos = self._rows(start_index=2, num_scheduled=[1], row_lead=[2])
        assert ids[0, :3].tolist() == [33, 34, 35]
        assert pos[0, :3].tolist() == [3, 4, 5]

    def test_padding_rows_left_untouched(self):
        # target_num_reqs can exceed the pass size; extra rows stay zero.
        ids, pos = self._rows(start_index=1, num_scheduled=[1], rows=3)
        assert ids[1].tolist() == [0] * 8
        assert pos[2].tolist() == [0] * 8

    def test_zero_scheduled_row_writes_nothing(self):
        ids, _ = self._rows(start_index=0, num_scheduled=[0, 2])
        assert ids[0].tolist() == [0] * 8
        assert ids[1, :2].tolist() == [23, 24]
