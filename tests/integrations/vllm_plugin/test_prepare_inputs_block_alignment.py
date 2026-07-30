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
from vllm_tt.model_runner import _row_lead, _same_block_run

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
