# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner batch selection.

``TTModelRunnerV2._order_scheduled_reqs`` / ``_select_batch`` (see
vllm_tt/model_runner_v2.py) are pure host logic: they order a step's scheduled
requests decodes-first and carve out one pass's sub-batch under the SMEM row
caps. They run on cpu with no TT hardware; the SMEM-cap scalars are injected.

They pin the selection math TT owns: decode-first ordering, the max/most model-len
row cap, the prefill-cap
multi-pass split, and the decode/prefill target-bucket + padded query length.
The outputs feed ``_prepare_input_tokens`` and ``_prepare_attn_tensors``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState

VOCAB = 1000
PADDINGS = [1, 32, 64, 128, 256]


def make_runner(**scalars):
    r = object.__new__(TTModelRunnerV2)
    max_num_reqs = scalars.get("max_num_reqs", 4)
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=512,
        max_num_batched_tokens=1024,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.num_reqs_max_model_len = scalars.get("num_reqs_max_model_len", 3)
    r.max_prefill_num_reqs = scalars.get("max_prefill_num_reqs", 2)
    r.min_num_reqs = scalars.get("min_num_reqs", 1)
    r.max_num_reqs = scalars.get("max_num_reqs", 4)
    r.num_tokens_paddings = scalars.get("num_tokens_paddings", PADDINGS)
    r.dp_size = scalars.get("dp_size", 1)
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_order_scheduled_reqs_decodes_first():
    r = make_runner()
    for rid, prompt in [("A", [1]), ("B", [1, 2]), ("C", [3])]:
        r.req_states.add_request(rid, len(prompt), prompt, 0)
    s_a = r.req_states.req_id_to_index["A"]
    s_b = r.req_states.req_id_to_index["B"]
    s_c = r.req_states.req_id_to_index["C"]

    so = SimpleNamespace(num_scheduled_tokens={"A": 1, "B": 10, "C": 1})
    slots, ntoks = r._order_scheduled_reqs(so)

    # Ascending by scheduled tokens: decodes (A, C) before the prefill (B).
    assert ntoks.tolist() == [1, 1, 10]
    assert slots.tolist() == [s_a, s_c, s_b]


@pytest.mark.push
@pytest.mark.cpu
def test_order_skips_reqs_without_slot():
    r = make_runner()
    r.req_states.add_request("A", 1, [1], 0)
    s_a = r.req_states.req_id_to_index["A"]
    so = SimpleNamespace(num_scheduled_tokens={"A": 1, "GHOST": 5})
    slots, ntoks = r._order_scheduled_reqs(so)
    assert slots.tolist() == [s_a]
    assert ntoks.tolist() == [1]


@pytest.mark.push
@pytest.mark.cpu
def test_order_dp_full_grid_position_equals_slot():
    # Under DP the batch position picks the DP replica, so it must equal the
    # request's slot. Ordering returns the FULL replica grid (every slot
    # 0..max_num_reqs-1, position == slot) with per-slot token counts, even when
    # only a subset of slots holds a request -- a lone prefilling request must
    # stay at its own slot's position, not slide to position 0.
    r = make_runner(dp_size=2, max_num_reqs=4)
    for rid, prompt in [("A", [1]), ("B", [1, 2]), ("C", [3])]:
        r.req_states.add_request(rid, len(prompt), prompt, 0)
    s_b = r.req_states.req_id_to_index["B"]

    # Only B is scheduled this step; every other grid position pads to 0 tokens.
    so = SimpleNamespace(num_scheduled_tokens={"B": 5})
    slots, ntoks = r._order_scheduled_reqs(so)

    assert slots.tolist() == [0, 1, 2, 3]
    assert ntoks[s_b] == 5
    assert [ntoks[i] for i in range(4) if i != s_b] == [0, 0, 0]


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_dp_single_full_width_pass():
    # DP: one full-width pass, target == max_num_reqs, no reorder/re-clamp; the
    # padded query length follows the widest scheduled row.
    r = make_runner(dp_size=2, max_num_reqs=4)
    slots = np.array([0, 1, 2, 3], dtype=np.int32)
    ntoks = np.array([0, 5, 0, 1], dtype=np.int32)
    idx, nst, target, padded, end = r._select_batch(slots, ntoks, 0)
    assert idx.tolist() == [0, 1, 2, 3]
    assert nst.tolist() == [0, 5, 0, 1]
    assert target == 4
    assert padded == 32  # first bucket >= 5
    assert end == 4  # single pass consumes the whole grid


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_decode_runs_at_max_bucket():
    r = make_runner(num_reqs_max_model_len=3, max_num_reqs=4)
    slots = np.array([0, 1, 2], dtype=np.int32)
    ntoks = np.array([1, 1, 1], dtype=np.int32)

    idx, nst, target, padded, end = r._select_batch(slots, ntoks, 0)
    assert idx.tolist() == [0, 1, 2]
    assert nst.tolist() == [1, 1, 1]
    assert target == 4  # decode -> max_num_reqs
    assert padded == 1
    assert end == 3


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_prefill_cap_multipass():
    r = make_runner(
        num_reqs_max_model_len=3,
        max_prefill_num_reqs=2,
        min_num_reqs=1,
    )
    slots = np.array([0, 1, 2, 3], dtype=np.int32)
    ntoks = np.array([10, 20, 30, 40], dtype=np.int32)

    idx0, nst0, target0, padded0, end0 = r._select_batch(slots, ntoks, 0)
    assert idx0.tolist() == [0, 1]  # trimmed to the prefill cap
    assert nst0.tolist() == [10, 20]
    assert target0 == 2  # prefill bucket (actual > min_num_reqs)
    assert padded0 == 32  # ceil-bucket of 20
    assert end0 == 2

    idx1, nst1, target1, padded1, end1 = r._select_batch(slots, ntoks, end0)
    assert idx1.tolist() == [2, 3]
    assert padded1 == 64  # ceil-bucket of 40
    assert end1 == 4


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_row_cap_is_always_max_model_len():
    r = make_runner(
        num_reqs_max_model_len=2,
        max_prefill_num_reqs=4,
        min_num_reqs=1,
    )
    slots = np.array([0, 1, 2], dtype=np.int32)
    ntoks = np.array([5, 5, 200], dtype=np.int32)

    idx0, nst0, target0, padded0, end0 = r._select_batch(slots, ntoks, 0)
    # Row cap is always max-model-len cap.
    assert idx0.tolist() == [0, 1]
    assert end0 == 2
    assert padded0 == 32  # ceil-bucket of 5

    idx1, nst1, target1, padded1, end1 = r._select_batch(slots, ntoks, end0)
    assert idx1.tolist() == [2]
    assert target1 == 1  # actual (1) <= min_num_reqs
    assert padded1 == 256  # ceil-bucket of 200
    assert end1 == 3


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_short_rows_still_use_max_model_len_cap():
    r = make_runner(
        num_reqs_max_model_len=2,
        max_prefill_num_reqs=4,
        min_num_reqs=1,
    )
    slots = np.array([0, 1, 2, 3], dtype=np.int32)
    ntoks = np.array([5, 5, 5, 5], dtype=np.int32)

    idx0, nst0, target0, padded0, end0 = r._select_batch(slots, ntoks, 0)
    assert idx0.tolist() == [0, 1]
    assert nst0.tolist() == [5, 5]
    assert target0 == 2
    assert padded0 == 32
    assert end0 == 2
