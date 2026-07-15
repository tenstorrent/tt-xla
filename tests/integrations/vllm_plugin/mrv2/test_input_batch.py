# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU smoke test for the MRv2 ``TTInputBatch.make_dummy`` builder.

``TTInputBatch`` (see vllm_tt/input_batch_v2.py) is a near-verbatim port of
upstream's transient per-step view, so coverage here is intentionally light --
one shape/layout invariant check on the dummy builder used for warmup. The
real per-step population logic lives in the Phase 3 runner and is tested there.
Runs on cpu (device=torch.device("cpu")), no TT hardware.
"""

import pytest
import torch

from vllm_tt.input_batch_v2 import TTInputBatch, TTInputBuffers


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize(
    "num_reqs, num_tokens, expected_scheduled, expected_qsl, expected_logits",
    [
        (2, 6, [3, 3], [0, 3, 6], [2, 5]),  # even split
        (2, 7, [3, 4], [0, 3, 7], [2, 6]),  # remainder lands on the last req
    ],
)
def test_make_dummy_layout(
    num_reqs, num_tokens, expected_scheduled, expected_qsl, expected_logits
):
    bufs = TTInputBuffers(max_num_reqs=4, max_num_tokens=16, device=torch.device("cpu"))
    ib = TTInputBatch.make_dummy(
        num_reqs=num_reqs, num_tokens=num_tokens, input_buffers=bufs
    )

    assert ib.num_reqs == num_reqs
    assert ib.num_tokens == num_tokens
    assert ib.num_scheduled_tokens.tolist() == expected_scheduled
    assert int(ib.num_scheduled_tokens.sum()) == num_tokens
    assert ib.query_start_loc.tolist() == expected_qsl
    # One logit per request, at each request's last token.
    assert ib.logits_indices.tolist() == expected_logits
    assert ib.idx_mapping.tolist() == list(range(num_reqs))
    assert tuple(ib.input_ids.shape) == (num_tokens,)
    assert tuple(ib.positions.shape) == (num_tokens,)
    # TT has no spec decode / DCP -> these stay at their degenerate defaults.
    assert ib.num_draft_tokens == 0
    assert ib.dcp_local_seq_lens is None
