# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner host input-token preparation.

``TTModelRunnerV2._prepare_input_tokens`` (see vllm_tt/model_runner_v2.py) is
pure numpy over ``TTRequestState``, so it runs on cpu with no TT hardware and no
model: only ``req_states`` is injected.

They pin the index math that is TT's own responsibility: the unified
prefill/decode gather, the computed-token position offset (chunked prefill), the
2D forward layout with zero padding, and the query_start_loc cumsum + pad.
"""

import numpy as np
import pytest
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState

VOCAB = 1000


def runner_with_reqs(max_num_reqs=4, max_model_len=32):
    r = object.__new__(TTModelRunnerV2)
    r.uses_mrope = False
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_input_tokens_prefill_and_decode_batch():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request(
        "P", prompt_len=5, all_token_ids=[1, 2, 3, 4, 5], num_computed_tokens=0
    )
    rs.add_request("D", prompt_len=3, all_token_ids=[6, 7, 8], num_computed_tokens=0)
    slot_p = rs.req_id_to_index["P"]
    slot_d = rs.req_id_to_index["D"]
    # Simulate D having decoded one token: the writeback grew its token table.
    rs.all_token_ids[slot_d, 3] = 9
    rs.total_len[slot_d] = 4
    rs.num_computed_tokens[slot_d] = 3

    # TT orders decodes first, then prefills; padded batch of 4.
    idx = np.array([slot_d, slot_p], dtype=np.int32)
    nst = np.array([1, 5], dtype=np.int32)
    input_ids, positions, qsl, seq_lens, logits = r._prepare_input_tokens(
        idx, nst, target_num_reqs=4, padded_query_len=5
    )

    assert input_ids.shape == (4, 5)
    # Decode row: single last-sampled token read from all_token_ids.
    assert input_ids[0].tolist() == [9, 0, 0, 0, 0]
    # Prefill row: the whole prompt.
    assert input_ids[1].tolist() == [1, 2, 3, 4, 5]
    # Padding rows stay zero.
    assert input_ids[2].tolist() == [0, 0, 0, 0, 0]
    assert input_ids[3].tolist() == [0, 0, 0, 0, 0]

    assert positions[0].tolist() == [3, 0, 0, 0, 0]
    assert positions[1].tolist() == [0, 1, 2, 3, 4]

    assert seq_lens.tolist() == [4, 5, 0, 0]  # computed + scheduled
    assert logits.tolist() == [0, 4, 0, 0]  # n - 1 per request
    # cumsum([1, 5]) = [1, 6]; tail padded with the last (non-decreasing).
    assert qsl.tolist() == [0, 1, 6, 6, 6]


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_input_tokens_chunked_prefill_offset():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request(
        "C", prompt_len=6, all_token_ids=[10, 11, 12, 13, 14, 15], num_computed_tokens=0
    )
    slot = rs.req_id_to_index["C"]
    rs.num_computed_tokens[slot] = 2  # first chunk of 2 already computed

    idx = np.array([slot], dtype=np.int32)
    nst = np.array([3], dtype=np.int32)  # next chunk of 3 tokens
    input_ids, positions, qsl, seq_lens, logits = r._prepare_input_tokens(
        idx, nst, target_num_reqs=2, padded_query_len=4
    )

    # Gather starts at the computed offset.
    assert input_ids[0].tolist() == [12, 13, 14, 0]
    assert positions[0].tolist() == [2, 3, 4, 0]
    assert seq_lens.tolist() == [5, 0]  # 2 + 3
    assert logits.tolist() == [2, 0]  # n - 1
    assert qsl.tolist() == [0, 3, 3]


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_input_tokens_mrope_three_identical_planes():
    # M-RoPE builds 3D [3, reqs, tokens] positions; text-only inputs have three
    # identical planes (equivalent to 1D RoPE).
    r = runner_with_reqs()
    r.uses_mrope = True
    rs = r.req_states
    rs.add_request("P", prompt_len=4, all_token_ids=[1, 2, 3, 4], num_computed_tokens=0)
    slot = rs.req_id_to_index["P"]
    idx = np.array([slot], dtype=np.int32)
    nst = np.array([4], dtype=np.int32)
    input_ids, positions, _qsl, _seq, _logits = r._prepare_input_tokens(
        idx, nst, target_num_reqs=2, padded_query_len=4
    )

    assert input_ids.shape == (2, 4)  # input_ids stays 2D
    assert positions.shape == (3, 2, 4)
    plane = [0, 1, 2, 3]
    for p in range(3):
        assert positions[p, 0].tolist() == plane
    # All three planes identical.
    assert (positions[0] == positions[1]).all()
    assert (positions[0] == positions[2]).all()
