# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner sampled-token writeback.

``TTModelRunnerV2.postprocess`` (see vllm_tt/model_runner_v2.py) is the host
substitute for upstream's post_update Triton kernel. It is pure numpy over
``TTRequestState``, so it runs on cpu with no TT hardware; only ``req_states``
is injected.

They pin the writeback math TT owns: appending the sampled token at total_len so
the next step's input-prep reads it, discarding tokens from still-prefilling
(partial-chunk) requests, and leaving num_computed for the scheduler to advance.
"""

import numpy as np
import pytest
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState

VOCAB = 1000


def runner_with_reqs(max_num_reqs=4, max_model_len=32):
    r = object.__new__(TTModelRunnerV2)
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
def test_postprocess_appends_decode_token():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request("D", prompt_len=3, all_token_ids=[1, 2, 3], num_computed_tokens=0)
    slot = rs.req_id_to_index["D"]
    rs.num_computed_tokens[slot] = 3  # prefill already consumed -> decoding

    idx = np.array([slot], dtype=np.int32)
    nst = np.array([1], dtype=np.int32)  # one decode token scheduled
    valid = r.postprocess(idx, nst, [[99]])

    assert valid == [[99]]
    # Token landed at the old total_len; the table grew by one.
    assert rs.all_token_ids[slot, 3] == 99
    assert rs.total_len[slot] == 4
    assert rs.last_sampled_tokens[slot, 0] == 99
    # num_computed is left for the scheduler (update_requests), not advanced here.
    assert rs.num_computed_tokens[slot] == 3


@pytest.mark.push
@pytest.mark.cpu
def test_postprocess_discards_partial_prefill_token():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request(
        "P", prompt_len=6, all_token_ids=[10, 11, 12, 13, 14, 15], num_computed_tokens=0
    )
    slot = rs.req_id_to_index["P"]
    # First chunk of 4 tokens: seq_len (0 + 4) < prefill_len (6) -> still prefilling.
    idx = np.array([slot], dtype=np.int32)
    nst = np.array([4], dtype=np.int32)
    before = rs.all_token_ids[slot].copy()

    valid = r.postprocess(idx, nst, [[99]])

    assert valid == [[]]  # discarded
    assert rs.total_len[slot] == 6  # unchanged
    np.testing.assert_array_equal(rs.all_token_ids[slot], before)


@pytest.mark.push
@pytest.mark.cpu
def test_postprocess_final_prefill_chunk_appends():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request(
        "F", prompt_len=6, all_token_ids=[10, 11, 12, 13, 14, 15], num_computed_tokens=0
    )
    slot = rs.req_id_to_index["F"]
    rs.num_computed_tokens[slot] = 4  # first chunk already computed
    # Final chunk of 2: seq_len (4 + 2) == prefill_len (6) -> produces a token.
    idx = np.array([slot], dtype=np.int32)
    nst = np.array([2], dtype=np.int32)

    valid = r.postprocess(idx, nst, [[77]])

    assert valid == [[77]]
    assert rs.all_token_ids[slot, 6] == 77
    assert rs.total_len[slot] == 7


@pytest.mark.push
@pytest.mark.cpu
def test_postprocess_mixed_batch():
    r = runner_with_reqs()
    rs = r.req_states
    rs.add_request("D", prompt_len=2, all_token_ids=[1, 2], num_computed_tokens=0)
    rs.add_request(
        "P", prompt_len=5, all_token_ids=[3, 4, 5, 6, 7], num_computed_tokens=0
    )
    slot_d = rs.req_id_to_index["D"]
    slot_p = rs.req_id_to_index["P"]
    rs.num_computed_tokens[slot_d] = 2  # decoding
    # P still prefilling: 3 of 5 tokens this chunk.

    idx = np.array([slot_d, slot_p], dtype=np.int32)
    nst = np.array([1, 3], dtype=np.int32)
    valid = r.postprocess(idx, nst, [[55], [66]])

    assert valid == [[55], []]  # decode kept, partial prefill discarded
    assert rs.all_token_ids[slot_d, 2] == 55
    assert rs.total_len[slot_d] == 3
    assert rs.total_len[slot_p] == 5  # unchanged
