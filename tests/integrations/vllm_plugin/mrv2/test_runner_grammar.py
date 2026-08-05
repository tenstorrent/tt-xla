# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 structured-output (grammar) host logic.

``TTModelRunnerV2._apply_grammar_bitmask`` / ``structured_decode`` mask logits to
grammar-allowed tokens using a bitwise_and unpack (TT has no bitwise_right_shift),
and ``prepare_structured_decoding_input`` places each structured request's bitmask
row at its per-pass batch position (keyed by idx_mapping). These run purely on CPU.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm_tt.model_runner_v2 import TTModelRunnerV2

VOCAB = 40


def make_runner(max_num_reqs=4):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device("cpu")
    r.vocab_size = VOCAB
    r.max_num_reqs = max_num_reqs
    r.grammar_bitmask_cpu = torch.zeros(
        (max_num_reqs, cdiv(VOCAB, 32)), dtype=torch.int32
    )
    r.require_structured_out_cpu = torch.zeros((max_num_reqs, 1), dtype=torch.bool)
    bm = np.array([1 << i for i in range(32)], dtype=np.uint32).view(np.int32)
    r.structured_decode_bitmasks = torch.from_numpy(bm.copy())
    r.req_states = SimpleNamespace(index_to_req_id={})
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_apply_grammar_bitmask_allows_only_set_bits():
    r = make_runner()
    # Allow tokens 0, 3, 33 -> word0 bits {0,3}=0b1001=9, word1 bit (33-32=1)=2.
    grammar_bitmask = torch.tensor([[9, 2]], dtype=torch.int32)
    logits = torch.ones(1, VOCAB)
    out = r._apply_grammar_bitmask(
        logits, grammar_bitmask, r.structured_decode_bitmasks
    )
    allowed = {i for i in range(VOCAB) if out[0, i].item() != float("-inf")}
    assert allowed == {0, 3, 33}


@pytest.mark.push
@pytest.mark.cpu
def test_structured_decode_only_masks_flagged_rows():
    r = make_runner()
    # Row 0 requires grammar (allow only token 1); row 1 is free.
    require = torch.tensor([[True], [False]])
    grammar_bitmask = torch.tensor([[0b10, 0], [0, 0]], dtype=torch.int32)
    logits = torch.ones(2, VOCAB)
    out = r.structured_decode(
        require, grammar_bitmask, logits, r.structured_decode_bitmasks
    )
    # Row 0: only token 1 survives.
    assert out[0, 1].item() == 1.0
    assert out[0, 0].item() == float("-inf")
    # Row 1: untouched.
    assert torch.all(out[1] == 1.0)


@pytest.mark.push
@pytest.mark.cpu
def test_prepare_structured_decoding_input_keys_by_batch_position():
    r = make_runner()
    # Slots -> req_ids; this pass batches slots [2, 0] at positions [0, 1].
    r.req_states.index_to_req_id = {0: "A", 2: "B", 3: "C"}
    idx_mapping = np.array([2, 0], dtype=np.int32)

    # grammar_bitmask has one row per structured request, in scheduler order.
    # "C" is structured but not in this pass -> its row must be skipped by id,
    # not by position (enumerate index keeps rows aligned).
    grammar_output = SimpleNamespace(
        grammar_bitmask=np.array([[7], [0], [1]], dtype=np.int32),
        structured_output_request_ids=["B", "C", "A"],
    )
    require, bitmask, bitmasks = r.prepare_structured_decoding_input(
        grammar_output, idx_mapping, num_reqs=2
    )

    # B at position 0 (mask row 0 = 7), A at position 1 (mask row 2 = 1).
    assert require[:, 0].tolist() == [True, True]
    assert bitmask[0, 0].item() == 7
    assert bitmask[1, 0].item() == 1
    assert bitmasks.shape == (32,)
