# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner _run_model_pass plumbing.

``TTModelRunnerV2._run_model_pass`` (see vllm_tt/model_runner_v2.py) copies the
host arrays to the device, builds attention + sampling metadata, runs the
compiled forward/sample graph, and returns the sampled tokens. The metadata
builds (TTModelState.prepare_attn + XLASupportedSamplingMetadata.from_v2_states)
are real and run on cpu; the compiled graph itself needs a model + TT device, so
it is faked here (and validated on real hardware separately).

They pin the plumbing: host->device buffer copies, the real metadata builds, the
compiled-graph dispatch, and output slicing to the active requests.
"""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.model_state import TTModelState
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates

VOCAB = 1000
BLOCK_MAP = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [0, 0, 0, 0]]


class FakeBlockTable:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, group):
        return self

    def get_cpu_tensor(self):
        return self._arr


def make_runner(max_num_reqs=4, max_model_len=32):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device("cpu")
    r.sampling_device = torch.device("cpu")
    r.vocab_size = VOCAB
    r.dp_size = 1
    r.block_size = 16
    r.max_num_blocks_per_req = 4
    r.attention_layer_names = ("layer.0", "layer.1")
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.sampling_states = TTSamplingStates(max_num_reqs=max_num_reqs, vocab_size=VOCAB)
    r.block_table = FakeBlockTable(torch.tensor(BLOCK_MAP, dtype=torch.int32))
    # prepare_attn uses no instance state.
    r.model_state = object.__new__(TTModelState)
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_run_model_pass_builds_metadata_and_returns_tokens():
    r = make_runner()
    # Two decoding requests occupying stable slots.
    r.req_states.add_request(
        "A", prompt_len=2, all_token_ids=[1, 2], num_computed_tokens=0
    )
    r.req_states.add_request(
        "B", prompt_len=1, all_token_ids=[3], num_computed_tokens=0
    )
    slot_a = r.req_states.req_id_to_index["A"]
    slot_b = r.req_states.req_id_to_index["B"]
    for slot in (slot_a, slot_b):
        r.sampling_states.add_request(slot, SamplingParams(temperature=0.0))

    captured = {}

    def fake_forward(
        input_ids, positions, logits_indices, attn_metadata, sampling_metadata
    ):
        # The real metadata builds must have produced usable objects.
        captured["attn_layers"] = set(attn_metadata)
        captured["all_greedy"] = sampling_metadata.all_greedy
        captured["input_ids_device"] = input_ids.device.type
        n = input_ids.shape[0]
        return torch.tensor([[100 + i] for i in range(n)], dtype=torch.int64)

    r._forward_and_sample = fake_forward

    idx = np.array([slot_a, slot_b], dtype=np.int32)
    nst = np.array([1, 1], dtype=np.int32)
    input_ids = np.zeros((2, 1), dtype=np.int32)
    positions = np.zeros((2, 1), dtype=np.int32)
    logits_indices = np.zeros(2, dtype=np.int32)
    seq_lens = np.array([1, 1], dtype=np.int32)
    page_table, fill_page_table, cache_position = r._prepare_attn_tensors(
        idx, nst, seq_lens, target_num_reqs=2, num_blocks_per_req=4
    )

    out = r._run_model_pass(
        idx,
        nst,
        2,
        1,
        input_ids,
        positions,
        logits_indices,
        page_table,
        fill_page_table,
        cache_position,
    )

    # Attn metadata fanned out to every layer; greedy sampling metadata built.
    assert captured["attn_layers"] == {"layer.0", "layer.1"}
    assert captured["all_greedy"] is True
    assert captured["input_ids_device"] == "cpu"
    # Two active requests -> two token rows, padding dropped.
    assert out == [[100], [101]]


@pytest.mark.push
@pytest.mark.cpu
def test_run_model_pass_drops_padding_rows():
    r = make_runner()
    r.req_states.add_request(
        "A", prompt_len=1, all_token_ids=[5], num_computed_tokens=0
    )
    slot = r.req_states.req_id_to_index["A"]
    r.sampling_states.add_request(slot, SamplingParams(temperature=0.0))

    # target_num_reqs (4) padded beyond the single active request.
    r._forward_and_sample = lambda ii, pos, li, am, sm: torch.arange(
        4, dtype=torch.int64
    ).reshape(4, 1)

    idx = np.array([slot], dtype=np.int32)
    nst = np.array([1], dtype=np.int32)
    page_table, fill_page_table, cache_position = r._prepare_attn_tensors(
        idx, nst, np.array([1], dtype=np.int32), target_num_reqs=4, num_blocks_per_req=4
    )
    out = r._run_model_pass(
        idx,
        nst,
        4,
        1,
        np.zeros((4, 1), dtype=np.int32),
        np.zeros((4, 1), dtype=np.int32),
        np.zeros(4, dtype=np.int32),
        page_table,
        fill_page_table,
        cache_position,
    )
    assert out == [[0]]  # only the one active request's row


@pytest.mark.push
@pytest.mark.cpu
def test_sample_from_logits_greedy_is_argmax():
    r = object.__new__(TTModelRunnerV2)
    logits = torch.tensor([[0.1, 0.9, 0.2], [0.5, 0.1, 0.3]])
    md = type(
        "M",
        (),
        dict(
            all_greedy=True,
            no_penalties=True,
            no_logit_bias=True,
            no_bad_words=True,
            no_allowed_token_ids=True,
            no_min_tokens=True,
            no_generators=True,
        ),
    )()
    out = r._sample_from_logits(logits, md)
    assert out.tolist() == [[1], [0]]
