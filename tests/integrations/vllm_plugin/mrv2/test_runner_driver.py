# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU integration tests for the MRv2 runner two-phase driver.

``TTModelRunnerV2.execute_model`` / ``sample_tokens`` (see
vllm_tt/model_runner_v2.py) compose the tested host stages around the single
hardware leaf ``_run_model_pass``. By injecting a fake ``_run_model_pass`` (the
only device-coupled call), the whole host driver -- state update, decode-first
ordering, multi-pass loop, writeback, and output assembly -- runs on cpu with no
TT hardware and no model.

They pin the driver composition: the two-phase stash/hand-off, the decode-first
output ordering, and that the sampled tokens flow through postprocess into both
the ModelRunnerOutput and the request token tables.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates

VOCAB = 1000


class FakeBlockTable:
    """Host block table: stores rows on add and serves them to _prepare_attn_tensors."""

    def __init__(self, max_num_reqs, max_blocks):
        self._arr = torch.zeros((max_num_reqs, max_blocks), dtype=torch.int32)

    def add_row(self, block_ids, slot):
        ids = list(block_ids[0])  # single kv group
        if ids:
            self._arr[slot, : len(ids)] = torch.tensor(ids, dtype=torch.int32)

    def append_row(self, block_ids, slot):
        pass  # not exercised in these single-step scenarios

    def clear_row(self, slot):
        self._arr[slot] = 0

    def __getitem__(self, group):
        return self

    def get_cpu_tensor(self):
        return self._arr


def make_runner(max_num_reqs=8, max_model_len=32, max_num_blocks_per_req=4):
    r = object.__new__(TTModelRunnerV2)
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=128,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.sampling_states = TTSamplingStates(max_num_reqs=max_num_reqs, vocab_size=VOCAB)
    r.block_table = FakeBlockTable(max_num_reqs, max_num_blocks_per_req)
    r.encoder_cache = {}
    r.num_prompt_logprobs = {}
    r.model_state = None
    r.vocab_size = VOCAB
    r.scheduler_output = None
    r.supports_mm_inputs = False
    r.max_num_blocks_per_req = max_num_blocks_per_req
    r.block_size = 16
    # SMEM-cap scalars: generous so scenarios run in a single pass.
    r.most_model_len = None
    r.num_reqs_max_model_len = max_num_reqs
    r.num_reqs_most_model_len = None
    r.min_num_reqs = 1
    r.max_prefill_num_reqs = max_num_reqs
    r.max_num_reqs = max_num_reqs
    r.num_tokens_paddings = [1, 32, 64, 128]
    return r


def new_req(req_id, prompt, block_ids=([0],)):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SamplingParams(temperature=0.0),
        prompt_token_ids=prompt,
        prefill_token_ids=prompt,
        num_computed_tokens=0,
        block_ids=block_ids,
    )


def make_sched(new=(), num_sched=None, total=None):
    num_sched = dict(num_sched or {})
    cached = SimpleNamespace(req_ids=[], num_computed_tokens=[], new_block_ids=[])
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_sched,
        total_num_scheduled_tokens=(
            sum(num_sched.values()) if total is None else total
        ),
        finished_req_ids=[],
        preempted_req_ids=[],
        free_encoder_mm_hashes=[],
    )


@pytest.mark.push
@pytest.mark.cpu
def test_execute_model_stashes_and_returns_none():
    r = make_runner()
    so = make_sched(new=[new_req("A", [1, 2, 3])], num_sched={"A": 3})
    assert r.execute_model(so) is None
    assert r.scheduler_output is so
    assert r.req_states.num_reqs == 1  # add_requests ran


@pytest.mark.push
@pytest.mark.cpu
def test_execute_model_no_tokens_returns_empty_output():
    r = make_runner()
    out = r.execute_model(make_sched(num_sched={}, total=0))
    assert out is not None
    assert out.req_ids == []
    assert r.scheduler_output is None  # nothing stashed


@pytest.mark.push
@pytest.mark.cpu
def test_execute_model_rejects_reentry_before_sampling():
    r = make_runner()
    r.scheduler_output = object()  # a step already pending
    with pytest.raises(AssertionError):
        r.execute_model(make_sched(new=[new_req("A", [1])], num_sched={"A": 1}))


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_composes_full_prefill_batch():
    r = make_runner()
    so = make_sched(
        new=[new_req("A", [1, 2, 3]), new_req("B", [4, 5])],
        num_sched={"A": 3, "B": 2},
    )
    assert r.execute_model(so) is None

    slot_a = r.req_states.req_id_to_index["A"]
    slot_b = r.req_states.req_id_to_index["B"]

    # Fake the one hardware leaf: return a distinct token per active req.
    r._run_model_pass = lambda idx, ns, tnr, pql, ii, pos, li, pt, fpt, cp: [
        [1000 + int(idx[b])] for b in range(len(idx))
    ]

    out = r.sample_tokens(None)

    # Decode-first ordering puts the shorter request (B, 2 tokens) first.
    assert out.req_ids == ["B", "A"]
    assert out.sampled_token_ids == [[1000 + slot_b], [1000 + slot_a]]
    assert out.req_id_to_index == {"B": 0, "A": 1}

    # Writeback grew both token tables (both finished prefill this step).
    assert r.req_states.total_len[slot_a] == 4
    assert r.req_states.all_token_ids[slot_a, 3] == 1000 + slot_a
    assert r.req_states.total_len[slot_b] == 3
    assert r.req_states.all_token_ids[slot_b, 2] == 1000 + slot_b

    # Step fully consumed -> ready for the next execute_model.
    assert r.scheduler_output is None


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_returns_none_without_stashed_step():
    r = make_runner()
    assert r.sample_tokens(None) is None
