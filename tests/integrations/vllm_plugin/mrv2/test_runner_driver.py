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
    r.uses_mrope = False
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
    r.lora_requests_by_slot = {}
    r.mm_features_by_slot = {}
    r.lora_config = None
    r.model_state = None
    r.vocab_size = VOCAB
    r.scheduler_output = None
    r.supports_mm_inputs = False
    r.max_num_blocks_per_req = max_num_blocks_per_req
    r.block_size = 16
    r._prefix_sdpa_usable = max_num_blocks_per_req % 8 == 0
    # Chunking off by default (budget == max_model_len); the chunked-prefill
    # scenarios flip _chunked_sdpa_active themselves.
    r.prefill_chunk_budget = max_model_len
    r._chunked_sdpa_active = False
    # Single full-attention KV group, as initialize_kv_cache would leave it.
    r._num_kv_cache_groups = 1
    r._group_block_sizes = [r.block_size]
    r._group_is_sliding = [False]
    r._group_window_blocks = [0]
    r._layer_to_group = {}
    # SMEM-cap scalars: generous so scenarios run in a single pass.
    r.num_reqs_max_model_len = max_num_reqs
    r.min_num_reqs = 1
    r.max_prefill_num_reqs = max_num_reqs
    r.max_num_reqs = max_num_reqs
    r.num_tokens_paddings = [1, 32, 64, 128]
    r.dp_size = 1
    # Spec decode off by default; scenarios that want it set these.
    r.device = torch.device("cpu")
    r.max_model_len = max_model_len
    r.num_spec_tokens = 0
    r.drafter = None
    r._draft_token_ids = None
    r._draft_token_req_ids = None
    return r


def new_req(req_id, prompt, block_ids=([0],)):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SamplingParams(temperature=0.0),
        prompt_token_ids=prompt,
        prefill_token_ids=prompt,
        num_computed_tokens=0,
        block_ids=block_ids,
        lora_request=None,
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
    # Returns (sampled, hidden_states, logprobs).
    r._run_model_pass = lambda idx, ns, tnr, pql, ii, pos, li, pt, fpt, cp, *a, **k: (
        [[1000 + int(idx[b])] for b in range(len(idx))],
        None,
        None,
    )

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


def _lp(rows):
    """LogprobsLists carrying one identifiable value per row."""
    import numpy as np
    from vllm.v1.outputs import LogprobsLists

    return LogprobsLists(
        logprob_token_ids=np.array([[v] for v in rows], dtype=np.int32),
        logprobs=np.array([[float(v)] for v in rows], dtype=np.float32),
        sampled_token_ranks=np.array(rows, dtype=np.int32),
    )


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_reports_logprobs_aligned_with_req_ids():
    # ModelRunnerOutput.logprobs rows must line up 1:1 with req_ids, which are in
    # decode-first order, not slot order.
    r = make_runner()
    so = make_sched(
        new=[new_req("A", [1, 2, 3]), new_req("B", [4, 5])],
        num_sched={"A": 3, "B": 2},
    )
    r.execute_model(so)
    slot_a = r.req_states.req_id_to_index["A"]
    slot_b = r.req_states.req_id_to_index["B"]

    r._run_model_pass = lambda idx, ns, tnr, pql, ii, pos, li, pt, fpt, cp, *a, **k: (
        [[1000 + int(idx[b])] for b in range(len(idx))],
        None,
        _lp([int(idx[b]) for b in range(len(idx))]),
    )

    out = r.sample_tokens(None)

    assert out.req_ids == ["B", "A"]
    assert out.logprobs is not None
    assert out.logprobs.sampled_token_ranks.tolist() == [slot_b, slot_a]


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_logprobs_drop_unscheduled_rows():
    # A row scheduled 0 tokens (DP padding) is not reported, so its gathered
    # logprob row must be dropped too or every later row is off by one.
    r = make_runner()
    so = make_sched(
        new=[new_req("A", [1, 2, 3]), new_req("B", [4, 5])],
        num_sched={"A": 3, "B": 0},
        total=3,
    )
    r.execute_model(so)
    slot_a = r.req_states.req_id_to_index["A"]

    r._run_model_pass = lambda idx, ns, tnr, pql, ii, pos, li, pt, fpt, cp, *a, **k: (
        [[1000 + int(idx[b])] for b in range(len(idx))],
        None,
        _lp([int(idx[b]) for b in range(len(idx))]),
    )

    out = r.sample_tokens(None)

    assert out.req_ids == ["A"]
    assert out.logprobs.sampled_token_ranks.tolist() == [slot_a]


@pytest.mark.push
@pytest.mark.cpu
def test_sample_tokens_logprobs_none_when_not_requested():
    r = make_runner()
    so = make_sched(new=[new_req("A", [1, 2, 3])], num_sched={"A": 3})
    r.execute_model(so)
    r._run_model_pass = lambda idx, ns, tnr, pql, ii, pos, li, pt, fpt, cp, *a, **k: (
        [[7] for _ in range(len(idx))],
        None,
        None,
    )
    assert r.sample_tokens(None).logprobs is None
