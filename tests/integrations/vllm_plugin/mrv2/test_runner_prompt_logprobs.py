# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 prompt-logprobs host logic.

``TTModelRunnerV2._get_prompt_logprobs_dict`` reprocesses each prefilling
request's prompt positions through compute_logits / gather_logprobs and packs
the per-position top-k logprobs into ``LogprobsTensors``. These tests pin the
host bookkeeping (position math, chunked continuation, completion / cleanup)
with the two device leaves (compute_logits, gather_logprobs) faked.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm_tt.model_runner_v2 import TTModelRunnerV2

VOCAB = 50
HID = 8
MAX_LOGPROBS = 5


class FakeReqStates:
    def __init__(self, slot, prompt_tokens, num_computed):
        self.req_id_to_index = {"r0": slot}
        self.index_to_req_id = {slot: "r0"}
        n = len(prompt_tokens)
        self.prompt_len = np.zeros(slot + 1, dtype=np.int32)
        self.prompt_len[slot] = n
        self.num_computed_tokens = np.zeros(slot + 1, dtype=np.int32)
        self.num_computed_tokens[slot] = num_computed
        self.all_token_ids = np.zeros((slot + 1, max(n, 8)), dtype=np.int32)
        self.all_token_ids[slot, :n] = prompt_tokens


def make_runner(req_states, num_plp):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device("cpu")
    r.max_num_reqs = 8
    r.model_config = SimpleNamespace(max_logprobs=MAX_LOGPROBS)
    r.req_states = req_states
    r.num_prompt_logprobs = {"r0": num_plp}
    r.in_progress_prompt_logprobs = {}

    # compute_logits: hidden [B, HID] -> deterministic logits [B, VOCAB]. Stub
    # the compiled wrapper too: the class-level torch.compile(backend="tt")
    # version would init the device on a CPU runner.
    def fake_compute_logits(hs, replicate=True):
        return torch.arange(VOCAB, dtype=torch.float32).repeat(hs.shape[0], 1)

    r.compute_logits = fake_compute_logits
    r.compute_logits_compiled = fake_compute_logits

    # sampler.gather_logprobs: return a marker per row so placement is checkable.
    # The token id column 0 encodes the batch row (so dest-slice writes show up).
    def fake_gather(logprobs, num_logprobs, token_ids):
        b = logprobs.shape[0]
        ids = (
            torch.arange(b, dtype=torch.int32).unsqueeze(1).repeat(1, num_logprobs + 1)
        )
        lps = torch.full((b, num_logprobs + 1), -0.5, dtype=torch.float32)
        ranks = torch.ones(b, dtype=torch.int32)
        return LogprobsTensors(ids, lps, ranks)

    r.sampler = SimpleNamespace(
        compute_logprobs=lambda logits: logits,
        gather_logprobs=fake_gather,
    )
    return r


@pytest.mark.push
@pytest.mark.cpu
def test_prompt_logprobs_single_step_full_prefill():
    # 4-token prompt, computed in one step. Result has num_prompt_tokens-1 rows.
    rs = FakeReqStates(slot=2, prompt_tokens=[10, 20, 30, 40], num_computed=0)
    r = make_runner(rs, num_plp=3)
    prompt_lp_hs = {2: torch.ones(4, HID)}
    sched = SimpleNamespace(num_scheduled_tokens={"r0": 4})

    out = r._get_prompt_logprobs_dict(prompt_lp_hs, sched)

    assert set(out.keys()) == {"r0"}
    lt = out["r0"]
    assert lt.logprob_token_ids.shape == (3, 4)  # num_plp+1 columns
    assert lt.logprobs.shape == (3, 4)
    # Completed request is cleaned up.
    assert "r0" not in r.num_prompt_logprobs
    assert "r0" not in r.in_progress_prompt_logprobs


@pytest.mark.push
@pytest.mark.cpu
def test_prompt_logprobs_num_plp_clamped_to_max():
    rs = FakeReqStates(slot=0, prompt_tokens=[1, 2, 3], num_computed=0)
    r = make_runner(rs, num_plp=999)  # exceeds MAX_LOGPROBS
    prompt_lp_hs = {0: torch.ones(3, HID)}
    sched = SimpleNamespace(num_scheduled_tokens={"r0": 3})

    out = r._get_prompt_logprobs_dict(prompt_lp_hs, sched)
    # Trimmed to MAX_LOGPROBS + 1 columns.
    assert out["r0"].logprob_token_ids.shape == (2, MAX_LOGPROBS + 1)


@pytest.mark.push
@pytest.mark.cpu
def test_prompt_logprobs_chunked_not_yet_complete():
    # 6-token prompt, only 3 scheduled this step -> not complete, stays in-progress.
    rs = FakeReqStates(slot=1, prompt_tokens=[1, 2, 3, 4, 5, 6], num_computed=0)
    r = make_runner(rs, num_plp=2)
    prompt_lp_hs = {1: torch.ones(3, HID)}
    sched = SimpleNamespace(num_scheduled_tokens={"r0": 3})

    out = r._get_prompt_logprobs_dict(prompt_lp_hs, sched)
    # Not yet completed: no result emitted, still tracked.
    assert out == {}
    assert "r0" in r.num_prompt_logprobs
    assert "r0" in r.in_progress_prompt_logprobs


@pytest.mark.push
@pytest.mark.cpu
def test_prompt_logprobs_empty_when_none_requested():
    rs = FakeReqStates(slot=0, prompt_tokens=[1, 2], num_computed=0)
    r = make_runner(rs, num_plp=1)
    r.num_prompt_logprobs = {}  # nothing requested
    out = r._get_prompt_logprobs_dict({}, SimpleNamespace(num_scheduled_tokens={}))
    assert out == {}
