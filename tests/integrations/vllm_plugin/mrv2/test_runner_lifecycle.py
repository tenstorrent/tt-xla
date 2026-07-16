# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner request lifecycle.

``TTModelRunnerV2.add_requests`` / ``update_requests`` / ``finish_requests``
(see vllm_tt/model_runner_v2.py) are pure host bookkeeping that drive the split
v2 state (TTRequestState + TTSamplingStates + a block table). They run on cpu
with no TT hardware and no model: the runner is allocated without ``__init__``
and the state it touches is injected, and the block table is a recording fake
so the tests assert the calls the lifecycle makes rather than device layout.

They pin the v2 semantics that differ from the v1 fork: stable slots (no
condense), preempted-request removal, and re-add clearing a stale slot.
"""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates

VOCAB = 1000


class FakeBlockTable:
    """Records add/append/clear calls per slot (stands in for MultiGroupBlockTable)."""

    def __init__(self):
        self.rows: dict[int, tuple[str, object]] = {}
        self.cleared: list[int] = []

    def add_row(self, block_ids, slot):
        self.rows[slot] = ("add", block_ids)

    def append_row(self, block_ids, slot):
        self.rows[slot] = ("append", block_ids)

    def clear_row(self, slot):
        self.cleared.append(slot)


def make_runner(max_num_reqs=4, max_model_len=32):
    r = object.__new__(TTModelRunnerV2)
    r.req_states = TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.sampling_states = TTSamplingStates(max_num_reqs=max_num_reqs, vocab_size=VOCAB)
    r.block_table = FakeBlockTable()
    r.encoder_cache = {}
    r.num_prompt_logprobs = {}
    r.model_state = None
    r.vocab_size = VOCAB
    return r


def new_req(req_id, prompt, block_ids=([0],), num_computed=0, sp=None, prefill=None):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=sp if sp is not None else SamplingParams(temperature=0.0),
        prompt_token_ids=prompt,
        prefill_token_ids=prefill if prefill is not None else prompt,
        num_computed_tokens=num_computed,
        block_ids=block_ids,
    )


def sched(
    new=(),
    cached_ids=(),
    cached_computed=(),
    cached_blocks=(),
    finished=(),
    preempted=(),
    free_mm=(),
):
    cached = SimpleNamespace(
        req_ids=list(cached_ids),
        num_computed_tokens=list(cached_computed),
        new_block_ids=list(cached_blocks),
    )
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached,
        finished_req_ids=list(finished),
        preempted_req_ids=list(preempted),
        free_encoder_mm_hashes=list(free_mm),
    )


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_populates_all_tables():
    r = make_runner()
    sp = SamplingParams(temperature=0.7, top_p=0.9)
    r.add_requests(sched(new=[new_req("A", [10, 11, 12], block_ids=([1, 2],), sp=sp)]))

    slot = r.req_states.req_id_to_index["A"]
    assert r.req_states.num_reqs == 1
    assert r.req_states.prompt_len[slot] == 3
    assert r.req_states.total_len[slot] == 3
    # Sampling params landed in the sampling table at the same slot.
    assert bool(r.sampling_states.is_greedy[slot]) is False
    assert r.sampling_states.temperature[slot] == pytest.approx(0.7)
    # Block ids handed to the block table under the stable slot.
    assert r.block_table.rows[slot] == ("add", ([1, 2],))


@pytest.mark.push
@pytest.mark.cpu
def test_finish_frees_slot_across_tables_and_reuses():
    r = make_runner()
    r.add_requests(sched(new=[new_req("A", [1, 2]), new_req("B", [3])]))
    slot_a = r.req_states.req_id_to_index["A"]

    r.finish_requests(sched(finished=["A"]))

    assert "A" not in r.req_states.req_id_to_index
    assert slot_a in r.block_table.cleared
    assert slot_a in r.req_states.free_indices
    # Sampling slot reset to greedy default.
    assert bool(r.sampling_states.is_greedy[slot_a]) is True
    # Freed slot is reused by the next request.
    r.add_requests(sched(new=[new_req("C", [7, 8])]))
    assert r.req_states.req_id_to_index["C"] == slot_a


@pytest.mark.push
@pytest.mark.cpu
def test_preempted_request_is_removed():
    r = make_runner()
    r.add_requests(sched(new=[new_req("A", [1, 2])]))
    slot_a = r.req_states.req_id_to_index["A"]

    r.finish_requests(sched(preempted=["A"]))
    assert "A" not in r.req_states.req_id_to_index
    assert slot_a in r.req_states.free_indices


@pytest.mark.push
@pytest.mark.cpu
def test_readd_same_id_clears_stale_slot():
    r = make_runner()
    r.add_requests(sched(new=[new_req("A", [1, 2, 3])]))
    # Re-add the same id (abort+resubmit) in a later step.
    r.add_requests(sched(new=[new_req("A", [9])]))

    assert r.req_states.num_reqs == 1
    slot = r.req_states.req_id_to_index["A"]
    assert r.req_states.prompt_len[slot] == 1
    assert r.req_states.total_len[slot] == 1


@pytest.mark.push
@pytest.mark.cpu
def test_update_advances_computed_and_appends_blocks():
    r = make_runner()
    r.add_requests(sched(new=[new_req("A", [1, 2, 3])]))
    slot = r.req_states.req_id_to_index["A"]

    r.update_requests(
        sched(cached_ids=["A"], cached_computed=[2], cached_blocks=[([5],)])
    )
    assert r.req_states.num_computed_tokens[slot] == 2
    assert r.block_table.rows[slot] == ("append", ([5],))

    # No new blocks -> no append recorded (row unchanged from the append above).
    r.update_requests(
        sched(cached_ids=["A"], cached_computed=[3], cached_blocks=[None])
    )
    assert r.req_states.num_computed_tokens[slot] == 3
    assert r.block_table.rows[slot] == ("append", ([5],))


@pytest.mark.push
@pytest.mark.cpu
def test_update_clamps_num_computed_prefill_tokens():
    r = make_runner()
    r.add_requests(sched(new=[new_req("A", [1, 2, 3])]))  # prefill_len == 3
    slot = r.req_states.req_id_to_index["A"]

    r.update_requests(
        sched(cached_ids=["A"], cached_computed=[5], cached_blocks=[None])
    )
    assert r.req_states.num_computed_tokens[slot] == 5
    # Prefill progress is clamped to the prefill length.
    assert r.req_states.num_computed_prefill_tokens[slot] == 3


@pytest.mark.push
@pytest.mark.cpu
def test_prompt_logprobs_full_vocab_sentinel():
    r = make_runner()
    r.add_requests(
        sched(new=[new_req("A", [1], sp=SamplingParams(prompt_logprobs=-1))])
    )
    assert r.num_prompt_logprobs["A"] == VOCAB

    r.add_requests(sched(new=[new_req("B", [2], sp=SamplingParams(prompt_logprobs=3))]))
    assert r.num_prompt_logprobs["B"] == 3


@pytest.mark.push
@pytest.mark.cpu
def test_free_encoder_mm_hashes():
    r = make_runner()
    r.encoder_cache["hash0"] = object()
    r.finish_requests(sched(free_mm=["hash0"]))
    assert "hash0" not in r.encoder_cache


@pytest.mark.push
@pytest.mark.cpu
def test_seeded_request_records_generator():
    r = make_runner()
    r.add_requests(
        sched(new=[new_req("A", [1], sp=SamplingParams(temperature=0.8, seed=42))])
    )
    slot = r.req_states.req_id_to_index["A"]
    assert slot in r.sampling_states.generators
