# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner's DP block-pool wiring.

Under DP+TP each replica holds its own slice of the KV block pool, so the ids the
runner writes into the block table must be replica-local slots rather than
vLLM's global ids (see ``_to_physical`` in vllm_tt/model_runner_v2.py). These
tests drive the lifecycle on cpu with a recording block table and assert on the
ids that reach it. ``ReplicaBlockPool`` itself is covered in
test_replica_block_pool.py.
"""

from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm_tt.model_runner_v2 import TTModelRunnerV2
from vllm_tt.replica_block_pool import ReplicaBlockPool
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates

VOCAB = 1000
MAX_NUM_REQS = 4
DP_SIZE = 2
SLOTS_PER_REPLICA = 8


class RecordingBlockTable:
    def __init__(self):
        self.rows: dict[int, list[int]] = {}
        self.cleared: list[int] = []

    def add_row(self, block_ids, slot):
        self.rows[slot] = list(block_ids[0])

    def append_row(self, block_ids, slot):
        self.rows.setdefault(slot, []).extend(block_ids[0])

    def clear_row(self, slot):
        self.cleared.append(slot)
        self.rows.pop(slot, None)


def make_runner(with_pool=True):
    r = object.__new__(TTModelRunnerV2)
    r.req_states = TTRequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=32,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=VOCAB,
        device="cpu",
    )
    r.sampling_states = TTSamplingStates(max_num_reqs=MAX_NUM_REQS, vocab_size=VOCAB)
    r.block_table = RecordingBlockTable()
    r.encoder_cache = {}
    r.num_prompt_logprobs = {}
    r.lora_requests_by_slot = {}
    r.mm_features_by_slot = {}
    r.lora_config = None
    r.supports_mm_inputs = False
    r.model_state = None
    r.vocab_size = VOCAB
    r._replica_pool = (
        ReplicaBlockPool(DP_SIZE, SLOTS_PER_REPLICA, MAX_NUM_REQS // DP_SIZE)
        if with_pool
        else None
    )
    return r


def new_req(req_id, block_ids):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SamplingParams(temperature=0.0),
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3],
        num_computed_tokens=0,
        block_ids=(block_ids,),
        lora_request=None,
    )


def sched(new=(), cached_ids=(), cached_computed=(), cached_blocks=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(cached_ids),
            num_computed_tokens=list(cached_computed),
            new_block_ids=list(cached_blocks),
        ),
        finished_req_ids=[],
        preempted_req_ids=[],
        free_encoder_mm_hashes=[],
    )


def add(runner, req_id, block_ids):
    """Add one request, returning its slot."""
    runner.add_requests(sched(new=[new_req(req_id, block_ids)]))
    return runner.req_states.req_id_to_index[req_id]


def test_ids_reaching_the_table_are_replica_local():
    r = make_runner()
    slot = add(r, "a", [101, 102, 103])
    written = r.block_table.rows[slot]
    assert len(written) == 3
    assert all(0 < b < SLOTS_PER_REPLICA for b in written)


def test_two_rows_on_one_replica_get_disjoint_slots():
    # Stride-apart global ids are what a plain modulo would fold together.
    r = make_runner()
    first = add(r, "a", [1, 2, 3])
    second = add(r, "b", [1 + SLOTS_PER_REPLICA, 2 + SLOTS_PER_REPLICA])
    pool = r._replica_pool
    assert pool.replica_of_row(first) == pool.replica_of_row(second)
    assert not set(r.block_table.rows[first]) & set(r.block_table.rows[second])


def test_append_claims_further_slots():
    r = make_runner()
    slot = add(r, "a", [1, 2])
    head = list(r.block_table.rows[slot])
    r.block_table.rows[slot] = head  # keep the recorded prefix
    r.update_requests(
        sched(cached_ids=["a"], cached_computed=[2], cached_blocks=[([50],)])
    )
    grown = r.block_table.rows[slot]
    assert len(grown) == 3
    assert len(set(grown)) == 3


def test_finish_returns_slots_to_the_replica():
    r = make_runner()
    slot = add(r, "a", [1, 2, 3])
    replica = r._replica_pool.replica_of_row(slot)
    free_while_held = r._replica_pool.free_slots(replica)
    r.finish_requests(
        SimpleNamespace(
            finished_req_ids=["a"],
            preempted_req_ids=[],
            free_encoder_mm_hashes=[],
        )
    )
    assert r._replica_pool.free_slots(replica) == free_while_held + 3
    assert slot in r.block_table.cleared


def test_ids_pass_through_untouched_without_a_pool():
    r = make_runner(with_pool=False)
    slot = add(r, "a", [101, 102, 103])
    assert r.block_table.rows[slot] == [101, 102, 103]
