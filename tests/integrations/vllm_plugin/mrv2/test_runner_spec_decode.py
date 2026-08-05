# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for MRv2 speculative decode (ngram) host plumbing.

The rejection-sampling math lives in rejection_sampler.py and is covered by
tests/integrations/vllm_plugin/sampling/test_speculative_decode.py. What is
tested here is the v2-specific plumbing around it:

* staging scheduled drafts into the slot table without committing them
* the per-pass SpecDecodeMetadata (flat target/bonus indices over the packed
  [reqs, spec_width] logits layout)
* the packed logits_indices those metadata index into
* multi-token writeback (spec decode accepts 1..spec_width tokens per request)
* the ngram proposal / take_draft_token_ids handoff
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_runner_driver import make_runner, make_sched, new_req


def spec_runner(num_spec_tokens=3, **kw):
    r = make_runner(**kw)
    r.num_spec_tokens = num_spec_tokens
    # Re-size the draft table for the requested spec width.
    r.req_states.num_speculative_steps = num_spec_tokens
    r.req_states.draft_tokens = np.zeros(
        (r.req_states.max_num_reqs, num_spec_tokens), dtype=np.int64
    )
    return r


def seed_decoding_req(r, req_id="A", prompt=(1, 2, 3), sampled=9):
    """Add a request and advance it past prefill by one sampled token."""
    r.execute_model(
        make_sched(new=[new_req(req_id, list(prompt))], num_sched={req_id: len(prompt)})
    )
    r.scheduler_output = None
    slot = r.req_states.req_id_to_index[req_id]
    rs = r.req_states
    pos = int(rs.total_len[slot])
    rs.all_token_ids[slot, pos] = sampled
    rs.total_len[slot] = pos + 1
    rs.num_computed_tokens[slot] = pos
    rs.num_computed_prefill_tokens[slot] = int(rs.prefill_len[slot])
    return slot


@pytest.mark.push
@pytest.mark.cpu
def test_apply_scheduled_drafts_stages_without_committing():
    r = spec_runner()
    slot = seed_decoding_req(r)
    rs = r.req_states
    total_before = int(rs.total_len[slot])

    so = make_sched(num_sched={"A": 3}, total=3)
    so.scheduled_spec_decode_tokens = {"A": [11, 12]}
    r._apply_scheduled_drafts(so)

    assert int(rs.num_draft_tokens[slot]) == 2
    assert list(rs.draft_tokens[slot, :2]) == [11, 12]
    # Drafts are readable by the input gather...
    assert list(rs.all_token_ids[slot, total_before : total_before + 2]) == [11, 12]
    # ...but uncommitted: total_len must not move.
    assert int(rs.total_len[slot]) == total_before


@pytest.mark.push
@pytest.mark.cpu
def test_apply_scheduled_drafts_clears_slots_without_drafts():
    r = spec_runner()
    slot = seed_decoding_req(r)
    so = make_sched(num_sched={"A": 3}, total=3)
    so.scheduled_spec_decode_tokens = {"A": [11, 12]}
    r._apply_scheduled_drafts(so)

    so2 = make_sched(num_sched={"A": 1}, total=1)
    so2.scheduled_spec_decode_tokens = {}
    r._apply_scheduled_drafts(so2)
    assert int(r.req_states.num_draft_tokens[slot]) == 0


@pytest.mark.push
@pytest.mark.cpu
def test_apply_scheduled_drafts_noop_when_spec_disabled():
    r = make_runner()  # num_spec_tokens == 0
    slot = seed_decoding_req(r)
    so = make_sched(num_sched={"A": 1}, total=1)
    so.scheduled_spec_decode_tokens = {"A": [11]}
    r._apply_scheduled_drafts(so)
    assert int(r.req_states.num_draft_tokens[slot]) == 0


@pytest.mark.push
@pytest.mark.cpu
def test_spec_metadata_is_none_when_disabled_or_no_drafts():
    r = make_runner()
    slot = seed_decoding_req(r)
    assert r._build_spec_decode_metadata(np.array([slot])) is None

    r2 = spec_runner()
    slot2 = seed_decoding_req(r2)
    # Spec enabled but nothing drafted this step -> stay on the flat layout.
    assert r2._build_spec_decode_metadata(np.array([slot2])) is None


@pytest.mark.push
@pytest.mark.cpu
def test_spec_metadata_indices_over_packed_layout():
    r = spec_runner(num_spec_tokens=3)
    slot_a = seed_decoding_req(r, "A", (1, 2, 3))
    slot_b = seed_decoding_req(r, "B", (4, 5), sampled=8)

    so = make_sched(num_sched={"A": 3, "B": 1}, total=4)
    # A drafts 2 tokens, B drafts none.
    so.scheduled_spec_decode_tokens = {"A": [11, 12]}
    r._apply_scheduled_drafts(so)

    md = r._build_spec_decode_metadata(np.array([slot_a, slot_b]))
    assert md is not None
    assert md.num_draft_tokens == [2, 0]
    assert list(md.draft_token_ids.numpy()) == [11, 12]

    spec_width = 4  # num_spec_tokens + 1
    # A's two draft logits sit at packed rows 0..1; bonus is the last column of
    # each request's block.
    assert list(md.target_logits_indices.numpy()) == [0, 1]
    assert list(md.bonus_logits_indices.numpy()) == [
        spec_width - 1,
        2 * spec_width - 1,
    ]
    assert list(md.cu_num_draft_tokens.numpy()) == [2, 2]
    assert list(md.cu_num_sampled_tokens.numpy()) == [3, 4]


@pytest.mark.push
@pytest.mark.cpu
def test_packed_logits_indices_point_at_drafts_then_bonus():
    r = spec_runner(num_spec_tokens=3)
    slot = seed_decoding_req(r, "A", (1, 2, 3))
    so = make_sched(num_sched={"A": 3}, total=3)
    so.scheduled_spec_decode_tokens = {"A": [11, 12]}
    r._apply_scheduled_drafts(so)

    idx = np.array([slot])
    md = r._build_spec_decode_metadata(idx)
    num_sched = np.array([3], dtype=np.int32)
    *_, logits_indices = r._prepare_input_tokens(idx, num_sched, 1, 8, md)

    assert logits_indices.shape == (1, 4)
    # First two columns index this request's two draft positions; the rest all
    # point at the bonus (last scheduled) position.
    assert list(logits_indices[0]) == [0, 1, 2, 2]


@pytest.mark.push
@pytest.mark.cpu
def test_flat_logits_indices_without_spec_metadata():
    r = spec_runner()
    slot = seed_decoding_req(r, "A", (1, 2, 3))
    idx = np.array([slot])
    *_, logits_indices = r._prepare_input_tokens(
        idx, np.array([1], dtype=np.int32), 1, 8, None
    )
    assert logits_indices.shape == (1,)
    assert int(logits_indices[0]) == 0


@pytest.mark.push
@pytest.mark.cpu
def test_postprocess_commits_all_accepted_tokens():
    r = spec_runner()
    slot = seed_decoding_req(r, "A", (1, 2, 3))
    rs = r.req_states
    total_before = int(rs.total_len[slot])

    # Three accepted tokens (two drafts confirmed + bonus).
    valid = r.postprocess(np.array([slot]), np.array([3]), [[21, 22, 23]])

    assert valid[0] == [21, 22, 23]
    assert int(rs.total_len[slot]) == total_before + 3
    assert list(rs.all_token_ids[slot, total_before : total_before + 3]) == [21, 22, 23]
    # last_sampled seeds the next step's input: must be the final accepted token.
    assert int(rs.last_sampled_tokens[slot, 0]) == 23


@pytest.mark.push
@pytest.mark.cpu
def test_postprocess_single_token_path_unchanged():
    r = make_runner()
    slot = seed_decoding_req(r, "A", (1, 2, 3))
    rs = r.req_states
    total_before = int(rs.total_len[slot])
    r.postprocess(np.array([slot]), np.array([1]), [[42]])
    assert int(rs.total_len[slot]) == total_before + 1
    assert int(rs.last_sampled_tokens[slot, 0]) == 42


@pytest.mark.push
@pytest.mark.cpu
def test_propose_draft_token_ids_caches_and_hands_off():
    r = spec_runner()
    seed_decoding_req(r, "A", (1, 2, 3))

    class FakeProposer:
        def __init__(self):
            self.seen = None

        # Signature must match the real NgramProposer.propose, leading
        # num_speculative_tokens included: a laxer fake hides a call-site drift.
        def propose(
            self,
            num_speculative_tokens,
            sampled,
            num_tokens_no_spec,
            token_ids_cpu,
            slot_mappings=None,
        ):
            self.seen = (num_speculative_tokens, sampled, num_tokens_no_spec.copy())
            return [[71, 72]]

    r.drafter = FakeProposer()
    r.propose_draft_token_ids(["A"], [[21]])
    assert r.drafter.seen[0] == r.num_spec_tokens

    out = r.take_draft_token_ids()
    assert out.req_ids == ["A"]
    assert out.draft_token_ids == [[71, 72]]
    # Taking twice must not re-emit the same drafts.
    assert r.take_draft_token_ids() is None


@pytest.mark.push
@pytest.mark.cpu
def test_propose_draft_token_ids_noop_without_drafter():
    r = spec_runner()
    seed_decoding_req(r, "A", (1, 2, 3))
    r.propose_draft_token_ids(["A"], [[21]])
    assert r.take_draft_token_ids() is None


@pytest.mark.push
@pytest.mark.cpu
def test_propose_draft_token_ids_skips_when_nothing_accepted():
    r = spec_runner()
    seed_decoding_req(r, "A", (1, 2, 3))

    class BoomProposer:
        def propose(self, *a):
            raise AssertionError("must not draft with no accepted tokens")

    r.drafter = BoomProposer()
    r.propose_draft_token_ids(["A"], [[]])
    assert r.take_draft_token_ids() is None


# --- KV block alignment for speculative rows -------------------------------
#
# paged_fill_cache writes from the start of the block fill_page_table points at,
# while the chunked SDPA read offsets by an exact token position. They agree only
# when num_computed is block aligned, which a prefill chunk always is and a spec
# row never is. The row is therefore extended left to its boundary (row_lead) and
# every within-row index shifts with it.


def aligned_spec_runner(num_spec_tokens=3):
    """Spec runner whose page-table layout supports the chunked SDPA op."""
    r = spec_runner(
        num_spec_tokens=num_spec_tokens, max_model_len=128, max_num_blocks_per_req=8
    )
    assert r._prefix_sdpa_usable, "test needs num_blocks_per_req % 8 == 0"
    return r


def seed_at(r, computed, n_sched, req_id="A"):
    """Request decoding at ``computed`` with ``n_sched`` tokens this step."""
    rs = r.req_states
    ids = [(i % 97) + 1 for i in range(computed + n_sched)]
    rs.add_request(req_id, computed, ids, computed)
    return rs.req_id_to_index[req_id], ids


@pytest.mark.push
@pytest.mark.cpu
def test_spec_row_extends_left_to_its_block_boundary():
    r = aligned_spec_runner()
    slot, ids = seed_at(r, computed=20, n_sched=3)  # block 16 -> lead 4

    input_ids, positions, _, seq_lens, logits_indices = r._prepare_input_tokens(
        np.array([slot], dtype=np.int32), np.array([3], dtype=np.int32), 1, 32
    )

    # Row starts on the boundary (16) and re-feeds the 4 computed tokens.
    assert positions[0, :7].tolist() == list(range(16, 23))
    assert input_ids[0, :7].tolist() == ids[16:23]
    assert positions[0, 7:].tolist() == [0] * 25
    # The sampled logit moved right by the lead.
    assert int(logits_indices[0]) == 6
    # seq_len stays the physical context length, unaffected by the re-feed.
    assert int(seq_lens[0]) == 23


@pytest.mark.push
@pytest.mark.cpu
def test_packed_logits_indices_shift_by_row_lead():
    r = aligned_spec_runner()
    slot, _ = seed_at(r, computed=20, n_sched=3)  # lead 4, 2 drafts + bonus
    spec_md = SimpleNamespace(num_draft_tokens=[2])

    *_, logits_indices = r._prepare_input_tokens(
        np.array([slot], dtype=np.int32),
        np.array([3], dtype=np.int32),
        1,
        32,
        spec_md,
    )

    assert logits_indices[0, :2].tolist() == [4, 5]  # drafts, shifted by lead
    assert logits_indices[0, 2:].tolist() == [6, 6]  # bonus at row_len - 1


@pytest.mark.push
@pytest.mark.cpu
def test_chunk_start_idx_is_the_row_block_boundary():
    r = aligned_spec_runner()
    slot, _ = seed_at(r, computed=20, n_sched=3)

    start = r._chunk_start_idx_for(
        np.array([slot], dtype=np.int32), np.array([3], dtype=np.int32), 32
    )
    # Must equal where paged_fill_cache writes, not the exact decode position.
    assert start is not None and start.tolist() == [16]


@pytest.mark.push
@pytest.mark.cpu
def test_chunk_start_idx_none_for_plain_decode():
    r = aligned_spec_runner()
    slot, _ = seed_at(r, computed=20, n_sched=1)
    assert (
        r._chunk_start_idx_for(
            np.array([slot], dtype=np.int32), np.array([1], dtype=np.int32), 1
        )
        is None
    )


@pytest.mark.push
@pytest.mark.cpu
def test_select_batch_trims_spec_pass_to_one_block_boundary():
    r = aligned_spec_runner()
    s_a, _ = seed_at(r, computed=20, n_sched=3, req_id="A")  # boundary 16
    s_b, _ = seed_at(r, computed=40, n_sched=3, req_id="B")  # boundary 32

    slots = np.array([s_a, s_b], dtype=np.int32)
    ntoks = np.array([3, 3], dtype=np.int32)
    idx0, nst0, _, padded0, end0 = r._select_batch(slots, ntoks, 0)

    # One shared read offset cannot describe both boundaries: take A now, B next.
    assert idx0.tolist() == [s_a]
    assert end0 == 1
    # Bucket covers the re-fed length (4 + 3), not just the 3 scheduled tokens.
    assert padded0 == 32

    idx1, _, _, _, end1 = r._select_batch(slots, ntoks, end0)
    assert idx1.tolist() == [s_b]
    assert end1 == 2


@pytest.mark.push
@pytest.mark.cpu
def test_no_row_lead_when_spec_decode_is_off():
    """The whole alignment path must be inert without spec decode."""
    r = aligned_spec_runner()
    r.num_spec_tokens = 0
    slot, ids = seed_at(r, computed=20, n_sched=3)

    input_ids, positions, _, _, logits_indices = r._prepare_input_tokens(
        np.array([slot], dtype=np.int32), np.array([3], dtype=np.int32), 1, 32
    )

    assert positions[0, :3].tolist() == [20, 21, 22]
    assert input_ids[0, :3].tolist() == ids[20:23]
    assert int(logits_indices[0]) == 2
    assert (
        r._chunk_start_idx_for(
            np.array([slot], dtype=np.int32), np.array([3], dtype=np.int32), 32
        )
        is None
    )
