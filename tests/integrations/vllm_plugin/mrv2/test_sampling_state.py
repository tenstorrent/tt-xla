# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 sampling bridge.

``TTSamplingStates`` (see vllm_tt/sampling_state_v2.py) is the persistent
per-slot sampling-param table, and ``XLASupportedSamplingMetadata.from_v2_states``
gathers a batch-ordered view from it + ``TTRequestState``. Both are pure
host-side (numpy/torch-on-cpu), so these run with no TT hardware and no model.

They pin the ``SamplingParams`` extraction, the stable-slot reset on removal, and
the batch->slot gather (via ``idx_mapping``) that feeds the sampler.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm_tt.metadata import DEFAULT_SAMPLING_PARAMS, XLASupportedSamplingMetadata
from vllm_tt.request_state import TTRequestState
from vllm_tt.sampling_state_v2 import TTSamplingStates

VOCAB = 1000


def make_rs(max_num_reqs=4, max_model_len=32, num_speculative_steps=0):
    return TTRequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        num_speculative_steps=num_speculative_steps,
        vocab_size=VOCAB,
        device="cpu",
    )


def make_ss(max_num_reqs=4):
    return TTSamplingStates(max_num_reqs=max_num_reqs, vocab_size=VOCAB)


def ib(idx_list):
    """Batch view stand-in: make_batch_view only reads these two."""
    return SimpleNamespace(
        num_reqs=len(idx_list),
        idx_mapping_np=np.array(idx_list, dtype=np.int32),
    )


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_greedy_extraction():
    ss = make_ss()
    ss.add_request(2, SamplingParams(temperature=0.0))
    assert ss.temperature[2] == 0.0
    assert bool(ss.is_greedy[2]) is True
    # Disabled top_k (default 0) maps to vocab_size, matching the v1 fork.
    assert ss.top_k[2] == VOCAB


@pytest.mark.push
@pytest.mark.cpu
def test_add_request_random_extraction():
    ss = make_ss()
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        min_p=0.1,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        repetition_penalty=1.3,
        min_tokens=3,
        logprobs=2,
        logit_bias={5: 2.0},
        allowed_token_ids=[1, 2, 3],
    )
    ss.add_request(1, sp)
    assert bool(ss.is_greedy[1]) is False
    assert ss.temperature[1] == pytest.approx(0.7)
    assert ss.top_p[1] == pytest.approx(0.9)
    assert ss.top_k[1] == 50  # in-range top_k kept verbatim
    assert ss.min_p[1] == pytest.approx(0.1)
    assert ss.frequency_penalties[1] == pytest.approx(0.5)
    assert ss.presence_penalties[1] == pytest.approx(0.2)
    assert ss.repetition_penalties[1] == pytest.approx(1.3)
    assert ss.min_tokens[1][0] == 3
    assert ss.num_logprobs[1] == 2
    assert ss.logit_bias[1] == {5: 2.0}
    assert ss.allowed_token_ids[1] == [1, 2, 3]


@pytest.mark.push
@pytest.mark.cpu
def test_remove_request_resets_slot():
    ss = make_ss()
    ss.add_request(
        0,
        SamplingParams(
            temperature=0.7,
            min_tokens=2,
            logprobs=1,
            logit_bias={1: 1.0},
            allowed_token_ids=[3],
        ),
    )
    ss.remove_request(0)
    assert ss.temperature[0] == DEFAULT_SAMPLING_PARAMS["temperature"]
    assert ss.top_k[0] == DEFAULT_SAMPLING_PARAMS["top_k"]
    assert ss.repetition_penalties[0] == DEFAULT_SAMPLING_PARAMS["repetition_penalties"]
    assert bool(ss.is_greedy[0]) is True
    assert 0 not in ss.min_tokens
    assert 0 not in ss.num_logprobs
    assert 0 not in ss.allowed_token_ids
    assert ss.logit_bias[0] is None


@pytest.mark.push
@pytest.mark.cpu
def test_make_batch_view_gather_order_and_padding():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", prompt_len=2, all_token_ids=[10, 11, 12], num_computed_tokens=0)
    rs.add_request("B", prompt_len=1, all_token_ids=[20], num_computed_tokens=0)
    slot_a = rs.req_id_to_index["A"]
    slot_b = rs.req_id_to_index["B"]
    ss.add_request(slot_a, SamplingParams(temperature=0.5))
    ss.add_request(slot_b, SamplingParams(temperature=0.9))

    # Batch position 0 -> slot_b, position 1 -> slot_a (reversed vs slot order).
    view = ss.make_batch_view(rs, ib([slot_b, slot_a]), padded_num_reqs=4)

    assert view.num_reqs == 2
    assert view.temperature_cpu_tensor[0].item() == pytest.approx(0.9)
    assert view.temperature_cpu_tensor[1].item() == pytest.approx(0.5)
    # Padding rows carry the sampler default.
    assert (
        view.temperature_cpu_tensor[2].item() == DEFAULT_SAMPLING_PARAMS["temperature"]
    )
    # num_prompt_tokens follows batch order; padding rows are zero.
    assert view.num_prompt_tokens.tolist() == [1, 2, 0, 0]
    # Output tokens derived from all_token_ids[prompt_len:total_len].
    assert view.req_output_token_ids[0] == []
    assert view.req_output_token_ids[1] == [12]


@pytest.mark.push
@pytest.mark.cpu
def test_make_batch_view_carries_draft_tokens():
    rs = make_rs(num_speculative_steps=3)
    ss = make_ss()
    rs.add_request("A", prompt_len=2, all_token_ids=[10, 11, 12], num_computed_tokens=0)
    rs.add_request("B", prompt_len=1, all_token_ids=[20], num_computed_tokens=0)
    slot_a = rs.req_id_to_index["A"]
    slot_b = rs.req_id_to_index["B"]
    ss.add_request(slot_a, SamplingParams(temperature=0.0))
    ss.add_request(slot_b, SamplingParams(temperature=0.0))
    rs.set_draft_tokens(slot_a, [71, 72])

    view = ss.make_batch_view(rs, ib([slot_a, slot_b]), padded_num_reqs=4)

    # Drafts follow batch order; undrafted and padding rows stay empty.
    assert view.spec_token_ids[0] == [71, 72]
    assert view.spec_token_ids[1] == []
    assert view.spec_token_ids[2:] == [[], []]
    # Drafts are unverified, so they stay out of the committed output tokens.
    assert view.req_output_token_ids[0] == [12]


@pytest.mark.push
@pytest.mark.cpu
def test_make_batch_view_drops_cleared_draft_tokens():
    rs = make_rs(num_speculative_steps=3)
    ss = make_ss()
    rs.add_request("A", prompt_len=2, all_token_ids=[10, 11, 12], num_computed_tokens=0)
    slot_a = rs.req_id_to_index["A"]
    ss.add_request(slot_a, SamplingParams(temperature=0.0))
    rs.set_draft_tokens(slot_a, [71, 72])
    rs.clear_draft_tokens(slot_a)

    view = ss.make_batch_view(rs, ib([slot_a]), padded_num_reqs=2)

    assert view.spec_token_ids[0] == []


@pytest.mark.push
@pytest.mark.cpu
def test_make_batch_view_rekeys_dicts_to_batch_index():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 1, [1], 0)
    rs.add_request("B", 1, [2], 0)
    slot_a = rs.req_id_to_index["A"]
    slot_b = rs.req_id_to_index["B"]
    # Populate slot-keyed dicts directly (bad_words_token_ids is a read-only
    # SamplingParams property, so it can't come through add_request in a unit test).
    ss.bad_words_token_ids[slot_b] = [[9]]
    ss.min_tokens[slot_b] = (5, {0})
    gen = torch.Generator()
    ss.generators[slot_b] = gen

    # Batch position 0 -> slot_a, position 1 -> slot_b.
    view = ss.make_batch_view(rs, ib([slot_a, slot_b]), padded_num_reqs=4)

    assert view.bad_words_token_ids == {1: [[9]]}
    assert view.min_tokens == {1: (5, {0})}
    assert view.generators == {1: gen}


@pytest.mark.push
@pytest.mark.cpu
def test_flags_mixed_greedy_random():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 1, [1], 0)
    rs.add_request("B", 1, [2], 0)
    slot_a = rs.req_id_to_index["A"]
    slot_b = rs.req_id_to_index["B"]
    ss.add_request(slot_a, SamplingParams(temperature=0.0))  # greedy
    ss.add_request(slot_b, SamplingParams(temperature=0.8))  # random

    view = ss.make_batch_view(rs, ib([slot_a, slot_b]), padded_num_reqs=2)
    assert view.all_greedy is False
    assert view.all_random is False
    assert view.no_penalties is True


@pytest.mark.push
@pytest.mark.cpu
def test_from_v2_states_all_greedy_minimal():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 2, [10, 11], 0)
    slot_a = rs.req_id_to_index["A"]
    ss.add_request(slot_a, SamplingParams(temperature=0.0))

    md = XLASupportedSamplingMetadata.from_v2_states(
        rs,
        ss,
        ib([slot_a]),
        padded_num_reqs=4,
        xla_device=torch.device("cpu"),
        vocab_size=VOCAB,
    )
    assert md.all_greedy is True
    assert md.no_penalties is True
    # Early-return minimal path leaves the param tensors unset.
    assert md.temperature is None


@pytest.mark.push
@pytest.mark.cpu
def test_from_v2_states_penalties_path_carries_drafts():
    """Drafts must reach the metadata: the rejection sampler expands the output
    tokens over them, and a request with an empty draft list is dropped from
    that expansion while the penalty path still builds a row per draft."""
    rs = make_rs(num_speculative_steps=3)
    ss = make_ss()
    rs.add_request("A", 2, [10, 11], 0)
    slot_a = rs.req_id_to_index["A"]
    ss.add_request(slot_a, SamplingParams(temperature=0.7, frequency_penalty=0.5))
    rs.set_draft_tokens(slot_a, [71, 72])

    md = XLASupportedSamplingMetadata.from_v2_states(
        rs,
        ss,
        ib([slot_a]),
        padded_num_reqs=4,
        xla_device=torch.device("cpu"),
        vocab_size=VOCAB,
    )
    assert md.no_penalties is False
    assert md.spec_token_ids == [[71, 72]]


@pytest.mark.push
@pytest.mark.cpu
def test_from_v2_states_penalties_path():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 2, [10, 11], 0)
    slot_a = rs.req_id_to_index["A"]
    ss.add_request(slot_a, SamplingParams(temperature=0.7, frequency_penalty=0.5))

    md = XLASupportedSamplingMetadata.from_v2_states(
        rs,
        ss,
        ib([slot_a]),
        padded_num_reqs=4,
        xla_device=torch.device("cpu"),
        vocab_size=VOCAB,
    )
    assert md.no_penalties is False
    assert md.temperature is not None
    assert md.output_token_counts is not None
    assert tuple(md.output_token_counts.shape) == (4, VOCAB)
    assert md.prompt_token_mask is not None
    assert tuple(md.prompt_token_mask.shape) == (4, VOCAB)


@pytest.mark.push
@pytest.mark.cpu
def test_from_v2_states_seeded_generators_build_q_samples():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 2, [10, 11], 0)
    slot_a = rs.req_id_to_index["A"]
    gen = torch.Generator()
    gen.manual_seed(123)
    ss.add_request(slot_a, SamplingParams(temperature=0.7), generator=gen)

    md = XLASupportedSamplingMetadata.from_v2_states(
        rs,
        ss,
        ib([slot_a]),
        padded_num_reqs=4,
        xla_device=torch.device("cpu"),
        vocab_size=VOCAB,
    )
    assert md.no_generators is False
    assert md.q_samples is not None
    assert tuple(md.q_samples.shape) == (4, VOCAB)


@pytest.mark.push
@pytest.mark.cpu
def test_from_v2_states_allowed_token_ids_mask():
    rs = make_rs()
    ss = make_ss()
    rs.add_request("A", 1, [10], 0)
    slot_a = rs.req_id_to_index["A"]
    ss.add_request(slot_a, SamplingParams(temperature=0.7, allowed_token_ids=[5, 7]))

    md = XLASupportedSamplingMetadata.from_v2_states(
        rs,
        ss,
        ib([slot_a]),
        padded_num_reqs=2,
        xla_device=torch.device("cpu"),
        vocab_size=VOCAB,
    )
    assert md.no_allowed_token_ids is False
    bool_mask = md.allowed_token_ids_mask  # True == disallowed
    assert tuple(bool_mask.shape) == (2, VOCAB)
    assert bool_mask[0, 5].item() is False
    assert bool_mask[0, 7].item() is False
    assert bool_mask[0, 6].item() is True

    additive = md.allowed_token_ids_additive_mask  # 0.0 allowed, -inf disallowed
    assert tuple(additive.shape) == (2, VOCAB)
    assert additive[0, 5].item() == 0.0
    assert additive[0, 7].item() == 0.0
    assert additive[0, 6].item() == float("-inf")
