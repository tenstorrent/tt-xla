# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.model_runner import TTModelRunner
from vllm_tt.rejection_sampler import _PLACEHOLDER_TOKEN_ID, RejectionSampler


def _make_target_logits(argmax_token_ids: list[int], vocab_size: int) -> torch.Tensor:
    logits = torch.full((len(argmax_token_ids), vocab_size), -1e9, dtype=torch.float32)
    for row, token_id in enumerate(argmax_token_ids):
        logits[row, token_id] = 1.0
    return logits


def _make_rejection_sampler() -> RejectionSampler:
    return RejectionSampler(torch.nn.Identity())


@pytest.mark.push
@pytest.mark.cpu
def test_speculative_greedy_fallback_accepts_all_matching_drafts():
    """Example acceptance case: all drafted tokens match target argmax."""
    draft_token_ids = torch.tensor([7, 9, 11], dtype=torch.int32)
    target_logits = _make_target_logits([7, 9, 11], vocab_size=32)
    bonus_token_ids = torch.tensor([[13]], dtype=torch.int32)

    output_token_ids = _make_rejection_sampler()._rejection_sample_fallback(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[3],
        max_spec_len=3,
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=None,
    )

    assert output_token_ids.tolist() == [[7, 9, 11, 13]]


@pytest.mark.push
@pytest.mark.cpu
def test_speculative_greedy_fallback_keeps_accepted_prefix_before_rejection():
    draft_token_ids = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    target_logits = _make_target_logits([5, 6, 21, 8], vocab_size=32)
    bonus_token_ids = torch.tensor([[31]], dtype=torch.int32)

    output_token_ids = _make_rejection_sampler()._rejection_sample_fallback(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[4],
        max_spec_len=4,
        cu_num_draft_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=None,
    )

    assert output_token_ids.tolist() == [
        [5, 6, 21, _PLACEHOLDER_TOKEN_ID, _PLACEHOLDER_TOKEN_ID]
    ]


@pytest.mark.push
@pytest.mark.cpu
def test_speculative_nongreedy_fallback_recovers_on_rejection():
    draft_token_ids = torch.tensor([5], dtype=torch.int32)
    target_logits = _make_target_logits([6], vocab_size=16)
    bonus_token_ids = torch.tensor([[9]], dtype=torch.int32)

    sampling_metadata = XLASupportedSamplingMetadata(
        temperature=torch.tensor([1.0], dtype=torch.float32),
        top_k=torch.tensor([0], dtype=torch.int32),
        top_p=torch.tensor([1.0], dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        _generators={0: torch.Generator(device="cpu").manual_seed(0)},
        no_generators=False,
    )

    output_token_ids = _make_rejection_sampler()._rejection_sample_fallback(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[1],
        max_spec_len=1,
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
    )

    assert output_token_ids[0, 0].item() == 6
    assert output_token_ids[0, 1].item() == _PLACEHOLDER_TOKEN_ID


@pytest.mark.push
@pytest.mark.cpu
def test_speculative_greedy_fallback_batch_mixed_draft_lengths_uses_cu_offsets():
    """Exercise multi-request flat packing via cu offsets.

    req0 has draft_len=2 and req1 has draft_len=1. This validates that req1
    reads from its own flat slice (start = cu[0]) instead of req0's tokens.
    """
    draft_token_ids = torch.tensor([7, 9, 21], dtype=torch.int32)
    target_logits = _make_target_logits([7, 9, 21], vocab_size=32)
    bonus_token_ids = torch.tensor([[13], [29]], dtype=torch.int32)

    output_token_ids = _make_rejection_sampler()._rejection_sample_fallback(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[2, 1],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32),
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=None,
    )

    assert output_token_ids.tolist() == [
        [7, 9, 13],
        [21, 29, _PLACEHOLDER_TOKEN_ID],
    ]


@pytest.mark.push
@pytest.mark.cpu
def test_speculative_greedy_fallback_handles_zero_draft_len_with_bonus_only():
    """Cover draft_len == 0 branch that emits only the bonus token."""
    draft_token_ids = torch.tensor([5, 6], dtype=torch.int32)
    target_logits = _make_target_logits([5, 6], vocab_size=16)
    bonus_token_ids = torch.tensor([[42], [11]], dtype=torch.int32)

    output_token_ids = _make_rejection_sampler()._rejection_sample_fallback(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[0, 2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([0, 2], dtype=torch.int32),
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=None,
    )

    assert output_token_ids.tolist() == [
        [42, _PLACEHOLDER_TOKEN_ID, _PLACEHOLDER_TOKEN_ID],
        [5, 6, 11],
    ]


@pytest.mark.push
@pytest.mark.cpu
def test_propose_draft_token_ids_ignores_discarded_rows():
    class _FakeDrafter:
        def __init__(self):
            self.last_sampled = None

        def propose(self, sampled_token_ids_list, num_tokens_no_spec, token_ids_cpu):
            self.last_sampled = sampled_token_ids_list
            return [[] for _ in sampled_token_ids_list]

    fake_drafter = _FakeDrafter()
    fake_input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens_no_spec=np.array([3, 3], dtype=np.int32),
        token_ids_cpu=np.array(
            [[11, 12, 13, 0, 0], [21, 22, 23, 0, 0]],
            dtype=np.int32,
        ),
        req_ids=["req-0", "req-1"],
    )
    fake_runner = SimpleNamespace(
        _draft_token_req_ids=None,
        _draft_token_ids=None,
        drafter=fake_drafter,
        num_spec_tokens=3,
        input_batch=fake_input_batch,
        max_model_len=5,
    )

    sampled_token_ids = torch.tensor([[100], [200]], dtype=torch.int32)
    TTModelRunner.propose_draft_token_ids(
        fake_runner,
        scheduler_output=SimpleNamespace(),
        sampled_token_ids=sampled_token_ids,
        discard_req_indices=(1,),
    )

    assert fake_drafter.last_sampled == [[100], []]
