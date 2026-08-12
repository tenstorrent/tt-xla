# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for MRv2 host-side sampling (sample_from_logits_cpu).

``TTModelRunnerV2.sample_from_logits_cpu`` is the cpu_sampling path (Gumbel-max
instead of a compiled device sampling graph). These tests pin the greedy,
temperature/top-k, and penalty behaviour purely on CPU (no device).
"""
from types import SimpleNamespace

import pytest
import torch
from vllm_tt.model_runner_v2 import TTModelRunnerV2


def runner():
    return object.__new__(TTModelRunnerV2)


def greedy_meta():
    return SimpleNamespace(no_penalties=True, all_greedy=True)


@pytest.mark.push
@pytest.mark.cpu
def test_cpu_sampling_greedy_is_argmax():
    r = runner()
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])
    out = r.sample_from_logits_cpu(logits, greedy_meta())
    assert out.shape == (2, 1)
    assert out[:, 0].tolist() == [1, 0]


@pytest.mark.push
@pytest.mark.cpu
def test_cpu_sampling_uniform_topk_stays_in_topk():
    r = runner()
    torch.manual_seed(0)
    vocab = 20
    logits = torch.randn(3, vocab)
    k = 4
    meta = SimpleNamespace(
        no_penalties=True,
        all_greedy=False,
        temperature=torch.ones(3),
        top_k=torch.full((3,), k, dtype=torch.int32),
        top_p=None,
    )
    out = r.sample_from_logits_cpu(logits, meta)
    assert out.shape == (3, 1)
    # Every sampled token must be within that row's top-k set.
    for i in range(3):
        topk_idx = torch.topk(logits[i], k).indices.tolist()
        assert int(out[i, 0]) in topk_idx


@pytest.mark.push
@pytest.mark.cpu
def test_cpu_sampling_near_zero_temp_is_greedy_even_in_random_path():
    r = runner()
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
    meta = SimpleNamespace(
        no_penalties=True,
        all_greedy=False,  # forces the random branch, but temp ~ 0 -> greedy
        temperature=torch.tensor([1e-9]),
        top_k=None,
        top_p=None,
    )
    out = r.sample_from_logits_cpu(logits, meta)
    assert int(out[0, 0]) == 1  # argmax


@pytest.mark.push
@pytest.mark.cpu
def test_cpu_sampling_repetition_penalty_suppresses_token():
    r = runner()
    # Token 0 is the argmax; a strong repetition penalty on it should flip the
    # greedy pick to the next-best token.
    logits = torch.tensor([[5.0, 4.0, 1.0]])
    meta = SimpleNamespace(
        no_penalties=False,
        all_greedy=True,
        output_token_counts=torch.tensor([[3.0, 0.0, 0.0]]),
        prompt_token_mask=torch.tensor([[False, False, False]]),
        repetition_penalties=torch.tensor([2.0]),
        frequency_penalties=torch.tensor([0.0]),
        presence_penalties=torch.tensor([0.0]),
    )
    out = r.sample_from_logits_cpu(logits, meta)
    # Positive logit / rep_pen -> 5.0/2 = 2.5 < 4.0, so token 1 wins.
    assert int(out[0, 0]) == 1
