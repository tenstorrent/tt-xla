# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for Sampler.compute_logprobs and Sampler.gather_logprobs.

Pure CPU tests — no TT hardware, no torch_xla, no compilation needed.

compute_logprobs:
  Validates that log_softmax output forms a valid log-probability distribution:
  non-positive values, exp sums to 1.0, and ordering preserved from logits.

gather_logprobs:
  Validates the structure and values of the returned LogprobsTensors:
  - sampled/prompt token at index 0
  - top-k entries at indices 1..k, sorted descending by log-prob
  - top-k values are the actual highest log-probs in the distribution
  - rank is 1-based (rank 1 = most probable token)
  - int32 dtypes for indices and ranks (serialization regression guard)
  - correct behavior on known small-vocabulary inputs
"""

import pytest
import torch
from vllm_tt.sampler import Sampler

# Tests do not need silicon runner, but CI only runs on silicon runners.
pytestmark = pytest.mark.single_device


# ---------------------------------------------------------------------------
# compute_logprobs
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_compute_logprobs_non_positive():
    """All log-probabilities must be <= 0."""
    torch.manual_seed(0)
    logits = torch.randn(4, 1000)
    lp = Sampler().compute_logprobs(logits)
    assert (lp <= 0).all(), f"max log-prob is {lp.max().item():.4f}, expected <= 0"


@pytest.mark.push
def test_compute_logprobs_exp_sums_to_one():
    """exp(log_softmax) must sum to 1.0 across the vocab dimension."""
    torch.manual_seed(1)
    logits = torch.randn(3, 500)
    lp = Sampler().compute_logprobs(logits)
    sums = lp.exp().sum(dim=-1)
    assert torch.allclose(
        sums, torch.ones(3), atol=1e-4
    ), f"probability sums: {sums.tolist()}, expected ~1.0"


@pytest.mark.push
def test_compute_logprobs_preserves_ordering():
    """Tokens with higher logits must have higher log-probabilities."""
    logits = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
    lp = Sampler().compute_logprobs(logits)
    ranked = lp.argsort(dim=-1, descending=True)[0]
    # token 0 (logit 3.0) must be most probable, then token 2, then token 1, then token 3
    assert ranked[0].item() == 0
    assert ranked[1].item() == 2
    assert ranked[2].item() == 1
    assert ranked[3].item() == 3


# ---------------------------------------------------------------------------
# gather_logprobs
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_gather_logprobs_sampled_token_at_index_zero():
    """The sampled/prompt token ID must appear at column index 0 of the output."""
    torch.manual_seed(2)
    batch, vocab = 4, 200
    logits = torch.randn(batch, vocab)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.randint(0, vocab, (batch,))

    result = Sampler().gather_logprobs(lp, num_logprobs=5, token_ids=token_ids)

    for i in range(batch):
        assert result.logprob_token_ids[i, 0].item() == token_ids[i].item(), (
            f"row {i}: expected token {token_ids[i].item()} at index 0, "
            f"got {result.logprob_token_ids[i, 0].item()}"
        )


@pytest.mark.push
def test_gather_logprobs_token_logprob_at_index_zero():
    """The log-prob at column 0 must be the log-prob of the sampled token."""
    torch.manual_seed(3)
    batch, vocab = 3, 100
    logits = torch.randn(batch, vocab)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.tensor([5, 42, 99])

    result = Sampler().gather_logprobs(lp, num_logprobs=3, token_ids=token_ids)

    for i in range(batch):
        expected = lp[i, token_ids[i]].item()
        actual = result.logprobs[i, 0].item()
        assert abs(actual - expected) < 1e-5, (
            f"row {i}: logprob at index 0 is {actual:.6f}, "
            f"expected {expected:.6f} for token {token_ids[i].item()}"
        )


@pytest.mark.push
def test_gather_logprobs_topk_are_sorted_descending():
    """Top-k log-probs (columns 1..k) must be in descending order."""
    torch.manual_seed(4)
    batch, vocab, k = 5, 300, 8
    logits = torch.randn(batch, vocab)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.zeros(batch, dtype=torch.long)

    result = Sampler().gather_logprobs(lp, num_logprobs=k, token_ids=token_ids)

    topk_lp = result.logprobs[:, 1:]  # columns 1..k are the top-k entries
    diffs = topk_lp[:, :-1] - topk_lp[:, 1:]
    assert (diffs >= -1e-5).all(), "Top-k log-probs are not sorted descending"


@pytest.mark.push
def test_gather_logprobs_topk_are_actual_topk():
    """The top-k entries must be the actual k highest log-probs in the distribution."""
    torch.manual_seed(5)
    batch, vocab, k = 3, 150, 5
    logits = torch.randn(batch, vocab)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.zeros(batch, dtype=torch.long)

    result = Sampler().gather_logprobs(lp, num_logprobs=k, token_ids=token_ids)

    for i in range(batch):
        actual_topk, _ = torch.topk(lp[i], k)
        returned_topk = result.logprobs[i, 1:].sort(descending=True).values
        assert torch.allclose(
            returned_topk, actual_topk, atol=1e-5
        ), f"row {i}: returned top-{k} log-probs don't match actual top-{k}"


@pytest.mark.push
def test_gather_logprobs_rank_semantics():
    """Rank must be 1-based: rank 1 for the most probable token, rank N for the Nth."""
    # Controlled vocabulary: logits strictly ordered so ranks are deterministic.
    vocab = 5
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])  # token 0 is most probable
    lp = Sampler().compute_logprobs(logits)

    for expected_rank, token_id in enumerate([0, 1, 2, 3, 4], start=1):
        tid = torch.tensor([token_id])
        result = Sampler().gather_logprobs(lp, num_logprobs=1, token_ids=tid)
        actual_rank = result.selected_token_ranks[0].item()
        assert (
            actual_rank == expected_rank
        ), f"token {token_id}: expected rank {expected_rank}, got {actual_rank}"


@pytest.mark.push
def test_gather_logprobs_known_values():
    """Exact log-prob and rank values on a tiny controlled vocabulary."""
    # vocab of 3: logits [2, 1, 0] → softmax ≈ [0.665, 0.245, 0.090]
    # log-softmax ≈ [-0.408, -1.408, -2.408]
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    lp = Sampler().compute_logprobs(logits)
    expected_lp = torch.log_softmax(logits, dim=-1)

    # Sample token 1 (2nd most probable → rank 2)
    token_ids = torch.tensor([1])
    result = Sampler().gather_logprobs(lp, num_logprobs=2, token_ids=token_ids)

    # Index 0: token 1's log-prob
    assert result.logprob_token_ids[0, 0].item() == 1
    assert abs(result.logprobs[0, 0].item() - expected_lp[0, 1].item()) < 1e-5

    # Rank of token 1 = 2 (token 0 has higher log-prob)
    assert result.selected_token_ranks[0].item() == 2

    # Top-2 (columns 1,2): should be tokens 0 and 1 (highest and 2nd highest)
    assert result.logprob_token_ids[0, 1].item() == 0  # most probable
    assert result.logprob_token_ids[0, 2].item() == 1  # 2nd most probable


# ---------------------------------------------------------------------------
# dtype regression tests
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_gather_logprobs_ranks_are_int32():
    """selected_token_ranks must be int32, not int64.

    Regression guard: sum(-1) on a boolean tensor returns int64, which caused
    vLLM's msgpack serializer to crash with 'Object of type numpy.int64 is not
    serializable' (#3310).
    """
    torch.manual_seed(6)
    logits = torch.randn(4, 1000)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.randint(0, 1000, (4,))
    result = Sampler().gather_logprobs(lp, num_logprobs=5, token_ids=token_ids)
    assert result.selected_token_ranks.dtype == torch.int32, (
        f"selected_token_ranks must be int32 to avoid numpy.int64 serialization "
        f"crash (#3310), got {result.selected_token_ranks.dtype}"
    )


@pytest.mark.push
def test_gather_logprobs_indices_are_int32():
    """logprob_token_ids must be int32 to reduce tensor size."""
    torch.manual_seed(7)
    logits = torch.randn(2, 500)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.randint(0, 500, (2,))
    result = Sampler().gather_logprobs(lp, num_logprobs=3, token_ids=token_ids)
    assert (
        result.logprob_token_ids.dtype == torch.int32
    ), f"logprob_token_ids must be int32, got {result.logprob_token_ids.dtype}"


@pytest.mark.push
@pytest.mark.parametrize("vocab_size", [32000, 128256])
def test_gather_logprobs_large_vocab(vocab_size):
    """Logprob gathering works correctly at production vocabulary sizes."""
    torch.manual_seed(8)
    batch = 4
    logits = torch.randn(batch, vocab_size)
    lp = Sampler().compute_logprobs(logits)
    token_ids = torch.randint(0, vocab_size, (batch,))
    result = Sampler().gather_logprobs(lp, num_logprobs=5, token_ids=token_ids)

    assert result.logprob_token_ids.dtype == torch.int32
    assert result.selected_token_ranks.dtype == torch.int32
    assert (result.logprobs <= 0).all()
    assert (result.logprob_token_ids >= 0).all()
    assert (result.logprob_token_ids < vocab_size).all()
    assert (result.selected_token_ranks >= 1).all()
