# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for Sampler.apply_logit_bias and Sampler.apply_bad_words.

Pure CPU tests — no TT hardware, no torch_xla, no compilation needed.

apply_logit_bias:
  Validates that per-request bias values are added to the correct token
  positions, positive/negative/zero biases behave as expected, and that
  rows in a batch are independently biased.

apply_bad_words:
  Validates that single-token bad words get -inf in the logits, non-banned
  tokens are not touched.

_compute_bad_words_mask:
  Validates the mask-building logic for both single-token and multi-token
  bad words, including prefix matching against output token history.
"""

import math

import pytest
import torch
from vllm_tt.sampler import Sampler

# Tests do not need silicon runner, but CI only runs on silicon runners.
pytestmark = pytest.mark.single_device

# ---------------------------------------------------------------------------
# apply_logit_bias
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_apply_logit_bias_exact_values():
    """Bias values are added to exactly the right token positions."""
    batch, vocab = 2, 50
    logits = torch.zeros(batch, vocab)
    bias = torch.zeros(batch, vocab)
    bias[0, 5] = 3.0
    bias[0, 10] = -2.5
    bias[1, 20] = 1.0

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert torch.isclose(
        out[0, 5], torch.tensor(3.0), atol=1e-6
    ), "positive bias not applied"
    assert torch.isclose(
        out[0, 10], torch.tensor(-2.5), atol=1e-6
    ), "negative bias not applied"
    assert torch.isclose(
        out[1, 20], torch.tensor(1.0), atol=1e-6
    ), "row-1 bias not applied"
    # Non-biased positions must be unchanged
    assert out[0, 0] == 0.0
    assert out[1, 5] == 0.0


@pytest.mark.push
def test_apply_logit_bias_zero_is_noop():
    """An all-zeros bias tensor must leave logits completely unchanged."""
    torch.manual_seed(0)
    batch, vocab = 3, 128
    logits = torch.randn(batch, vocab)
    bias = torch.zeros(batch, vocab)

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert torch.allclose(out, logits, atol=1e-6), "zero bias changed logits"


@pytest.mark.push
def test_apply_logit_bias_promotes_token():
    """A large positive bias forces argmax to the biased token regardless of original logits."""
    torch.manual_seed(1)
    batch, vocab = 1, 1000
    logits = torch.randn(batch, vocab)
    bias = torch.zeros(batch, vocab)
    target = 42
    bias[0, target] = 1000.0

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert out.argmax(dim=-1).item() == target, "large positive bias should win argmax"


@pytest.mark.push
def test_apply_logit_bias_suppresses_token():
    """A large negative bias prevents argmax from selecting the biased token."""
    torch.manual_seed(2)
    batch, vocab = 1, 1000
    logits = torch.randn(batch, vocab)
    # Identify which token would normally win, then bias it away
    original_winner = logits.argmax(dim=-1).item()
    bias = torch.zeros(batch, vocab)
    bias[0, original_winner] = -1000.0

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert (
        out.argmax(dim=-1).item() != original_winner
    ), "large negative bias should not win"


@pytest.mark.push
def test_apply_logit_bias_per_request_independence():
    """Each row in the batch uses its own bias dict independently."""
    batch, vocab = 3, 100
    logits = torch.ones(batch, vocab)
    bias = torch.zeros(batch, vocab)
    # Each row biases a different token
    bias[0, 10] = 5.0
    bias[1, 20] = 5.0
    bias[2, 30] = 5.0

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert out[0].argmax().item() == 10
    assert out[1].argmax().item() == 20
    assert out[2].argmax().item() == 30


@pytest.mark.push
@pytest.mark.parametrize("vocab_size", [32000, 128256])
def test_apply_logit_bias_large_vocab(vocab_size):
    """Bias applied correctly at production vocab sizes."""
    torch.manual_seed(3)
    logits = torch.randn(1, vocab_size)
    bias = torch.zeros(1, vocab_size)
    target = vocab_size - 1
    bias[0, target] = 1000.0

    out = Sampler().apply_logit_bias(logits.clone(), bias)

    assert out.argmax(dim=-1).item() == target


# ---------------------------------------------------------------------------
# apply_bad_words
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_apply_bad_words_banned_tokens_become_neg_inf():
    """Banned positions receive -inf; all other logits are unchanged."""
    batch, vocab = 2, 50
    logits = torch.zeros(batch, vocab)
    mask = torch.zeros(batch, vocab)
    mask[0, 5] = float("-inf")
    mask[0, 10] = float("-inf")
    mask[1, 3] = float("-inf")

    out = Sampler().apply_bad_words(logits.clone(), mask)

    assert math.isinf(out[0, 5]) and out[0, 5] < 0, "token 5 row 0 must be -inf"
    assert math.isinf(out[0, 10]) and out[0, 10] < 0, "token 10 row 0 must be -inf"
    assert math.isinf(out[1, 3]) and out[1, 3] < 0, "token 3 row 1 must be -inf"
    # Non-banned positions unchanged
    assert out[0, 0] == 0.0
    assert out[1, 5] == 0.0


@pytest.mark.push
def test_apply_bad_words_unbanned_logits_unchanged():
    """Non-masked positions must have exactly the original logit values."""
    torch.manual_seed(4)
    batch, vocab = 2, 200
    logits = torch.randn(batch, vocab)
    mask = torch.zeros(batch, vocab)
    mask[0, 0] = float("-inf")
    mask[0, 1] = float("-inf")

    out = Sampler().apply_bad_words(logits.clone(), mask)

    # Every position except [0,0] and [0,1] must be identical to original
    unchanged_mask = torch.ones(batch, vocab, dtype=torch.bool)
    unchanged_mask[0, 0] = False
    unchanged_mask[0, 1] = False
    assert torch.allclose(out[unchanged_mask], logits[unchanged_mask], atol=1e-6)


@pytest.mark.push
def test_apply_bad_words_zero_mask_is_noop():
    """An all-zeros mask must leave logits completely unchanged."""
    torch.manual_seed(5)
    batch, vocab = 3, 128
    logits = torch.randn(batch, vocab)
    mask = torch.zeros(batch, vocab)

    out = Sampler().apply_bad_words(logits.clone(), mask)

    assert torch.allclose(out, logits, atol=1e-6), "zero mask changed logits"


@pytest.mark.push
def test_apply_bad_words_prevents_argmax():
    """Banning the argmax token forces a different token to be selected."""
    torch.manual_seed(6)
    batch, vocab = 1, 500
    logits = torch.randn(batch, vocab)
    original_winner = logits.argmax(dim=-1).item()

    mask = torch.zeros(batch, vocab)
    mask[0, original_winner] = float("-inf")

    out = Sampler().apply_bad_words(logits.clone(), mask)

    assert out.argmax(dim=-1).item() != original_winner


@pytest.mark.push
def test_apply_bad_words_per_request_independence():
    """Each row bans a different token; rows don't interfere with each other."""
    batch, vocab = 3, 100
    logits = torch.ones(batch, vocab)
    # Make token 10/20/30 the natural winner in each row before banning
    logits[0, 10] = 5.0
    logits[1, 20] = 5.0
    logits[2, 30] = 5.0

    mask = torch.zeros(batch, vocab)
    mask[0, 10] = float("-inf")
    mask[1, 20] = float("-inf")
    mask[2, 30] = float("-inf")

    out = Sampler().apply_bad_words(logits.clone(), mask)

    assert out[0].argmax().item() != 10, "row 0: banned token must not win"
    assert out[1].argmax().item() != 20, "row 1: banned token must not win"
    assert out[2].argmax().item() != 30, "row 2: banned token must not win"


# ---------------------------------------------------------------------------
# _compute_bad_words_mask (from metadata.py)
# ---------------------------------------------------------------------------
# Tests for the mask-building logic that supports both single-token and
# multi-token bad words via prefix matching against output token history.

from vllm_tt.metadata import XLASupportedSamplingMetadata

_compute = XLASupportedSamplingMetadata._compute_bad_words_mask


@pytest.mark.push
def test_bad_words_mask_building_single_token():
    """Single-token bad words are set to -inf in the mask."""
    mask = _compute({0: [[5], [17]]}, [[]], 1, 100)

    assert math.isinf(mask[0, 5]) and mask[0, 5] < 0
    assert math.isinf(mask[0, 17]) and mask[0, 17] < 0
    assert mask[0, 0] == 0.0


@pytest.mark.push
def test_bad_words_mask_building_empty_seq_silently_skipped():
    """Empty token sequences (len == 0) must not crash or affect the mask."""
    mask = _compute({0: [[], [7]]}, [[]], 1, 100)

    assert math.isinf(mask[0, 7]) and mask[0, 7] < 0
    assert mask[0, 0] == 0.0


@pytest.mark.push
def test_multi_token_prefix_match():
    """Multi-token bad word bans the last token when prefix matches history."""
    # Bad word [10, 20, 30]: ban 30 when last 2 output tokens are [10, 20]
    mask = _compute({0: [[10, 20, 30]]}, [[10, 20]], 1, 100)
    assert math.isinf(mask[0, 30]) and mask[0, 30] < 0


@pytest.mark.push
def test_multi_token_prefix_no_match():
    """Multi-token bad word must NOT ban when prefix doesn't match."""
    mask = _compute({0: [[10, 20, 30]]}, [[10, 99]], 1, 100)
    assert mask[0, 30] == 0.0


@pytest.mark.push
def test_multi_token_not_enough_history():
    """Multi-token bad word must NOT ban when history is too short."""
    mask = _compute({0: [[10, 20, 30]]}, [[10]], 1, 100)
    assert mask[0, 30] == 0.0


@pytest.mark.push
def test_two_token_bad_word():
    """Two-token bad word [A, B] bans B when last output token is A."""
    mask = _compute({0: [[5, 15]]}, [[5]], 1, 100)
    assert math.isinf(mask[0, 15]) and mask[0, 15] < 0


@pytest.mark.push
def test_multi_token_empty_history():
    """Multi-token bad word must NOT ban when history is empty."""
    mask = _compute({0: [[10, 20]]}, [[]], 1, 100)
    assert mask[0, 20] == 0.0


@pytest.mark.push
def test_mixed_single_and_multi_token():
    """Single-token and multi-token bad words coexist correctly."""
    mask = _compute(
        {0: [[42], [5, 15], [5, 25], [99, 88]]},
        [[5]],
        1,
        100,
    )
    assert math.isinf(mask[0, 42]) and mask[0, 42] < 0, "single-token always banned"
    assert math.isinf(mask[0, 15]) and mask[0, 15] < 0, "[5,15] prefix matches"
    assert math.isinf(mask[0, 25]) and mask[0, 25] < 0, "[5,25] prefix matches"
    assert mask[0, 88] == 0.0, "[99,88] prefix doesn't match"
    assert (mask[0] == float("-inf")).sum() == 3


@pytest.mark.push
def test_multiple_requests_independent():
    """Bad words are applied per-request, not globally."""
    mask = _compute({0: [[10]], 2: [[20]]}, [[], [], []], 3, 100)
    assert math.isinf(mask[0, 10]) and mask[0, 10] < 0
    assert mask[0, 20] == 0.0
    assert not (mask[1] != 0).any()
    assert math.isinf(mask[2, 20]) and mask[2, 20] < 0
    assert mask[2, 10] == 0.0


@pytest.mark.push
def test_padding_rows_are_zero():
    """Padding rows beyond actual requests should be all zeros."""
    mask = _compute({0: [[42]]}, [[]], 4, 100)
    for i in range(1, 4):
        assert not (mask[i] != 0).any(), f"padding row {i} should be all zeros"


@pytest.mark.push
def test_token_id_out_of_vocab_range():
    """Bad word token IDs beyond vocab_size should be silently skipped."""
    mask = _compute({0: [[999]]}, [[]], 1, 50)
    assert not (mask[0] != 0).any()


@pytest.mark.push
def test_matches_upstream_reference():
    """Mask must produce the same bans as upstream vLLM bad_words logic."""
    from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch

    vocab_size = 1000
    bad_words_0 = [[100], [200, 300], [400, 500, 600], [200, 301]]
    bad_words_1 = [[50], [60, 70]]
    output_ids = [[200, 400, 500], [60]]

    mask = _compute({0: bad_words_0, 1: bad_words_1}, output_ids, 2, vocab_size)

    for req_idx, (bw, past) in enumerate(
        [(bad_words_0, output_ids[0]), (bad_words_1, output_ids[1])]
    ):
        ref_logits = torch.zeros(vocab_size)
        _apply_bad_words_single_batch(ref_logits, bw, past)
        ref_banned = ref_logits == float("-inf")
        actual_banned = mask[req_idx] == float("-inf")
        assert torch.equal(actual_banned, ref_banned), (
            f"Request {req_idx}: mask mismatch with upstream.\n"
            f"  ours: {actual_banned.nonzero().flatten().tolist()}\n"
            f"  ref:  {ref_banned.nonzero().flatten().tolist()}"
        )
