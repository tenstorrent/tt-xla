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
  tokens are not touched, and that multi-token sequences raise
  NotImplementedError (on-device enforcement only supports single-token bans).
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
# Mask-building logic (from metadata.py from_input_batch)
# ---------------------------------------------------------------------------
# These tests replicate the mask-building loop from
# XLASupportedSamplingMetadata.from_input_batch to verify the multi-token
# skip behaviour without requiring a full InputBatch.


def _build_bad_words_mask(bad_words_token_ids, padded_num_reqs, vocab_size):
    """Replicate the from_input_batch mask-building logic for isolated testing."""
    bad_words_cpu = torch.zeros(padded_num_reqs, vocab_size, dtype=torch.float32)
    for req_idx, token_seqs in bad_words_token_ids.items():
        for token_seq in token_seqs:
            if len(token_seq) > 1:
                raise NotImplementedError(
                    "Multi-token bad_words sequences are not yet supported "
                    "in the TT sampler. Only single-token bad words can be "
                    "enforced on device. "
                    "https://github.com/tenstorrent/tt-xla/issues/3363"
                )
            if len(token_seq) == 1:
                bad_words_cpu[req_idx, token_seq[0]] = float("-inf")
    return bad_words_cpu


@pytest.mark.push
def test_bad_words_mask_building_single_token():
    """Single-token bad words are set to -inf in the mask."""
    mask = _build_bad_words_mask({0: [[5], [17]]}, padded_num_reqs=1, vocab_size=100)

    assert math.isinf(mask[0, 5]) and mask[0, 5] < 0
    assert math.isinf(mask[0, 17]) and mask[0, 17] < 0
    # Other tokens untouched
    assert mask[0, 0] == 0.0


@pytest.mark.push
def test_bad_words_mask_building_multi_token_raises():
    """Multi-token sequences (len > 1) must raise NotImplementedError.

    On-device enforcement only supports single-token bans.  Silently skipping
    multi-token sequences would give the user no signal that their bad_words
    entry did nothing.
    """
    import pytest as _pytest

    with _pytest.raises(NotImplementedError, match="Multi-token bad_words"):
        _build_bad_words_mask(
            {0: [[5], [10, 20]]},
            padded_num_reqs=1,
            vocab_size=100,
        )


@pytest.mark.push
def test_bad_words_mask_building_empty_seq_silently_skipped():
    """Empty token sequences (len == 0) must not crash or affect the mask."""
    mask = _build_bad_words_mask(
        {0: [[], [7]]},
        padded_num_reqs=1,
        vocab_size=100,
    )

    assert math.isinf(mask[0, 7]) and mask[0, 7] < 0
    # No IndexError for empty seq, and nothing else affected
    assert mask[0, 0] == 0.0
