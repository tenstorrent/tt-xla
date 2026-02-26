# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for bad words mask computation and application.

Pure CPU tests — no TT hardware, no torch_xla, no compilation needed.
Validates that _compute_bad_words_mask produces the correct mask for
single-token and multi-token bad words, and that the Sampler applies
the mask correctly (banned tokens receive -inf logits).

The mask logic mirrors vllm.v1.sample.ops.bad_words but materialises
the result as a fixed-shape [batch, vocab] bool tensor.
"""

import pytest
import torch
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import Sampler

# ---------------------------------------------------------------------------
# _compute_bad_words_mask unit tests
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_single_token_bad_word_always_banned():
    """A single-token bad word should always be banned regardless of history."""
    vocab_size = 100
    padded = 2
    bad_words = {0: [[42]]}  # request 0 bans token 42
    output_ids = [[], []]  # no history

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 42], "Single-token bad word should be banned"
    assert not mask[1, 42], "Request 1 has no bad words"
    assert mask[0].sum() == 1, "Only one token should be banned"


@pytest.mark.push
def test_multi_token_bad_word_prefix_match():
    """Multi-token bad word bans the last token only when prefix matches."""
    vocab_size = 100
    padded = 1
    # Bad word [10, 20, 30]: ban token 30 only when last 2 output tokens are [10, 20]
    bad_words = {0: [[10, 20, 30]]}
    output_ids = [[10, 20]]  # prefix matches

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 30], "Token 30 should be banned (prefix [10, 20] matches)"


@pytest.mark.push
def test_multi_token_bad_word_prefix_no_match():
    """Multi-token bad word should NOT ban when prefix doesn't match."""
    vocab_size = 100
    padded = 1
    bad_words = {0: [[10, 20, 30]]}
    output_ids = [[10, 99]]  # prefix doesn't match

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert not mask[0, 30], "Token 30 should NOT be banned (prefix mismatch)"


@pytest.mark.push
def test_multi_token_bad_word_not_enough_history():
    """Multi-token bad word should NOT ban when history is too short."""
    vocab_size = 100
    padded = 1
    bad_words = {0: [[10, 20, 30]]}
    output_ids = [[10]]  # only 1 token, need 2 for prefix

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert not mask[0, 30], "Not enough history to match 2-token prefix"


@pytest.mark.push
def test_multi_token_bad_word_empty_history():
    """Multi-token bad word should NOT ban when history is empty."""
    vocab_size = 100
    padded = 1
    bad_words = {0: [[10, 20]]}
    output_ids = [[]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert not mask[0, 20], "Empty history can't match any prefix"


@pytest.mark.push
def test_two_token_bad_word_prefix_match():
    """Two-token bad word [A, B] bans B when last output token is A."""
    vocab_size = 100
    padded = 1
    bad_words = {0: [[5, 15]]}
    output_ids = [[5]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 15], "Token 15 should be banned (prefix [5] matches last output)"


@pytest.mark.push
def test_multiple_bad_words_same_request():
    """Multiple bad words on the same request should all be evaluated."""
    vocab_size = 100
    padded = 1
    bad_words = {
        0: [
            [42],  # single-token: always banned
            [5, 15],  # two-token: banned if last output is 5
            [5, 25],  # two-token: banned if last output is 5
            [99, 88],  # two-token: NOT banned (last output is 5, not 99)
        ]
    }
    output_ids = [[5]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 42], "Single-token bad word always banned"
    assert mask[0, 15], "Two-token [5,15]: prefix matches"
    assert mask[0, 25], "Two-token [5,25]: prefix matches"
    assert not mask[0, 88], "Two-token [99,88]: prefix doesn't match"
    assert mask[0].sum() == 3, "Exactly 3 tokens should be banned"


@pytest.mark.push
def test_multiple_requests():
    """Bad words are applied per-request, not globally."""
    vocab_size = 100
    padded = 3
    bad_words = {
        0: [[10]],  # request 0: ban token 10
        2: [[20]],  # request 2: ban token 20
    }
    output_ids = [[], [], []]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 10], "Request 0 should ban token 10"
    assert not mask[0, 20], "Request 0 should NOT ban token 20"
    assert not mask[1].any(), "Request 1 has no bad words"
    assert mask[2, 20], "Request 2 should ban token 20"
    assert not mask[2, 10], "Request 2 should NOT ban token 10"


@pytest.mark.push
def test_padding_rows_are_false():
    """Padding rows beyond actual requests should be all False."""
    vocab_size = 100
    padded = 4
    bad_words = {0: [[42]]}
    output_ids = [[]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    for i in range(1, padded):
        assert not mask[i].any(), f"Padding row {i} should be all False"


@pytest.mark.push
def test_token_id_out_of_vocab_range():
    """Bad word token IDs beyond vocab_size should be silently skipped."""
    vocab_size = 50
    padded = 1
    bad_words = {0: [[999]]}  # 999 >= vocab_size
    output_ids = [[]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert not mask[0].any(), "Out-of-range token should be skipped"


@pytest.mark.push
def test_empty_bad_word_sequence_skipped():
    """Empty bad word sequences should be silently skipped."""
    vocab_size = 100
    padded = 1
    bad_words = {0: [[], [42]]}
    output_ids = [[]]

    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words, output_ids, padded, vocab_size
    )

    assert mask[0, 42], "Non-empty bad word should still be applied"
    assert mask[0].sum() == 1, "Only the non-empty bad word should produce a ban"


@pytest.mark.push
def test_matches_upstream_reference():
    """Mask-based approach must produce the same bans as the upstream
    vLLM _apply_bad_words_single_batch for a realistic scenario."""
    from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch

    vocab_size = 1000
    padded = 2
    bad_words_list_0 = [[100], [200, 300], [400, 500, 600], [200, 301]]
    bad_words_list_1 = [[50], [60, 70]]
    bad_words_dict = {0: bad_words_list_0, 1: bad_words_list_1}
    output_ids = [[200, 400, 500], [60]]

    # Our mask
    mask = XLASupportedSamplingMetadata._compute_bad_words_mask(
        bad_words_dict, output_ids, padded, vocab_size
    )

    # Upstream reference: apply to logits and check which became -inf
    for req_idx, (bw_list, past) in enumerate(
        [(bad_words_list_0, output_ids[0]), (bad_words_list_1, output_ids[1])]
    ):
        ref_logits = torch.zeros(vocab_size)
        _apply_bad_words_single_batch(ref_logits, bw_list, past)
        ref_banned = ref_logits == float("-inf")

        assert torch.equal(mask[req_idx], ref_banned), (
            f"Request {req_idx}: mask mismatch with upstream.\n"
            f"  mask bans: {mask[req_idx].nonzero().flatten().tolist()}\n"
            f"  ref  bans: {ref_banned.nonzero().flatten().tolist()}"
        )


# ---------------------------------------------------------------------------
# Sampler integration: bad_words_mask applied correctly
# ---------------------------------------------------------------------------


@pytest.mark.push
def test_sampler_bans_masked_tokens():
    """Sampler.sample() should never select a token that is banned by bad_words_mask."""
    vocab_size = 100
    batch = 2
    torch.manual_seed(0)

    # Make token 42 very likely for request 0 and token 7 for request 1
    logits = torch.randn(batch, vocab_size)
    logits[0, 42] = 100.0
    logits[1, 7] = 100.0

    # Ban the dominant tokens
    bad_words_mask = torch.zeros(batch, vocab_size, dtype=torch.bool)
    bad_words_mask[0, 42] = True
    bad_words_mask[1, 7] = True

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((batch,), 0.8),
        top_k=torch.full((batch,), vocab_size, dtype=torch.int32),
        top_p=torch.full((batch,), 1.0),
        min_p=torch.full((batch,), 0.0),
        all_greedy=False,
        no_bad_words=False,
        bad_words_mask=bad_words_mask,
    )

    sampler = Sampler()
    # Run multiple times to reduce flakiness
    for _ in range(10):
        sampled = sampler.sample(logits.clone(), metadata)
        assert sampled[0].item() != 42, "Request 0 should never sample banned token 42"
        assert sampled[1].item() != 7, "Request 1 should never sample banned token 7"


@pytest.mark.push
def test_sampler_greedy_respects_bad_words():
    """Greedy decoding (temperature=0) should also respect bad_words_mask."""
    vocab_size = 100
    batch = 1

    logits = torch.zeros(batch, vocab_size)
    logits[0, 50] = 10.0  # highest logit
    logits[0, 30] = 5.0  # second highest

    # Ban token 50
    bad_words_mask = torch.zeros(batch, vocab_size, dtype=torch.bool)
    bad_words_mask[0, 50] = True

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((batch,), 0.0),  # greedy
        top_k=torch.full((batch,), vocab_size, dtype=torch.int32),
        top_p=torch.full((batch,), 1.0),
        min_p=torch.full((batch,), 0.0),
        all_greedy=False,  # even though temp=0, route through sampler
        no_bad_words=False,
        bad_words_mask=bad_words_mask,
    )

    sampler = Sampler()
    sampled = sampler.sample(logits, metadata)
    assert (
        sampled[0].item() == 30
    ), f"Greedy should pick second-best token 30 (got {sampled[0].item()})"
