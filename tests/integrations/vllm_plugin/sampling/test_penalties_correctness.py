# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for Sampler.apply_penalties.

Pure CPU tests — no TT hardware, no torch_xla, no compilation needed.
Validates that the penalty math matches the vLLM GPU spec:
  - repetition_penalty: applied to tokens in prompt ∪ output
  - frequency_penalty:  applied to output tokens, scaled by count
  - presence_penalty:   applied to output tokens, flat per token
"""

import pytest
import torch
from vllm_tt.sampler import Sampler


def apply_penalties_reference(
    logits,
    output_token_counts,
    prompt_token_mask,
    presence_penalties,
    frequency_penalties,
    repetition_penalties,
):
    """Independent CPU reference implementation of the vLLM GPU penalty spec.

    Written differently from Sampler.apply_penalties (plain / instead of
    reciprocal) to catch algebraic bugs.
    """
    logits = logits.clone().float()
    occurred_output = output_token_counts > 0  # [batch, vocab] bool

    # Repetition penalty: prompt ∪ output tokens
    rep_mask = occurred_output | prompt_token_mask
    rep = repetition_penalties.unsqueeze(1)  # [batch, 1]
    logits = torch.where(
        rep_mask,
        torch.where(logits > 0, logits / rep, logits * rep),
        logits,
    )

    # Frequency penalty: subtract penalty scaled by output occurrence count
    logits -= frequency_penalties.unsqueeze(1) * output_token_counts.float()

    # Presence penalty: subtract flat penalty for each output-occurring token
    logits -= presence_penalties.unsqueeze(1) * occurred_output.float()

    return logits


PENALTY_CASES = [
    # neutral: identity values — logits must be unchanged
    pytest.param(0.0, 0.0, 1.0, id="neutral"),
    pytest.param(0.5, 0.3, 1.2, id="moderate"),
    pytest.param(2.0, 1.0, 3.0, id="high"),
    pytest.param(0.0, 1.0, 2.0, id="freq_rep_only"),  # no presence
]


@pytest.mark.push
@pytest.mark.parametrize("vocab_size", [32000, 128256])
@pytest.mark.parametrize("presence,frequency,repetition", PENALTY_CASES)
def test_apply_penalties_matches_reference(vocab_size, presence, frequency, repetition):
    """apply_penalties must match the reference for all penalty combinations."""
    batch = 4
    torch.manual_seed(42)

    logits = torch.randn(batch, vocab_size)
    original_logits = logits.clone()

    # Tokens 0–9 each appear 1–4 times in output
    output_token_counts = torch.zeros(batch, vocab_size)
    output_token_counts[:, :10] = torch.randint(1, 5, (batch, 10)).float()

    # Tokens 5–14 appeared in prompt (overlaps output at 5–9)
    prompt_token_mask = torch.zeros(batch, vocab_size, dtype=torch.bool)
    prompt_token_mask[:, 5:15] = True

    presence_penalties = torch.full((batch,), presence)
    frequency_penalties = torch.full((batch,), frequency)
    repetition_penalties = torch.full((batch,), repetition)

    sampler = Sampler()
    actual = sampler.apply_penalties(
        logits.clone(),
        output_token_counts,
        prompt_token_mask,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    expected = apply_penalties_reference(
        logits,
        output_token_counts,
        prompt_token_mask,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    match = torch.isclose(actual, expected, atol=1e-5)
    assert match.all(), (
        f"max delta={( actual - expected).abs().max():.6f}, "
        f"first mismatch at {torch.where(~match)}"
    )

    # For neutral penalties the logits must be completely unchanged
    if presence == 0.0 and frequency == 0.0 and repetition == 1.0:
        assert torch.allclose(
            actual, original_logits, atol=1e-5
        ), "neutral penalties must leave logits unchanged"


@pytest.mark.push
def test_repetition_penalty_scope_prompt_only_token():
    """Repetition penalty must cover prompt-only tokens (not just output tokens).

    This directly validates the ``rep_mask = occurred_output | prompt_token_mask``
    fix: the old code (rep_mask = occurred_output only) would leave the
    prompt-only token at 3.0 instead of penalising it to 1.5.
    """
    batch = 1
    vocab_size = 100

    PROMPT_ONLY = 42  # appeared in prompt, not in output; positive logit 3.0
    OUTPUT_ONLY = 7  # appeared in output, not in prompt; positive logit 2.0
    UNSEEN = 50  # appeared nowhere; logit 0.0

    logits = torch.zeros(batch, vocab_size)
    logits[0, PROMPT_ONLY] = 3.0
    logits[0, OUTPUT_ONLY] = 2.0
    logits[0, UNSEEN] = 0.0

    output_token_counts = torch.zeros(batch, vocab_size)
    output_token_counts[0, OUTPUT_ONLY] = 1.0

    prompt_token_mask = torch.zeros(batch, vocab_size, dtype=torch.bool)
    prompt_token_mask[0, PROMPT_ONLY] = True

    presence_penalties = torch.tensor([1.0])
    frequency_penalties = torch.tensor([0.5])
    repetition_penalties = torch.tensor([2.0])

    sampler = Sampler()
    out = sampler.apply_penalties(
        logits,
        output_token_counts,
        prompt_token_mask,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    # Prompt-only token: positive logit → divided by rep (3.0 / 2.0 = 1.5).
    # No freq/presence penalty because it's not in output.
    assert torch.isclose(
        out[0, PROMPT_ONLY], torch.tensor(1.5), atol=1e-5
    ), f"prompt-only token: expected 1.5, got {out[0, PROMPT_ONLY].item():.4f}"

    # Output-only token: positive logit → rep penalty + freq + presence.
    # 2.0/2.0 - 0.5*1 - 1.0*1 = 1.0 - 0.5 - 1.0 = -0.5
    expected_output_only = 2.0 / 2.0 - 0.5 * 1.0 - 1.0 * 1.0
    assert torch.isclose(
        out[0, OUTPUT_ONLY], torch.tensor(expected_output_only), atol=1e-5
    ), (
        f"output-only token: expected {expected_output_only:.4f}, "
        f"got {out[0, OUTPUT_ONLY].item():.4f}"
    )

    # Unseen token: no penalties applied, logit stays at 0.0
    assert torch.isclose(
        out[0, UNSEEN], torch.tensor(0.0), atol=1e-5
    ), f"unseen token: expected 0.0, got {out[0, UNSEEN].item():.4f}"
