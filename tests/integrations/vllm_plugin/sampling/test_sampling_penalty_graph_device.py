# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate apply_penalties math on TT device (compiled graph, no model needed).

Tests that the compiled TTNN kernel produces numerically correct penalty outputs —
not just a valid token index. Isolates the penalty subgraph from the rest of
sample_from_logits so we can observe exact logit values after penalization.

Run with:
    pytest -svv tests/integrations/vllm_plugin/sampling/test_penalty_graph_device.py
"""

import pytest
import torch
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import Sampler

PROD_VOCAB_SIZES = [
    pytest.param(128256, id="llama3_8b"),
    pytest.param(151936, id="qwen3_32b"),
]


@pytest.fixture
def device():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def _apply_penalties(
    logits,
    output_token_counts,
    prompt_token_mask,
    presence_penalties,
    frequency_penalties,
    repetition_penalties,
):
    return Sampler().apply_penalties(
        logits,
        output_token_counts,
        prompt_token_mask,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _run_sampler(logits, metadata):
    return Sampler()(logits, metadata).sampled_token_ids


# ---------------------------------------------------------------------------
# Repetition penalty: positive logits divided, negative logits multiplied.
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
def test_repetition_penalty_occurred_tokens(device):
    """Repetition penalty must divide positive logits and multiply negative logits.

    Setup:
      - logits[0] = +4.0  (positive, occurred in output → divided by rep)
      - logits[1] = -4.0  (negative, occurred in output → multiplied by rep)
      - logits[2] = +4.0  (positive, NOT occurred → unchanged)
      - repetition_penalty = 2.0

    Expected:
      - logits[0] == +2.0  (4.0 / 2.0)
      - logits[1] == -8.0  (-4.0 * 2.0)
      - logits[2] == +4.0  (unchanged)
    """
    VOCAB = 100
    rep = 2.0

    logits = torch.zeros(1, VOCAB, dtype=torch.float32)
    logits[0, 0] = 4.0
    logits[0, 1] = -4.0
    logits[0, 2] = 4.0

    output_token_counts = torch.zeros(1, VOCAB, dtype=torch.float32)
    output_token_counts[0, 0] = 1.0
    output_token_counts[0, 1] = 1.0

    compiled_fn = torch.compile(
        _apply_penalties, backend="tt", fullgraph=True, dynamic=False
    )
    result = compiled_fn(
        logits.to(device),
        output_token_counts.to(device),
        torch.zeros(1, VOCAB, dtype=torch.bool).to(device),
        torch.zeros(1, dtype=torch.float32).to(device),
        torch.zeros(1, dtype=torch.float32).to(device),
        torch.tensor([rep], dtype=torch.float32).to(device),
    ).cpu()

    print(f"[TESTOUT] logits[0]={result[0,0].item():.4f} (expected 2.0)")
    print(f"[TESTOUT] logits[1]={result[0,1].item():.4f} (expected -8.0)")
    print(f"[TESTOUT] logits[2]={result[0,2].item():.4f} (expected 4.0)")

    assert torch.isclose(
        result[0, 0], torch.tensor(2.0), atol=1e-4
    ), f"Positive occurred logit must be divided by rep: expected 2.0, got {result[0,0].item()}"
    assert torch.isclose(
        result[0, 1], torch.tensor(-8.0), atol=1e-4
    ), f"Negative occurred logit must be multiplied by rep: expected -8.0, got {result[0,1].item()}"
    assert torch.isclose(
        result[0, 2], torch.tensor(4.0), atol=1e-4
    ), f"Non-occurred logit must be unchanged: expected 4.0, got {result[0,2].item()}"


# ---------------------------------------------------------------------------
# Presence penalty: flat subtraction for tokens that appeared in output
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
def test_presence_penalty_reduces_occurred_tokens(device):
    """Presence penalty must subtract a flat value from occurred-token logits.

    Setup:
      - logits: all 0.0
      - tokens 0-9: appeared in output once (output_token_counts = 1.0)
      - tokens 10+: never appeared
      - presence_penalty = 2.0

    Expected:
      - logits[0:10]  == -2.0  (presence_penalty subtracted once)
      - logits[10:]   ==  0.0  (unchanged)
    """
    VOCAB = 100
    OCCURRED = 10
    PENALTY = 2.0

    output_token_counts = torch.zeros(1, VOCAB, dtype=torch.float32)
    output_token_counts[0, :OCCURRED] = 1.0

    compiled_fn = torch.compile(
        _apply_penalties, backend="tt", fullgraph=True, dynamic=False
    )
    result = compiled_fn(
        torch.zeros(1, VOCAB, dtype=torch.float32).to(device),
        output_token_counts.to(device),
        torch.zeros(1, VOCAB, dtype=torch.bool).to(device),
        torch.tensor([PENALTY], dtype=torch.float32).to(device),
        torch.zeros(1, dtype=torch.float32).to(device),
        torch.ones(1, dtype=torch.float32).to(device),
    ).cpu()

    penalized = result[0, :OCCURRED]
    unaffected = result[0, OCCURRED:]

    print(f"[TESTOUT] penalized tokens (0-{OCCURRED-1}): {penalized[:5].tolist()}")
    print(f"[TESTOUT] unaffected tokens ({OCCURRED}+): {unaffected[:5].tolist()}")

    assert torch.allclose(
        penalized, torch.full_like(penalized, -PENALTY), atol=1e-4
    ), f"Occurred tokens must have logit -= {PENALTY}, got: {penalized[:5].tolist()}"
    assert torch.allclose(
        unaffected, torch.zeros_like(unaffected), atol=1e-4
    ), f"Non-occurred tokens must be unchanged (0.0), got: {unaffected[:5].tolist()}"


# ---------------------------------------------------------------------------
# Production vocab sizes — same test but at full production scale
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", PROD_VOCAB_SIZES)
def test_presence_penalty_prod_vocab(device, vocab_size):
    """Presence penalty correct at production vocab sizes (Llama/Qwen scale)."""
    OCCURRED = 10
    PENALTY = 2.0

    output_token_counts = torch.zeros(1, vocab_size, dtype=torch.float32)
    output_token_counts[0, :OCCURRED] = 1.0

    compiled_fn = torch.compile(
        _apply_penalties, backend="tt", fullgraph=True, dynamic=False
    )
    result = compiled_fn(
        torch.zeros(1, vocab_size, dtype=torch.float32).to(device),
        output_token_counts.to(device),
        torch.zeros(1, vocab_size, dtype=torch.bool).to(device),
        torch.tensor([PENALTY], dtype=torch.float32).to(device),
        torch.zeros(1, dtype=torch.float32).to(device),
        torch.ones(1, dtype=torch.float32).to(device),
    ).cpu()

    penalized = result[0, :OCCURRED]
    unaffected = result[0, OCCURRED:]

    print(
        f"[TESTOUT] vocab={vocab_size} penalized (0-{OCCURRED-1}): {penalized[:5].tolist()}"
    )
    print(
        f"[TESTOUT] vocab={vocab_size} unaffected ({OCCURRED}+): {unaffected[:5].tolist()}"
    )

    assert torch.allclose(
        penalized, torch.full_like(penalized, -PENALTY), atol=1e-3
    ), f"vocab={vocab_size}: occurred tokens must have logit == -{PENALTY}, got {penalized.tolist()}"
    assert torch.allclose(
        unaffected, torch.zeros_like(unaffected), atol=1e-3
    ), f"vocab={vocab_size}: non-occurred tokens must be unchanged (0.0), got {unaffected[:10].tolist()}"


# ---------------------------------------------------------------------------
# Full sample_from_logits graph: penalty must steer greedy selection away
# from occurred tokens. This tests the interaction of apply_penalties with
# the rest of the sampling graph (temperature, top_k, softmax, rand, argmax).
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", PROD_VOCAB_SIZES)
def test_presence_penalty_steers_full_sampler(device, vocab_size):
    """With strong presence_penalty, greedy must not select any occurred token.

    Setup:
      - logits[0:10] = 10.0  (highest, would win without penalty)
      - logits[10:]  =  0.0
      - tokens 0-9 occurred once in output
      - presence_penalty = 50.0 (strong enough to make 10.0 - 50.0 = -40.0)
      - temperature = 0.0 (greedy)

    Expected: sampled token >= 10 (occurred tokens eliminated by penalty).
    """
    OCCURRED = 10
    PENALTY = 50.0

    logits_cpu = torch.zeros(1, vocab_size, dtype=torch.float32)
    logits_cpu[0, :OCCURRED] = 10.0

    output_token_counts = torch.zeros(1, vocab_size, dtype=torch.float32)
    output_token_counts[0, :OCCURRED] = 1.0

    metadata = XLASupportedSamplingMetadata(
        temperature=torch.zeros(1, device=device),
        top_k=None,
        top_p=None,
        min_p=None,
        all_greedy=False,
        no_penalties=False,
        presence_penalties=torch.tensor([PENALTY], dtype=torch.float32, device=device),
        frequency_penalties=torch.zeros(1, dtype=torch.float32, device=device),
        repetition_penalties=torch.ones(1, dtype=torch.float32, device=device),
        output_token_counts=output_token_counts.to(device),
        prompt_token_mask=torch.zeros(1, vocab_size, dtype=torch.bool, device=device),
    )

    compiled_fn = torch.compile(
        _run_sampler, backend="tt", fullgraph=True, dynamic=False
    )
    token = compiled_fn(logits_cpu.to(device), metadata).cpu().item()

    print(
        f"[TESTOUT] vocab={vocab_size} presence_penalty={PENALTY}: sampled token={token}"
    )

    assert token >= OCCURRED, (
        f"vocab={vocab_size}: presence_penalty={PENALTY} must steer away from tokens 0-{OCCURRED-1}, "
        f"but got token {token}. Occurred tokens had logit=10.0; after penalty should be -40.0."
    )
