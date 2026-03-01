# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for seed / q_samples pre-sampling logic.

Pure CPU tests — no TT hardware, no torch_xla, no compilation needed.
Validates that:
  - random_sample() uses q_samples when provided (not on-device noise)
  - same seed -> same Generator -> same noise -> same token
  - mixed batch: seeded rows deterministic, un-seeded rows use global RNG
  - q_samples=None falls through to the original exponential_ path
"""

import pytest
import torch
from vllm_tt.sampler import Sampler

# Tests do not need silicon runner, but CI only runs on silicon runners.
pytestmark = pytest.mark.single_device

_VOCAB = 1000  # small vocab keeps CPU tests fast


def _build_q_samples(batch_size, vocab_size, generators=None):
    """Reproduce the q_samples construction from metadata.py on CPU."""
    q = torch.empty(batch_size, vocab_size, dtype=torch.float32)
    q.exponential_()
    if generators:
        for idx, gen in generators.items():
            q[idx].exponential_(generator=gen)
    return q


@pytest.mark.push
def test_seed_random_sample_uses_q_samples():
    """random_sample() with q_samples must use the provided noise, not generate new.

    Constructs known probs and known q, then verifies the output matches
    the manual (probs / q).argmax() — proving the q_samples branch is taken.
    """
    sampler = Sampler()
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, _VOCAB), dim=-1)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    q = torch.empty(1, _VOCAB, dtype=torch.float32)
    q.exponential_(generator=gen)

    expected = probs.clone().div_(q.clone()).argmax(dim=-1).view(-1)
    actual = sampler.random_sample(probs.clone(), generators={}, q_samples=q.clone())

    assert (
        actual.item() == expected.item()
    ), f"random_sample must use q_samples: expected {expected.item()}, got {actual.item()}"


@pytest.mark.push
def test_seed_q_samples_determinism():
    """Same seed -> same Generator -> same exponential noise -> same token."""
    sampler = Sampler()
    torch.manual_seed(0)
    logits = torch.randn(1, _VOCAB, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    tokens = []
    for _ in range(3):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        q = torch.empty(1, _VOCAB, dtype=torch.float32)
        q.exponential_(generator=gen)
        tok = sampler.random_sample(probs.clone(), generators={}, q_samples=q)
        tokens.append(tok.item())

    assert (
        tokens[0] == tokens[1] == tokens[2]
    ), f"Same seed must produce identical tokens across runs: {tokens}"

    # Different seed should produce different noise (and very likely a different token).
    gen_diff = torch.Generator(device="cpu")
    gen_diff.manual_seed(999)
    q_diff = torch.empty(1, _VOCAB, dtype=torch.float32)
    q_diff.exponential_(generator=gen_diff)
    tok_diff = sampler.random_sample(probs.clone(), generators={}, q_samples=q_diff)

    # With vocab=1000 and different noise, collision is very unlikely.
    assert tok_diff.item() != tokens[0], (
        f"Different seed should (almost certainly) produce a different token: "
        f"seed=42 -> {tokens[0]}, seed=999 -> {tok_diff.item()}"
    )


@pytest.mark.push
def test_seed_mixed_batch_q_samples():
    """Mixed batch: seeded rows are deterministic, un-seeded rows use global RNG.

    Simulates the metadata.py construction: batch of 4 requests where
    rows 1 and 3 have per-request seeds, rows 0 and 2 use global RNG.
    The seeded rows must be identical across two independent constructions.
    """
    sampler = Sampler()
    batch_size = 4
    torch.manual_seed(0)
    logits = torch.randn(batch_size, _VOCAB, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    seeded_indices = {1: 42, 3: 123}

    results = []
    for _ in range(2):
        generators = {}
        for idx, seed in seeded_indices.items():
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            generators[idx] = g
        q = _build_q_samples(batch_size, _VOCAB, generators)
        tokens = sampler.random_sample(probs.clone(), generators={}, q_samples=q)
        results.append(tokens.tolist())

    # Seeded rows must be identical across runs.
    for idx in seeded_indices:
        assert results[0][idx] == results[1][idx], (
            f"Seeded row {idx} (seed={seeded_indices[idx]}) must be deterministic: "
            f"run1={results[0][idx]}, run2={results[1][idx]}"
        )

    # Un-seeded rows use global RNG so they'll differ across runs
    # (since we don't reset torch.manual_seed between runs).
    # We can't assert inequality reliably, but we can assert the tokens are valid.
    for run in results:
        for tok in run:
            assert 0 <= tok < _VOCAB, f"Token {tok} out of range"


@pytest.mark.push
def test_seed_q_samples_none_falls_through():
    """When q_samples is None, random_sample generates noise on-device (existing path).

    Two calls without q_samples should produce different tokens (with high
    probability) because the on-device exponential_ draws from the global RNG.
    """
    sampler = Sampler()
    torch.manual_seed(0)
    logits = torch.randn(1, _VOCAB, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)

    # q_samples=None triggers the original on-device exponential path.
    tok1 = sampler.random_sample(probs.clone(), generators={}, q_samples=None)
    tok2 = sampler.random_sample(probs.clone(), generators={}, q_samples=None)

    # With vocab=1000 and independent RNG draws, collision is very unlikely.
    assert tok1.item() != tok2.item(), (
        f"Without q_samples, successive calls should produce different tokens "
        f"(global RNG): got {tok1.item()} both times"
    )
