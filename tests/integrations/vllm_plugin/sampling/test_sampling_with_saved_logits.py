# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate TT device sampling using real saved model logits.

Loads a fixture captured by capture_logits.py (real Llama-3.2-3B logits at
decode step 1) and runs stage-by-stage validation of the sampling pipeline on
TT device vs CPU golden.

This test is more realistic than test_sampling_pipeline.py (which uses random
logits) because real model logits have a peaked distribution that may expose
bugs random inputs miss.

Run capture_logits.py first to generate the fixture:
    python tests/integrations/vllm_plugin/sampling/capture_logits.py
"""

import os

import pytest
import torch
import torch_xla.core.xla_model as xm
from vllm_tt.sampler import Sampler, apply_top_k_top_p_fast

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "llama3_2_3b_decode_step1.pt"
)


@pytest.fixture(scope="module")
def fixture():
    if not os.path.exists(FIXTURE_PATH):
        pytest.skip(
            f"Fixture not found: {FIXTURE_PATH}\n"
            "Run capture_logits.py first to generate it."
        )
    return torch.load(FIXTURE_PATH, weights_only=False)


@pytest.fixture(scope="module")
def device():
    return xm.xla_device()


# ---------------------------------------------------------------------------
# Stage 1: chunked topk — same top-k values selected on device as on CPU?
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_stage1_topk_values(fixture, device):
    """Top-k candidate VALUES should match CPU (ordering may differ)."""
    logits = fixture["logits"]  # [1, vocab_size] float32 CPU

    vals_cpu, idx_cpu = apply_top_k_top_p_fast(logits, None, None)
    vals_dev, idx_dev = apply_top_k_top_p_fast(logits.to(device), None, None)

    vals_gathered_cpu = torch.gather(logits, -1, idx_cpu)
    vals_gathered_dev = torch.gather(logits, -1, idx_dev.cpu())
    cos_sim = torch.nn.functional.cosine_similarity(
        vals_gathered_cpu.flatten().unsqueeze(0),
        vals_gathered_dev.flatten().unsqueeze(0),
    )
    idx_match = (idx_cpu == idx_dev.cpu()).float().mean().item()
    print(
        f"\n  vocab={logits.shape[-1]}  candidates={vals_cpu.shape[-1]}"
        f"  values_cos_sim={cos_sim.item():.6f}"
        f"  index_exact_match={idx_match:.3f} (ordering non-deterministic)"
    )
    assert cos_sim > 0.99, f"topk values wrong: cos_sim={cos_sim.item():.6f}"


# ---------------------------------------------------------------------------
# Stage 2: softmax on candidate set
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_stage2_softmax(fixture, device):
    """Softmax on the candidate set should match CPU."""
    logits = fixture["logits"]
    vals_cpu, _ = apply_top_k_top_p_fast(logits, None, None)

    probs_cpu = vals_cpu.softmax(dim=-1, dtype=torch.float32)
    probs_dev = vals_cpu.to(device).softmax(dim=-1, dtype=torch.float32).cpu()

    cos_sim = torch.nn.functional.cosine_similarity(
        probs_cpu.flatten().unsqueeze(0),
        probs_dev.flatten().unsqueeze(0),
    )
    print(f"\n  softmax cos_sim={cos_sim.item():.6f}")
    assert cos_sim > 0.99, f"softmax wrong: cos_sim={cos_sim.item():.6f}"


# ---------------------------------------------------------------------------
# Stage 3: gather — map local candidate index to global vocab token
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_stage3_gather(fixture, device):
    """Gather: mapping local→global vocab should return the same token on device."""
    logits = fixture["logits"]
    vals_cpu, idx_cpu = apply_top_k_top_p_fast(logits, None, None)
    probs_cpu = vals_cpu.softmax(dim=-1, dtype=torch.float32)

    torch.manual_seed(42)
    q = torch.empty_like(probs_cpu).exponential_()
    local_idx = probs_cpu.div(q).argmax(dim=-1)  # [1]

    # Test gather with native XLA int64 tensors (never transferred from CPU).
    idx_dev = idx_cpu.to(device)
    local_dev = local_idx.to(device)

    global_cpu = idx_cpu.gather(1, local_idx.unsqueeze(-1)).squeeze(-1).item()
    global_dev = idx_dev.gather(1, local_dev.unsqueeze(-1)).squeeze(-1).cpu().item()

    print(
        f"\n  local_idx={local_idx.item()}  "
        f"cpu_token={global_cpu}  dev_token={global_dev}  match={global_cpu == global_dev}"
    )
    assert global_cpu == global_dev, (
        f"gather mismatch: cpu={global_cpu} dev={global_dev} "
        f"(local_idx={local_idx.item()})"
    )


# ---------------------------------------------------------------------------
# End-to-end greedy: device argmax must match CPU exactly
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_greedy_token_matches(fixture, device):
    """Greedy sampling (argmax) on real logits must exactly match CPU."""
    logits = fixture["logits"]
    expected = fixture["greedy_token"]

    def run_greedy(logits, _):
        return torch.argmax(logits, dim=-1, keepdim=True)

    compiled = torch.compile(run_greedy, backend="tt", dynamic=False)
    actual = compiled(logits.to(device), None).cpu().item()

    print(f"\n  expected={expected}  actual={actual}  match={expected == actual}")
    assert actual == expected, f"Greedy mismatch: expected={expected} got={actual}"


# ---------------------------------------------------------------------------
# End-to-end non-greedy: sampled token must be within top-K candidates
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_nongreedy_token_in_topk(fixture, device):
    """Non-greedy sampled token must come from the top-K candidate set."""
    logits = fixture["logits"]
    _, idx_cpu = apply_top_k_top_p_fast(logits, None, None)
    top_k_tokens = set(idx_cpu.squeeze(0).tolist())

    def run_sampler(logits, metadata):
        sampler = Sampler()
        return sampler(logits, metadata).sampled_token_ids

    from vllm_tt.metadata import XLASupportedSamplingMetadata

    dev = device
    metadata = XLASupportedSamplingMetadata(
        temperature=torch.full((1,), 0.8, device=dev),
        top_k=None,
        top_p=None,
        min_p=torch.zeros(1, device=dev),
    )

    compiled = torch.compile(run_sampler, backend="tt", dynamic=False)
    result = compiled(logits.to(device), metadata).cpu()
    token = result.item()

    in_topk = token in top_k_tokens
    print(
        f"\n  sampled_token={token}  in_top_{len(top_k_tokens)}={in_topk}"
        f"  vocab_size={logits.shape[-1]}"
    )
    assert 0 <= token < logits.shape[-1], f"Token {token} out of vocab range"
    assert in_topk, (
        f"Sampled token {token} not in top-{len(top_k_tokens)} candidates — "
        "sampling is broken"
    )
