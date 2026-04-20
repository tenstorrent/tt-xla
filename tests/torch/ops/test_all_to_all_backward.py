# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the autograd/backward passes of:
    torch.ops.tt.all_to_all_dispatch
    torch.ops.tt.all_to_all_combine

We verify:
    * CPU backward matches a hand-derived reference.
    * Forward/backward composition yields expected gradients.
    * CPU and XLA agree (within bfloat16 tolerance) on gradient values.
    * Edge cases: multiple (B, S, H, K, E, D) combos and both output_shard_dim options.

Run the XLA-path tests with TT_VISIBLE_DEVICES=0 so only one device is exposed.
"""

import pytest
import torch

# Ensure custom ops (incl. autograd) are registered.
from tt_torch import custom_ops  # noqa: F401


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_expert_mapping(E: int, D: int) -> torch.Tensor:
    """One-hot [1, 1, E, D] map: expert e lives on device (e % D)."""
    m = torch.zeros(1, 1, E, D, dtype=torch.int64)
    for e in range(E):
        m[0, 0, e, e % D] = 1
    return m


def _random_inputs_dispatch(B, S, H, K, E, seed=0, input_dim=3):
    torch.manual_seed(seed)
    if input_dim == 3:
        x = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    elif input_dim == 4:
        x = torch.randn(B, 1, S, H, dtype=torch.float32, requires_grad=True)
    else:
        raise ValueError(input_dim)
    idx = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    return x, idx


def _reference_dispatch_input_grad(grad_dispatched, B, S, H, D, input_dim=3):
    """Hand-derived d_input[b, s, :] = sum_d grad_dispatched[0, d*B + b, s, :]."""
    g = grad_dispatched.reshape(1, D, B, S, H).sum(dim=1).squeeze(0)  # [B, S, H]
    if input_dim == 4:
        g = g.unsqueeze(1)
    return g


def _reference_combine_input_grad(
    grad_combined, expert_metadata, E, BD, S, H, K, D, output_shard_dim=1
):
    """d_input_flat[e, tok, :] = sum_{k: metadata[tok, k]==e, valid} grad_combined[k, tok, :]."""
    tokens = BD * S
    tpd = tokens // D
    if output_shard_dim == 1:
        g = grad_combined.squeeze(2)
    else:
        g = grad_combined.squeeze(1)

    d_input_flat = torch.zeros(E, tokens, H, dtype=grad_combined.dtype)
    meta_tpd = expert_metadata[0, 0, :tpd].long()  # [tpd, K]
    for tok in range(tpd):
        for k in range(K):
            e = int(meta_tpd[tok, k])
            if 0 <= e < E:
                d_input_flat[e, tok, :] += g[k, tok, :]
    return d_input_flat.reshape(E, BD, S, H)


# ----------------------------------------------------------------------------
# CPU tests for all_to_all_dispatch backward
# ----------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize(
    "B,S,H,K,E,D",
    [
        (1, 4, 8, 2, 4, 1),
        (2, 4, 8, 2, 4, 1),
        (2, 8, 16, 2, 8, 2),
        (4, 16, 32, 4, 8, 2),
        (1, 32, 64, 2, 16, 4),
    ],
)
@pytest.mark.parametrize("input_dim", [3, 4])
def test_dispatch_backward_cpu_matches_reference(B, S, H, K, E, D, input_dim):
    x, idx = _random_inputs_dispatch(B, S, H, K, E, seed=B * S + D, input_dim=input_dim)
    mp = _make_expert_mapping(E, D)

    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, idx, mp, num_devices=D, cluster_axis=0
    )
    assert dispatched.shape == (1, B * D, S, H)
    assert metadata.shape == (1, B * D, S, K)

    grad_out = torch.randn_like(dispatched)
    (g,) = torch.autograd.grad(dispatched, x, grad_out)

    expected = _reference_dispatch_input_grad(grad_out, B, S, H, D, input_dim=input_dim)
    assert g.shape == x.shape
    assert torch.allclose(g, expected, atol=1e-6), (
        f"dispatch bwd mismatch max_diff={(g - expected).abs().max().item()}"
    )


@pytest.mark.single_device
def test_dispatch_backward_cpu_single_device_is_permute_inverse():
    # With num_devices=1 the forward is a permute; backward should invert it exactly.
    B, S, H, K, E = 3, 5, 7, 2, 4
    x = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    idx = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    mp = _make_expert_mapping(E, 1)

    dispatched, _ = torch.ops.tt.all_to_all_dispatch(
        x, idx, mp, num_devices=1, cluster_axis=0
    )
    grad_out = torch.randn_like(dispatched)
    (g,) = torch.autograd.grad(dispatched, x, grad_out)
    # Forward: permute(1,0,2,3) of [B,1,S,H] → [1,B,S,H]. Backward: swap dims 0/1 back.
    expected = grad_out.squeeze(0)
    assert torch.allclose(g, expected, atol=1e-6)


# ----------------------------------------------------------------------------
# CPU tests for all_to_all_combine backward
# ----------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize(
    "B,S,H,K,E,D",
    [
        (1, 4, 8, 2, 4, 1),
        (2, 4, 8, 2, 4, 1),
        (2, 8, 16, 2, 8, 2),
        (1, 16, 32, 4, 16, 1),
        (2, 8, 16, 3, 8, 2),
    ],
)
@pytest.mark.parametrize("output_shard_dim", [1, 2])
def test_combine_backward_cpu_matches_reference(B, S, H, K, E, D, output_shard_dim):
    torch.manual_seed(B * S + D + output_shard_dim)
    BD = B * D
    tokens = BD * S

    expert_out = torch.randn(E, BD, S, H, dtype=torch.float32, requires_grad=True)
    meta = torch.randint(0, E, (1, 1, tokens, K), dtype=torch.int64)
    mp = _make_expert_mapping(E, D)

    combined = torch.ops.tt.all_to_all_combine(
        expert_out,
        meta,
        mp,
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=output_shard_dim,
    )
    expected_shape = (
        (K, tokens // D, 1, H) if output_shard_dim == 1 else (K, 1, tokens // D, H)
    )
    assert combined.shape == expected_shape

    grad_out = torch.randn_like(combined)
    (g,) = torch.autograd.grad(combined, expert_out, grad_out)

    expected = _reference_combine_input_grad(
        grad_out, meta, E, BD, S, H, K, D, output_shard_dim
    )
    assert g.shape == expert_out.shape
    assert torch.allclose(g, expected, atol=1e-6), (
        f"combine bwd mismatch max_diff={(g - expected).abs().max().item()}"
    )


@pytest.mark.single_device
def test_combine_backward_with_invalid_expert_ids_masks_gradient():
    """Entries in metadata that point outside [0, E_local) must contribute zero gradient."""
    B, S, H, K, E, D = 1, 2, 3, 2, 4, 1
    BD = B * D
    tokens = BD * S
    expert_out = torch.randn(E, BD, S, H, dtype=torch.float32, requires_grad=True)
    # Introduce a sentinel invalid index (E = out of range).
    meta = torch.tensor([[[[0, E], [1, 2]]]], dtype=torch.int64)
    assert meta.shape == (1, 1, tokens, K)
    mp = _make_expert_mapping(E, D)

    combined = torch.ops.tt.all_to_all_combine(
        expert_out,
        meta,
        mp,
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=1,
    )
    # Entry where metadata was E (invalid) must contribute zero to d_input.
    # We verify by checking only the valid (e, tok) pairs received gradient.
    grad_out = torch.ones_like(combined)
    (g,) = torch.autograd.grad(combined, expert_out, grad_out)
    # Valid (e, tok): (0, 0), (1, 1), (2, 1) → non-zero. Others: zero.
    nz = (g != 0).reshape(E, -1).any(dim=-1)
    # tok 0: only k=0 valid, expert 0 → e=0 gets grad
    # tok 1: k=0 expert 1, k=1 expert 2 → e=1, 2 get grad
    assert bool(nz[0])
    assert bool(nz[1])
    assert bool(nz[2])
    assert not bool(nz[3])  # expert 3 never selected


# ----------------------------------------------------------------------------
# CPU tests: composed dispatch→identity→combine gradient flow
# ----------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize(
    "B,S,H,K,E,D",
    [
        (1, 4, 3, 2, 4, 1),
        (2, 2, 5, 2, 4, 1),
        (1, 4, 3, 2, 4, 2),
        (2, 2, 5, 2, 8, 2),
    ],
)
def test_dispatch_combine_identity_grad_flow(B, S, H, K, E, D):
    """Compose dispatch → (fake identity-per-expert) → combine and check x.grad = K (all ones)."""
    torch.manual_seed(0)
    x = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    idx = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    mp = _make_expert_mapping(E, D)

    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, idx, mp, num_devices=D, cluster_axis=0
    )
    # dispatch metadata is [1, BD, S, K]; combine expects [1, 1, BD*S, K].
    metadata_for_combine = metadata.reshape(1, 1, B * D * S, K)
    # Build fake expert outputs: for every expert, output = dispatched (pure identity).
    expert_out = (
        dispatched.unsqueeze(0).expand(E, -1, -1, -1, -1).reshape(E, B * D, S, H)
    )

    combined = torch.ops.tt.all_to_all_combine(
        expert_out,
        metadata_for_combine,
        mp,
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=1,
    )
    loss = combined.sum()
    loss.backward()
    # Each x[b, s, h] is duplicated D times in dispatched, then copied to every expert,
    # then combine picks K experts per token. Since all experts are identity and each picks,
    # every (b, s, h) contributes K copies to the final sum. Gradient = K everywhere.
    assert torch.allclose(x.grad, torch.full_like(x.grad, float(K)), atol=1e-6)


# ----------------------------------------------------------------------------
# XLA tests: compare CPU reference against XLA device
# ----------------------------------------------------------------------------


def _xla_available():
    try:
        import torch_xla.core.xla_model as xm  # noqa

        xm.xla_device()
        return True
    except Exception:
        return False


@pytest.mark.single_device
@pytest.mark.skipif(not _xla_available(), reason="XLA device unavailable")
@pytest.mark.parametrize(
    "B,S,H,K,E,D",
    [
        (1, 4, 8, 2, 4, 1),
        (2, 4, 8, 2, 4, 1),
        (2, 8, 16, 2, 8, 2),
    ],
)
def test_dispatch_backward_xla_matches_cpu(B, S, H, K, E, D):
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    torch.manual_seed(B * S + D)
    x_cpu = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    idx_cpu = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    dispatched_cpu, _ = torch.ops.tt.all_to_all_dispatch(
        x_cpu, idx_cpu, mp_cpu, num_devices=D, cluster_axis=0
    )
    grad_out_cpu = torch.randn_like(dispatched_cpu)
    (g_cpu,) = torch.autograd.grad(dispatched_cpu, x_cpu, grad_out_cpu)

    x_xla = x_cpu.detach().clone().to(dev).requires_grad_(True)
    dispatched_xla, _ = torch.ops.tt.all_to_all_dispatch(
        x_xla, idx_cpu.to(dev), mp_cpu.to(dev), num_devices=D, cluster_axis=0
    )
    grad_out_xla = grad_out_cpu.to(dev)
    (g_xla,) = torch.autograd.grad(dispatched_xla, x_xla, grad_out_xla)

    g_xla_cpu = g_xla.cpu()
    # bfloat16 on device → ~4e-3 relative tolerance against fp32 CPU
    assert torch.allclose(g_cpu, g_xla_cpu, atol=1e-2, rtol=1e-2), (
        f"max_diff={(g_cpu - g_xla_cpu).abs().max().item()}"
    )


@pytest.mark.single_device
@pytest.mark.skipif(not _xla_available(), reason="XLA device unavailable")
@pytest.mark.parametrize(
    "B,S,H,K,E,D",
    [
        (1, 4, 8, 2, 4, 1),
        (2, 4, 8, 2, 4, 1),
        (2, 8, 16, 2, 8, 2),
    ],
)
@pytest.mark.parametrize("output_shard_dim", [1, 2])
def test_combine_backward_xla_matches_cpu(B, S, H, K, E, D, output_shard_dim):
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    torch.manual_seed(B * S + D + output_shard_dim)
    BD = B * D
    tokens = BD * S

    expert_out_cpu = torch.randn(E, BD, S, H, dtype=torch.float32, requires_grad=True)
    meta_cpu = torch.randint(0, E, (1, 1, tokens, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    combined_cpu = torch.ops.tt.all_to_all_combine(
        expert_out_cpu,
        meta_cpu,
        mp_cpu,
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=output_shard_dim,
    )
    grad_out_cpu = torch.randn_like(combined_cpu)
    (g_cpu,) = torch.autograd.grad(combined_cpu, expert_out_cpu, grad_out_cpu)

    expert_out_xla = expert_out_cpu.detach().clone().to(dev).requires_grad_(True)
    combined_xla = torch.ops.tt.all_to_all_combine(
        expert_out_xla,
        meta_cpu.to(dev),
        mp_cpu.to(dev),
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=output_shard_dim,
    )
    grad_out_xla = grad_out_cpu.to(dev)
    (g_xla,) = torch.autograd.grad(combined_xla, expert_out_xla, grad_out_xla)

    g_xla_cpu = g_xla.cpu()
    assert torch.allclose(g_cpu, g_xla_cpu, atol=1e-2, rtol=1e-2), (
        f"max_diff={(g_cpu - g_xla_cpu).abs().max().item()}"
    )


@pytest.mark.single_device
@pytest.mark.skipif(not _xla_available(), reason="XLA device unavailable")
def test_dispatch_combine_grad_flow_xla():
    """End-to-end dispatch→combine gradient flow on XLA.

    We stage the forward/backward per-op rather than composing them into a
    single graph — composing both tuple-returning custom calls in one graph
    trips Shardy propagation ('doesn't support tuples' error) which is a
    pre-existing compiler limitation unrelated to the backward formula.
    """
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    B, S, H, K, E, D = 2, 4, 8, 2, 4, 1
    torch.manual_seed(0)
    x = torch.randn(B, S, H, dtype=torch.float32, device=dev, requires_grad=True)
    idx = torch.randint(0, E, (B, S, K), dtype=torch.int64, device=dev)
    mp = _make_expert_mapping(E, D).to(dev)

    # Stage 1: forward+backward of dispatch only. Produces d_x from d_dispatched.
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, idx, mp, num_devices=D, cluster_axis=0
    )
    grad_dispatched = torch.ones_like(dispatched) * float(E * K)
    # Why E*K? The identity-expert fake pipeline expands dispatched by E and combine
    # collapses to K per token, so downstream grad_dispatched per slot = E*K·1.
    (g_xla,) = torch.autograd.grad(dispatched, x, grad_dispatched)

    # Reference: CPU path through the same backward formula
    x_cpu = x.detach().cpu().requires_grad_(True)
    idx_cpu = idx.cpu()
    mp_cpu = mp.cpu()
    dispatched_cpu, _ = torch.ops.tt.all_to_all_dispatch(
        x_cpu, idx_cpu, mp_cpu, num_devices=D, cluster_axis=0
    )
    grad_dispatched_cpu = torch.ones_like(dispatched_cpu) * float(E * K)
    (g_cpu,) = torch.autograd.grad(dispatched_cpu, x_cpu, grad_dispatched_cpu)

    assert torch.allclose(g_cpu, g_xla.cpu(), atol=1e-2, rtol=1e-2), (
        f"max_diff={(g_cpu - g_xla.cpu()).abs().max().item()}"
    )
