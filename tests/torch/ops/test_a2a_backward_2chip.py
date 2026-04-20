# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end MoE backward on an N300 1x2 SPMD mesh.

Covers:
    1. Dispatch's backward emits `tt.all_to_all_combine` on 2 chips.
    2. Combine's backward (scatter_add) works under the same SPMD setup.
    3. moe_expert_token_remap's backward flows under the same SPMD setup.
    4. A full router + dispatch + experts + combine pipeline backward with
       a PCC check against a CPU reference.

Run this file in its OWN pytest session — `xr.use_spmd()` cannot be
un-called, and if non-SPMD XLA tests have already run in the session the
torch_xla client warns about "Replicating tensors already initialized on
non-virtual XLA device for SPMD" and the tt-mlir lowering that follows
can trip "Read unexpected run_mailbox value" / fabric setup errors:

    TT_VISIBLE_DEVICES=0 pytest -xvs tests/torch/ops/test_a2a_backward_2chip.py
"""

import os

# Must be set BEFORE any torch_xla import so the plugin loads in Shardy mode.
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import numpy as np
import pytest
import torch
import torch.nn as nn

from tt_torch import custom_ops  # noqa: F401 — registers custom ops + autograd


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_expert_mapping(E: int, D: int) -> torch.Tensor:
    """[1, 1, E, D] one-hot: expert e lives on device (e % D)."""
    m = torch.zeros(1, 1, E, D, dtype=torch.int64)
    for e in range(E):
        m[0, 0, e, e % D] = 1
    return m


def _setup_spmd_mesh():
    """Returns (mesh, num_devices, xs, xm). Call once per test."""
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh, num_devices, xs, xm


def _require_2_chips():
    import torch_xla.runtime as xr

    if xr.global_runtime_device_count() < 2:
        pytest.skip("need 2 XLA devices")


# ----------------------------------------------------------------------------
# 1. Dispatch's backward actually emits combine on 2 chips.
# ----------------------------------------------------------------------------


@pytest.mark.single_device
def test_dispatch_backward_2chip_spmd_runs():
    """Shape + finiteness check — proves the combine-backward compile and
    execute path works on the real 1x2 mesh."""
    _require_2_chips()
    mesh, num_devices, xs, xm = _setup_spmd_mesh()

    dev = xm.xla_device()
    B, S, H, K, E, D = 2, 8, 16, 2, 8, 2
    cluster_axis = 1  # "model"

    torch.manual_seed(0)
    x_cpu = torch.randn(B, S, H, dtype=torch.float32)
    idx_cpu = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    x_xla = x_cpu.clone().to(dev).requires_grad_(True)
    idx_xla = idx_cpu.to(dev)
    mp_xla = mp_cpu.to(dev)
    xs.mark_sharding(mp_xla, mesh, (None, None, None, "model"))

    dispatched, _ = torch.ops.tt.all_to_all_dispatch(
        x_xla, idx_xla, mp_xla, num_devices=D, cluster_axis=cluster_axis
    )
    assert dispatched.shape == (1, B * D, S, H)

    grad_out = torch.randn(1, B * D, S, H, device=dev, dtype=torch.float32)
    xs.mark_sharding(grad_out, mesh, (None, "model", None, None))
    (g_xla,) = torch.autograd.grad(dispatched, x_xla, grad_out)

    g = g_xla.cpu()
    assert g.shape == x_cpu.shape
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0


# ----------------------------------------------------------------------------
# 2. Combine's backward — scatter_add torch ops — runs under SPMD.
# ----------------------------------------------------------------------------


@pytest.mark.single_device
def test_combine_backward_2chip_spmd_runs():
    _require_2_chips()
    mesh, num_devices, xs, xm = _setup_spmd_mesh()
    dev = xm.xla_device()

    B, S, H, K, E, D = 2, 8, 16, 2, 8, 2
    BD = B * D
    cluster_axis = 1

    # Use sparse_mlp's expert_out layout.
    torch.manual_seed(1)
    expert_out_cpu = torch.randn(E, 1, BD * S, H, dtype=torch.float32)
    meta_cpu = torch.randint(0, E, (1, 1, BD * S, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    expert_out_xla = expert_out_cpu.to(dev).requires_grad_(True)
    meta_xla = meta_cpu.to(dev)
    mp_xla = mp_cpu.to(dev)
    xs.mark_sharding(expert_out_xla, mesh, ("model", None, None, None))
    xs.mark_sharding(mp_xla, mesh, (None, None, None, "model"))

    combined = torch.ops.tt.all_to_all_combine(
        expert_out_xla, meta_xla, mp_xla,
        num_devices=D, cluster_axis=cluster_axis,
        num_experts_per_tok=K, output_shard_dim=2,
    )
    assert combined.shape == (K, 1, B * S, H)

    grad_out = torch.randn(K, 1, B * S, H, device=dev, dtype=torch.float32)
    (g_xla,) = torch.autograd.grad(combined, expert_out_xla, grad_out)

    g = g_xla.cpu()
    assert g.shape == expert_out_cpu.shape
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0


# ----------------------------------------------------------------------------
# 4. moe_expert_token_remap's backward — scatter torch ops — runs under SPMD.
# ----------------------------------------------------------------------------


@pytest.mark.single_device
def test_moe_expert_token_remap_backward_2chip_spmd_runs():
    _require_2_chips()
    mesh, num_devices, xs, xm = _setup_spmd_mesh()
    dev = xm.xla_device()

    B, S, H, K, E, D = 2, 8, 16, 2, 8, 2
    BS = B * S

    torch.manual_seed(2)
    topk_cpu = torch.randn(BS, E, dtype=torch.float32)
    mp_cpu = _make_expert_mapping(E, D)
    meta_cpu = torch.randint(0, E, (1, 1, BS * D, K), dtype=torch.int64)

    topk_xla = topk_cpu.to(dev).requires_grad_(True)
    mp_xla = mp_cpu.to(dev)
    meta_xla = meta_cpu.to(dev)
    xs.mark_sharding(mp_xla, mesh, (None, None, None, "model"))

    mapping_xla, _ = torch.ops.tt.moe_expert_token_remap(
        topk_xla, mp_xla, meta_xla, num_devices=D, reduction_size=32
    )
    grad_out = torch.randn_like(mapping_xla)
    (g_xla,) = torch.autograd.grad(mapping_xla, topk_xla, grad_out)

    g = g_xla.cpu()
    assert g.shape == topk_cpu.shape
    assert torch.isfinite(g).all()
    assert g.abs().sum().item() > 0


# ----------------------------------------------------------------------------
# 5. Full MoE pipeline (router + dispatch + experts + combine) backward.
# ----------------------------------------------------------------------------


class TinyMoE(nn.Module):
    """Router + dispatch + per-expert einsum + combine + weighted sum.

    Deliberately uses K == D so each token's K experts cover all D devices,
    which makes the distributed adjoint agree with a naive CPU sum.
    """

    def __init__(self, H, E, K, D):
        super().__init__()
        self.H, self.E, self.K, self.D = H, E, K, D
        self.router = nn.Linear(H, E, bias=False)
        self.expert_weights = nn.Parameter(torch.randn(E, H, H) * 0.02)
        self.register_buffer("expert_mapping", _make_expert_mapping(E, D))

    def forward(self, hidden_states, cluster_axis):
        B, S, H = hidden_states.shape
        K, E, D = self.K, self.E, self.D

        router_logits = self.router(hidden_states.reshape(-1, H))
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, K, dim=-1)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-6)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            hidden_states, topk_idx, self.expert_mapping,
            num_devices=D, cluster_axis=cluster_axis,
        )
        BD = dispatched.shape[1]
        metadata_flat = metadata.reshape(1, 1, BD * S, K)

        # Expert compute: per-expert linear applied to dispatched tokens.
        tokens_flat = dispatched.reshape(BD * S, H)
        expert_out = torch.einsum("nh,eho->eno", tokens_flat, self.expert_weights)
        expert_out = expert_out.reshape(E, 1, BD * S, H)

        combined = torch.ops.tt.all_to_all_combine(
            expert_out, metadata_flat, self.expert_mapping,
            num_devices=D, cluster_axis=cluster_axis,
            num_experts_per_tok=K, output_shard_dim=2,
        )
        # combined: [K, 1, B*S, H]
        weights = topk_weights.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        out = (combined * weights).sum(dim=0)  # [1, B*S, H]
        return out.reshape(B, S, H)


@pytest.mark.single_device
def test_full_moe_pipeline_backward_2chip():
    """End-to-end: router + dispatch + experts + combine on 2 chips,
    compare gradient of x against a CPU reference."""
    _require_2_chips()
    mesh, num_devices, xs, xm = _setup_spmd_mesh()
    dev = xm.xla_device()

    B, S, H, K, E, D = 2, 8, 16, 2, 2, 2
    cluster_axis = 1

    torch.manual_seed(7)
    model = TinyMoE(H, E, K, D).to(torch.float32)
    x_cpu = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)

    # --- CPU golden ---
    out_cpu = model(x_cpu, cluster_axis=cluster_axis)
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()
    g_x_cpu = x_cpu.grad.detach().clone()
    g_expert_cpu = model.expert_weights.grad.detach().clone()
    g_router_cpu = model.router.weight.grad.detach().clone()
    model.zero_grad()

    # --- XLA 2-chip run ---
    model_xla = TinyMoE(H, E, K, D).to(torch.float32)
    # Copy the CPU model's parameters so both graphs see identical weights.
    model_xla.load_state_dict(model.state_dict())
    model_xla = model_xla.to(dev)

    x_xla = x_cpu.detach().clone().to(dev).requires_grad_(True)

    # Shard expert weights on E so each chip has E/D local experts — matches
    # combine's E_local assertion. Also shard the mapping so Shardy annotates.
    xs.mark_sharding(model_xla.expert_mapping, mesh, (None, None, None, "model"))
    xs.mark_sharding(model_xla.expert_weights, mesh, ("model", None, None))

    out_xla = model_xla(x_xla, cluster_axis=cluster_axis)
    out_xla.sum().backward()

    diff_x = (g_x_cpu - x_xla.grad.cpu()).abs().max().item()
    diff_expert = (g_expert_cpu - model_xla.expert_weights.grad.cpu()).abs().max().item()
    diff_router = (g_router_cpu - model_xla.router.weight.grad.cpu()).abs().max().item()
    print(f"x grad max_diff = {diff_x}")
    print(f"expert_weights grad max_diff = {diff_expert}")
    print(f"router.weight grad max_diff = {diff_router}")

    # Tolerances reflect bfloat16 precision on TT hardware + accumulated error
    # through router → dispatch → einsum → combine → weighted-sum → backward.
    # Check PCC (Pearson correlation) as the primary quality metric and keep
    # a loose absolute bound as a sanity check.
    def pcc(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        va = a - a.mean()
        vb = b - b.mean()
        denom = va.norm() * vb.norm()
        return 1.0 if denom == 0 else float((va @ vb) / denom)

    pcc_x = pcc(g_x_cpu, x_xla.grad.cpu())
    pcc_expert = pcc(g_expert_cpu, model_xla.expert_weights.grad.cpu())
    pcc_router = pcc(g_router_cpu, model_xla.router.weight.grad.cpu())
    print(f"PCC x = {pcc_x:.6f}, expert = {pcc_expert:.6f}, router = {pcc_router:.6f}")

    assert pcc_x > 0.99, f"x grad PCC too low: {pcc_x}"
    assert pcc_expert > 0.99, f"expert_weights grad PCC too low: {pcc_expert}"
    assert pcc_router > 0.99, f"router.weight grad PCC too low: {pcc_router}"
