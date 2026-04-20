# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end MoE backward correctness on a 2-chip Wormhole N300 host.

    TT_VISIBLE_DEVICES=0 on this box already exposes a (1, 2) mesh — the two
    Wormhole chips on the N300 board. xr.global_runtime_device_count() == 2.

Pipeline under test (same shape as A2aSparseMLP):
    hidden_states [B, S, H]
        ├─ router (Linear+softmax+topk) → (scores, indices)
        ├─ moe_expert_token_remap → (mapping, sparsity)
        ├─ all_to_all_dispatch → (dispatched, metadata)
        ├─ expert MLPs (einsum over [E, H, H])
        ├─ all_to_all_combine → [K, 1, B*S, H]
        └─ weighted sum by router scores → [B, S, H]

Testing strategy:
    1.  CPU golden: the whole forward/backward on CPU with the same weights
        (this is the correctness reference).
    2.  Device: per-op forward + torch.autograd.grad for each of the three
        custom ops (dispatch, combine, moe_expert_token_remap). The backward
        graphs compiled on device have NO tuple-returning custom calls — only
        the pure torch ops my autograd rules emit — so Shardy's
        "propagation doesn't support tuples" limitation in the tt-mlir
        pipeline doesn't trip. We compare each per-op gradient to the CPU
        reference produced by feeding the same grad_output through the same
        op on CPU.
    3.  Full-pipeline forward on device: xfail-marked, because once dispatch's
        tuple output is fed into downstream ops in a single compiled graph,
        Shardy refuses with "tuple<...>" errors. The fix is on the compiler
        side — make Shardy propagate through tuple-typed custom_call results.

To see the compiled IR:
    TTXLA_LOGGER_LEVEL=DEBUG pytest -xvs tests/torch/ops/test_moe_backward_distributed.py
"""

import pytest
import torch
import torch.nn as nn

# Must import before torch_xla so custom op autograds are registered.
from tt_torch import custom_ops  # noqa: F401


# NOTE: do not call torch_xla.runtime functions at decoration-time — that
# initializes the computation client before the conftest fixture can set the
# device type, and torch_xla aborts with
#   "InitializeComputationClient() can only be called once"
def _require_2_xla_devices():
    import torch_xla.runtime as xr

    n = xr.global_runtime_device_count()
    if n < 2:
        pytest.skip(f"need ≥2 XLA devices (N300 1x2), got {n}")


def _make_expert_mapping(E, D):
    m = torch.zeros(1, 1, E, D, dtype=torch.int64)
    for e in range(E):
        m[0, 0, e, e % D] = 1
    return m


# ---------------------------------------------------------------------------
# A small MoE module (router + dispatch + experts + combine) used for the
# CPU golden forward/backward. The distributed tests stage the same ops but
# without running the full forward on device (see file docstring).
# ---------------------------------------------------------------------------


class TinyMoE(nn.Module):
    def __init__(self, H, E, K, D):
        super().__init__()
        self.H, self.E, self.K, self.D = H, E, K, D
        self.router = nn.Linear(H, E, bias=False)
        self.expert_weights = nn.Parameter(torch.randn(E, H, H) * 0.02)
        self.register_buffer("expert_mapping", _make_expert_mapping(E, D))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        K, E, D = self.K, self.E, self.D
        router_logits = self.router(hidden_states.reshape(-1, H))
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, K, dim=-1)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-6)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            hidden_states, topk_idx, self.expert_mapping, num_devices=D, cluster_axis=0
        )
        BD = dispatched.shape[1]
        metadata_flat = metadata.reshape(1, 1, BD * S, K)

        # moe_expert_token_remap — exercise the op (its `mapping` grad doesn't
        # feed into the primary expert computation here; we use it to verify
        # the op composes in the MoE pipeline and doesn't break autograd).
        remap_mapping, _ = torch.ops.tt.moe_expert_token_remap(
            router_probs, self.expert_mapping, metadata_flat, num_devices=D
        )

        tokens_flat = dispatched.reshape(BD * S, H)
        expert_out = torch.einsum("nh,eho->eno", tokens_flat, self.expert_weights)
        expert_out = expert_out.reshape(E, 1, BD * S, H)

        combined = torch.ops.tt.all_to_all_combine(
            expert_out,
            metadata_flat,
            self.expert_mapping,
            num_devices=D,
            cluster_axis=0,
            num_experts_per_tok=K,
            output_shard_dim=2,
        )
        # combined: [K, 1, B*S, H]
        weights = topk_weights.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        # Fold remap_mapping in with a trivial addition so its backward path is
        # also on the hook for router gradients — acts like a bias term.
        remap_bias = remap_mapping.sum(dim=-1, keepdim=True)  # [1, 1, BD*S, 1]
        # Only take the first B*S tokens (the "local" device's view).
        combined = combined + remap_bias[..., : B * S, :]
        out = (combined * weights).sum(dim=0)  # [1, B*S, H]
        return out.reshape(B, S, H)


def _forward_backward_on_cpu(B, S, H, E, K, D, seed=0):
    torch.manual_seed(seed)
    model = TinyMoE(H, E, K, D)
    x = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    return (
        out.detach().clone(),
        {
            "x": x.grad.detach().clone(),
            "expert_weights": model.expert_weights.grad.detach().clone(),
            "router": model.router.weight.grad.detach().clone(),
        },
        model,
    )


# ---------------------------------------------------------------------------
# 1. CPU E2E — the golden reference.
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_tiny_moe_forward_backward_cpu():
    B, S, H, E, K, D = 2, 8, 16, 4, 2, 2
    out, grads, model = _forward_backward_on_cpu(B, S, H, E, K, D)
    assert out.shape == (B, S, H)
    for name, g in grads.items():
        assert g.abs().sum().item() > 0, f"CPU grad for {name} is all zero — pipeline broken"


# ---------------------------------------------------------------------------
# 2. Per-op backward on the 2-chip host.
#
# These tests work around the Shardy tuple limitation by running forward and
# backward separately: autograd.grad with an external grad_output does NOT
# require compiling a graph that chains dispatch's tuple output into
# downstream ops, so the tt-mlir stablehlo pipeline accepts it.
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_moe_expert_token_remap_backward_distributed():
    _require_2_xla_devices()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    BS, E, K, D = 16, 8, 2, 2
    torch.manual_seed(0)
    topk_cpu = torch.randn(BS, E, dtype=torch.float32, requires_grad=True)
    mp_cpu = _make_expert_mapping(E, D)
    # metadata for combine/remap is [1, 1, tokens=BS*D, K]
    meta_cpu = torch.randint(0, E, (1, 1, BS * D, K), dtype=torch.int64)

    # CPU reference backward.
    mapping_cpu, _ = torch.ops.tt.moe_expert_token_remap(
        topk_cpu, mp_cpu, meta_cpu, num_devices=D, reduction_size=32
    )
    grad_mapping = torch.randn_like(mapping_cpu)
    (g_cpu,) = torch.autograd.grad(mapping_cpu, topk_cpu, grad_mapping)

    # XLA: same forward setup, same grad_mapping. Backward runs on 2-device host.
    topk_xla = topk_cpu.detach().clone().to(dev).requires_grad_(True)
    mapping_xla, _ = torch.ops.tt.moe_expert_token_remap(
        topk_xla, mp_cpu.to(dev), meta_cpu.to(dev), num_devices=D, reduction_size=32
    )
    grad_mapping_xla = grad_mapping.to(dev)
    (g_xla,) = torch.autograd.grad(mapping_xla, topk_xla, grad_mapping_xla)

    diff = (g_cpu - g_xla.cpu()).abs().max().item()
    assert diff < 5e-2, f"moe_expert_token_remap backward mismatch on 2-device: diff={diff}"


@pytest.mark.single_device
def test_dispatch_backward_distributed_2device():
    _require_2_xla_devices()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    B, S, H, K, E, D = 2, 8, 16, 2, 4, 2

    torch.manual_seed(0)
    x_cpu = torch.randn(B, S, H, dtype=torch.float32, requires_grad=True)
    idx_cpu = torch.randint(0, E, (B, S, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    dispatched_cpu, _ = torch.ops.tt.all_to_all_dispatch(
        x_cpu, idx_cpu, mp_cpu, num_devices=D, cluster_axis=0
    )
    grad_out = torch.randn_like(dispatched_cpu)
    (g_cpu,) = torch.autograd.grad(dispatched_cpu, x_cpu, grad_out)

    x_xla = x_cpu.detach().clone().to(dev).requires_grad_(True)
    dispatched_xla, _ = torch.ops.tt.all_to_all_dispatch(
        x_xla, idx_cpu.to(dev), mp_cpu.to(dev), num_devices=D, cluster_axis=0
    )
    (g_xla,) = torch.autograd.grad(dispatched_xla, x_xla, grad_out.to(dev))

    diff = (g_cpu - g_xla.cpu()).abs().max().item()
    assert diff < 5e-2, f"dispatch backward mismatch on 2-device: diff={diff}"


@pytest.mark.single_device
def test_combine_backward_distributed_2device():
    _require_2_xla_devices()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    B, S, H, K, E, D = 2, 8, 16, 2, 8, 2
    BD = B * D

    torch.manual_seed(0)
    expert_out_cpu = torch.randn(E, BD, S, H, dtype=torch.float32, requires_grad=True)
    meta_cpu = torch.randint(0, E, (1, 1, BD * S, K), dtype=torch.int64)
    mp_cpu = _make_expert_mapping(E, D)

    combined_cpu = torch.ops.tt.all_to_all_combine(
        expert_out_cpu,
        meta_cpu,
        mp_cpu,
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=2,
    )
    grad_out = torch.randn_like(combined_cpu)
    (g_cpu,) = torch.autograd.grad(combined_cpu, expert_out_cpu, grad_out)

    expert_out_xla = expert_out_cpu.detach().clone().to(dev).requires_grad_(True)
    combined_xla = torch.ops.tt.all_to_all_combine(
        expert_out_xla,
        meta_cpu.to(dev),
        mp_cpu.to(dev),
        num_devices=D,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=2,
    )
    (g_xla,) = torch.autograd.grad(combined_xla, expert_out_xla, grad_out.to(dev))

    diff = (g_cpu - g_xla.cpu()).abs().max().item()
    assert diff < 5e-2, f"combine backward mismatch on 2-device: diff={diff}"


# ---------------------------------------------------------------------------
# 3. Full pipeline forward on device — currently blocked by a compiler issue.
# ---------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Compiler limit: Shardy's propagation pass (run by tt-mlir's stablehlo "
        "pipeline on multi-device compiled graphs) does not handle "
        "tuple-returning custom_call results. all_to_all_dispatch returns "
        "(dispatched, metadata); any graph that chains it into downstream ops "
        "fails at compile with 'Shardy propagation doesn't support tuples'. "
        "Fix is on the compiler side (teach Shardy tuple propagation)."
    ),
)
def test_tiny_moe_full_pipeline_xla_2device():
    _require_2_xla_devices()
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    B, S, H, E, K, D = 2, 8, 16, 4, 2, 2
    torch.manual_seed(0)
    model_xla = TinyMoE(H, E, K, D).to(dev)
    x_xla = torch.randn(B, S, H, dtype=torch.float32, device=dev, requires_grad=True)

    out = model_xla(x_xla)
    out.sum().backward()
    _ = x_xla.grad.cpu()  # force compile+fetch
