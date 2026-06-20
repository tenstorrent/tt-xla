# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device numerical check for the expert-parallel (EP) MoE path.

``tt_experts_forward`` routes tokens through their top-k experts via
``all_to_all_dispatch`` -> ``sparse_matmul`` -> ``all_to_all_combine`` when a
multi-chip SPMD mesh with an EP axis > 1 is set. This path is what
DeepSeek-V2-Lite's 26 MoE layers hit during the ``-k mla`` generation test on a
1D ``(1, N)`` mesh (``use_2d_mesh=False``).

The MLA attention backend is already validated numerically by
``test_mla_attention_impl.py``; this test isolates the *other* big e2e-only
component so an incoherent end-to-end generation can be attributed (or not) to
the EP MoE path. We compare, at DeepSeek-V2-Lite-like dims:

* device:   ``tt_experts_forward`` on the ``(1, N)`` SPMD mesh (the EP kernels),
* golden:   ``tt_experts_forward`` on CPU (HF ``batched_mm`` reference), and
* ref:      an independent explicit top-k expert loop.

A low PCC means the EP dispatch / expert-to-device mapping / combine is wrong
(e.g. tokens routed through the wrong experts), not the MLA attention.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import set_global_mesh

from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

REQUIRED_PCC = 0.99


class DummyFusedExperts(nn.Module):
    """Minimal fused-gate-up experts module compatible with the tt_moe backend.

    Mirrors vLLM's stacked-expert layout consumed by ``TTFusedMoE`` via its
    ``gate_up_proj`` / ``down_proj`` adapter properties:
        gate_up_proj : [E, 2*I, H]   (hidden -> gate||up)
        down_proj    : [E, H, I]     (intermediate -> hidden)
    ``is_transposed=False`` matches vLLM's row-major [E, out, in] orientation.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.is_transposed = False
        # Attrs read by the HF batched_mm CPU fallback.
        self.has_gate = True
        self.has_bias = False
        scale = 1.0 / math.sqrt(hidden_dim)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * scale
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * scale
        )

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return F.silu(gate) * up


class _SharedExpertsMLP(nn.Module):
    """Plain fused gate/up + down SiLU MLP, modelling DeepSeek shared experts.

    DeepSeek-V2-Lite runs ``n_shared_experts`` always-on experts as an ordinary
    MLP alongside the routed experts; ``SharedFusedMoE.forward`` computes it via
    ``self._shared_experts(hidden_states)``.
    """

    def __init__(self, hidden_dim, shared_intermediate_dim):
        super().__init__()
        scale = 1.0 / math.sqrt(hidden_dim)
        self.gate_up = nn.Parameter(
            torch.randn(2 * shared_intermediate_dim, hidden_dim) * scale
        )
        self.down = nn.Parameter(
            torch.randn(hidden_dim, shared_intermediate_dim) * scale
        )

    def forward(self, hidden):
        gate_up = hidden @ self.gate_up.t()
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down.t()


class FakeSharedRoutedMoE(DummyFusedExperts):
    """Replicates ``TTSharedFusedMoE.forward`` (non-overlapped, no TP reduce).

    The MRO that runs in production is
    ``TTSharedFusedMoE -> SharedFusedMoE -> TTFusedMoE -> FusedMoE``: with
    ``use_overlapped=False``, ``SharedFusedMoE.forward`` computes the shared
    experts eagerly and routes the rest through ``super().forward()`` ->
    ``TTFusedMoE.forward_native``. We can't cheaply construct the real class
    (vLLM ``FusedMoE.__init__`` needs distributed + quant setup), so this mirror
    drives the *actual* ``TTFusedMoE.forward_native`` (real routing-from-logits
    + EP kernels) and the real shared/routed structure. Only the always-1-device
    TP all-reduce branch (stock vLLM, not TT code) is omitted.
    """

    def __init__(self, num_experts, hidden_dim, intermediate_dim, shared_intermediate_dim, top_k):
        super().__init__(num_experts, hidden_dim, intermediate_dim)
        # FusedMoE attrs read by TTFusedMoE.forward_native's routing.
        self.top_k = top_k
        self.global_num_experts = num_experts
        self.renormalize = False  # DeepSeek-V2-Lite: norm_topk_prob=False
        self.custom_routing_function = None
        self._shared_experts = _SharedExpertsMLP(hidden_dim, shared_intermediate_dim)

    def forward(self, hidden_states, router_logits):
        from vllm_tt.layers.fused_moe import TTFusedMoE

        shared_out = self._shared_experts(hidden_states)
        routed_out = TTFusedMoE.forward_native(self, hidden_states, router_logits)
        return shared_out, routed_out


def _reference_shared_routed(
    hidden, gate_up, down, router_logits, shared, top_k, renormalize
):
    """Independent reference for the full shared + routed MoE combination.

    Mirrors ``TTFusedMoE.forward_native`` routing (softmax scores -> top-k ->
    optional renorm) plus the shared-expert MLP, combined as ``DeepseekV2MoE``
    does (routed_scaling_factor == 1.0 for V2-Lite)."""
    scores = F.softmax(router_logits.float(), dim=-1)
    top_k_weights, top_k_index = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp(
            min=1e-9
        )
    routed = _reference(hidden, gate_up, down, top_k_index, top_k_weights)
    shared_out = shared(hidden.float()).float()
    return shared_out + routed


def _reference(hidden, gate_up, down, top_k_index, top_k_weights):
    """Independent dense top-k expert reference (nn.Linear semantics)."""
    hidden = hidden.float()
    gate_up = gate_up.float()
    down = down.float()
    w = top_k_weights.float()
    T, H = hidden.shape
    K = top_k_index.shape[1]
    out = torch.zeros(T, H, dtype=torch.float32)
    for k in range(K):
        e = top_k_index[:, k]  # [T]
        gu = torch.einsum("th,toh->to", hidden, gate_up[e])  # [T, 2I]
        gate, up = gu.chunk(2, dim=-1)
        act = F.silu(gate) * up  # [T, I]
        o = torch.einsum("ti,thi->th", act, down[e])  # [T, H]
        out += w[:, k : k + 1] * o
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().float()
    y = b.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [32])
def test_ep_moe_matches_reference(seq_len):
    """EP MoE on a 1D ``(1, N)`` mesh must match the CPU/explicit reference."""
    enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    # DeepSeek-V2-Lite: 64 routed experts, top-6. Keep hidden/intermediate small
    # for compile speed; routing/dispatch behaviour is independent of width.
    num_experts = 64
    top_k = 6
    hidden_dim = 512
    intermediate_dim = 256
    if num_experts % num_devices != 0:
        pytest.skip(f"num_experts {num_experts} not divisible by {num_devices} devices")

    mesh = get_mesh((1, num_devices), ("batch", "model"))
    set_global_mesh(mesh)

    torch.manual_seed(0)
    experts = DummyFusedExperts(num_experts, hidden_dim, intermediate_dim)
    hidden = torch.randn(seq_len, hidden_dim) / math.sqrt(hidden_dim)
    top_k_index = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(seq_len)]
    ).to(torch.int32)
    top_k_weights = torch.rand(seq_len, top_k)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    from tt_torch.moe_backend import tt_experts_forward

    # ----- Independent explicit reference -----
    ref = _reference(hidden, experts.gate_up_proj, experts.down_proj, top_k_index, top_k_weights)

    # ----- CPU golden (HF batched_mm fallback) -----
    golden = tt_experts_forward(experts, hidden, top_k_index, top_k_weights)
    golden_pcc = _pcc(golden, ref)
    assert golden_pcc >= REQUIRED_PCC, (
        f"CPU batched_mm reference disagrees with explicit reference "
        f"(PCC {golden_pcc:.5f}); test setup is wrong, not the EP path."
    )

    # ----- Device: EP path on the (1, N) mesh -----
    device = torch_xla.device()
    dev_experts = DummyFusedExperts(num_experts, hidden_dim, intermediate_dim).to(device)
    dev_experts.gate_up_proj.data.copy_(experts.gate_up_proj)
    dev_experts.down_proj.data.copy_(experts.down_proj)
    # Shard expert weights along the expert dim exactly as partition_fused_moe.
    expert_axis = tuple(mesh.axis_names)
    xs.mark_sharding(dev_experts.gate_up_proj, mesh, (expert_axis, None, None))
    xs.mark_sharding(dev_experts.down_proj, mesh, (expert_axis, None, None))

    h_dev = hidden.to(device)
    idx_dev = top_k_index.to(device)
    w_dev = top_k_weights.to(device)

    device_out = tt_experts_forward(dev_experts, h_dev, idx_dev, w_dev)
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == ref.shape == (seq_len, hidden_dim)
    pcc = _pcc(device_out, ref)
    assert pcc >= REQUIRED_PCC, f"EP MoE PCC {pcc:.5f} < {REQUIRED_PCC} (routed wrong?)"


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("seq_len", [32])
def test_tt_shared_fused_moe_matches_reference(seq_len):
    """``TTSharedFusedMoE`` path (shared + routed) on a 1D ``(1, N)`` mesh.

    Drives the real ``TTFusedMoE.forward_native`` routing-from-logits + EP
    kernels and the shared-expert MLP combine, exactly as DeepSeek-V2-Lite's
    MoE layers do (``n_shared_experts=2``, ``norm_topk_prob=False``,
    ``routed_scaling_factor=1.0``). The earlier test fed explicit top-k indices;
    this one exercises the routing computed inside ``forward_native``.
    """
    enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    num_experts = 64
    top_k = 6
    hidden_dim = 512
    intermediate_dim = 256
    shared_intermediate_dim = 256  # n_shared_experts * moe_intermediate_size (scaled)
    if num_experts % num_devices != 0:
        pytest.skip(f"num_experts {num_experts} not divisible by {num_devices} devices")

    mesh = get_mesh((1, num_devices), ("batch", "model"))
    set_global_mesh(mesh)

    torch.manual_seed(0)
    moe = FakeSharedRoutedMoE(
        num_experts, hidden_dim, intermediate_dim, shared_intermediate_dim, top_k
    )
    hidden = torch.randn(seq_len, hidden_dim) / math.sqrt(hidden_dim)
    router_logits = torch.randn(seq_len, num_experts)

    # ----- Independent reference: shared + routed -----
    ref = _reference_shared_routed(
        hidden,
        moe.gate_up_proj,
        moe.down_proj,
        router_logits,
        moe._shared_experts,
        top_k,
        moe.renormalize,
    )

    # ----- Device: TTSharedFusedMoE.forward semantics on the (1, N) mesh -----
    device = torch_xla.device()
    dev_moe = FakeSharedRoutedMoE(
        num_experts, hidden_dim, intermediate_dim, shared_intermediate_dim, top_k
    ).to(device)
    dev_moe.gate_up_proj.data.copy_(moe.gate_up_proj)
    dev_moe.down_proj.data.copy_(moe.down_proj)
    dev_moe._shared_experts.gate_up.data.copy_(moe._shared_experts.gate_up)
    dev_moe._shared_experts.down.data.copy_(moe._shared_experts.down)
    expert_axis = tuple(mesh.axis_names)
    xs.mark_sharding(dev_moe.gate_up_proj, mesh, (expert_axis, None, None))
    xs.mark_sharding(dev_moe.down_proj, mesh, (expert_axis, None, None))

    shared_dev, routed_dev = dev_moe.forward(
        hidden.to(device), router_logits.to(device)
    )
    final_dev = shared_dev + routed_dev  # routed_scaling_factor == 1.0
    torch_xla.sync()
    final_dev = final_dev.cpu()

    assert final_dev.shape == ref.shape == (seq_len, hidden_dim)
    pcc = _pcc(final_dev, ref)
    assert pcc >= REQUIRED_PCC, (
        f"TTSharedFusedMoE PCC {pcc:.5f} < {REQUIRED_PCC} "
        "(shared/routed combine or routing-from-logits wrong?)"
    )
