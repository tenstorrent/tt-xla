"""
Focused test to debug dispatch backward.

Tests dispatch+combine forward+backward with simple operations.
Reports grad PCC for dispatch backward.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_dispatch_backward_only.py 2>&1 | tee out_disp_bwd.txt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt_torch.sparse_mlp import build_expert_mapping


def P(*a, **kw):
    print(*a, **kw, flush=True)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return 1.0
    a_mean = a - a.mean()
    b_mean = b - b.mean()
    num = (a_mean * b_mean).sum()
    den = (a_mean.norm() * b_mean.norm()).clamp(min=1e-12)
    return (num / den).item()


class DispatchBiasCombine(nn.Module):
    """Dispatch → per-expert bias → combine → weighted sum.
    Tests dispatch+combine fwd+bwd with a simple differentiable operation.
    """
    def __init__(self, H, E, K, num_devices, dispatch_devices, mesh_shape):
        super().__init__()
        self.E = E
        self.K = K
        self.dispatch_devices = dispatch_devices
        self.cluster_axis = 0

        self.router_weight = nn.Parameter(torch.randn(E, H))
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.expert_bias = nn.Parameter(torch.randn(E, H) * 0.1)

        mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        K, E = self.K, self.E

        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(scores, K, dim=-1)

        x = hidden_states.view(B, 1, S, H)
        expert_indices = topk_indices.view(B, 1, S, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
        )

        BD = dispatched.shape[1]

        # Per-expert bias between dispatch and combine
        expert_out = dispatched.squeeze(0).unsqueeze(0).expand(E, -1, -1, -1)
        expert_out = expert_out + self.expert_bias.view(E, 1, 1, H)

        combined = torch.ops.tt.all_to_all_combine(
            expert_out, metadata, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
            num_experts_per_tok=K, output_shard_dim=1,
            expert_indices=expert_indices,
        )

        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        topk_weights = topk_weights.view(B, S, K).permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)
        return output


class DispatchEinsumCombine(nn.Module):
    """Dispatch → einsum expert MLP → combine → weighted sum.
    Tests the realistic expert computation path without sparse_matmul.
    """
    def __init__(self, H, E, K, inter, num_devices, dispatch_devices, mesh_shape):
        super().__init__()
        self.E = E
        self.K = K
        self.intermediate_size = inter
        self.dispatch_devices = dispatch_devices
        self.cluster_axis = 0
        self.alpha = 1.702
        self.limit = 7.0

        self.router_weight = nn.Parameter(torch.randn(E, H))
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_bias = nn.Parameter(torch.zeros(E, H))

        mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        K, E, inter = self.K, self.E, self.intermediate_size
        M = 32

        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(scores, K, dim=-1)

        x = hidden_states.view(B, 1, S, H)
        expert_indices = topk_indices.view(B, 1, S, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
        )

        BD = dispatched.shape[1]
        split_seq = S % M == 0 and S >= M
        if split_seq:
            dim_a, dim_b = BD, S // M
            hidden_4d = dispatched.view(BD, S // M, M, H)
        else:
            dim_a, dim_b = BD // M, S
            hidden_4d = dispatched.view(BD // M, M, S, H).permute(0, 2, 1, 3)

        # Gate+Up via einsum
        gate_up_out = torch.einsum('abmh,ehn->abemn', hidden_4d, self.gate_up_proj)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, N]
        gate_up_out = gate_up_out + self.gate_up_bias

        # Activation
        gate_out = gate_up_out[..., :inter]
        up_out = gate_up_out[..., inter:]
        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # Down via einsum
        activated_r = activated.permute(0, 1, 3, 2, 4).contiguous()
        activated_r = activated_r.view(dim_a * dim_b, E, M, inter)
        down_out = torch.einsum('aemk,ekn->aemn', activated_r, self.down_proj)

        # Reshape for combine: [E, BD, S, H]
        down_out = down_out.view(dim_a, dim_b, E, M, H)
        down_out = down_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, H]
        down_out = down_out + self.down_bias
        if split_seq:
            down_out = down_out.permute(3, 0, 1, 2, 4).contiguous().view(E, BD, S, H)
        else:
            down_out = down_out.permute(3, 0, 2, 1, 4).contiguous().view(E, BD, S, H)

        combined = torch.ops.tt.all_to_all_combine(
            down_out, metadata, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
            num_experts_per_tok=K, output_shard_dim=1,
            expert_indices=expert_indices,
        )

        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        topk_weights = topk_weights.view(B, S, K).permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)
        return output


def run_test(label, model, input_tensor, mesh, shard_specs_fn):
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    # CPU
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_input = input_tensor.clone().requires_grad_(True)
    cpu_out = cpu_compiled(cpu_input)
    cpu_out.sum().backward()
    cpu_fwd = cpu_out.detach()
    cpu_grad = cpu_input.grad.detach()

    cpu_wgrads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_wgrads[name] = p.grad.detach().clone()
    model.zero_grad()

    P(f"  [cpu] fwd norm={cpu_fwd.norm():.4f}  grad_input norm={cpu_grad.norm():.4f}")

    # TT
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    if shard_specs_fn:
        for t, spec in shard_specs_fn(model).items():
            xs.mark_sharding(t, mesh, spec)

    tt_input = input_tensor.clone().to(device).requires_grad_(True)
    tt_out = tt_compiled(tt_input)
    tt_out.sum().backward()
    torch_xla.sync(wait=True)

    tt_fwd = tt_out.detach().to("cpu")
    tt_grad = tt_input.grad.detach().to("cpu")

    P(f"  [tt]  fwd norm={tt_fwd.norm():.4f}  grad_input norm={tt_grad.norm():.4f}")
    P(f"  Forward PCC  = {pcc(cpu_fwd, tt_fwd):.6f}")
    P(f"  Backward PCC (grad_input) = {pcc(cpu_grad, tt_grad):.6f}")

    P(f"  Weight gradients:")
    for name in sorted(cpu_wgrads.keys()):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            cpu_g = cpu_wgrads[name]
            P(f"    {name:30s} PCC={pcc(cpu_g, tt_g):.6f}  cpu_norm={cpu_g.norm():.4f}  tt_norm={tt_g.norm():.4f}")

    return pcc(cpu_fwd, tt_fwd), pcc(cpu_grad, tt_grad)


def main():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    if n >= 32:
        mesh_shape = (4, 8)
    elif n >= 8:
        mesh_shape = (2, 4)
    else:
        raise RuntimeError(f"Need >=8 devices, got {n}")

    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))
    num_devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[0]
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    B, S, H, E, K, inter = 1, 32, 512, 32, 4, 512
    torch.manual_seed(123)
    input_tensor = torch.randn(B, S, H, dtype=torch.float32)

    # Test 1: Dispatch + bias + Combine
    P("\n" + "=" * 60)
    P("TEST 1: Dispatch + bias + Combine (fwd+bwd)")
    P("=" * 60)
    torch.manual_seed(42)
    model1 = DispatchBiasCombine(H, E, K, num_devices, dispatch_devices, mesh_shape)
    with torch.no_grad():
        model1.router_bias.fill_(-100.0)
        for k in range(K):
            model1.router_bias[k] = 100.0 + k * 10.0

    def shard1(m):
        return {
            m.router_weight: (None, "batch"),
            m.expert_bias: (("model", "batch"), None),
        }

    fwd1, bwd1 = run_test("D+Bias+C", model1, input_tensor, mesh, shard1)

    # Test 2: Dispatch + Einsum + Combine
    P("\n" + "=" * 60)
    P("TEST 2: Dispatch + Einsum MLP + Combine (fwd+bwd)")
    P("=" * 60)
    torch.manual_seed(42)
    model2 = DispatchEinsumCombine(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    with torch.no_grad():
        model2.router_bias.fill_(-100.0)
        for k in range(K):
            model2.router_bias[k] = 100.0 + k * 10.0

    def shard2(m):
        return {
            m.router_weight: (None, "batch"),
            m.gate_up_proj: (("model", "batch"), None, None),
            m.gate_up_bias: (("model", "batch"), None),
            m.down_proj: (("model", "batch"), None, None),
            m.down_bias: (("model", "batch"), None),
        }

    fwd2, bwd2 = run_test("D+Einsum+C", model2, input_tensor, mesh, shard2)

    P(f"\n{'=' * 60}")
    P("SUMMARY:")
    P(f"  Test 1 (D+bias+C):    Fwd={fwd1:.6f}  Bwd={bwd1:.6f}")
    P(f"  Test 2 (D+einsum+C):  Fwd={fwd2:.6f}  Bwd={bwd2:.6f}")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
