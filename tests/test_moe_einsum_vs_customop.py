"""
Test full MoE pipeline: sparse_matmul custom_op vs raw einsum.

Compares a full single-layer MoE pipeline using:
A) torch.ops.tt.sparse_matmul (custom_op with einsum inside)
B) Raw einsum (no custom_op, same math, no sparsity mask)

This isolates whether the issue is in the custom_op boundary or elsewhere.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_moe_einsum_vs_customop.py 2>&1 | tee out_moe_einsum.txt
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


def pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0 or a.numel() != b.numel():
        return float('nan')
    a_m = a - a.mean()
    b_m = b - b.mean()
    num = (a_m * b_m).sum()
    den = (a_m.norm() * b_m.norm()).clamp(min=1e-12)
    return (num / den).item()


def compare(name, cpu_t, tt_t):
    P(f"  {name}:")
    P(f"    shapes: cpu={list(cpu_t.shape)}  tt={list(tt_t.shape)}")
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()
    if cpu_f.shape != tt_f.shape:
        P(f"    SHAPE MISMATCH! cpu={cpu_f.numel()} vs tt={tt_f.numel()}")
        P(f"    cpu_norm={cpu_t.norm():.4f}  tt_norm={tt_t.norm():.4f}")
        return
    if cpu_f.numel() > 1:
        diff = (cpu_f - tt_f).abs()
        p = pcc(cpu_t, tt_t)
        P(f"    PCC={p:.6f}  atol={diff.max():.6f}  mae={diff.mean():.6f}  "
          f"cpu_norm={cpu_t.norm():.4f}  tt_norm={tt_t.norm():.4f}  "
          f"ratio={tt_t.norm()/(cpu_t.norm()+1e-12):.4f}")
    else:
        P(f"    cpu_val={cpu_f.item():.6f}  tt_val={tt_f.item():.6f}")


class MoEWithCustomOp(nn.Module):
    """Full MoE using torch.ops.tt.sparse_matmul (custom_op)."""
    def __init__(self, H, E, K, inter, num_devices, dispatch_devices, mesh_shape):
        super().__init__()
        self.E = E
        self.K = K
        self.intermediate_size = inter
        self.dispatch_devices = dispatch_devices
        self.cluster_axis = 0
        self.alpha = 1.702
        self.limit = 7.0
        self.M = 32

        self.router_weight = nn.Parameter(torch.randn(E, H))
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.gate_up_proj = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(1, E, inter, H) * 0.02)
        self.down_bias = nn.Parameter(torch.zeros(E, H))

        mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        K, E, inter = self.K, self.E, self.intermediate_size
        M = self.M

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

        # Build sparsity from metadata
        if split_seq:
            metadata_full = metadata[0].view(BD, S // M, M, K)
            metadata_flat = metadata_full.reshape(BD, (S // M) * M, K)
            sparsity_flat = torch.zeros(
                BD, (S // M) * M, 1, E,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sparsity_flat.scatter_(
                dim=-1, index=metadata_flat.unsqueeze(2),
                src=torch.ones_like(metadata_flat.unsqueeze(2), dtype=hidden_states.dtype),
            )
            sparsity = sparsity_flat.view(BD, S // M, M, E).sum(dim=2).clamp(max=1.0)
            sparsity = sparsity.unsqueeze(2)
        else:
            metadata_full = metadata[0].view(BD // M, M, S, K)
            metadata_flat = metadata_full.reshape((BD // M) * M, S, K)
            sparsity_flat = torch.zeros(
                (BD // M) * M, S, 1, E,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sparsity_flat.scatter_(
                dim=-1, index=metadata_flat.unsqueeze(2),
                src=torch.ones_like(metadata_flat.unsqueeze(2), dtype=hidden_states.dtype),
            )
            sparsity = sparsity_flat.view(BD // M, M, S, E).sum(dim=1).clamp(max=1.0)
            sparsity = sparsity.unsqueeze(2)

        # Gate+Up via sparse_matmul custom_op
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d, self.gate_up_proj, sparsity,
            nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
        )
        gate_up_out = gate_up_out.squeeze(2)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)
        gate_up_out = gate_up_out + self.gate_up_bias

        gate_out = gate_up_out[..., :inter]
        up_out = gate_up_out[..., inter:]
        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # Down via sparse_matmul custom_op
        activated_reshaped = (
            activated.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(dim_a * dim_b, E, M, inter)
        )
        sparsity_down = sparsity.view(1, 1, dim_a * dim_b, E)
        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped, self.down_proj, sparsity_down,
            nnz=0, is_input_a_sparse=True, is_input_b_sparse=False,
        )

        down_out = down_out.view(dim_a, dim_b, E, M, H)
        down_out = down_out.permute(0, 1, 3, 2, 4)
        down_out = down_out + self.down_bias
        if split_seq:
            down_out = down_out.permute(3, 0, 1, 2, 4).contiguous()
            down_out = down_out.view(E, BD, S, H)
        else:
            down_out = down_out.permute(3, 0, 2, 1, 4).contiguous()
            down_out = down_out.view(E, BD, S, H)

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


class MoEWithRawEinsum(nn.Module):
    """Full MoE using raw einsum (no custom_op, no sparsity mask).

    Same architecture as test_dispatch_backward_only.py DispatchEinsumCombine
    which achieved PCC=0.999. Uses 3D weights [E, K, N] instead of 4D [1, E, K, N].
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
        self.M = 32

        self.router_weight = nn.Parameter(torch.randn(E, H))
        self.router_bias = nn.Parameter(torch.zeros(E))
        # 3D weights (no leading 1 dim) for raw einsum
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_bias = nn.Parameter(torch.zeros(E, H))

        mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        K, E, inter = self.K, self.E, self.intermediate_size
        M = self.M

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

        # Gate+Up via raw einsum (no sparsity mask, all experts computed)
        gate_up_out = torch.einsum('abmh,ehn->abemn', hidden_4d, self.gate_up_proj)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, N]
        gate_up_out = gate_up_out + self.gate_up_bias

        gate_out = gate_up_out[..., :inter]
        up_out = gate_up_out[..., inter:]
        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # Down via raw einsum
        activated_r = activated.permute(0, 1, 3, 2, 4).contiguous()
        activated_r = activated_r.view(dim_a * dim_b, E, M, inter)
        down_out = torch.einsum('aemk,ekn->aemn', activated_r, self.down_proj)

        down_out = down_out.view(dim_a, dim_b, E, M, H)
        down_out = down_out.permute(0, 1, 3, 2, 4)
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

    compare("forward_output", cpu_fwd, tt_fwd)
    compare("grad_input", cpu_grad, tt_grad)

    P(f"  Weight gradients:")
    for name in sorted(cpu_wgrads.keys()):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            cpu_g = cpu_wgrads[name]
            compare(f"  grad_{name}", cpu_g, tt_g)

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

    # Test A: Full MoE with sparse_matmul custom_op
    P("\n" + "=" * 60)
    P("TEST A: Full MoE with sparse_matmul custom_op")
    P("=" * 60)
    torch.manual_seed(42)
    modelA = MoEWithCustomOp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    with torch.no_grad():
        modelA.router_bias.fill_(-100.0)
        for k in range(K):
            modelA.router_bias[k] = 100.0 + k * 10.0

    def shardA(m):
        return {
            m.router_weight: (None, "batch"),
            m.gate_up_proj: (None, ("model", "batch"), None, None),
            m.gate_up_bias: (("model", "batch"), None),
            m.down_proj: (None, ("model", "batch"), None, None),
            m.down_bias: (("model", "batch"), None),
        }
    fwdA, bwdA = run_test("customop_moe", modelA, input_tensor, mesh, shardA)

    # Test B: Full MoE with raw einsum (no custom_op, no sparsity)
    P("\n" + "=" * 60)
    P("TEST B: Full MoE with raw einsum (no custom_op)")
    P("=" * 60)
    torch.manual_seed(42)
    modelB = MoEWithRawEinsum(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    with torch.no_grad():
        modelB.router_bias.fill_(-100.0)
        for k in range(K):
            modelB.router_bias[k] = 100.0 + k * 10.0

    def shardB(m):
        return {
            m.router_weight: (None, "batch"),
            m.gate_up_proj: (("model", "batch"), None, None),
            m.gate_up_bias: (("model", "batch"), None),
            m.down_proj: (("model", "batch"), None, None),
            m.down_bias: (("model", "batch"), None),
        }
    fwdB, bwdB = run_test("raw_einsum_moe", modelB, input_tensor, mesh, shardB)

    P(f"\n{'=' * 60}")
    P("SUMMARY:")
    P(f"  Test A (custom_op MoE):  Fwd={fwdA:.6f}  Bwd={bwdA:.6f}")
    P(f"  Test B (raw einsum MoE): Fwd={fwdB:.6f}  Bwd={bwdB:.6f}")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
