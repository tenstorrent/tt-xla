"""
Test individual MoE ops and single-layer backward on TT.

Tests:
1. sparse_matmul via custom_op (torch.ops.tt.sparse_matmul)
2. Same computation via raw einsum (no custom_op) — to isolate custom_op issues
3. Full single-layer MoE (dispatch → sparse_matmul → combine) forward+backward

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_moe_layer_backward.py 2>&1 | tee out_moe_layer.txt
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


def compare_tensors(name: str, cpu_t: torch.Tensor, tt_t: torch.Tensor):
    """Print PCC, rtol, atol, and norms for a pair of tensors."""
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()

    p = pcc(cpu_t, tt_t)
    diff = (cpu_f - tt_f).abs()
    atol = diff.max().item()
    cpu_abs = cpu_f.abs().clamp(min=1e-12)
    rtol = (diff / cpu_abs).max().item()
    # Also compute mean absolute error for context
    mae = diff.mean().item()

    P(f"    {name:30s} PCC={p:.6f}  atol={atol:.6f}  rtol={rtol:.6f}  mae={mae:.6f}  "
      f"cpu_norm={cpu_t.norm():.4f}  tt_norm={tt_t.norm():.4f}")
    return p


class SparseMatmulOnly(nn.Module):
    """Test sparse_matmul forward+backward via custom_op."""
    def __init__(self, H, E, K, inter, num_devices, dispatch_devices, mesh_shape):
        super().__init__()
        self.E = E
        self.K = K
        self.inter = inter
        self.gate_up_proj = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, hidden_4d, sparsity):
        out = torch.ops.tt.sparse_matmul(
            hidden_4d, self.gate_up_proj, sparsity,
            nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
        )
        return out.sum()


class RawEinsumOnly(nn.Module):
    """Same computation as SparseMatmulOnly but using raw einsum (no custom_op).

    This bypasses the torch.library.custom_op boundary to test whether
    Shardy can correctly partition the einsum when it's a standard op
    in the FX graph rather than hidden inside a custom_op.
    """
    def __init__(self, H, E, K, inter, num_devices, dispatch_devices, mesh_shape):
        super().__init__()
        self.E = E
        self.K = K
        self.inter = inter
        self.gate_up_proj = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, hidden_4d, sparsity):
        # Same math as sparse_matmul for not_a_sparse + b_sparse mode:
        # input_tensor_a: [A, B, M, K]  @ input_tensor_b[0]: [E, K, N] -> [A, B, 1, E, M, N]
        ib = self.gate_up_proj[0]  # [E, H, inter*2]
        sp_mask = sparsity[:, :, 0, :]  # [A, B, E]
        out = torch.einsum('abmk,ekn->abemn', hidden_4d, ib)
        out = out * sp_mask.unsqueeze(-1).unsqueeze(-1)
        out = out.unsqueeze(2)  # [A, B, 1, E, M, N]
        return out.sum()


class SingleLayerMoE(nn.Module):
    """Full dispatch → sparse_matmul → combine single layer."""
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

        # Build sparsity from metadata (manual path)
        if split_seq:
            metadata_full = metadata[0].view(BD, S // M, M, K)
            metadata_flat = metadata_full.reshape(BD, (S // M) * M, K)
            sparsity_flat = torch.zeros(
                BD, (S // M) * M, 1, E,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sparsity_flat.scatter_(
                dim=-1,
                index=metadata_flat.unsqueeze(2),
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
                dim=-1,
                index=metadata_flat.unsqueeze(2),
                src=torch.ones_like(metadata_flat.unsqueeze(2), dtype=hidden_states.dtype),
            )
            sparsity = sparsity_flat.view(BD // M, M, S, E).sum(dim=1).clamp(max=1.0)
            sparsity = sparsity.unsqueeze(2)

        # Gate+Up
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

        # Down
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


def run_test(label, model, inputs, mesh, shard_specs_fn):
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    # CPU
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_inputs = [i.clone().requires_grad_(True) if i.is_floating_point() else i.clone() for i in inputs]
    cpu_out = cpu_compiled(*cpu_inputs)
    if cpu_out.dim() == 0:
        cpu_out.backward()
    else:
        cpu_out.sum().backward()
    cpu_fwd = cpu_out.detach()
    cpu_grad = cpu_inputs[0].grad.detach() if cpu_inputs[0].grad is not None else torch.zeros(1)

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

    tt_inputs = [i.clone().to(device).requires_grad_(True) if i.is_floating_point() else i.clone().to(device) for i in inputs]
    tt_out = tt_compiled(*tt_inputs)
    if tt_out.dim() == 0:
        tt_out.backward()
    else:
        tt_out.sum().backward()
    torch_xla.sync(wait=True)

    tt_fwd = tt_out.detach().to("cpu")
    tt_grad = tt_inputs[0].grad.detach().to("cpu") if tt_inputs[0].grad is not None else torch.zeros(1)

    P(f"  [tt]  fwd norm={tt_fwd.norm():.4f}  grad_input norm={tt_grad.norm():.4f}")
    P(f"  [shapes] cpu_fwd={list(cpu_fwd.shape)}  tt_fwd={list(tt_fwd.shape)}  "
      f"cpu_grad={list(cpu_grad.shape)}  tt_grad={list(tt_grad.shape)}")

    if list(cpu_fwd.shape) != list(tt_fwd.shape):
        P(f"  *** SHAPE MISMATCH in forward output! ***")
        P(f"  *** TT .sum() was likely eliminated by compiler ***")

    P(f"  Forward:")
    compare_tensors("fwd_output", cpu_fwd, tt_fwd)
    P(f"  Backward (grad_input):")
    compare_tensors("grad_input", cpu_grad, tt_grad)

    P(f"  Weight gradients:")
    for name in sorted(cpu_wgrads.keys()):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            cpu_g = cpu_wgrads[name]
            compare_tensors(name, cpu_g, tt_g)

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

    # Test 1: sparse_matmul via custom_op
    P("\n" + "=" * 60)
    P("TEST 1: sparse_matmul via custom_op (isolated)")
    P("=" * 60)
    torch.manual_seed(42)
    model1 = SparseMatmulOnly(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    hidden_4d = torch.randn(4, 1, 32, H)
    sparsity = torch.ones(4, 1, 1, E)
    def shard1(m):
        return {m.gate_up_proj: (None, ("model", "batch"), None, None)}
    fwd1, bwd1 = run_test("sparse_matmul_customop", model1, [hidden_4d, sparsity], mesh, shard1)

    # Test 2: Same computation via raw einsum (no custom_op)
    P("\n" + "=" * 60)
    P("TEST 2: Raw einsum (same math, no custom_op)")
    P("=" * 60)
    torch.manual_seed(42)
    model2 = RawEinsumOnly(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    hidden_4d2 = torch.randn(4, 1, 32, H)
    sparsity2 = torch.ones(4, 1, 1, E)
    def shard2(m):
        return {m.gate_up_proj: (None, ("model", "batch"), None, None)}
    fwd2, bwd2 = run_test("raw_einsum", model2, [hidden_4d2, sparsity2], mesh, shard2)

    # Test 3: Full single-layer MoE
    P("\n" + "=" * 60)
    P("TEST 3: Full single-layer MoE (dispatch+sparse_matmul+combine)")
    P("=" * 60)
    torch.manual_seed(42)
    model3 = SingleLayerMoE(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    with torch.no_grad():
        model3.router_bias.fill_(-100.0)
        for k in range(K):
            model3.router_bias[k] = 100.0 + k * 10.0
    input_tensor = torch.randn(B, S, H)
    def shard3(m):
        return {
            m.router_weight: (None, "batch"),
            m.gate_up_proj: (None, ("model", "batch"), None, None),
            m.gate_up_bias: (("model", "batch"), None),
            m.down_proj: (None, ("model", "batch"), None, None),
            m.down_bias: (("model", "batch"), None),
        }
    fwd3, bwd3 = run_test("full_moe", model3, [input_tensor], mesh, shard3)

    P(f"\n{'=' * 60}")
    P("SUMMARY:")
    P(f"  Test 1 (custom_op sparse_matmul):  Fwd={fwd1:.6f}  Bwd={bwd1:.6f}")
    P(f"  Test 2 (raw einsum):               Fwd={fwd2:.6f}  Bwd={bwd2:.6f}")
    P(f"  Test 3 (full MoE):                 Fwd={fwd3:.6f}  Bwd={bwd3:.6f}")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
