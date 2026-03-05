"""
Thorough sparse_matmul testing + hardcoded output bypass.

Tests:
  1. CPU-only sparse_matmul verification with known inputs
  2. sparse_matmul on TT via torch.compile (compound-sharded weights)
  3. Full MLP but with sparse_matmul replaced by standard torch.matmul
  4. Full MLP but with sparse_matmul replaced by hardcoded ones output

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_ops_isolated.py 2>&1 | tee out_ops.txt
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
from tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping
from tt_torch.sharding import sharding_constraint_tensor


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


# ---- Test 1: CPU-only verification of sparse_matmul shapes/logic ----
def test_sparse_matmul_cpu():
    """Verify sparse_matmul gives correct results on CPU with known inputs."""
    P("\n=== TEST 1: sparse_matmul CPU verification ===")

    # Shapes matching what A2aSparseMLP uses for gate_up:
    # input_a: [A=BD, B=S/M, M, K=H] = [4, 1, 32, 512]
    # input_b: [1, E, K=H, N=inter*2] = [1, 32, 512, 1024]
    # sparsity: [A=4, B=1, 1, E=32]
    # mode: is_input_a_sparse=False, is_input_b_sparse=True
    A, B, M, K, E, N = 4, 1, 32, 512, 32, 1024

    torch.manual_seed(42)
    input_a = torch.randn(A, B, M, K, dtype=torch.float32)
    input_b = torch.randn(1, E, K, N, dtype=torch.float32) * 0.02

    # Sparsity: experts 0-3 active
    sparsity = torch.zeros(A, B, 1, E, dtype=torch.float32)
    for k in range(4):
        sparsity[:, :, 0, k] = 1.0

    P(f"  input_a: {input_a.shape}")
    P(f"  input_b: {input_b.shape}")
    P(f"  sparsity: {sparsity.shape}, active experts: {(sparsity[0,0,0,:] > 0).sum().item()}")

    # Call sparse_matmul on CPU
    out = torch.ops.tt.sparse_matmul(
        input_a, input_b, sparsity,
        nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
    )
    P(f"  output: {out.shape}, norm={out.float().norm():.4f}")

    # Manual reference: for each active expert e, out[:,:,0,e,:,:] = input_a @ input_b[0,e]
    ref = torch.zeros_like(out)
    for e in range(E):
        if sparsity[0, 0, 0, e] > 0:
            ref[:, :, 0, e] = torch.matmul(input_a, input_b[0, e])

    ref_pcc = pcc(out, ref)
    P(f"  sparse_matmul vs manual reference PCC = {ref_pcc:.6f}")
    P(f"  output[0,0,0,0,:4,:4]: {out[0,0,0,0,:4,:4].tolist()}")
    P(f"  ref[0,0,0,0,:4,:4]:    {ref[0,0,0,0,:4,:4].tolist()}")

    # Verify inactive experts are zero
    inactive_norm = out[:, :, 0, 4:].norm().item()
    P(f"  inactive experts (4-31) norm = {inactive_norm:.6f} (should be 0)")

    return ref_pcc


# ---- Test 2: sparse_matmul on TT with compound sharding ----
class SparseMatmulModule(nn.Module):
    """Wraps sparse_matmul in a Module for torch.compile."""
    def __init__(self, E, H, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, E, H, N) * 0.02)
        self.E = E

    def forward(self, input_a, sparsity):
        out = torch.ops.tt.sparse_matmul(
            input_a, self.weight, sparsity,
            nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
        )
        # out: [A, B, 1, E, M, N] - sum to scalar-ish for comparison
        return out.sum(dim=(2, 4, 5))  # [A, B, E]


def test_sparse_matmul_tt(mesh):
    P("\n=== TEST 2: sparse_matmul on TT (compound-sharded E) ===")

    A, B, M, H, E, N = 4, 1, 32, 512, 32, 1024

    torch.manual_seed(42)
    input_a = torch.randn(A, B, M, H, dtype=torch.float32)
    sparsity = torch.zeros(A, B, 1, E, dtype=torch.float32)
    for k in range(4):
        sparsity[:, :, 0, k] = 1.0

    model = SparseMatmulModule(E, H, N)
    model = model.to(torch.float32)

    # CPU
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_out = cpu_compiled(input_a, sparsity)
    P(f"  [cpu] shape={cpu_out.shape} norm={cpu_out.float().norm():.4f}")
    P(f"  [cpu] vals[0,0,:8]: {cpu_out[0,0,:8].tolist()}")

    # TT
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    # Compound-shard the weight on E dim (same as real model)
    xs.mark_sharding(model.weight, mesh, (None, ("model", "batch"), None, None))

    tt_input_a = input_a.to(device)
    tt_sparsity = sparsity.to(device)
    tt_out = tt_compiled(tt_input_a, tt_sparsity)
    torch_xla.sync(wait=True)

    tt_out_cpu = tt_out.to("cpu")
    P(f"  [tt]  shape={tt_out_cpu.shape} norm={tt_out_cpu.float().norm():.4f}")
    P(f"  [tt]  vals[0,0,:8]: {tt_out_cpu[0,0,:8].tolist()}")

    fwd_pcc = pcc(cpu_out.detach(), tt_out_cpu)
    P(f"  PCC = {fwd_pcc:.6f}")
    return fwd_pcc


# ---- Shared MLP infrastructure ----
class FakeRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_tok):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.top_k = num_experts_per_tok

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.weight, self.bias)
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return scores, topk_indices


class FakeExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size * 2) * 0.02
        )
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size * 2))
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.alpha = 1.702
        self.limit = 7.0


class FakeMLP(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_tok, intermediate_size):
        super().__init__()
        self.router = FakeRouter(hidden_size, num_experts, num_experts_per_tok)
        self.experts = FakeExperts(num_experts, hidden_size, intermediate_size)


# ---- Test 3: Full MLP with standard matmul replacing sparse_matmul ----
class MLPStdMatmul(nn.Module):
    """A2aSparseMLP but with torch.matmul instead of sparse_matmul."""
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        mlp = self.mlp
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = mlp.num_experts_per_tok
        E = mlp.num_experts

        router_scores, router_indices = mlp.router(hidden_states)
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = router_indices.view(batch_size, 1, seq_len, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, mlp.expert_mapping,
            num_devices=mlp.dispatch_devices, cluster_axis=mlp.cluster_axis,
        )

        BD = dispatched.shape[1]
        M = 32

        split_seq = seq_len % M == 0 and seq_len >= M
        if split_seq:
            dim_a, dim_b = BD, seq_len // M
        else:
            dim_a, dim_b = BD // M, seq_len

        if split_seq:
            hidden_4d = dispatched.view(BD, seq_len // M, M, hidden_size)
        elif dim_b == 1:
            hidden_4d = dispatched.view(BD // M, 1, M, hidden_size)
        else:
            hidden_4d = dispatched.view(BD // M, M, seq_len, hidden_size)
            hidden_4d = hidden_4d.permute(0, 2, 1, 3)

        # *** REPLACE sparse_matmul with standard matmul ***
        # hidden_4d: [dim_a, dim_b, M, H]
        # gate_up_proj: [E, H, inter*2] → need [1, E, H, inter*2]
        gate_up_proj = mlp.experts.gate_up_proj  # [E, H, inter*2]
        # Broadcast matmul: [dim_a, dim_b, M, H] @ [E, H, inter*2]^T → need einsum
        # Output should be [dim_a, dim_b, E, M, inter*2]
        gate_up_out = torch.einsum('abmh,ehn->abemn', hidden_4d, gate_up_proj)
        # Permute to match sparse_matmul output order: [dim_a, dim_b, E, M, N]
        # Already in that order from einsum
        gate_up_out = gate_up_out + mlp.experts.gate_up_proj_bias.unsqueeze(1)

        # Activation
        gate_out = gate_up_out[..., :mlp.intermediate_size]
        up_out = gate_up_out[..., mlp.intermediate_size:]
        gate_out = gate_out.clamp(max=mlp.limit)
        up_out = up_out.clamp(-mlp.limit, mlp.limit)
        glu = gate_out * torch.sigmoid(gate_out * mlp.alpha)
        activated = (up_out + 1) * glu

        # Down projection with standard matmul
        # activated: [dim_a, dim_b, E, M, inter]
        down_proj = mlp.experts.down_proj  # [E, inter, H]
        down_out = torch.einsum('abeml,elh->abemh', activated, down_proj)
        # down_out: [dim_a, dim_b, E, M, H]
        down_out = down_out + mlp.experts.down_proj_bias.unsqueeze(1)

        # Reshape for combine: [E, BD, S, H]
        if split_seq:
            down_out = down_out.permute(2, 0, 1, 3, 4).contiguous()
            down_out = down_out.view(E, BD, seq_len, hidden_size)
        elif dim_b == 1:
            down_out = down_out.squeeze(1)
            down_out = down_out.permute(1, 0, 2, 3).contiguous()
            down_out = down_out.view(E, BD, hidden_size).unsqueeze(1)
        else:
            down_out = down_out.permute(2, 0, 3, 1, 4).contiguous()
            down_out = down_out.view(E, BD, seq_len, hidden_size)

        # Combine
        decode_mode = dim_b == 1 and not split_seq
        combined = torch.ops.tt.all_to_all_combine(
            down_out, metadata, mlp.expert_mapping,
            num_devices=mlp.dispatch_devices, cluster_axis=mlp.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=2 if decode_mode else 1,
            expert_indices=expert_indices,
        )

        # Weighted sum
        topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)
        topk_weights = topk_weights.view(batch_size, seq_len, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)

        return output


# ---- Test 4: Full MLP with hardcoded ones from sparse_matmul ----
class MLPHardcodedOnes(nn.Module):
    """A2aSparseMLP but sparse_matmul returns ones (known non-zero value)."""
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        mlp = self.mlp
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = mlp.num_experts_per_tok
        E = mlp.num_experts

        router_scores, router_indices = mlp.router(hidden_states)
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = router_indices.view(batch_size, 1, seq_len, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, mlp.expert_mapping,
            num_devices=mlp.dispatch_devices, cluster_axis=mlp.cluster_axis,
        )

        BD = dispatched.shape[1]

        # Skip ALL expert computation. Just produce hardcoded down_out.
        # down_out shape for combine: [E, BD, S, H]
        down_out = torch.ones(E, BD, seq_len, hidden_size,
                              dtype=hidden_states.dtype, device=hidden_states.device)

        # Combine
        combined = torch.ops.tt.all_to_all_combine(
            down_out, metadata, mlp.expert_mapping,
            num_devices=mlp.dispatch_devices, cluster_axis=mlp.cluster_axis,
            num_experts_per_tok=K, output_shard_dim=1,
            expert_indices=expert_indices,
        )

        # Weighted sum
        topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)
        topk_weights = topk_weights.view(batch_size, seq_len, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)

        return output


def run_test(label, model, input_tensor, mesh, shard_specs_fn):
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    cpu_compiled = torch.compile(model, backend="inductor")
    with torch.set_grad_enabled(True):
        cpu_out = cpu_compiled(input_tensor)
    P(f"  [cpu] norm={cpu_out.float().norm():.4f}")

    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    shard_specs = shard_specs_fn(model) if shard_specs_fn else {}
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    tt_input = input_tensor.detach().clone().to(device)
    with torch.set_grad_enabled(True):
        tt_out = tt_compiled(tt_input)
    torch_xla.sync(wait=True)

    tt_out_cpu = tt_out.detach().to("cpu")
    fwd_pcc = pcc(cpu_out.detach(), tt_out_cpu)
    P(f"  [tt]  norm={tt_out_cpu.float().norm():.4f}")
    P(f"  PCC = {fwd_pcc:.6f}")
    return fwd_pcc


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
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    # Test 1: CPU-only verification
    pcc1 = test_sparse_matmul_cpu()

    # Test 2: sparse_matmul on TT
    pcc2 = test_sparse_matmul_tt(mesh)

    # Setup for tests 3, 4
    B = 1; S = 32; H = 512; E = 32; K = 4; inter = 512
    num_devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[0]

    torch.manual_seed(123)
    input_tensor = torch.randn(B, S, H, dtype=torch.float32)

    def make_mlp():
        torch.manual_seed(42)
        fake_mlp = FakeMLP(H, E, K, inter)
        class FakeConfig:
            hidden_size = H
            num_local_experts = E
            num_experts_per_tok = K
        mlp = A2aSparseMLP(
            fake_mlp, num_experts=E, num_experts_per_tok=K,
            num_devices=num_devices, dispatch_devices=dispatch_devices,
            cluster_axis=0, config=FakeConfig(),
        )
        with torch.no_grad():
            mlp.router.bias.fill_(-100.0)
            for k in range(K):
                mlp.router.bias[k] = 100.0 + k * 10.0
        mlp.use_fused_remap = False
        return mlp

    def mlp_shard_specs(model):
        mlp = None
        for name, mod in model.named_modules():
            if isinstance(mod, A2aSparseMLP):
                mlp = mod
                break
        if mlp is None:
            return {}
        return {
            mlp.router.weight: (None, "batch"),
            mlp.router.bias: (None,),
            mlp.experts.gate_up_proj: (("model", "batch"), None, None),
            mlp.experts.gate_up_proj_bias: (("model", "batch"), None),
            mlp.experts.down_proj: (("model", "batch"), None, None),
            mlp.experts.down_proj_bias: (("model", "batch"), None),
        }

    # Test 3: MLP with standard matmul (no sparse_matmul)
    pcc3 = run_test("MLP with torch.matmul (no sparse_matmul)",
                     MLPStdMatmul(make_mlp()), input_tensor, mesh, mlp_shard_specs)

    # Test 4: MLP with hardcoded ones (skip all expert computation)
    pcc4 = run_test("MLP with hardcoded ones (dispatch+combine only)",
                     MLPHardcodedOnes(make_mlp()), input_tensor, mesh, mlp_shard_specs)

    P(f"\n{'='*60}")
    P("SUMMARY:")
    P(f"  1. sparse_matmul CPU:      PCC = {pcc1:.6f}")
    P(f"  2. sparse_matmul TT:       PCC = {pcc2:.6f}")
    P(f"  3. MLP + std matmul:       PCC = {pcc3:.6f}")
    P(f"  4. MLP + hardcoded ones:   PCC = {pcc4:.6f}")
    P(f"{'='*60}")


if __name__ == "__main__":
    main()
