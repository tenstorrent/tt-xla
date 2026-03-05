"""
Test sparse_matmul vs einsum replacement on TT with compound sharding.

Goal: Determine if replacing sparse_matmul with standard einsum fixes the
all-zeros issue on TT, and whether GSPMD correctly handles the compound
sharding through einsum.

Tests:
  1. Isolated sparse_matmul (gate_up) — compound-sharded E
  2. Isolated einsum replacement (gate_up) — compound-sharded E
  3. Full A2aSparseMLP with sparse_matmul (original)
  4. Full A2aSparseMLP with einsum replacing sparse_matmul
  5. sparse_matmul with different E sharding levels (E_local=1,4,8,32)

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_sparse_matmul_vs_einsum.py 2>&1 | tee out_einsum.txt
"""

import os
import sys
import copy
import math

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


# ---- Test 1 & 2: Isolated sparse_matmul vs einsum ----

class SparseMatmulGateUp(nn.Module):
    """Gate+Up projection using sparse_matmul."""
    def __init__(self, E, H, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, E, H, N) * 0.02)

    def forward(self, hidden_4d, sparsity):
        out = torch.ops.tt.sparse_matmul(
            hidden_4d, self.weight, sparsity,
            nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
        )
        # out: [A, B, 1, E, M, N] → squeeze → [A, B, E, M, N] → sum(M,N) → [A, B, E]
        return out.squeeze(2).sum(dim=(-2, -1))


class EinsumGateUp(nn.Module):
    """Gate+Up projection using standard einsum (no sparse_matmul)."""
    def __init__(self, E, H, N):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, E, H, N) * 0.02)

    def forward(self, hidden_4d, sparsity):
        # hidden_4d: [A, B, M, H], weight[0]: [E, H, N]
        # einsum: [A, B, M, H] @ [E, H, N] -> [A, B, E, M, N]
        out = torch.einsum('abmh,ehn->abemn', hidden_4d, self.weight[0])
        # Mask by sparsity: sparsity [A, B, 1, E] -> [A, B, E, 1, 1]
        mask = sparsity.squeeze(2).unsqueeze(-1).unsqueeze(-1)
        out = out * mask
        return out.sum(dim=(-2, -1))  # [A, B, E]


def run_isolated_test(label, model, hidden_4d, sparsity, mesh, shard_spec_E):
    """Run a model with compound-sharded weight on E dim."""
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)

    # CPU reference
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_out = cpu_compiled(hidden_4d, sparsity)
    P(f"  [cpu] norm={cpu_out.float().norm():.4f}")
    P(f"  [cpu] active expert 0 sum: {cpu_out[0,0,0].item():.4f}")
    P(f"  [cpu] inactive expert 5 sum: {cpu_out[0,0,min(5, cpu_out.shape[2]-1)].item():.4f}")

    # TT
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    # Shard weight on E dim
    xs.mark_sharding(model.weight, mesh, shard_spec_E)

    tt_hidden = hidden_4d.to(device)
    tt_sparsity = sparsity.to(device)
    tt_out = tt_compiled(tt_hidden, tt_sparsity)
    torch_xla.sync(wait=True)

    tt_out_cpu = tt_out.detach().to("cpu")
    P(f"  [tt]  norm={tt_out_cpu.float().norm():.4f}")
    P(f"  [tt]  active expert 0 sum: {tt_out_cpu[0,0,0].item():.4f}")
    P(f"  [tt]  inactive expert 5 sum: {tt_out_cpu[0,0,min(5, tt_out_cpu.shape[2]-1)].item():.4f}")

    fwd_pcc = pcc(cpu_out.detach(), tt_out_cpu)
    P(f"  PCC = {fwd_pcc:.6f}")
    return fwd_pcc


# ---- Test 3 & 4: Full MLP with sparse_matmul vs einsum ----

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


class EinsumMLP(nn.Module):
    """A2aSparseMLP but with einsum instead of sparse_matmul.

    This tests whether the full dispatch/einsum/combine pipeline works on TT
    when sparse_matmul is bypassed entirely.
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        mlp = self.mlp
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = mlp.num_experts_per_tok
        E = mlp.num_experts

        # Router
        router_scores, router_indices = mlp.router(hidden_states)
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = router_indices.view(batch_size, 1, seq_len, K)

        # Dispatch
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

        # Build sparsity from metadata (manual path for clarity)
        if split_seq:
            metadata_full = metadata[0].view(BD, seq_len // M, M, K)
            metadata_flat = metadata_full.reshape(BD, (seq_len // M) * M, K)
            sparsity_flat = torch.zeros(
                BD, (seq_len // M) * M, 1, E,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sparsity_flat.scatter_(
                dim=-1,
                index=metadata_flat.unsqueeze(2),
                src=torch.ones_like(metadata_flat.unsqueeze(2), dtype=hidden_states.dtype),
            )
            sparsity = sparsity_flat.view(BD, seq_len // M, M, E).sum(dim=2).clamp(max=1.0)
            sparsity = sparsity.unsqueeze(2)  # [dim_a, dim_b, 1, E]
        else:
            metadata_full = metadata[0].view(BD // M, M, seq_len, K)
            metadata_flat = metadata_full.reshape((BD // M) * M, seq_len, K)
            sparsity_flat = torch.zeros(
                (BD // M) * M, seq_len, 1, E,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sparsity_flat.scatter_(
                dim=-1,
                index=metadata_flat.unsqueeze(2),
                src=torch.ones_like(metadata_flat.unsqueeze(2), dtype=hidden_states.dtype),
            )
            sparsity = sparsity_flat.view(BD // M, M, seq_len, E).sum(dim=1).clamp(max=1.0)
            sparsity = sparsity.unsqueeze(2)

        gate_up_proj = mlp.experts.gate_up_proj  # [E, H, inter*2] (de-interleaved in __init__)
        down_proj_w = mlp.experts.down_proj       # [E, inter, H]
        gate_up_bias = mlp.experts.gate_up_proj_bias
        down_bias = mlp.experts.down_proj_bias

        if split_seq:
            hidden_4d = dispatched.view(BD, seq_len // M, M, hidden_size)
        elif dim_b == 1:
            hidden_4d = dispatched.view(BD // M, 1, M, hidden_size)
        else:
            hidden_4d = dispatched.view(BD // M, M, seq_len, hidden_size)
            hidden_4d = hidden_4d.permute(0, 2, 1, 3)

        # *** EINSUM instead of sparse_matmul for gate_up ***
        # hidden_4d: [A, B, M, H], gate_up_proj: [E, H, N]
        # → [A, B, E, M, N]
        gate_up_out = torch.einsum('abmh,ehn->abemn', hidden_4d, gate_up_proj)
        # Mask by sparsity [A, B, 1, E] → [A, B, E, 1, 1]
        sp_mask = sparsity.squeeze(2).unsqueeze(-1).unsqueeze(-1)
        gate_up_out = gate_up_out * sp_mask

        # Permute to [A, B, M, E, N] (matches production sparse_matmul path)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)
        gate_up_out = gate_up_out + gate_up_bias

        # Activation
        gate_out = gate_up_out[..., :mlp.intermediate_size]
        up_out = gate_up_out[..., mlp.intermediate_size:]
        gate_out = gate_out.clamp(max=mlp.limit)
        up_out = up_out.clamp(-mlp.limit, mlp.limit)
        glu = gate_out * torch.sigmoid(gate_out * mlp.alpha)
        activated = (up_out + 1) * glu

        # *** EINSUM instead of sparse_matmul for down ***
        # activated: [A, B, M, E, inter] → reshape to [A*B, E, M, inter]
        activated_reshaped = (
            activated.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(dim_a * dim_b, E, M, mlp.intermediate_size)
        )
        # down_proj_w: [E, inter, H]
        # → [A*B, E, M, H]
        down_out = torch.einsum('aemk,ekn->aemn', activated_reshaped, down_proj_w)

        # Mask by sparsity_down
        sparsity_down = sparsity.view(dim_a * dim_b, 1, E).unsqueeze(-1)  # [A*B, 1, E, 1]
        down_out = down_out * sparsity_down.squeeze(1).unsqueeze(2)  # [A*B, E, 1, H] broadcast

        # Reshape for combine: [E, BD, S, H]
        down_out = down_out.view(dim_a, dim_b, E, M, hidden_size)
        down_out = down_out.permute(0, 1, 3, 2, 4)  # [dim_a, dim_b, M, E, H]
        down_out = down_out + down_bias
        if split_seq:
            down_out = down_out.permute(3, 0, 1, 2, 4).contiguous()
            down_out = down_out.view(E, BD, seq_len, hidden_size)
        elif dim_b == 1:
            down_out = down_out.squeeze(1)
            down_out = down_out.permute(2, 0, 1, 3).contiguous()
            down_out = down_out.view(E, BD, hidden_size).unsqueeze(1)
        else:
            down_out = down_out.permute(3, 0, 2, 1, 4).contiguous()
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


def run_mlp_test(label, model, input_tensor, mesh, mesh_shape):
    """Run full MLP test with proper compound sharding."""
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    # CPU reference
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_out = cpu_compiled(input_tensor)
    P(f"  [cpu] norm={cpu_out.float().norm():.4f}")

    # TT
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    # Apply shard specs after model is on device
    mlp = None
    for name, mod in model.named_modules():
        if isinstance(mod, A2aSparseMLP):
            mlp = mod
            break
    if mlp:
        xs.mark_sharding(mlp.router.weight, mesh, (None, "batch"))
        xs.mark_sharding(mlp.router.bias, mesh, (None,))
        xs.mark_sharding(mlp.experts.gate_up_proj, mesh, (("model", "batch"), None, None))
        xs.mark_sharding(mlp.experts.gate_up_proj_bias, mesh, (("model", "batch"), None))
        xs.mark_sharding(mlp.experts.down_proj, mesh, (("model", "batch"), None, None))
        xs.mark_sharding(mlp.experts.down_proj_bias, mesh, (("model", "batch"), None))

    tt_input = input_tensor.detach().clone().to(device)
    tt_out = tt_compiled(tt_input)
    torch_xla.sync(wait=True)

    tt_out_cpu = tt_out.detach().to("cpu")
    fwd_pcc = pcc(cpu_out.detach(), tt_out_cpu)
    P(f"  [tt]  norm={tt_out_cpu.float().norm():.4f}")
    P(f"  PCC = {fwd_pcc:.6f}")
    return fwd_pcc


def make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape):
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
        mesh_shape=mesh_shape,
    )
    # Force experts 0-3 to be always selected
    with torch.no_grad():
        mlp.router.bias.fill_(-100.0)
        for k in range(K):
            mlp.router.bias[k] = 100.0 + k * 10.0
    mlp.use_fused_remap = False
    return mlp


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

    # Common shapes
    A, B, M, H, E, N = 4, 1, 32, 512, 32, 1024

    torch.manual_seed(42)
    hidden_4d = torch.randn(A, B, M, H, dtype=torch.float32)
    sparsity = torch.zeros(A, B, 1, E, dtype=torch.float32)
    for k in range(4):
        sparsity[:, :, 0, k] = 1.0

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 1: Isolated sparse_matmul (compound-sharded E)")
    P("=" * 60)
    torch.manual_seed(42)
    pcc1 = run_isolated_test(
        "sparse_matmul gate_up",
        SparseMatmulGateUp(E, H, N),
        hidden_4d.clone(), sparsity.clone(), mesh,
        (None, ("model", "batch"), None, None),
    )

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 2: Isolated einsum (compound-sharded E)")
    P("=" * 60)
    torch.manual_seed(42)
    pcc2 = run_isolated_test(
        "einsum gate_up",
        EinsumGateUp(E, H, N),
        hidden_4d.clone(), sparsity.clone(), mesh,
        (None, ("model", "batch"), None, None),
    )

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 3: Full A2aSparseMLP with sparse_matmul (original)")
    P("=" * 60)
    B_size, S, H_size, K_top, inter = 1, 32, 512, 4, 512
    torch.manual_seed(123)
    input_tensor = torch.randn(B_size, S, H_size, dtype=torch.float32)

    mlp3 = make_mlp(H_size, E, K_top, inter, num_devices, dispatch_devices, mesh_shape)
    pcc3 = run_mlp_test(
        "A2aSparseMLP + sparse_matmul",
        mlp3, input_tensor, mesh, mesh_shape,
    )

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 4: Full A2aSparseMLP with einsum (replacing sparse_matmul)")
    P("=" * 60)
    mlp4 = make_mlp(H_size, E, K_top, inter, num_devices, dispatch_devices, mesh_shape)
    pcc4 = run_mlp_test(
        "EinsumMLP (dispatch+einsum+combine)",
        EinsumMLP(mlp4), input_tensor, mesh, mesh_shape,
    )

    # ==================================================================
    P(f"\n{'=' * 60}")
    P("SUMMARY:")
    P(f"  1. Isolated sparse_matmul (compound E):  PCC = {pcc1:.6f}")
    P(f"  2. Isolated einsum (compound E):          PCC = {pcc2:.6f}")
    P(f"  3. Full MLP + sparse_matmul:              PCC = {pcc3:.6f}")
    P(f"  4. Full MLP + einsum:                     PCC = {pcc4:.6f}")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
