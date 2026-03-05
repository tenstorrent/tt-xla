"""
Diagnose einsum + E-sharding issue.

Tests increasingly complex operations with E-sharded weight to find
where forward diverges from CPU.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_einsum_sharding_diag.py 2>&1 | tee out_einsum_diag.txt
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def P(*a, **kw):
    print(*a, **kw, flush=True)


def compare(name, cpu_val, tt_val):
    """Print shapes, PCC, rtol, atol, and norms."""
    P(f"  {name}:")
    P(f"    shapes: cpu={list(cpu_val.shape)}  tt={list(tt_val.shape)}")
    cpu_f = cpu_val.detach().float().flatten()
    tt_f = tt_val.detach().float().flatten()
    if cpu_f.numel() == 0:
        P(f"    empty tensors")
        return

    # If shapes don't match, note it
    if cpu_f.shape != tt_f.shape:
        P(f"    WARNING: shape mismatch! cpu numel={cpu_f.numel()} vs tt numel={tt_f.numel()}")

    # For mismatched shapes, compare norms only
    cpu_norm = cpu_val.norm().item()
    tt_norm = tt_val.norm().item()
    P(f"    cpu_norm={cpu_norm:.6f}  tt_norm={tt_norm:.6f}  ratio={tt_norm/(cpu_norm+1e-12):.4f}")

    # If shapes match, compute PCC and errors
    if cpu_f.shape == tt_f.shape and cpu_f.numel() > 1:
        diff = (cpu_f - tt_f).abs()
        atol = diff.max().item()
        mae = diff.mean().item()
        a_m = cpu_f - cpu_f.mean()
        b_m = tt_f - tt_f.mean()
        num = (a_m * b_m).sum()
        den = (a_m.norm() * b_m.norm()).clamp(min=1e-12)
        p = (num / den).item()
        P(f"    PCC={p:.6f}  atol={atol:.6f}  mae={mae:.6f}")

    # Print actual values if small
    if cpu_f.numel() <= 4:
        P(f"    cpu_vals={cpu_f.tolist()}")
    if tt_f.numel() <= 4:
        P(f"    tt_vals={tt_f.tolist()}")


class Test_WeightSum(nn.Module):
    """Sum the E-sharded weight → scalar."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        return self.w.sum()


class Test_EinsumSlice(nn.Module):
    """Full einsum, return expert-0 slice [N] (no E reduction)."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]
        out = torch.einsum('abmk,ekn->abemn', x, ib)
        return out[0, 0, 0, 0]  # [N] from expert 0


class Test_EinsumSum(nn.Module):
    """Full einsum + global sum → scalar."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]
        out = torch.einsum('abmk,ekn->abemn', x, ib)
        return out.sum()


class Test_EinsumPerExpertSum(nn.Module):
    """Einsum then sum per expert → [E]."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]
        out = torch.einsum('abmk,ekn->abemn', x, ib)
        return out.sum(dim=(0, 1, 3, 4))  # [E]


class Test_EinsumSumNonE(nn.Module):
    """Einsum then sum only over NON-E dims → [E, 1].
    Tests if reducing non-E dims works correctly."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]
        out = torch.einsum('abmk,ekn->abemn', x, ib)
        # Sum only A, B, M, N (keep E) but return as [E]
        # Equivalent to sum(dim=(0,1,3,4)) but let me try a different way
        return out.sum(dim=0).sum(dim=0).sum(dim=-1).sum(dim=-1)  # [E]


class Test_EinsumContractE(nn.Module):
    """Use einsum to CONTRACT E (instead of explicit sum).
    This tests if einsum contraction over E works where explicit sum doesn't."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]  # [E, H, N]
        # Contract E in the einsum itself → no separate E reduction needed
        out = torch.einsum('abmk,ekn->abmn', x, ib)  # E contracted
        return out.sum()


class Test_SingleAxisShard(nn.Module):
    """Same einsum+sum but with E sharded on SINGLE axis (not compound).
    Tests if compound sharding is the issue."""
    def __init__(self, H, E, inter):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, E, H, inter * 2) * 0.02)

    def forward(self, x):
        ib = self.w[0]
        out = torch.einsum('abmk,ekn->abemn', x, ib)
        return out.sum()


def run_one(label, model, x_cpu, mesh, shard_fn):
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.eval()

    # CPU
    cpu_x = x_cpu.clone()
    with torch.no_grad():
        cpu_out = model(cpu_x)
    cpu_out_val = cpu_out.detach()

    # TT
    tt_model = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    if shard_fn:
        for t, spec in shard_fn(model).items():
            xs.mark_sharding(t, mesh, spec)

    tt_x = x_cpu.clone().to(device)
    with torch.no_grad():
        tt_out = tt_model(tt_x)
    torch_xla.sync(wait=True)
    tt_out_val = tt_out.detach().to("cpu")

    compare(label, cpu_out_val, tt_out_val)
    return cpu_out_val, tt_out_val


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
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    H, E, inter = 512, 32, 512
    torch.manual_seed(42)
    x = torch.randn(4, 1, 32, H)

    def shard_compound(m):
        return {m.w: (None, ("model", "batch"), None, None)}

    def shard_single(m):
        # Shard E on "model" axis only (8 devices), E_local=4
        return {m.w: (None, "model", None, None)}

    # Test A: sum of E-sharded weight → scalar
    P("\n" + "=" * 60)
    P("TEST A: Sum of E-compound-sharded weight → scalar")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("weight_sum", Test_WeightSum(H, E, inter), x, mesh, shard_compound)

    # Test B: einsum, return slice (no E reduction)
    P("\n" + "=" * 60)
    P("TEST B: Einsum, return expert-0 slice (no E reduction)")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("einsum_slice", Test_EinsumSlice(H, E, inter), x, mesh, shard_compound)

    # Test C: einsum + global sum → scalar
    P("\n" + "=" * 60)
    P("TEST C: Einsum + global sum → scalar (broken case)")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("einsum_sum", Test_EinsumSum(H, E, inter), x, mesh, shard_compound)

    # Test D: einsum + per-expert sum → [E]
    P("\n" + "=" * 60)
    P("TEST D: Einsum + per-expert sum → [E]")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("per_expert_sum", Test_EinsumPerExpertSum(H, E, inter), x, mesh, shard_compound)

    # Test E: einsum contract E in the einsum itself → scalar
    P("\n" + "=" * 60)
    P("TEST E: Einsum with E contracted (no free E dim) → scalar")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("einsum_contract_E", Test_EinsumContractE(H, E, inter), x, mesh, shard_compound)

    # Test F: Same as Test C but with SINGLE-axis shard on E
    P("\n" + "=" * 60)
    P("TEST F: Einsum + sum, E sharded on SINGLE axis ('model' only)")
    P("=" * 60)
    torch.manual_seed(42)
    run_one("single_axis_sum", Test_SingleAxisShard(H, E, inter), x, mesh, shard_single)

    P(f"\n{'=' * 60}")
    P("DONE")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
