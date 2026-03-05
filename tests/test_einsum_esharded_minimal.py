"""
Minimal test: single einsum with E-sharded weight, backward through E-reduction.

Tests increasingly complex cases to find where E-sharding breaks backward:
1. Single einsum, sum over E in model → no E in output
2. Single einsum, keep E in output, use E-sharded target
3. Full gate_up + activation + down chain

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_einsum_esharded_minimal.py 2>&1 | tee out_esharded.txt
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


def P(*a, **kw):
    print(*a, **kw, flush=True)


def pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() < 2 or a.numel() != b.numel():
        return float('nan')
    a_m = a - a.mean()
    b_m = b - b.mean()
    return ((a_m * b_m).sum() / (a_m.norm() * b_m.norm()).clamp(min=1e-12)).item()


def compare(name, cpu_t, tt_t):
    P(f"  {name}:")
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()
    if cpu_f.shape != tt_f.shape:
        P(f"    SHAPE MISMATCH: {list(cpu_t.shape)} vs {list(tt_t.shape)}")
        return float('nan')
    if cpu_f.numel() > 1:
        p = pcc(cpu_t, tt_t)
        diff = (cpu_f - tt_f).abs()
        P(f"    PCC={p:.6f}  atol={diff.max():.4f}  "
          f"cpu_norm={cpu_t.float().norm():.4f}  tt_norm={tt_t.float().norm():.4f}  "
          f"ratio={tt_t.float().norm()/(cpu_t.float().norm()+1e-12):.4f}")
        return p
    return float('nan')


class Test_EinsumSumE(nn.Module):
    """Single einsum with E-sharded weight, output summed over E."""
    def __init__(self, E, H, N):
        super().__init__()
        self.w = nn.Parameter(torch.randn(E, H, N) * 0.02)

    def forward(self, x):
        # x: [A, M, H], w: [E, H, N]
        out = torch.einsum('amh,ehn->aemn', x, self.w)  # [A, E, M, N]
        return out.sum(dim=1)  # [A, M, N] — E summed away


class Test_EinsumKeepE(nn.Module):
    """Single einsum with E-sharded weight, E kept in output."""
    def __init__(self, E, H, N):
        super().__init__()
        self.w = nn.Parameter(torch.randn(E, H, N) * 0.02)

    def forward(self, x):
        # x: [A, M, H], w: [E, H, N]
        return torch.einsum('amh,ehn->aemn', x, self.w)  # [A, E, M, N]


class Test_TwoEinsum(nn.Module):
    """Two einsums (like gate_up then down), E kept."""
    def __init__(self, E, H, N):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(E, H, N) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, N, H) * 0.02)

    def forward(self, x):
        # x: [A, M, H]
        mid = torch.einsum('amh,ehn->aemn', x, self.w1)  # [A, E, M, N]
        out = torch.einsum('aemn,enh->aemh', mid, self.w2)  # [A, E, M, H]
        return out.sum(dim=1)  # [A, M, H]


def run_test(label, model, x, mesh, E, shard=True):
    P(f"\n{'='*60}")
    P(f"TEST: {label}  (shard={'YES' if shard else 'NO'})")
    P(f"{'='*60}")

    model = model.to(torch.float32)
    model.train()

    # CPU
    cpu_model = torch.compile(model, backend="inductor")
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = cpu_model(cpu_x)
    cpu_loss = cpu_out.sum()
    cpu_loss.backward()

    cpu_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    cpu_grad_x = cpu_x.grad.detach().clone()
    model.zero_grad()

    P(f"  CPU loss: {cpu_loss.item():.4f}")

    # TT
    tt_model = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    if shard:
        for name, p in model.named_parameters():
            if p.shape[0] == E:
                spec = [None] * len(p.shape)
                spec[0] = ("model", "batch")
                xs.mark_sharding(p, mesh, tuple(spec))

    tt_x = x.clone().to(device).requires_grad_(True)
    tt_out = tt_model(tt_x)
    tt_loss = tt_out.sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    P(f"  TT loss: {tt_loss.detach().to('cpu').item():.4f}")
    tt_grad_x = tt_x.grad.detach().to("cpu")

    compare("grad_x", cpu_grad_x, tt_grad_x)
    min_pcc = 1.0
    for name in sorted(cpu_grads):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            p_val = compare(f"grad_{name}", cpu_grads[name], tt_g)
            if not (p_val != p_val):
                min_pcc = min(min_pcc, p_val)

    P(f"  MIN PCC: {min_pcc:.6f}")
    return min_pcc


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

    E, H, N = 32, 128, 256
    A, M = 4, 32
    torch.manual_seed(42)
    x = torch.randn(A, M, H, dtype=torch.float32)

    # Test 1: Single einsum, sum over E
    torch.manual_seed(42)
    run_test("Single einsum, sum over E, SHARDED",
             Test_EinsumSumE(E, H, N), x, mesh, E, shard=True)

    # Test 2: Single einsum, keep E (output has E dim, loss = .sum())
    torch.manual_seed(42)
    run_test("Single einsum, keep E, SHARDED",
             Test_EinsumKeepE(E, H, N), x, mesh, E, shard=True)

    # Test 3: Two einsums, sum over E, sharded
    torch.manual_seed(42)
    run_test("Two einsums, sum over E, SHARDED",
             Test_TwoEinsum(E, H, N), x, mesh, E, shard=True)

    # Test 4: Two einsums, NO sharding (baseline)
    torch.manual_seed(42)
    run_test("Two einsums, NO sharding",
             Test_TwoEinsum(E, H, N), x, mesh, E, shard=False)

    P(f"\n{'='*60}")
    P("DONE")
    P(f"{'='*60}")


if __name__ == "__main__":
    main()
