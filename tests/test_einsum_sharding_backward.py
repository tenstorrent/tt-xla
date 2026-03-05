"""
Test einsum backward with E-sharding. Single run, one config.

Change SHARD_MODE below: "none", "model", "batch", "compound"

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_einsum_sharding_backward.py 2>&1 | tee out_eshard_bwd.txt
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

# ===== CONFIG =====
SHARD_MODE = "model"  # "none", "model", "batch", "compound"
# ==================

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
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()
    if cpu_f.shape != tt_f.shape:
        P(f"  {name}: SHAPE MISMATCH {list(cpu_t.shape)} vs {list(tt_t.shape)}")
        return float('nan')
    if cpu_f.numel() > 1:
        p = pcc(cpu_t, tt_t)
        diff = (cpu_f - tt_f).abs()
        P(f"  {name}: PCC={p:.6f}  atol={diff.max():.4f}  "
          f"cpu_norm={cpu_t.float().norm():.4f}  tt_norm={tt_t.float().norm():.4f}")
        return p
    return float('nan')


class SimpleEinsum(nn.Module):
    """Single einsum: x @ w + b, keep E in output."""
    def __init__(self, E, H, N):
        super().__init__()
        self.w = nn.Parameter(torch.randn(E, H, N) * 0.02)
        self.b = nn.Parameter(torch.zeros(E, N))

    def forward(self, x):
        # x: [A, M, H], w: [E, H, N] -> out: [A, E, M, N]
        out = torch.einsum('amh,ehn->aemn', x, self.w)
        out = out + self.b.unsqueeze(0).unsqueeze(2)
        return out


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
    P(f"[setup] {n} devices, mesh={mesh_shape}, SHARD_MODE={SHARD_MODE}")

    E, H, N = 32, 128, 256
    A, M = 4, 32

    torch.manual_seed(42)
    model = SimpleEinsum(E, H, N).to(torch.float32)
    model.train()
    x = torch.randn(A, M, H, dtype=torch.float32)
    target = torch.randn(A, E, M, N, dtype=torch.float32)

    # CPU
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = model(cpu_x)
    cpu_loss = (cpu_out * target).sum()
    cpu_loss.backward()

    cpu_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    cpu_grad_x = cpu_x.grad.detach().clone()
    model.zero_grad()

    P(f"\nCPU loss: {cpu_loss.item():.4f}")

    # TT
    tt_model = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)
    tt_target = target.clone().to(device)

    shard_spec = {
        "none": None,
        "model": "model",
        "batch": "batch",
        "compound": ("model", "batch"),
    }[SHARD_MODE]

    if shard_spec is not None:
        for name, p in model.named_parameters():
            if p.shape[0] == E:
                spec = [None] * len(p.shape)
                spec[0] = shard_spec
                xs.mark_sharding(p, mesh, tuple(spec))
                P(f"  Sharded {name}: {tuple(spec)}")

    tt_x = x.clone().to(device).requires_grad_(True)
    tt_out = tt_model(tt_x)
    tt_loss = (tt_out * tt_target).sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    P(f"\nTT loss: {tt_loss.detach().to('cpu').item():.4f}")

    # Forward comparison
    compare("forward", cpu_out.detach(), tt_out.detach().to("cpu"))

    # Gradient comparison
    tt_grad_x = tt_x.grad.detach().to("cpu")
    compare("grad_x", cpu_grad_x, tt_grad_x)

    for name in sorted(cpu_grads):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            compare(f"grad_{name}", cpu_grads[name], tt_g)

    P(f"\n{'='*60}")
    P("DONE")
    P(f"{'='*60}")


if __name__ == "__main__":
    main()
