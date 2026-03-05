"""
Isolate einsum backward on TT WITHOUT dispatch/combine.

Tests the exact same einsum + activation + einsum chain as A2aSparseMLP,
but with direct input (no dispatch) and direct output (no combine).
This tells us whether the einsum backward itself works on TT with E-sharding.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_einsum_backward_isolated.py 2>&1 | tee out_einsum_bwd.txt
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
    num = (a_m * b_m).sum()
    den = (a_m.norm() * b_m.norm()).clamp(min=1e-12)
    return (num / den).item()


def compare(name, cpu_t, tt_t):
    P(f"  {name}:")
    P(f"    shapes: cpu={list(cpu_t.shape)}  tt={list(tt_t.shape)}")
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()
    if cpu_f.shape != tt_f.shape:
        P(f"    SHAPE MISMATCH!")
        return float('nan')
    if cpu_f.numel() > 1:
        p = pcc(cpu_t, tt_t)
        diff = (cpu_f - tt_f).abs()
        cn = cpu_t.float().norm().item()
        tn = tt_t.float().norm().item()
        P(f"    PCC={p:.6f}  atol={diff.max():.6f}  mae={diff.mean():.6f}  "
          f"cpu_norm={cn:.4f}  tt_norm={tn:.4f}  "
          f"ratio={tn/(cn+1e-12):.4f}")
        return p
    return float('nan')


class EinsumMLP(nn.Module):
    """Raw einsum expert computation — no dispatch/combine."""
    def __init__(self, E, H, inter):
        super().__init__()
        self.E = E
        self.intermediate_size = inter
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_bias = nn.Parameter(torch.zeros(E, H))

    def forward(self, x):
        # x: [A, B, M, H] — same shape as hidden_4d in A2aSparseMLP
        E = self.E
        A, B, M, H = x.shape

        # Gate+Up: [A, B, M, H] @ [E, H, N] -> [A, B, E, M, N]
        gate_up_out = torch.einsum('abmh,ehn->abemn', x, self.gate_up_proj)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, N]
        gate_up_out = gate_up_out + self.gate_up_bias

        # Activation (SiLU * up)
        gate = gate_up_out[..., :self.intermediate_size]
        up = gate_up_out[..., self.intermediate_size:]
        activated = F.silu(gate) * up

        # Down: [AB, E, M, inter] @ [E, inter, H] -> [AB, E, M, H]
        act_r = activated.permute(0, 1, 3, 2, 4).contiguous().view(A*B, E, M, self.intermediate_size)
        down_out = torch.einsum('aemk,ekn->aemn', act_r, self.down_proj)

        # Reshape back and add bias
        down_out = down_out.view(A, B, E, M, H)
        down_out = down_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, H]
        down_out = down_out + self.down_bias

        # Return the full output (caller computes loss)
        return down_out


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

    E, H, inter = 32, 512, 512
    A, B, M = 4, 1, 32  # Same as BD=4, seq//M=1, M=32

    torch.manual_seed(42)
    model = EinsumMLP(E, H, inter).to(torch.float32)
    model.train()
    x = torch.randn(A, B, M, H, dtype=torch.float32)

    # Fixed target for loss computation (dot product, not .sum())
    torch.manual_seed(99)
    target = torch.randn(A, B, M, E, H, dtype=torch.float32)

    # ---- CPU forward + backward ----
    cpu_model = torch.compile(model, backend="inductor")
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = cpu_model(cpu_x)
    cpu_loss = (cpu_out * target).sum()
    cpu_loss.backward()

    cpu_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    cpu_grad_x = cpu_x.grad.detach().clone()
    model.zero_grad()

    P(f"\nCPU loss: {cpu_loss.item():.4f}")
    P(f"CPU grad_x norm: {cpu_grad_x.norm():.4f}")
    for name in sorted(cpu_grads):
        g = cpu_grads[name]
        P(f"CPU grad_{name}: norm={g.norm():.4f}, shape={list(g.shape)}")

    # ---- TT forward + backward ----
    tt_model = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    tt_target = target.clone().to(device)

    # Shard E dimension with compound sharding
    # Shard E dimension — "model" for 8-way, or ("model","batch") for 32-way
    shard_mode = "compound"  # "none", "model", "compound"
    if shard_mode == "compound":
        e_spec = ("model", "batch")
    elif shard_mode == "model":
        e_spec = "model"
    else:
        e_spec = None

    if e_spec is not None:
        for name, p in model.named_parameters():
            if p.shape[0] == E:
                spec = [None] * len(p.shape)
                spec[0] = e_spec
                xs.mark_sharding(p, mesh, tuple(spec))
                P(f"Sharded {name} with spec {tuple(spec)}")
    else:
        P("NO SHARDING")
    tt_x = x.clone().to(device).requires_grad_(True)
    tt_out = tt_model(tt_x)
    tt_loss = (tt_out * tt_target).sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    tt_loss_val = tt_loss.detach().to("cpu")
    tt_grad_x = tt_x.grad.detach().to("cpu")

    P(f"\nTT loss: {tt_loss_val.item():.4f}")

    # Compare
    P(f"\n{'='*60}")
    P("COMPARISON")
    P(f"{'='*60}")

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
