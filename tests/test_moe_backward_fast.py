"""
Fast test: A2aSparseMLP backward on TT device without loading GPT-OSS model.

Uses a small synthetic model with the same MoE structure. Compares CPU vs TT
forward and backward PCC for individual MLP parameters.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_moe_backward_fast.py 2>&1 | tee out_fast.txt
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_package.tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping
from tt_torch.sharding import sharding_constraint_tensor


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


class FakeRouter(nn.Module):
    def __init__(self, E, H, K):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H) * 0.01)
        self.bias = nn.Parameter(torch.zeros(E))
        self.top_k = K

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        logits = flat @ self.weight.T + self.bias
        scores = torch.softmax(logits, -1)
        topk_s, topk_i = torch.topk(scores, self.top_k, -1)
        full = torch.zeros_like(scores)
        full.scatter_(1, topk_i, topk_s)
        return full, topk_i


class FakeExperts(nn.Module):
    def __init__(self, E, H, inter):
        super().__init__()
        # Interleaved gate_up_proj: [g0, u0, g1, u1, ...]
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.01)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.01)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H))
        self.alpha = 1.702
        self.limit = 7.0


class FakeMLP(nn.Module):
    def __init__(self, E, H, K, inter):
        super().__init__()
        self.router = FakeRouter(E, H, K)
        self.experts = FakeExperts(E, H, inter)


def P(*args, **kw):
    print(*args, **kw, flush=True)


def main():
    import time

    # ---- SPMD setup ----
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    P("[t] init_computation_client...")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    if n >= 32:
        mesh_shape = (4, 8)
    elif n >= 8:
        mesh_shape = (2, 4)
    else:
        raise RuntimeError(f"Need >=8 devices, got {n}")
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))
    rows, cols = mesh_shape
    D = n  # total devices
    D_dispatch = rows  # dispatch devices (along cluster_axis=0)

    P(f"[setup] {n} devices, mesh={mesh_shape}, dispatch_devices={D_dispatch}")

    # ---- Model params ----
    E = 32       # experts
    K = 4        # top-k
    H = 64       # hidden size (small for speed)
    inter = 64   # intermediate size
    B = 1        # batch
    S = 32       # seq_len (must be divisible by M=32)

    torch.manual_seed(42)

    fake_mlp = FakeMLP(E, H, K, inter)
    model = A2aSparseMLP(
        fake_mlp, E, K,
        num_devices=D,
        dispatch_devices=D_dispatch,
        cluster_axis=0,
    )
    model.train()

    hidden = torch.randn(B, S, H) * 0.1
    grad_out = torch.randn(B, S, H) * 0.01

    # ==== CPU forward + backward (eager, no compile) ====
    P("[t] CPU forward...")
    t0 = time.time()
    with torch.set_grad_enabled(True):
        cpu_out, _ = model(hidden)
    P(f"[t] CPU forward done: {time.time()-t0:.1f}s")
    t0 = time.time()
    cpu_out.backward(gradient=grad_out.clone())
    P(f"[t] CPU backward done: {time.time()-t0:.1f}s")

    cpu_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    P(f"[cpu] fwd shape: {cpu_out.shape}, norm: {cpu_out.norm():.6f}")
    P(f"[cpu] grads: {len(cpu_grads)} params")

    # ==== TT forward + backward ====
    P("[t] torch.compile(backend='tt')...")
    t0 = time.time()
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    P(f"[t] compile created: {time.time()-t0:.1f}s")

    device = torch_xla.device()
    P("[t] model.to(device)...")
    t0 = time.time()
    model.to(device)
    P(f"[t] to device done: {time.time()-t0:.1f}s")

    # Shard expert weights along expert dim, others replicated
    E_dim = ("batch", "model")  # compound shard E across both dims
    shard_specs = {}
    for name, param in model.named_parameters():
        if "gate_up_proj" in name and "bias" not in name:
            # [E, H, inter*2] — shard E
            shard_specs[param] = (E_dim, None, None)
        elif "gate_up_proj_bias" in name:
            # [E, inter*2] — shard E
            shard_specs[param] = (E_dim, None)
        elif "down_proj" in name and "bias" not in name:
            # [E, inter, H] — shard E
            shard_specs[param] = (E_dim, None, None)
        elif "down_proj_bias" in name:
            # [E, H] — shard E
            shard_specs[param] = (E_dim, None)
        elif "router" in name:
            if param.dim() == 2:
                shard_specs[param] = (None, None)
            else:
                shard_specs[param] = (None,)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    # Also shard expert_mapping buffer
    xs.mark_sharding(model.expert_mapping, mesh, (None, None, E_dim, None))

    hidden_dev = hidden.to(device)
    grad_dev = grad_out.to(device)

    P("[t] TT forward (triggers compile)...")
    t0 = time.time()
    with torch.set_grad_enabled(True):
        tt_out, _ = tt_compiled(hidden_dev)
    P(f"[t] TT forward done: {time.time()-t0:.1f}s")
    t0 = time.time()
    torch_xla.sync(wait=True)
    P(f"[t] sync done: {time.time()-t0:.1f}s")

    P("[t] TT backward...")
    t0 = time.time()
    with torch.set_grad_enabled(True):
        tt_out.backward(gradient=grad_dev)
    P(f"[t] TT backward done: {time.time()-t0:.1f}s")

    # Mark gradient sharding
    for param in model.parameters():
        if param.grad is None or param not in shard_specs:
            continue
        param.grad = sharding_constraint_tensor(
            param.grad, mesh, shard_specs[param],
        )

    wanted = [p.grad for p in model.parameters() if p.grad is not None]
    torch_xla._XLAC._xla_sync_multi(
        wanted, list({p.device.type for p in wanted}), wait=True,
    )

    tt_grads = {n: p.grad.to("cpu") for n, p in model.named_parameters() if p.grad is not None}

    # ==== Compare ====
    fwd_pcc = pcc(cpu_out.detach(), tt_out.detach().to("cpu"))
    print(f"\nForward PCC = {fwd_pcc:.6f}")

    mlp_pccs = []
    for name in sorted(cpu_grads.keys()):
        if name not in tt_grads:
            print(f"  {name}: MISSING on TT")
            continue
        p = pcc(cpu_grads[name], tt_grads[name])
        mlp_pccs.append(p)
        cn = cpu_grads[name].norm().item()
        tn = tt_grads[name].norm().item()
        print(f"  {name}: PCC={p:.6f}  cpu_norm={cn:.6f}  tt_norm={tn:.6f}")

    if mlp_pccs:
        print(f"\nAvg backward PCC = {sum(mlp_pccs)/len(mlp_pccs):.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
