"""
MoE forward+backward with NO custom ops (no dispatch, no combine).

Uses raw einsum + torch.gather to replace dispatch/combine custom_calls.
This lets Shardy see through everything and correctly partition the backward.

If this gives good PCC, the issue is in how Shardy handles custom_call backward.
If this also gives bad PCC, the issue is more fundamental.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_moe_no_customops.py 2>&1 | tee out_no_customops.txt
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


class MoENoCustomOps(nn.Module):
    """
    MoE with raw einsum + gather. No dispatch/combine custom ops.

    Forward:
    1. Router selects top-K experts per token
    2. Einsum computes ALL experts for ALL tokens (Shardy partitions E)
    3. torch.gather picks selected experts' outputs
    4. Weighted sum
    """
    def __init__(self, E, H, inter, K):
        super().__init__()
        self.E = E
        self.K = K
        self.inter = inter

        # Router
        self.router_weight = nn.Parameter(torch.randn(E, H) * 0.01)
        self.router_bias = nn.Parameter(torch.zeros(E))

        # Expert weights (E-shardable)
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_bias = nn.Parameter(torch.zeros(E, H))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        E, K = self.E, self.K
        BS = B * S

        # Router
        flat = hidden_states.view(BS, H)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)  # [BS, E]
        topk_weights, topk_indices = torch.topk(scores, K, dim=-1)  # [BS, K]

        # Expert computation: [B, S, H] @ [E, H, N] -> [B, S, E, N]
        gate_up = torch.einsum('bsh,ehn->bsen', hidden_states, self.gate_up_proj)
        gate_up = gate_up + self.gate_up_bias  # broadcast [E, N] to [B, S, E, N]

        gate = gate_up[..., :self.inter]
        up = gate_up[..., self.inter:]
        activated = F.silu(gate) * up  # [B, S, E, inter]

        down = torch.einsum('bsek,ekh->bseh', activated, self.down_proj)
        down = down + self.down_bias  # [B, S, E, H]

        # Gather selected experts: pick K from E dim
        # topk_indices: [BS, K] -> [B, S, K]
        idx = topk_indices.view(B, S, K)
        # Expand for gather: [B, S, K, H]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, H)
        gathered = torch.gather(down, dim=2, index=idx_expanded)  # [B, S, K, H]

        # Weighted sum: [B, S, K, H] * [B, S, K, 1] -> sum -> [B, S, H]
        weights = topk_weights.view(B, S, K, 1)
        output = (gathered * weights).sum(dim=2)  # [B, S, H]

        return output, scores


class WrappedMoE(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        out, _ = self.moe(x)
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
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    B, S, H, E, K, inter = 1, 32, 512, 32, 4, 512

    torch.manual_seed(42)
    moe = MoENoCustomOps(E, H, inter, K).to(torch.float32)

    # Force deterministic routing (same as A2aSparseMLP test)
    with torch.no_grad():
        moe.router_bias.fill_(-100.0)
        for k in range(K):
            moe.router_bias[k] = 100.0 + k * 10.0

    model = WrappedMoE(moe)
    model.train()

    torch.manual_seed(42)
    x = torch.randn(B, S, H, dtype=torch.float32)

    # Use a fixed target to make loss non-scalar (avoid .sum() scalar elimination)
    torch.manual_seed(99)
    target = torch.randn(B, S, H, dtype=torch.float32)

    # ---- CPU ----
    cpu_model = torch.compile(model, backend="inductor")
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = cpu_model(cpu_x)
    cpu_loss = (cpu_out * target).sum()
    cpu_loss.backward()

    cpu_fwd = cpu_out.detach()
    cpu_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    cpu_grad_x = cpu_x.grad.detach().clone()
    model.zero_grad()

    P(f"\nCPU loss: {cpu_loss.item():.4f}")
    P(f"CPU fwd norm: {cpu_fwd.norm():.4f}")

    # ---- TT ----
    tt_model = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    model.to(device)

    # Shard E dimension with compound sharding
    for name, p in model.named_parameters():
        if p.shape[0] == E:
            spec = [None] * len(p.shape)
            spec[0] = ("model", "batch")
            xs.mark_sharding(p, mesh, tuple(spec))
            P(f"Sharded {name}: {tuple(spec)}")

    tt_target = target.clone().to(device)
    tt_x = x.clone().to(device).requires_grad_(True)
    tt_out = tt_model(tt_x)
    tt_loss = (tt_out * tt_target).sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    tt_fwd = tt_out.detach().to("cpu")
    tt_grad_x = tt_x.grad.detach().to("cpu")

    P(f"\nTT loss: {tt_loss.detach().to('cpu').item():.4f}")
    P(f"TT fwd norm: {tt_fwd.norm():.4f}")

    # Compare
    P(f"\n{'='*60}")
    P("COMPARISON")
    P(f"{'='*60}")

    fwd_pcc = compare("forward", cpu_fwd, tt_fwd)
    gi_pcc = compare("grad_input", cpu_grad_x, tt_grad_x)

    P(f"\n  Weight gradients:")
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
