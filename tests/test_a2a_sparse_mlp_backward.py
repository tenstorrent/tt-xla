"""
Isolated test of A2aSparseMLP forward+backward on TT vs CPU.

Tests the actual A2aSparseMLP module (from sparse_mlp.py) with a minimal
MLP stub, comparing weight gradients between CPU and TT.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_a2a_sparse_mlp_backward.py 2>&1 | tee out_a2a_bwd.txt
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_package"))
from tt_torch.sparse_mlp import A2aSparseMLP


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
        if not torch.isfinite(cpu_t).all():
            P(f"    WARNING: CPU has non-finite values")
        if not torch.isfinite(tt_t).all():
            P(f"    WARNING: TT has non-finite values")
        return p
    return float('nan')


class FakeMLP(nn.Module):
    """Minimal MLP module with the interface A2aSparseMLP expects."""
    def __init__(self, E, H, inter, K):
        super().__init__()
        self.router = FakeRouter(E, H, K)
        self.experts = FakeExperts(E, H, inter)


class FakeRouter(nn.Module):
    def __init__(self, E, H, K):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H) * 0.01)
        self.bias = nn.Parameter(torch.zeros(E))
        self.top_k = K

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.weight, self.bias)
        scores = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return scores, topk_indices


class FakeExperts(nn.Module):
    def __init__(self, E, H, inter):
        super().__init__()
        # Interleaved gate_up_proj: [g0, u0, g1, u1, ...]
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H))
        self.alpha = 1.702
        self.limit = 7.0


class WrappedA2aSparseMLP(nn.Module):
    """Wraps A2aSparseMLP, returns only the output tensor (not router_scores)."""
    def __init__(self, a2a):
        super().__init__()
        self.a2a = a2a

    def forward(self, hidden_states):
        out, _ = self.a2a(hidden_states)
        return out


def run_test(model, input_tensor, mesh, shard_specs_fn, label="test"):
    P(f"\n{'='*60}")
    P(f"TEST: {label}")
    P(f"{'='*60}")

    model = model.to(torch.float32)
    model.train()

    # CPU forward + backward
    cpu_compiled = torch.compile(model, backend="inductor")
    cpu_input = input_tensor.clone().requires_grad_(True)
    cpu_out = cpu_compiled(cpu_input)
    cpu_out.sum().backward()

    cpu_fwd = cpu_out.detach()
    cpu_grad_input = cpu_input.grad.detach()
    cpu_wgrads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_wgrads[name] = p.grad.detach().clone()
    model.zero_grad()

    P(f"\n  CPU forward shape: {list(cpu_fwd.shape)}, norm: {cpu_fwd.norm():.4f}")
    P(f"  CPU grad_input shape: {list(cpu_grad_input.shape)}, norm: {cpu_grad_input.norm():.4f}")
    for name in sorted(cpu_wgrads.keys()):
        g = cpu_wgrads[name]
        P(f"  CPU grad_{name}: shape={list(g.shape)}, norm={g.float().norm():.4f}, "
          f"finite={torch.isfinite(g).all().item()}")

    # TT forward + backward
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
    tt_grad_input = tt_input.grad.detach().to("cpu")

    P(f"\n  TT forward shape: {list(tt_fwd.shape)}, norm: {tt_fwd.norm():.4f}")
    P(f"  TT grad_input shape: {list(tt_grad_input.shape)}, norm: {tt_grad_input.norm():.4f}")

    # Compare forward
    fwd_pcc = compare("forward_output", cpu_fwd, tt_fwd)
    gi_pcc = compare("grad_input", cpu_grad_input, tt_grad_input)

    # Compare weight gradients
    P(f"\n  Weight gradients:")
    min_pcc = 1.0
    for name in sorted(cpu_wgrads.keys()):
        p = dict(model.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            cpu_g = cpu_wgrads[name]
            p_val = compare(f"  grad_{name}", cpu_g, tt_g)
            if not (p_val != p_val):  # not NaN
                min_pcc = min(min_pcc, p_val)

    P(f"\n  SUMMARY: fwd_pcc={fwd_pcc:.6f}  gi_pcc={gi_pcc:.6f}  min_wgrad_pcc={min_pcc:.6f}")
    return fwd_pcc, gi_pcc, min_pcc


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
    P(f"[setup] {n} devices, mesh={mesh_shape}, dispatch_devices={dispatch_devices}")

    # Match the real model dims (20B uses H=2880, E=32, K=4, inter=2880)
    # Use smaller dims for faster test
    B, S, H, E, K, inter = 1, 32, 512, 32, 4, 512

    torch.manual_seed(42)
    input_tensor = torch.randn(B, S, H, dtype=torch.float32)

    # Create FakeMLP and A2aSparseMLP
    torch.manual_seed(42)
    fake_mlp = FakeMLP(E, H, inter, K)

    # Force same experts for all tokens (deterministic routing)
    with torch.no_grad():
        fake_mlp.router.bias.fill_(-100.0)
        for k in range(K):
            fake_mlp.router.bias[k] = 100.0 + k * 10.0

    a2a = A2aSparseMLP(
        fake_mlp,
        num_experts=E,
        num_experts_per_tok=K,
        num_devices=num_devices,
        dispatch_devices=dispatch_devices,
        cluster_axis=0,
        mesh_shape=mesh_shape,
    )
    model = WrappedA2aSparseMLP(a2a)

    def shard_fn(m):
        a = m.a2a
        return {
            a.experts.gate_up_proj: (("model", "batch"), None, None),
            a.experts.gate_up_proj_bias: (("model", "batch"), None),
            a.experts.down_proj: (("model", "batch"), None, None),
            a.experts.down_proj_bias: (("model", "batch"), None),
        }

    run_test(model, input_tensor, mesh, shard_fn, "A2aSparseMLP with raw einsum")

    P(f"\n{'='*60}")
    P("DONE")
    P(f"{'='*60}")


if __name__ == "__main__":
    main()
