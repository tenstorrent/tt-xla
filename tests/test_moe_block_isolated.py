"""
Test A2aSparseMLP block in isolation (forward + backward, CPU vs TT).

This isolates the MoE backward from the rest of the GPT model to determine
if the backward PCC issue is in the MoE block specifically.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_moe_block_isolated.py 2>&1 | tee out_moe_block.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# Import custom ops and sparse_mlp
import tt_torch.custom_moe_ops  # noqa: F401
from tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping


def P(*a, **kw):
    print(*a, **kw, flush=True)


def pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() < 2 or a.numel() != b.numel():
        return float("nan")
    a_m = a - a.mean()
    b_m = b - b.mean()
    return ((a_m * b_m).sum() / (a_m.norm() * b_m.norm()).clamp(min=1e-12)).item()


def compare(name, cpu_t, tt_t):
    cpu_f = cpu_t.detach().float().flatten()
    tt_f = tt_t.detach().float().flatten()
    if cpu_f.shape != tt_f.shape:
        P(f"  {name}: SHAPE MISMATCH {list(cpu_t.shape)} vs {list(tt_t.shape)}")
        return float("nan")
    if cpu_f.numel() > 1:
        p = pcc(cpu_t, tt_t)
        diff = (cpu_f - tt_f).abs()
        cn = cpu_t.float().norm().item()
        tn = tt_t.float().norm().item()
        P(
            f"  {name}: PCC={p:.6f}  atol={diff.max():.6f}  "
            f"cpu_norm={cn:.4f}  tt_norm={tn:.4f}  "
            f"ratio={tn / (cn + 1e-12):.4f}"
        )
        return p
    return float("nan")


class FakeMoEConfig:
    """Minimal config to satisfy A2aSparseMLP.__init__."""

    def __init__(self, hidden_size, num_experts, num_experts_per_tok, intermediate_size):
        self.hidden_size = hidden_size
        self.num_local_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok


class FakeExperts(nn.Module):
    """Fake experts module with the right parameter shapes."""

    def __init__(self, E, H, inter):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter * 2) * 0.02)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, inter * 2))
        self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.02)
        self.down_proj_bias = nn.Parameter(torch.zeros(E, H))


class FakeRouter(nn.Module):
    """Fake router that returns topk scores and indices."""

    def __init__(self, E, H, K):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H) * 0.01)
        self.bias = nn.Parameter(torch.zeros(E))
        self.K = K

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.weight, self.bias)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.K, dim=-1)
        return topk_scores, topk_indices


class FakeOriginalMLP(nn.Module):
    """Fake original MLP module that A2aSparseMLP wraps."""

    def __init__(self, E, H, inter, K):
        super().__init__()
        self.router = FakeRouter(E, H, K)
        self.experts = FakeExperts(E, H, inter)


class WrappedMoE(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        return self.moe(x)


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

    B, S, H, E, K, inter = 1, 32, 256, 32, 4, 256
    dispatch_devices = mesh_shape[0]  # 4

    torch.manual_seed(42)
    original_mlp = FakeOriginalMLP(E, H, inter, K)

    # Force deterministic routing
    with torch.no_grad():
        original_mlp.router.bias.fill_(-100.0)
        for k in range(K):
            original_mlp.router.bias[k] = 100.0 + k * 10.0

    # Create A2aSparseMLP (the actual module used in GPT-OSS)
    moe = A2aSparseMLP(
        original_mlp,
        num_experts=E,
        num_experts_per_tok=K,
        dispatch_devices=dispatch_devices,
        num_devices=n,
        cluster_axis=0,
        mesh_shape=mesh_shape,
        use_fused_remap=True,
    )
    model = WrappedMoE(moe)
    model.train()

    torch.manual_seed(42)
    x = torch.randn(B, S, H, dtype=torch.float32)
    torch.manual_seed(99)
    target = torch.randn(B, S, H, dtype=torch.float32)

    # ---- CPU ----
    P("\n=== CPU forward + backward ===")
    cpu_model = torch.compile(model, backend="inductor")
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = cpu_model(cpu_x)
    cpu_loss = (cpu_out * target).sum()
    cpu_loss.backward()

    cpu_fwd = cpu_out.detach()
    cpu_grad_x = cpu_x.grad.detach().clone()
    cpu_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    model.zero_grad()

    P(f"CPU loss: {cpu_loss.item():.4f}")
    P(f"CPU fwd norm: {cpu_fwd.norm():.4f}")
    P(f"CPU grad_x norm: {cpu_grad_x.norm():.4f}")
    for name in sorted(cpu_grads):
        P(f"  CPU grad_{name}: norm={cpu_grads[name].norm():.4f}")

    # ---- TT ----
    P("\n=== TT forward + backward ===")
    tt_model = torch.compile(
        model,
        backend="tt",
        options={
            "tt_experimental_compile": False,
            "tt_enable_torch_fx_fusion_pass": False,
        },
    )
    device = torch_xla.device()
    model.to(device)

    # Shard E dimension with compound sharding
    for name, p in model.named_parameters():
        if len(p.shape) >= 1 and p.shape[0] == E:
            spec = [None] * len(p.shape)
            spec[0] = ("model", "batch")
            xs.mark_sharding(p, mesh, tuple(spec))
            P(f"  Sharded {name}: {tuple(spec)}")

    # Shard expert_mapping
    em = model.moe.expert_mapping
    xs.mark_sharding(em, mesh, (None, None, ("model", "batch"), None))
    P(f"  Sharded expert_mapping")

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

    # ---- Compare ----
    P(f"\n{'='*60}")
    P("COMPARISON")
    P(f"{'='*60}")

    fwd_pcc = compare("forward", cpu_fwd, tt_fwd)
    gi_pcc = compare("grad_x", cpu_grad_x, tt_grad_x)

    P("\n  Weight gradients:")
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
