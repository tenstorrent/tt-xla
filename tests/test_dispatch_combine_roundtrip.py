"""
Test dispatch → combine roundtrip backward.

No sparse_matmul. Just dispatch + combine to isolate whether the
backward through these custom ops works correctly on TT.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_dispatch_combine_roundtrip.py 2>&1 | tee out_roundtrip.txt
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

# Import our custom ops
import tt_torch.custom_moe_ops  # noqa: F401


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


def build_expert_mapping(num_experts, num_devices, mesh_shape=None):
    """Build [1, 1, E, D] one-hot mapping."""
    mapping = torch.zeros(1, 1, num_experts, num_devices, dtype=torch.float32)
    if mesh_shape is not None:
        rows, cols = mesh_shape
        for i in range(num_experts):
            row = i % rows
            col = (i // rows) % cols
            device_id = row * cols + col
            mapping[0, 0, i, device_id] = 1
    else:
        for i in range(num_experts):
            mapping[0, 0, i, i % num_devices] = 1
    return mapping


class DispatchCombineModel(nn.Module):
    """
    Dispatch → simple linear transform → combine.
    Tests dispatch/combine backward without sparse_matmul complexity.
    """

    def __init__(self, E, H, K, num_devices, dispatch_devices, mesh_shape=None):
        super().__init__()
        self.E = E
        self.K = K
        self.num_devices = num_devices
        self.dispatch_devices = dispatch_devices

        # Simple linear per-expert weight (no sparsity, just a matmul)
        self.weight = nn.Parameter(torch.randn(E, H, H) * 0.02)
        self.bias = nn.Parameter(torch.zeros(E, H))

        # Router
        self.router_weight = nn.Parameter(torch.randn(E, H) * 0.01)
        self.router_bias = nn.Parameter(torch.zeros(E))

        mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        E, K = self.E, self.K

        # Router
        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(scores, K, dim=-1)  # [B*S, K]

        # Reshape for dispatch
        x = hidden_states.view(B, 1, S, H)
        expert_indices = topk_indices.view(B, 1, S, K)

        # Dispatch
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x,
            expert_indices,
            self.expert_mapping,
            num_devices=self.dispatch_devices,
            cluster_axis=0,
        )

        BD = dispatched.shape[1]

        # Simple expert computation (no sparse_matmul)
        # dispatched: [1, BD, S, H]
        # Expand to compute all experts: einsum
        disp = dispatched.squeeze(0)  # [BD, S, H]
        # All experts applied to all tokens
        expert_out = torch.einsum("bsh,ehk->bsek", disp, self.weight) + self.bias
        # expert_out: [BD, S, E, H]

        # Reshape to [E, BD, S, H] for combine
        expert_out = expert_out.permute(2, 0, 1, 3)  # [E, BD, S, H]

        # Combine: select expert outputs based on metadata
        combined = torch.ops.tt.all_to_all_combine(
            expert_out,
            metadata,
            self.expert_mapping,
            num_devices=self.dispatch_devices,
            cluster_axis=0,
            num_experts_per_tok=K,
            output_shard_dim=1,
            expert_indices=expert_indices,
        )
        # combined: [K, B, S, H]
        output = combined.sum(dim=0)  # [B, S, H]
        return output


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


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

    B, S, H, E, K = 1, 32, 256, 32, 4
    dispatch_devices = mesh_shape[0]  # 4
    num_devices = n  # 32

    torch.manual_seed(42)
    model = DispatchCombineModel(
        E, H, K, num_devices, dispatch_devices, mesh_shape=mesh_shape
    )

    # Force deterministic routing
    with torch.no_grad():
        model.router_bias.fill_(-100.0)
        for k in range(K):
            model.router_bias[k] = 100.0 + k * 10.0

    wrapped = WrappedModel(model)
    wrapped.train()

    torch.manual_seed(42)
    x = torch.randn(B, S, H, dtype=torch.float32)
    torch.manual_seed(99)
    target = torch.randn(B, S, H, dtype=torch.float32)

    # ---- CPU ----
    P("\n=== CPU forward + backward ===")
    cpu_model = torch.compile(wrapped, backend="inductor")
    cpu_x = x.clone().requires_grad_(True)
    cpu_out = cpu_model(cpu_x)
    cpu_loss = (cpu_out * target).sum()
    cpu_loss.backward()

    cpu_fwd = cpu_out.detach()
    cpu_grad_x = cpu_x.grad.detach().clone()
    cpu_grads = {}
    for name, p in wrapped.named_parameters():
        if p.grad is not None:
            cpu_grads[name] = p.grad.detach().clone()
    wrapped.zero_grad()

    P(f"CPU loss: {cpu_loss.item():.4f}")
    P(f"CPU fwd norm: {cpu_fwd.norm():.4f}")
    P(f"CPU grad_x norm: {cpu_grad_x.norm():.4f}")
    for name in sorted(cpu_grads):
        P(f"  CPU grad_{name}: norm={cpu_grads[name].norm():.4f}")

    # ---- TT ----
    P("\n=== TT forward + backward ===")
    tt_model = torch.compile(
        wrapped,
        backend="tt",
        options={
            "tt_experimental_compile": False,
            "tt_enable_torch_fx_fusion_pass": False,
        },
    )
    device = torch_xla.device()
    wrapped.to(device)

    # Shard E dimension with compound sharding
    for name, p in wrapped.named_parameters():
        if len(p.shape) >= 1 and p.shape[0] == E:
            spec = [None] * len(p.shape)
            spec[0] = ("model", "batch")
            xs.mark_sharding(p, mesh, tuple(spec))
            P(f"  Sharded {name}: {tuple(spec)}")

    # Also shard expert_mapping E dimension
    em = wrapped.model.expert_mapping
    xs.mark_sharding(em, mesh, (None, None, ("model", "batch"), None))
    P(f"  Sharded expert_mapping: (None, None, ('model', 'batch'), None)")

    tt_target = target.clone().to(device)
    tt_x = x.clone().to(device).requires_grad_(True)
    tt_out = tt_model(tt_x)
    tt_loss = (tt_out * tt_target).sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    tt_fwd = tt_out.detach().to("cpu")
    tt_grad_x = tt_x.grad.detach().to("cpu")

    P(f"TT loss: {tt_loss.detach().to('cpu').item():.4f}")
    P(f"TT fwd norm: {tt_fwd.norm():.4f}")

    # ---- Compare ----
    P(f"\n{'='*60}")
    P("COMPARISON")
    P(f"{'='*60}")

    compare("forward", cpu_fwd, tt_fwd)
    compare("grad_x", cpu_grad_x, tt_grad_x)

    P("\n  Weight gradients:")
    for name in sorted(cpu_grads):
        p = dict(wrapped.named_parameters())[name]
        if p.grad is not None:
            tt_g = p.grad.detach().to("cpu")
            compare(f"grad_{name}", cpu_grads[name], tt_g)

    P(f"\n{'='*60}")
    P("DONE")
    P(f"{'='*60}")


if __name__ == "__main__":
    main()
