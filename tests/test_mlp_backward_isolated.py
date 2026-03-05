"""
Isolated A2aSparseMLP backward PCC test: CPU (inductor) vs TT.
Tests MLP backward in isolation to determine if the issue is MLP-specific
or caused by interaction with the full model (attention, norms, residuals).

Usage:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_mlp_backward_isolated.py 2>&1 | tee out_mlp_bwd.txt
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_package.tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping


def pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() <= 1:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 0 else float("nan")


class FakeRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.view(B * S, H)
        logits = flat @ self.weight.T + self.bias
        scores = torch.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=4, dim=-1)
        full_scores = torch.zeros_like(scores)
        full_scores.scatter_(1, topk_indices, topk_scores)
        return full_scores, topk_indices


class FakeExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size * 2) * 0.01
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(num_experts, intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.01
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.alpha = 1.702
        self.limit = 7.0


class FakeMLP(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.router = FakeRouter(hidden_size, num_experts)
        self.experts = FakeExperts(num_experts, hidden_size, intermediate_size)


class MLPWrapper(nn.Module):
    """Wrapper that returns only the MLP output (not router scores) for torch.compile."""
    def __init__(self, a2a_mlp):
        super().__init__()
        self.mlp = a2a_mlp

    def forward(self, hidden_states):
        output, _ = self.mlp(hidden_states)
        return output


def run_cpu_backward(wrapper, hidden, grad_out):
    """Run forward+backward on CPU with inductor."""
    wrapper.train()
    hidden = hidden.clone().detach().requires_grad_(False)
    grad_out = grad_out.clone().detach()

    compiled = torch.compile(wrapper, backend="inductor")
    out = compiled(hidden)
    out.backward(grad_out)

    grads = {}
    for name, p in wrapper.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.clone().detach()
    return out.detach(), grads


def run_tt_backward(wrapper, hidden, grad_out, mesh, shard_specs):
    """Run forward+backward on TT."""
    device = torch_xla.device()

    wrapper.train()

    compiled = torch.compile(
        wrapper, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False}
    )

    # Build name→spec mapping before .to(device)
    name_to_spec = {}
    for name, param in wrapper.named_parameters():
        if param in shard_specs:
            name_to_spec[name] = shard_specs[param]

    wrapper.to(device)
    for name, param in wrapper.named_parameters():
        if name in name_to_spec:
            xs.mark_sharding(param, mesh, name_to_spec[name])

    hidden_dev = hidden.to(device)
    grad_dev = grad_out.to(device)

    out = compiled(hidden_dev)
    torch_xla.sync(wait=True)

    out.backward(grad_dev)

    # Mark gradient sharding
    for name, param in wrapper.named_parameters():
        if param.grad is not None and name in name_to_spec:
            param.grad = xs.mark_sharding(param.grad, mesh, name_to_spec[name])

    wanted_grads = [p.grad for p in wrapper.parameters() if p.grad is not None]
    torch_xla._XLAC._xla_sync_multi(
        wanted_grads,
        list(set([p.device.type for p in wanted_grads])),
        wait=True,
    )

    grads = {}
    for name, p in wrapper.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.clone().detach().cpu()

    out_cpu = out.detach().cpu()
    return out_cpu, grads


def main():
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8) if n >= 32 else (2, 4)
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))
    D = mesh_shape[0]
    total_devices = mesh_shape[0] * mesh_shape[1]

    torch.manual_seed(42)

    # Test configs
    configs = [
        {"name": "tiny", "E": 32, "K": 4, "H": 64, "inter": 64, "B": 1, "S": 32},
        {"name": "small", "E": 32, "K": 4, "H": 256, "inter": 256, "B": 1, "S": 32},
        {"name": "realistic", "E": 32, "K": 4, "H": 2880, "inter": 2880, "B": 1, "S": 32},
    ]

    for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        print(f"\n{'='*60}")
        print(f"Dtype: {dtype_name}")
        print(f"{'='*60}")

        for cfg in configs:
            torch.manual_seed(42)
            E, K, H, inter = cfg["E"], cfg["K"], cfg["H"], cfg["inter"]
            B, S = cfg["B"], cfg["S"]

            print(f"\n--- {cfg['name']}: E={E}, K={K}, H={H}, inter={inter}, B={B}, S={S} ---")

            # Create model
            fake_mlp = FakeMLP(E, H, inter)
            a2a = A2aSparseMLP(
                fake_mlp,
                num_experts=E,
                num_experts_per_tok=K,
                num_devices=total_devices,
                dispatch_devices=D,
                cluster_axis=0,
            )
            wrapper = MLPWrapper(a2a)
            wrapper = wrapper.to(dtype)

            # Input & grad
            hidden = torch.randn(B, S, H, dtype=dtype) * 0.1
            grad_out = torch.randn(B, S, H, dtype=dtype) * 0.01

            # CPU backward
            cpu_out, cpu_grads = run_cpu_backward(wrapper, hidden, grad_out)

            # Reset grads
            wrapper.zero_grad()

            # Shard specs (matching GPT-OSS pattern)
            shard_specs = {}
            for name, param in wrapper.named_parameters():
                if "gate_up_proj" in name and "bias" not in name:
                    shard_specs[param] = (("model", "batch"), None, None)
                elif "gate_up_proj_bias" in name:
                    shard_specs[param] = (("model", "batch"), None)
                elif "down_proj" in name and "bias" not in name:
                    shard_specs[param] = (("model", "batch"), None, None)
                elif "down_proj_bias" in name:
                    shard_specs[param] = (("model", "batch"), None)
                # Router weights: not sharded

            # TT backward
            try:
                tt_out, tt_grads = run_tt_backward(wrapper, hidden, grad_out, mesh, shard_specs)
            except Exception as ex:
                print(f"  TT FAILED: {ex}")
                continue

            # Forward PCC
            fwd_pcc = pcc(cpu_out, tt_out)
            print(f"  Forward PCC: {fwd_pcc:.6f}")

            # Per-parameter backward PCC
            min_pcc = 1.0
            for name in sorted(cpu_grads.keys()):
                if name not in tt_grads:
                    print(f"  {name}: MISSING on TT")
                    continue
                cg = cpu_grads[name]
                tg = tt_grads[name]
                p = pcc(cg, tg)
                cn = cg.float().norm().item()
                tn = tg.float().norm().item()
                ratio = cn / tn if tn > 0 else float("inf")
                tag = "MLP" if "mlp" in name else "ROUTER" if "router" in name else "OTHER"
                print(f"  {name}: PCC={p:.6f}  cpu_norm={cn:.6f}  tt_norm={tn:.6f}  ratio={ratio:.3f}  ({tag})")
                if not (p != p):  # not NaN
                    min_pcc = min(min_pcc, p)

            print(f"  MIN backward PCC: {min_pcc:.6f}")


if __name__ == "__main__":
    main()
