"""
Verify dispatch+combine correctness in both forward and backward.

Tests dispatch+combine with simple operations (bias add, einsum) between them,
WITHOUT sparse_matmul. Compares CPU vs TT for both forward output and
backward gradients.

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_dispatch_combine_fwd_bwd.py 2>&1 | tee out_dc_bwd.txt
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping


def P(*a, **kw):
    print(*a, **kw, flush=True)


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


class DispatchCombineOnlyModel(nn.Module):
    """
    Dispatch + per-expert bias + combine. No sparse_matmul.
    Tests the dispatch/combine pipeline with a simple differentiable operation.
    """
    def __init__(self, hidden_size, num_experts, num_experts_per_tok,
                 num_devices, dispatch_devices, cluster_axis, mesh_shape):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_devices = num_devices
        self.dispatch_devices = dispatch_devices
        self.cluster_axis = cluster_axis

        # Router
        self.router_weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.router_bias = nn.Parameter(torch.zeros(num_experts))
        self.top_k = num_experts_per_tok

        # Simple per-expert bias (differentiable, no sparse_matmul needed)
        self.expert_bias = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.1)

        # Expert mapping
        mapping = build_expert_mapping(num_experts, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok
        E = self.num_experts

        # Router
        flat = hidden_states.view(batch_size * seq_len, hidden_size)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, K, dim=-1)
        # scores: [B*S, E], topk_indices: [B*S, K]

        # Reshape for dispatch
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = topk_indices.view(batch_size, 1, seq_len, K)

        # Dispatch
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
        )

        BD = dispatched.shape[1]

        # Simple operation: add per-expert bias to dispatched tokens
        # For each token, add the bias of each expert → [E, BD, S, H]
        # This is a simple differentiable operation between dispatch and combine
        down_out = dispatched.squeeze(0)  # [BD, S, H]
        expert_out = down_out.unsqueeze(0).expand(E, -1, -1, -1)  # [E, BD, S, H]
        expert_out = expert_out + self.expert_bias.view(E, 1, 1, hidden_size)

        # Combine
        combined = torch.ops.tt.all_to_all_combine(
            expert_out, metadata, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=1,
            expert_indices=expert_indices,
        )
        # combined: [K, B, S, H]

        # Weighted sum
        topk_weights_reshaped = topk_weights.view(batch_size, seq_len, K)
        topk_weights_reshaped = topk_weights_reshaped.permute(2, 0, 1).unsqueeze(-1)  # [K, B, S, 1]
        output = (combined * topk_weights_reshaped).sum(dim=0)  # [B, S, H]

        return output


class DispatchEinsumCombineModel(nn.Module):
    """
    Dispatch + einsum expert computation + combine. No sparse_matmul.
    Tests a more realistic pipeline with actual weight multiplication.
    """
    def __init__(self, hidden_size, num_experts, num_experts_per_tok,
                 intermediate_size, num_devices, dispatch_devices,
                 cluster_axis, mesh_shape):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size = intermediate_size
        self.num_devices = num_devices
        self.dispatch_devices = dispatch_devices
        self.cluster_axis = cluster_axis

        # Router
        self.router_weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.router_bias = nn.Parameter(torch.zeros(num_experts))
        self.top_k = num_experts_per_tok

        # Expert weights (same layout as A2aSparseMLP)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size * 2) * 0.02
        )
        self.gate_up_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size * 2))
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

        self.alpha = 1.702
        self.limit = 7.0

        # Expert mapping
        mapping = build_expert_mapping(num_experts, num_devices, mesh_shape=mesh_shape)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok
        E = self.num_experts
        M = 32

        # Router
        flat = hidden_states.view(batch_size * seq_len, hidden_size)
        logits = F.linear(flat, self.router_weight, self.router_bias)
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, K, dim=-1)

        # Reshape for dispatch
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = topk_indices.view(batch_size, 1, seq_len, K)

        # Dispatch
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x, expert_indices, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
        )

        BD = dispatched.shape[1]
        split_seq = seq_len % M == 0 and seq_len >= M
        if split_seq:
            dim_a, dim_b = BD, seq_len // M
        else:
            dim_a, dim_b = BD // M, seq_len

        # Reshape dispatched
        if split_seq:
            hidden_4d = dispatched.view(BD, seq_len // M, M, hidden_size)
        elif dim_b == 1:
            hidden_4d = dispatched.view(BD // M, 1, M, hidden_size)
        else:
            hidden_4d = dispatched.view(BD // M, M, seq_len, hidden_size)
            hidden_4d = hidden_4d.permute(0, 2, 1, 3)

        # Gate+Up projection via einsum (no sparse_matmul)
        # hidden_4d: [A, B, M, H], gate_up_proj: [E, H, N]
        gate_up_out = torch.einsum('abmh,ehn->abemn', hidden_4d, self.gate_up_proj)
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, N]
        gate_up_out = gate_up_out + self.gate_up_bias

        # Activation
        gate_out = gate_up_out[..., :self.intermediate_size]
        up_out = gate_up_out[..., self.intermediate_size:]
        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # Down projection via einsum
        activated_reshaped = (
            activated.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(dim_a * dim_b, E, M, self.intermediate_size)
        )
        down_out = torch.einsum('aemk,ekn->aemn', activated_reshaped, self.down_proj)

        # Reshape for combine: [E, BD, S, H]
        down_out = down_out.view(dim_a, dim_b, E, M, hidden_size)
        down_out = down_out.permute(0, 1, 3, 2, 4)  # [A, B, M, E, H]
        down_out = down_out + self.down_bias
        if split_seq:
            down_out = down_out.permute(3, 0, 1, 2, 4).contiguous()
            down_out = down_out.view(E, BD, seq_len, hidden_size)
        elif dim_b == 1:
            down_out = down_out.squeeze(1)
            down_out = down_out.permute(2, 0, 1, 3).contiguous()
            down_out = down_out.view(E, BD, hidden_size).unsqueeze(1)
        else:
            down_out = down_out.permute(3, 0, 2, 1, 4).contiguous()
            down_out = down_out.view(E, BD, seq_len, hidden_size)

        # Combine
        decode_mode = dim_b == 1 and not split_seq
        combined = torch.ops.tt.all_to_all_combine(
            down_out, metadata, self.expert_mapping,
            num_devices=self.dispatch_devices, cluster_axis=self.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=2 if decode_mode else 1,
            expert_indices=expert_indices,
        )

        # Weighted sum
        topk_weights_reshaped = topk_weights.view(batch_size, seq_len, K)
        topk_weights_reshaped = topk_weights_reshaped.permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights_reshaped).sum(dim=0)

        return output


def run_fwd_bwd_test(label, model, input_tensor, mesh, shard_specs_fn):
    """Run forward + backward test, compare CPU vs TT."""
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    # CPU reference
    cpu_model = model
    cpu_compiled = torch.compile(cpu_model, backend="inductor")
    cpu_input = input_tensor.clone().requires_grad_(True)
    cpu_out = cpu_compiled(cpu_input)
    cpu_loss = cpu_out.sum()
    cpu_loss.backward()

    cpu_fwd = cpu_out.detach()
    cpu_grad_input = cpu_input.grad.detach()
    P(f"  [cpu] fwd norm={cpu_fwd.float().norm():.4f}")
    P(f"  [cpu] grad_input norm={cpu_grad_input.float().norm():.4f}")

    # Collect CPU weight gradients
    cpu_grads = {}
    for name, param in cpu_model.named_parameters():
        if param.grad is not None:
            cpu_grads[name] = param.grad.detach().clone()

    # Reset gradients for TT
    cpu_model.zero_grad()

    # TT
    tt_compiled = torch.compile(
        cpu_model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )
    device = torch_xla.device()
    cpu_model.to(device)

    if shard_specs_fn:
        shard_specs = shard_specs_fn(cpu_model)
        for tensor, spec in shard_specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    tt_input = input_tensor.clone().to(device).requires_grad_(True)
    tt_out = tt_compiled(tt_input)
    tt_loss = tt_out.sum()
    tt_loss.backward()
    torch_xla.sync(wait=True)

    tt_fwd_cpu = tt_out.detach().to("cpu")
    tt_grad_input_cpu = tt_input.grad.detach().to("cpu")

    fwd_pcc = pcc(cpu_fwd, tt_fwd_cpu)
    bwd_pcc = pcc(cpu_grad_input, tt_grad_input_cpu)

    P(f"  [tt]  fwd norm={tt_fwd_cpu.float().norm():.4f}")
    P(f"  [tt]  grad_input norm={tt_grad_input_cpu.float().norm():.4f}")
    P(f"  Forward PCC  = {fwd_pcc:.6f}")
    P(f"  Backward PCC (grad_input) = {bwd_pcc:.6f}")

    # Compare weight gradients
    P(f"\n  Weight gradient comparison:")
    for name, param in cpu_model.named_parameters():
        if param.grad is not None and name in cpu_grads:
            tt_grad = param.grad.detach().to("cpu")
            cpu_grad = cpu_grads[name]
            g_pcc = pcc(cpu_grad, tt_grad)
            P(f"    {name:40s}  PCC={g_pcc:.6f}  cpu_norm={cpu_grad.float().norm():.4f}  tt_norm={tt_grad.float().norm():.4f}")

    return fwd_pcc, bwd_pcc


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
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    # Common params
    B, S, H, E, K, inter = 1, 32, 512, 32, 4, 512

    torch.manual_seed(123)
    input_tensor = torch.randn(B, S, H, dtype=torch.float32)

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 1: Dispatch + per-expert bias + Combine (fwd+bwd)")
    P("  Fixed expert mapping, experts 0-3 forced active")
    P("=" * 60)

    torch.manual_seed(42)
    model1 = DispatchCombineOnlyModel(
        H, E, K, num_devices, dispatch_devices, 0, mesh_shape,
    )
    # Force router to always select experts 0-3
    with torch.no_grad():
        model1.router_bias.fill_(-100.0)
        for k in range(K):
            model1.router_bias[k] = 100.0 + k * 10.0

    def shard_specs_1(model):
        return {
            model.router_weight: (None, "batch"),
            model.router_bias: (None,),
            model.expert_bias: (("model", "batch"), None),
        }

    fwd1, bwd1 = run_fwd_bwd_test(
        "Dispatch+Bias+Combine",
        model1, input_tensor, mesh, shard_specs_1,
    )

    # ==================================================================
    P("\n" + "=" * 60)
    P("TEST 2: Dispatch + Einsum Expert MLP + Combine (fwd+bwd)")
    P("  Fixed expert mapping, experts 0-3 forced active")
    P("=" * 60)

    torch.manual_seed(42)
    model2 = DispatchEinsumCombineModel(
        H, E, K, inter, num_devices, dispatch_devices, 0, mesh_shape,
    )
    with torch.no_grad():
        model2.router_bias.fill_(-100.0)
        for k in range(K):
            model2.router_bias[k] = 100.0 + k * 10.0

    def shard_specs_2(model):
        return {
            model.router_weight: (None, "batch"),
            model.router_bias: (None,),
            model.gate_up_proj: (("model", "batch"), None, None),
            model.gate_up_bias: (("model", "batch"), None),
            model.down_proj: (("model", "batch"), None, None),
            model.down_bias: (("model", "batch"), None),
        }

    fwd2, bwd2 = run_fwd_bwd_test(
        "Dispatch+Einsum+Combine",
        model2, input_tensor, mesh, shard_specs_2,
    )

    # ==================================================================
    P(f"\n{'=' * 60}")
    P("SUMMARY:")
    P(f"  Test 1 (D+Bias+C):    Fwd PCC = {fwd1:.6f}  Bwd PCC = {bwd1:.6f}")
    P(f"  Test 2 (D+Einsum+C):  Fwd PCC = {fwd2:.6f}  Bwd PCC = {bwd2:.6f}")
    P(f"{'=' * 60}")


if __name__ == "__main__":
    main()
