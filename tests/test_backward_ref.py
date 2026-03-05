"""
Compare A2aSparseMLP backward on CPU against a reference dense MoE backward.
No XLA/TT involved - pure CPU test to validate custom op backward correctness.

Usage:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_backward_ref.py 2>&1 | tee out_bwd_ref.txt
"""

import os
import sys
import torch
import torch.nn as nn

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
        # Scatter into full scores
        full_scores = torch.zeros_like(scores)
        full_scores.scatter_(1, topk_indices, topk_scores)
        return full_scores, topk_indices


class FakeExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        # Interleaved gate_up_proj: [g0, u0, g1, u1, ...]
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


def reference_dense_forward_backward(mlp_module, hidden_states, grad_output, K=4):
    """
    Reference dense MoE forward + backward using only standard PyTorch ops.
    Uses the SAME weights as A2aSparseMLP (after de-interleave).
    """
    B, S, H = hidden_states.shape
    E = mlp_module.num_experts
    inter = mlp_module.intermediate_size
    alpha = mlp_module.alpha
    limit = mlp_module.limit

    hidden_states = hidden_states.detach().requires_grad_(True)

    # Router
    router_scores, router_indices = mlp_module.router(hidden_states)
    # router_scores: [B*S, E], router_indices: [B*S, K]

    # De-interleaved gate_up_proj (A2aSparseMLP does this in __init__)
    gate_up_w = mlp_module.experts.gate_up_proj  # [E, H, inter*2] (already de-interleaved)
    gate_up_b = mlp_module.experts.gate_up_proj_bias  # [E, inter*2]
    down_w = mlp_module.experts.down_proj  # [E, inter, H]
    down_b = mlp_module.experts.down_proj_bias  # [E, H]

    # Dense computation for all experts
    flat = hidden_states.view(B * S, H)  # [BS, H]

    # Gate+Up: [BS, H] @ [E, H, inter*2] -> [BS, E, inter*2]
    gate_up_out = torch.einsum("bh,ehn->ben", flat, gate_up_w) + gate_up_b

    # Split and activation
    gate_out = gate_up_out[..., :inter]
    up_out = gate_up_out[..., inter:]
    gate_out = gate_out.clamp(max=limit)
    up_out = up_out.clamp(-limit, limit)
    glu = gate_out * torch.sigmoid(gate_out * alpha)
    activated = (up_out + 1) * glu  # [BS, E, inter]

    # Down: [BS, E, inter] @ [E, inter, H] -> [BS, E, H]
    down_out = torch.einsum("bei,eih->beh", activated, down_w) + down_b

    # Weighted sum over top-K experts
    topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)  # [BS, K]
    expert_outputs = torch.gather(
        down_out, dim=1,
        index=router_indices.unsqueeze(-1).expand(-1, -1, H)
    )  # [BS, K, H]
    output = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)  # [BS, H]
    output = output.view(B, S, H)

    # Backward
    output.backward(grad_output)

    return {
        "output": output.detach(),
        "gate_up_proj": gate_up_w.grad.detach() if gate_up_w.grad is not None else None,
        "gate_up_proj_bias": gate_up_b.grad.detach() if gate_up_b.grad is not None else None,
        "down_proj": down_w.grad.detach() if down_w.grad is not None else None,
        "down_proj_bias": down_b.grad.detach() if down_b.grad is not None else None,
        "router_weight": mlp_module.router.weight.grad.detach() if mlp_module.router.weight.grad is not None else None,
        "hidden_grad": hidden_states.grad.detach() if hidden_states.grad is not None else None,
    }


def a2a_forward_backward(a2a_mlp, hidden_states, grad_output):
    """Run A2aSparseMLP forward + backward on CPU."""
    hidden_states = hidden_states.detach().requires_grad_(True)
    output, _ = a2a_mlp(hidden_states)
    output.backward(grad_output)

    return {
        "output": output.detach(),
        "gate_up_proj": a2a_mlp.experts.gate_up_proj.grad.detach() if a2a_mlp.experts.gate_up_proj.grad is not None else None,
        "gate_up_proj_bias": a2a_mlp.experts.gate_up_proj_bias.grad.detach() if a2a_mlp.experts.gate_up_proj_bias.grad is not None else None,
        "down_proj": a2a_mlp.experts.down_proj.grad.detach() if a2a_mlp.experts.down_proj.grad is not None else None,
        "down_proj_bias": a2a_mlp.experts.down_proj_bias.grad.detach() if a2a_mlp.experts.down_proj_bias.grad is not None else None,
        "router_weight": a2a_mlp.router.weight.grad.detach() if a2a_mlp.router.weight.grad is not None else None,
        "hidden_grad": hidden_states.grad.detach() if hidden_states.grad is not None else None,
    }


def main():
    torch.manual_seed(42)

    E = 32
    K = 4
    H = 64  # Small hidden size for quick test
    inter = 64
    B = 1
    S = 32
    D = 4  # dispatch_devices

    print(f"Config: E={E}, K={K}, H={H}, inter={inter}, B={B}, S={S}, D={D}")

    # Create fake MLP with interleaved weights
    fake_mlp = FakeMLP(E, H, inter)

    # Create A2aSparseMLP wrapping the fake MLP
    a2a = A2aSparseMLP(
        fake_mlp,
        num_experts=E,
        num_experts_per_tok=K,
        num_devices=D * 8,  # total devices = 4*8 = 32
        dispatch_devices=D,
        cluster_axis=0,
    )

    # Input
    hidden = torch.randn(B, S, H, dtype=torch.float32) * 0.1
    grad_out = torch.randn(B, S, H, dtype=torch.float32) * 0.01

    # Run A2aSparseMLP
    a2a_results = a2a_forward_backward(a2a, hidden, grad_out)

    # Zero grads before reference run
    a2a.zero_grad()

    # Run reference (reuses the SAME weight tensors since A2aSparseMLP stores references)
    ref_results = reference_dense_forward_backward(a2a, hidden, grad_out, K=K)

    # Compare
    print("\n=== Forward Output ===")
    p = pcc(a2a_results["output"], ref_results["output"])
    print(f"  PCC = {p:.6f}")
    print(f"  a2a_norm = {a2a_results['output'].norm():.4f}")
    print(f"  ref_norm = {ref_results['output'].norm():.4f}")

    print("\n=== Backward Gradients ===")
    for name in ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias",
                  "router_weight", "hidden_grad"]:
        a2a_g = a2a_results[name]
        ref_g = ref_results[name]
        if a2a_g is None or ref_g is None:
            print(f"  {name}: None (a2a={a2a_g is not None}, ref={ref_g is not None})")
            continue
        p = pcc(a2a_g, ref_g)
        ratio = a2a_g.norm().item() / ref_g.norm().item() if ref_g.norm() > 0 else float("inf")
        print(f"  {name}: PCC={p:.6f}  a2a_norm={a2a_g.norm():.4f}  ref_norm={ref_g.norm():.4f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
