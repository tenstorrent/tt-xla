"""
Targeted test: sparse_matmul backward on CPU only.
Compares custom op backward vs autograd through dense matmul.

Usage:
    cd ~/tt-xla && python3 tests/test_sparse_matmul_bwd.py
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_package.tt_torch.custom_moe_ops import sparse_matmul


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else float("nan")


def test_gate_up_backward():
    """Test not-a-sparse, b-sparse case (gate_up_proj matmul)."""
    print("=== gate_up sparse_matmul backward (not_a_sparse, b_sparse) ===")
    torch.manual_seed(42)

    A, B, M, K = 4, 1, 32, 64   # input: [A,B,M,K]
    E, N = 32, 128               # weight: [1,E,K,N]

    input_a = torch.randn(A, B, M, K, requires_grad=True)
    weight = torch.randn(1, E, K, N, requires_grad=True)

    # Sparsity: only 4 experts active per (a,b) pair
    sparsity = torch.zeros(A, B, 1, E)
    for a in range(A):
        for b in range(B):
            active = torch.randperm(E)[:4]
            sparsity[a, b, 0, active] = 1.0

    # Custom op forward+backward
    out_custom = torch.ops.tt.sparse_matmul(
        input_a, weight, sparsity, nnz=0,
        is_input_a_sparse=False, is_input_b_sparse=True
    )
    loss_custom = out_custom.sum()
    loss_custom.backward()
    grad_a_custom = input_a.grad.clone()
    grad_w_custom = weight.grad.clone()

    # Reference: dense matmul + masking
    input_a2 = input_a.detach().requires_grad_(True)
    weight2 = weight.detach().requires_grad_(True)
    # [A,B,M,K] @ [1,E,K,N] -> broadcast -> [A,B,E,M,N]
    out_ref = torch.einsum('abmk,dekn->abdemn', input_a2, weight2)
    # Apply sparsity mask
    mask = sparsity.unsqueeze(-1).unsqueeze(-1)  # [A,B,1,E,1,1]
    out_ref = out_ref * mask
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grad_a_ref = input_a2.grad.clone()
    grad_w_ref = weight2.grad.clone()

    print(f"  Forward PCC:  {pcc(out_custom, out_ref):.6f}")
    print(f"  grad_a PCC:   {pcc(grad_a_custom, grad_a_ref):.6f}")
    print(f"  grad_w PCC:   {pcc(grad_w_custom, grad_w_ref):.6f}")
    print(f"  grad_a norms: custom={grad_a_custom.norm():.4f} ref={grad_a_ref.norm():.4f}")
    print(f"  grad_w norms: custom={grad_w_custom.norm():.4f} ref={grad_w_ref.norm():.4f}")
    print(f"  grad_a[:2,:2,:2,:2] custom:\n    {grad_a_custom[:2,:2,:2,:2]}")
    print(f"  grad_a[:2,:2,:2,:2] ref:\n    {grad_a_ref[:2,:2,:2,:2]}")


def test_down_proj_backward():
    """Test a-sparse, not-b-sparse case (down_proj matmul)."""
    print("\n=== down_proj sparse_matmul backward (a_sparse, not_b_sparse) ===")
    torch.manual_seed(42)

    A, E, M, K = 4, 32, 32, 64   # input: [A,E,M,K]
    N = 64                         # weight: [1,E,K,N]

    # Sparsity: [1,1,A,E]
    sparsity = torch.zeros(1, 1, A, E)
    for a in range(A):
        active = torch.randperm(E)[:4]
        sparsity[0, 0, a, active] = 1.0

    input_a = torch.randn(A, E, M, K, requires_grad=True)
    weight = torch.randn(1, E, K, N, requires_grad=True)

    # Zero out non-active experts in input (simulating what TT does)
    with torch.no_grad():
        for a in range(A):
            for e in range(E):
                if sparsity[0, 0, a, e] == 0:
                    input_a.data[a, e] = 0

    # Custom op
    out_custom = torch.ops.tt.sparse_matmul(
        input_a, weight, sparsity, nnz=0,
        is_input_a_sparse=True, is_input_b_sparse=False
    )
    loss_custom = out_custom.sum()
    loss_custom.backward()
    grad_a_custom = input_a.grad.clone()
    grad_w_custom = weight.grad.clone()

    # Reference: dense matmul + masking
    input_a2 = input_a.detach().requires_grad_(True)
    weight2 = weight.detach().requires_grad_(True)
    out_ref = torch.einsum('aemk,dekn->aemn', input_a2, weight2)
    mask = sparsity.permute(2, 3, 0, 1)  # [A,E,1,1]
    out_ref = out_ref * mask
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grad_a_ref = input_a2.grad.clone()
    grad_w_ref = weight2.grad.clone()

    print(f"  Forward PCC:  {pcc(out_custom, out_ref):.6f}")
    print(f"  grad_a PCC:   {pcc(grad_a_custom, grad_a_ref):.6f}")
    print(f"  grad_w PCC:   {pcc(grad_w_custom, grad_w_ref):.6f}")
    print(f"  grad_a norms: custom={grad_a_custom.norm():.4f} ref={grad_a_ref.norm():.4f}")
    print(f"  grad_w norms: custom={grad_w_custom.norm():.4f} ref={grad_w_ref.norm():.4f}")


def test_down_proj_with_nonzero_inactive():
    """Test a-sparse backward when inactive experts have NON-ZERO input (CPU behavior).
    This simulates what happens on CPU: gate_up computes for all experts, then
    bias is added, then activation produces non-zero values for inactive experts.
    """
    print("\n=== down_proj backward WITH NON-ZERO INACTIVE experts ===")
    torch.manual_seed(42)

    A, E, M, K = 4, 32, 32, 64
    N = 64

    sparsity = torch.zeros(1, 1, A, E)
    for a in range(A):
        active = torch.randperm(E)[:4]
        sparsity[0, 0, a, active] = 1.0

    # Input has NON-ZERO values for ALL experts (simulating bias+activation on CPU)
    input_a = torch.randn(A, E, M, K, requires_grad=True)

    weight = torch.randn(1, E, K, N, requires_grad=True)

    # Custom op: sparsity masks output in forward, masks grad in backward
    out_custom = torch.ops.tt.sparse_matmul(
        input_a, weight, sparsity, nnz=0,
        is_input_a_sparse=True, is_input_b_sparse=False
    )
    loss_custom = out_custom.sum()
    loss_custom.backward()
    grad_a_custom = input_a.grad.clone()
    grad_w_custom = weight.grad.clone()

    # Same but with zeroed inactive experts (simulating TT behavior)
    input_a_tt = input_a.detach().clone()
    for a in range(A):
        for e in range(E):
            if sparsity[0, 0, a, e] == 0:
                input_a_tt[a, e] = 0
    input_a_tt.requires_grad_(True)
    weight_tt = weight.detach().requires_grad_(True)

    out_tt = torch.ops.tt.sparse_matmul(
        input_a_tt, weight_tt, sparsity, nnz=0,
        is_input_a_sparse=True, is_input_b_sparse=False
    )
    loss_tt = out_tt.sum()
    loss_tt.backward()
    grad_a_tt = input_a_tt.grad.clone()
    grad_w_tt = weight_tt.grad.clone()

    print(f"  Forward PCC (should be 1.0):  {pcc(out_custom, out_tt):.6f}")
    print(f"  grad_w PCC (cpu vs tt-like):   {pcc(grad_w_custom, grad_w_tt):.6f}")
    print(f"  grad_w norms: cpu={grad_w_custom.norm():.4f} tt-like={grad_w_tt.norm():.4f}")
    print(f"  grad_a PCC (cpu vs tt-like):   {pcc(grad_a_custom, grad_a_tt):.6f}")
    print(f"  grad_a norms: cpu={grad_a_custom.norm():.4f} tt-like={grad_a_tt.norm():.4f}")


def test_full_mlp_backward():
    """Test A2aSparseMLP forward+backward on CPU: compare against reference dense."""
    print("\n=== Full A2aSparseMLP CPU backward vs reference dense ===")
    from python_package.tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping

    torch.manual_seed(42)
    E, K, H, inter = 32, 4, 64, 64
    B, S, D = 1, 32, 4

    class FakeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(E, H) * 0.01)
            self.bias = nn.Parameter(torch.zeros(E))
        def forward(self, x):
            flat = x.view(-1, H)
            logits = flat @ self.weight.T + self.bias
            scores = torch.softmax(logits, -1)
            topk_s, topk_i = torch.topk(scores, K, -1)
            full = torch.zeros_like(scores)
            full.scatter_(1, topk_i, topk_s)
            return full, topk_i

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(E, H, inter*2) * 0.01)
            self.gate_up_proj_bias = nn.Parameter(torch.zeros(E, inter*2))
            self.down_proj = nn.Parameter(torch.randn(E, inter, H) * 0.01)
            self.down_proj_bias = nn.Parameter(torch.zeros(E, H))
            self.alpha = 1.702
            self.limit = 7.0

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter()
            self.experts = FakeExperts()

    fake_mlp = FakeMLP()
    a2a = A2aSparseMLP(fake_mlp, E, K, num_devices=D*8, dispatch_devices=D, cluster_axis=0)

    hidden = torch.randn(B, S, H) * 0.1
    grad_out = torch.randn(B, S, H) * 0.01

    # A2aSparseMLP backward
    hidden1 = hidden.clone().requires_grad_(False)
    out1, _ = a2a(hidden1)
    out1.backward(grad_out.clone())

    a2a_grads = {}
    for n, p in a2a.named_parameters():
        if p.grad is not None:
            a2a_grads[n] = p.grad.clone()

    # Reference dense backward (standard PyTorch ops, no custom ops)
    a2a.zero_grad()
    hidden2 = hidden.clone().requires_grad_(True)
    router_scores, router_indices = a2a.router(hidden2)
    flat = hidden2.view(B*S, H)
    gate_up_w = a2a.experts.gate_up_proj
    gate_up_b = a2a.experts.gate_up_proj_bias
    down_w = a2a.experts.down_proj
    down_b = a2a.experts.down_proj_bias
    alpha, limit = a2a.alpha, a2a.limit

    gate_up_out = torch.einsum('bh,ehn->ben', flat, gate_up_w) + gate_up_b
    gate_out = gate_up_out[..., :inter]
    up_out = gate_up_out[..., inter:]
    gate_out = gate_out.clamp(max=limit)
    up_out = up_out.clamp(-limit, limit)
    glu = gate_out * torch.sigmoid(gate_out * alpha)
    activated = (up_out + 1) * glu

    down_out = torch.einsum('bei,eih->beh', activated, down_w) + down_b
    topk_weights = torch.gather(router_scores, -1, router_indices)
    expert_outputs = torch.gather(down_out, 1, router_indices.unsqueeze(-1).expand(-1, -1, H))
    output = (expert_outputs * topk_weights.unsqueeze(-1)).sum(1).view(B, S, H)
    output.backward(grad_out.clone())

    ref_grads = {
        "experts.gate_up_proj": gate_up_w.grad,
        "experts.gate_up_proj_bias": gate_up_b.grad,
        "experts.down_proj": down_w.grad,
        "experts.down_proj_bias": down_b.grad,
        "router.weight": a2a.router.weight.grad,
        "router.bias": a2a.router.bias.grad,
    }

    print(f"  Forward PCC: {pcc(out1, output):.6f}")
    print(f"  Forward norms: a2a={out1.norm():.4f} ref={output.norm():.4f}")
    for name in sorted(a2a_grads.keys()):
        if name in ref_grads and ref_grads[name] is not None:
            p = pcc(a2a_grads[name], ref_grads[name])
            an = a2a_grads[name].norm().item()
            rn = ref_grads[name].norm().item()
            ratio = an / rn if rn > 0 else float("inf")
            print(f"  {name}: PCC={p:.6f}  a2a_norm={an:.4f}  ref_norm={rn:.4f}  ratio={ratio:.3f}")
        else:
            print(f"  {name}: missing in ref")


if __name__ == "__main__":
    test_gate_up_backward()
    test_down_proj_backward()
    test_down_proj_with_nonzero_inactive()
    test_full_mlp_backward()
