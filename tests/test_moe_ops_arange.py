"""
Arange-based diagnostic test for MoE custom ops.

Uses sequential/arange inputs so values are easy to trace and verify.
Tests each op individually on CPU and TT, printing actual tensor values
for side-by-side comparison.

Usage:
    # CPU only (no device needed):
    cd ~/tt-xla && python3 tests/test_moe_ops_arange.py

    # With TT device:
    cd ~/tt-xla && python3 tests/test_moe_ops_arange.py --device tt
"""

import argparse
import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_package.tt_torch.custom_moe_ops import (
    sparse_matmul,
    all_to_all_dispatch,
    all_to_all_combine,
    moe_expert_token_remap,
)
from python_package.tt_torch.sparse_mlp import build_expert_mapping


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else float("nan")


def print_tensor(name, t, max_elems=64):
    """Print tensor with shape and first few values."""
    flat = t.detach().float().flatten()
    vals = flat[:max_elems].tolist()
    print(f"  {name}: shape={list(t.shape)} dtype={t.dtype}")
    print(f"    values[:min({max_elems},len)]: {[round(v, 4) for v in vals]}")
    if flat.numel() > max_elems:
        print(f"    ... ({flat.numel()} total elements)")


# =============================================================================
# Test 1: sparse_matmul (not_a_sparse, b_sparse) — gate_up_proj case
# =============================================================================
def test_sparse_matmul_gate_up(device="cpu"):
    """
    Test sparse_matmul with not_a_sparse=True, b_sparse=True.
    This is the gate_up_proj matmul: [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
    """
    print(f"\n{'='*60}")
    print(f"TEST: sparse_matmul (gate_up) on {device}")
    print(f"{'='*60}")

    # Small shapes for readability
    A, B, M, K = 2, 1, 2, 3  # input_a: [2, 1, 2, 3]
    E, N = 4, 2               # weight: [1, 4, 3, 2]

    # Arange inputs
    input_a = torch.arange(A * B * M * K, dtype=torch.float32).reshape(A, B, M, K)
    weight = (torch.arange(E * K * N, dtype=torch.float32).reshape(1, E, K, N) + 1) * 0.1

    # Sparsity: experts 0,2 active for batch 0; experts 1,3 active for batch 1
    sparsity = torch.zeros(A, B, 1, E)
    sparsity[0, 0, 0, 0] = 1.0
    sparsity[0, 0, 0, 2] = 1.0
    sparsity[1, 0, 0, 1] = 1.0
    sparsity[1, 0, 0, 3] = 1.0

    if device != "cpu":
        input_a = input_a.to(device)
        weight = weight.to(device)
        sparsity = sparsity.to(device)

    input_a.requires_grad_(True)
    weight.requires_grad_(True)

    # Forward
    out = torch.ops.tt.sparse_matmul(
        input_a, weight, sparsity, nnz=0,
        is_input_a_sparse=False, is_input_b_sparse=True,
    )

    print_tensor("input_a", input_a)
    print_tensor("weight", weight)
    print_tensor("sparsity", sparsity)
    print_tensor("forward output", out)

    # Backward
    grad_out = torch.ones_like(out)
    out.backward(grad_out)

    print_tensor("grad_input_a", input_a.grad)
    print_tensor("grad_weight", weight.grad)

    return {
        "out": out.detach().cpu(),
        "grad_a": input_a.grad.detach().cpu(),
        "grad_w": weight.grad.detach().cpu(),
    }


# =============================================================================
# Test 2: sparse_matmul (a_sparse, not_b_sparse) — down_proj case
# =============================================================================
def test_sparse_matmul_down_proj(device="cpu"):
    """
    Test sparse_matmul with a_sparse=True, b_sparse=False.
    This is the down_proj matmul: [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
    """
    print(f"\n{'='*60}")
    print(f"TEST: sparse_matmul (down_proj) on {device}")
    print(f"{'='*60}")

    A, E, M, K = 2, 4, 2, 3
    N = 2

    input_a = torch.arange(A * E * M * K, dtype=torch.float32).reshape(A, E, M, K) * 0.1
    weight = (torch.arange(E * K * N, dtype=torch.float32).reshape(1, E, K, N) + 1) * 0.1

    # Sparsity: [1, 1, A, E]
    sparsity = torch.zeros(1, 1, A, E)
    sparsity[0, 0, 0, 0] = 1.0
    sparsity[0, 0, 0, 2] = 1.0
    sparsity[0, 0, 1, 1] = 1.0
    sparsity[0, 0, 1, 3] = 1.0

    if device != "cpu":
        input_a = input_a.to(device)
        weight = weight.to(device)
        sparsity = sparsity.to(device)

    input_a.requires_grad_(True)
    weight.requires_grad_(True)

    out = torch.ops.tt.sparse_matmul(
        input_a, weight, sparsity, nnz=0,
        is_input_a_sparse=True, is_input_b_sparse=False,
    )

    print_tensor("input_a", input_a)
    print_tensor("weight", weight)
    print_tensor("sparsity", sparsity)
    print_tensor("forward output", out)

    grad_out = torch.ones_like(out)
    out.backward(grad_out)

    print_tensor("grad_input_a", input_a.grad)
    print_tensor("grad_weight", weight.grad)

    return {
        "out": out.detach().cpu(),
        "grad_a": input_a.grad.detach().cpu(),
        "grad_w": weight.grad.detach().cpu(),
    }


# =============================================================================
# Test 3: all_to_all_dispatch forward + backward
# =============================================================================
def test_dispatch(device="cpu"):
    """
    Test all_to_all_dispatch: [B, 1, S, H] -> [1, BD, S, H]
    """
    print(f"\n{'='*60}")
    print(f"TEST: all_to_all_dispatch on {device}")
    print(f"{'='*60}")

    B, S, H = 2, 4, 3
    K = 2
    D = 2  # num_devices
    E = 8

    # Input: arange
    input_t = torch.arange(B * S * H, dtype=torch.float32).reshape(B, 1, S, H)
    expert_indices = torch.tensor([
        [[[0, 4], [1, 5], [2, 6], [3, 7]]],  # batch 0: experts across devices
        [[[4, 0], [5, 1], [6, 2], [7, 3]]],  # batch 1: reversed
    ], dtype=torch.int64)  # [B, 1, S, K]

    expert_mapping = build_expert_mapping(E, D)  # [1, 1, E, D]

    if device != "cpu":
        input_t = input_t.to(device)
        expert_indices = expert_indices.to(device)
        expert_mapping = expert_mapping.to(device)

    input_t.requires_grad_(True)

    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        input_t, expert_indices, expert_mapping,
        num_devices=D, cluster_axis=0,
    )

    print_tensor("input", input_t)
    print_tensor("expert_indices", expert_indices)
    print_tensor("dispatched", dispatched)
    print_tensor("metadata", metadata)

    # Backward
    grad_dispatched = torch.ones_like(dispatched)
    dispatched.backward(grad_dispatched)

    print_tensor("grad_input", input_t.grad)

    return {
        "dispatched": dispatched.detach().cpu(),
        "metadata": metadata.detach().cpu(),
        "grad_input": input_t.grad.detach().cpu(),
    }


# =============================================================================
# Test 4: all_to_all_combine forward + backward
# =============================================================================
def test_combine(device="cpu"):
    """
    Test all_to_all_combine: [E_local, BD, S, H] -> [K, B, S, H]
    """
    print(f"\n{'='*60}")
    print(f"TEST: all_to_all_combine on {device}")
    print(f"{'='*60}")

    E_local = 4
    B = 2
    D = 2
    BD = B * D
    S = 4
    H = 3
    K = 2

    # Expert outputs: arange
    input_t = torch.arange(E_local * BD * S * H, dtype=torch.float32).reshape(
        E_local, BD, S, H
    )

    # Metadata: which expert each token goes to
    # [1, BD, S, K] — for batch 0: experts 0,1; batch 1: experts 2,3
    metadata = torch.zeros(1, BD, S, K, dtype=torch.int64)
    for bd in range(BD):
        for s in range(S):
            metadata[0, bd, s, 0] = (bd + s) % E_local
            metadata[0, bd, s, 1] = (bd + s + 1) % E_local

    expert_mapping = build_expert_mapping(E_local, D)

    if device != "cpu":
        input_t = input_t.to(device)
        metadata = metadata.to(device)
        expert_mapping = expert_mapping.to(device)

    input_t.requires_grad_(True)

    combined = torch.ops.tt.all_to_all_combine(
        input_t, metadata, expert_mapping,
        num_devices=D, cluster_axis=0, num_experts_per_tok=K,
        output_shard_dim=1,
    )

    print_tensor("input (expert outputs)", input_t)
    print_tensor("metadata", metadata)
    print_tensor("combined", combined)

    # Backward
    grad_combined = torch.ones_like(combined)
    combined.backward(grad_combined)

    print_tensor("grad_input (expert outputs)", input_t.grad)

    return {
        "combined": combined.detach().cpu(),
        "grad_input": input_t.grad.detach().cpu(),
    }


# =============================================================================
# Test 5: Full dispatch -> sparse_matmul -> combine pipeline
# =============================================================================
def test_full_pipeline(device="cpu"):
    """
    Test the full dispatch -> sparse_matmul -> combine pipeline.
    This is what A2aSparseMLP does (simplified).
    """
    print(f"\n{'='*60}")
    print(f"TEST: Full pipeline (dispatch -> sparse_matmul -> combine) on {device}")
    print(f"{'='*60}")

    B, S, H = 2, 4, 8
    E = 8
    K = 2
    D = 2  # num_devices
    inter = 4

    torch.manual_seed(42)

    # Inputs
    hidden = torch.randn(B, S, H) * 0.1
    weight_up = torch.randn(1, E, H, inter) * 0.1
    weight_down = torch.randn(1, E, inter, H) * 0.1

    # Router: pick top-K experts
    router_logits = torch.randn(B * S, E)
    router_scores = torch.softmax(router_logits, -1)
    topk_scores, topk_indices = torch.topk(router_scores, K, -1)
    full_scores = torch.zeros_like(router_scores)
    full_scores.scatter_(1, topk_indices, topk_scores)

    expert_indices = topk_indices.view(B, 1, S, K)
    expert_mapping = build_expert_mapping(E, D)

    if device != "cpu":
        hidden = hidden.to(device)
        weight_up = weight_up.to(device)
        weight_down = weight_down.to(device)
        full_scores = full_scores.to(device)
        expert_indices = expert_indices.to(device)
        expert_mapping = expert_mapping.to(device)
        topk_indices = topk_indices.to(device)
        topk_scores = topk_scores.to(device)

    hidden.requires_grad_(True)
    weight_up.requires_grad_(True)
    weight_down.requires_grad_(True)

    # Step 1: Dispatch
    x = hidden.view(B, 1, S, H)
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, expert_indices, expert_mapping,
        num_devices=D, cluster_axis=0,
    )
    BD = dispatched.shape[1]

    # Step 2: Build sparsity from metadata
    meta_flat = metadata[0].view(BD * S, K)
    sparsity_flat = torch.zeros(BD * S, 1, E, dtype=hidden.dtype, device=hidden.device)
    sparsity_flat.scatter_(
        dim=-1,
        index=meta_flat.unsqueeze(1),
        src=torch.ones_like(meta_flat.unsqueeze(1), dtype=hidden.dtype),
    )
    sparsity = sparsity_flat.view(BD, S, 1, E)

    # Step 3: Up projection (sparse_matmul)
    hidden_4d = dispatched.view(BD, S, 1, H)
    up_out = torch.ops.tt.sparse_matmul(
        hidden_4d, weight_up, sparsity, nnz=0,
        is_input_a_sparse=False, is_input_b_sparse=True,
    )
    # [BD, S, 1, E, 1, inter] -> [BD*S, E, 1, inter]
    up_out = up_out.squeeze(2).view(BD * S, E, 1, inter)

    # Step 4: Activation (simple relu for testing)
    activated = torch.relu(up_out)

    # Step 5: Down projection (sparse_matmul)
    sparsity_down = sparsity.view(1, 1, BD * S, E)
    down_out = torch.ops.tt.sparse_matmul(
        activated, weight_down, sparsity_down, nnz=0,
        is_input_a_sparse=True, is_input_b_sparse=False,
    )
    # [BD*S, E, 1, H] -> [E, BD, S, H]
    down_out = down_out.squeeze(2).view(BD, S, E, H).permute(2, 0, 1, 3)

    # Step 6: Combine
    combined = torch.ops.tt.all_to_all_combine(
        down_out, metadata, expert_mapping,
        num_devices=D, cluster_axis=0, num_experts_per_tok=K,
        output_shard_dim=1,
    )
    # [K, B, S, H]

    # Weighted sum
    topk_w = topk_scores.view(B, S, K).permute(2, 0, 1).unsqueeze(-1)
    if device != "cpu":
        topk_w = topk_w.to(device)
    output = (combined * topk_w).sum(dim=0)  # [B, S, H]

    print_tensor("output", output)

    # Backward
    grad_out = torch.ones_like(output)
    output.backward(grad_out)

    print_tensor("grad_hidden", hidden.grad)
    print_tensor("grad_weight_up", weight_up.grad)
    print_tensor("grad_weight_down", weight_down.grad)

    return {
        "output": output.detach().cpu(),
        "grad_hidden": hidden.grad.detach().cpu(),
        "grad_w_up": weight_up.grad.detach().cpu(),
        "grad_w_down": weight_down.grad.detach().cpu(),
    }


def compare_results(cpu_results, tt_results, test_name):
    """Compare CPU vs TT results and print PCC."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {test_name}")
    print(f"{'='*60}")
    for key in cpu_results:
        cpu_t = cpu_results[key]
        tt_t = tt_results[key]
        p = pcc(cpu_t, tt_t)
        max_diff = (cpu_t.float() - tt_t.float()).abs().max().item()
        print(f"  {key}: PCC={p:.6f}  max_diff={max_diff:.6f}  "
              f"cpu_norm={cpu_t.float().norm():.4f}  tt_norm={tt_t.float().norm():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device: cpu or tt")
    args = parser.parse_args()

    print(f"Running MoE ops arange tests on device={args.device}")

    # Always run CPU
    cpu_results = {}
    cpu_results["gate_up"] = test_sparse_matmul_gate_up("cpu")
    cpu_results["down_proj"] = test_sparse_matmul_down_proj("cpu")
    cpu_results["dispatch"] = test_dispatch("cpu")
    cpu_results["combine"] = test_combine("cpu")
    cpu_results["pipeline"] = test_full_pipeline("cpu")

    if args.device != "cpu":
        # Run on TT device and compare
        tt_results = {}
        tt_results["gate_up"] = test_sparse_matmul_gate_up(args.device)
        tt_results["down_proj"] = test_sparse_matmul_down_proj(args.device)
        tt_results["dispatch"] = test_dispatch(args.device)
        tt_results["combine"] = test_combine(args.device)
        tt_results["pipeline"] = test_full_pipeline(args.device)

        for test_name in cpu_results:
            compare_results(cpu_results[test_name], tt_results[test_name], test_name)

    print("\nDone.")
