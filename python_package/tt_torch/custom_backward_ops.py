# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch_xla.experimental import stablehlo_custom_call

from .custom_ops import *


def _all_to_all_dispatch_setup_context(ctx, inputs, output):
    input_tensor, expert_indices, expert_mapping, num_devices, cluster_axis = inputs
    D = num_devices
    B = input_tensor.shape[0]
    _, _, S, _ = input_tensor.shape

    expert_ids = expert_indices.squeeze(1).to(torch.int64)  # [B, S, K]
    expert_to_device = expert_mapping[0, 0].to(torch.int64)  # [E, D]
    selected_device_map = torch.gather(
        expert_to_device.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1),
        2,
        expert_ids.unsqueeze(-1).expand(-1, -1, -1, D),
    )  # [B, S, K, D]
    goes_to_device = selected_device_map.amax(dim=2)  # [B, S, D]

    ctx.num_devices = D
    ctx.B = B
    ctx.input_shape = input_tensor.shape
    ctx.save_for_backward(goes_to_device)


def _all_to_all_dispatch_backward(ctx, grad_dispatched, _grad_metadata):
    grad_device = (
        grad_dispatched.device
        if grad_dispatched is not None
        else ctx.saved_tensors[0].device
    )
    if grad_dispatched is None:
        grad_input = torch.zeros(*ctx.input_shape, device=grad_device)
        return grad_input, None, None, None, None

    D = ctx.num_devices
    B = ctx.B
    _, _, S, H = grad_dispatched.shape
    goes_to_device = ctx.saved_tensors[0]  # [B, S, D]

    grad_5d = grad_dispatched.view(1, D, B, S, H)
    mask = (
        goes_to_device.permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(grad_dispatched.dtype)
    )  # [1, D, B, S, 1]
    grad_input = (grad_5d * mask).sum(dim=1).permute(1, 0, 2, 3)
    return grad_input, None, None, None, None


all_to_all_dispatch.register_autograd(
    _all_to_all_dispatch_backward,
    setup_context=_all_to_all_dispatch_setup_context,
)


def _all_to_all_combine_setup_context(ctx, inputs, output):
    (
        input_tensor,
        expert_metadata,
        expert_mapping,
        expert_locals,
        num_devices,
        cluster_axis,
        num_experts_per_tok,
        output_shard_dim,
    ) = inputs
    ctx.save_for_backward(expert_metadata, expert_mapping, expert_locals)
    ctx.num_devices = num_devices
    ctx.cluster_axis = cluster_axis
    ctx.num_experts_per_tok = num_experts_per_tok
    ctx.output_shard_dim = output_shard_dim
    ctx.input_shape = input_tensor.shape
    ctx.setup_device = input_tensor.device


def _all_to_all_combine_backward(ctx, grad_output):
    expert_metadata = ctx.saved_tensors[0]
    expert_mapping = ctx.saved_tensors[1]
    runtime_device = grad_output.device
    K = ctx.num_experts_per_tok

    if ctx.output_shard_dim == 1:
        num_experts, bd, seq, hidden = ctx.input_shape
        B = grad_output.shape[1]
    else:
        num_experts, seq, bd, hidden = ctx.input_shape
        B = grad_output.shape[2]

    total_devices = expert_mapping.shape[-1]
    mesh_cols = total_devices // ctx.num_devices
    expert_to_device = expert_mapping[0, 0].argmax(dim=-1).to(torch.int64)

    if ctx.output_shard_dim == 1:
        grad_input = torch.zeros(
            num_experts,
            bd,
            seq,
            hidden,
            device=runtime_device,
            dtype=grad_output.dtype,
        )
    else:
        grad_input = torch.zeros(
            num_experts,
            seq,
            bd,
            hidden,
            device=runtime_device,
            dtype=grad_output.dtype,
        )

    b_idx = torch.arange(B, device=runtime_device, dtype=torch.int64).view(B, 1)

    for k_idx in range(K):
        expert_ids = expert_metadata[0, :B, :, k_idx].to(torch.int64)
        safe_ids = expert_ids.clamp(0, num_experts - 1)
        rows = expert_to_device[safe_ids] // mesh_cols  # [B, S]
        bd_idx = rows * B + b_idx  # [B, S]

        oh_expert = torch.nn.functional.one_hot(
            safe_ids,
            num_experts,
        ).to(
            grad_output.dtype
        )  # [B, S, E]
        oh_bd = torch.nn.functional.one_hot(
            bd_idx,
            bd,
        ).to(
            grad_output.dtype
        )  # [B, S, BD]

        temp = oh_expert.unsqueeze(-1) * oh_bd.unsqueeze(-2)  # [B, S, E, BD]

        if ctx.output_shard_dim == 1:
            grad_k = grad_output[k_idx]  # [B, S, H]
            grad_input += torch.einsum("bsed,bsh->edsh", temp, grad_k)
        else:
            grad_k = grad_output[k_idx].permute(1, 0, 2)  # [S,B,H]->[B,S,H]
            grad_input += torch.einsum("bsed,bsh->esdh", temp, grad_k)

    return grad_input, None, None, None, None, None, None, None


all_to_all_combine.register_autograd(
    _all_to_all_combine_backward,
    setup_context=_all_to_all_combine_setup_context,
)


def _moe_expert_token_remap_setup_context(ctx, inputs, output):
    topk_tensor, expert_mapping, expert_metadata, reduction_size = inputs
    ctx.topk_shape = topk_tensor.shape
    ctx.topk_dtype = topk_tensor.dtype
    ctx.save_for_backward(expert_metadata)


def _moe_expert_token_remap_backward(ctx, grad_mapping, grad_reduced):
    expert_metadata = ctx.saved_tensors[0]
    grad_device = grad_mapping.device
    if grad_mapping is None and grad_reduced is not None:
        grad_device = grad_reduced.device

    grad_topk = torch.zeros(ctx.topk_shape, dtype=ctx.topk_dtype, device=grad_device)
    if grad_mapping is not None:
        gathered_grad = torch.gather(grad_mapping, dim=-1, index=expert_metadata)
        grad_topk.scatter_(dim=-1, index=expert_metadata, src=gathered_grad)

    return grad_topk, None, None, None


moe_expert_token_remap.register_autograd(
    _moe_expert_token_remap_backward,
    setup_context=_moe_expert_token_remap_setup_context,
)


def _sparse_matmul_setup_context(ctx, inputs, output):
    input_a, input_b, sparsity, nnz, is_input_a_sparse, is_input_b_sparse = inputs
    ctx.save_for_backward(input_a, input_b, sparsity)
    ctx.is_input_a_sparse = is_input_a_sparse
    ctx.is_input_b_sparse = is_input_b_sparse


def _sparse_matmul_backward(ctx, grad_output):
    input_a, input_b, sparsity = ctx.saved_tensors
    is_a_sparse = ctx.is_input_a_sparse
    is_b_sparse = ctx.is_input_b_sparse

    input_b_T = input_b.transpose(-2, -1).contiguous()  # [1, E, N, K]

    if is_a_sparse and is_b_sparse:
        # Forward: [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
        # sparsity: [1, 1, 1, E]

        # grad_a: [1,E,M,N] @ [1,E,N,K] -> [1,E,M,K] (both_sparse)
        grad_a = sparse_matmul(
            grad_output,
            input_b_T,
            sparsity,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
        )

        # grad_b: [1,E,K,M] @ [1,E,M,N] -> [1,E,K,N] (both_sparse)
        grad_b = sparse_matmul(
            input_a.transpose(-2, -1).contiguous(),
            grad_output,
            sparsity,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
        )

    elif not is_a_sparse and is_b_sparse:
        # Forward: [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
        # sparsity: [A, B, 1, E]
        A, B, M, K = input_a.shape
        E = input_b.shape[1]
        N = input_b.shape[-1]
        AB = A * B

        # grad_a: need [AB, E, M, N] @ [1, E, N, K] -> [AB, E, M, K] (a_sparse)
        # then sum over E -> [A, B, M, K]
        grad_3d = grad_output.squeeze(2).reshape(AB, E, M, N)
        sparsity_a = sparsity.reshape(1, 1, AB, E)

        grad_a_full = sparse_matmul(
            grad_3d,
            input_b_T,
            sparsity_a,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        # [AB, E, M, K] -> sum over E -> [A, B, M, K]
        grad_a = grad_a_full.sum(dim=1).reshape(A, B, M, K)

        # grad_b: sum_{ab} sp[ab,e] * input_a[ab,M,K]^T @ grad[ab,e,M,N] -> [1,E,K,N]
        # Fold AB into the M contraction dim so a single matmul replaces matmul+reduce.
        # reduce (stablehlo.reduce) may not be zero-initialised on TT.
        sp_mask = sparsity.reshape(AB, E, 1, 1).to(grad_3d.dtype)
        mg = grad_3d * sp_mask  # [AB, E, M, N]
        a_flat_T = input_a.reshape(AB, M, K).transpose(-2, -1)  # [AB, K, M]
        a_folded = a_flat_T.permute(1, 0, 2).reshape(K, AB * M)  # [K, AB*M]
        mg_folded = mg.permute(1, 0, 2, 3).reshape(E, AB * M, N)  # [E, AB*M, N]
        grad_b = torch.matmul(a_folded, mg_folded).unsqueeze(0)  # [1, E, K, N]

    elif is_a_sparse and not is_b_sparse:
        # Forward: [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
        # sparsity: [1, 1, A, E]

        # grad_a: [A, E, M, N] @ [1, E, N, K] -> [A, E, M, K] (a_sparse)
        grad_a = sparse_matmul(
            grad_output,
            input_b_T,
            sparsity,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )

        # grad_b: sum_a mask[a,e] * input_a[a,e,K,M] @ grad[a,e,M,N] -> [1,E,K,N]
        # Fold A into M to avoid separate reduce.
        mask = sparsity[0, 0]  # [A, E]
        masked_grad = grad_output * mask.unsqueeze(-1).unsqueeze(-1)  # [A, E, M, N]
        input_a_T = input_a.transpose(-2, -1)  # [A, E, K, M]
        A_, E_, K_, M_ = input_a_T.shape
        N_ = masked_grad.shape[3]
        a_T_folded = input_a_T.permute(1, 2, 0, 3).reshape(
            E_, K_, A_ * M_
        )  # [E, K, A*M]
        mg_folded = masked_grad.permute(1, 0, 2, 3).reshape(
            E_, A_ * M_, N_
        )  # [E, A*M, N]
        grad_b = torch.matmul(a_T_folded, mg_folded).unsqueeze(0)  # [1, E, K, N]

    else:
        raise ValueError("Invalid sparse mode")

    return grad_a, grad_b.to(input_b.dtype), None, None, None, None


sparse_matmul.register_autograd(
    _sparse_matmul_backward,
    setup_context=_sparse_matmul_setup_context,
)
