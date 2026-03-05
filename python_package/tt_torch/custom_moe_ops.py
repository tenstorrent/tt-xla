# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MoE (Mixture of Experts) custom operations for sparse computation.

Contains: sparse_matmul, all_to_all_dispatch, all_to_all_combine,
moe_expert_token_remap.
"""

from typing import Optional

import torch
from torch_xla.experimental import stablehlo_custom_call


# =============================================================================
# sparse_matmul
# =============================================================================


@torch.library.custom_op(
    "tt::sparse_matmul", mutates_args=[], device_types=["xla", "cpu"]
)
def sparse_matmul(
    input_tensor_a: torch.Tensor,
    input_tensor_b: torch.Tensor,
    sparsity: torch.Tensor,
    nnz: int = None,
    is_input_a_sparse: bool = False,
    is_input_b_sparse: bool = True,
) -> torch.Tensor:
    """
    Sparse matrix multiplication for MoE (Mixture of Experts) models.

    This operation performs matrix multiplication where computation is skipped
    for sparse (zero) blocks based on the sparsity tensor.

    Args:
        input_tensor_a: First input tensor. Shape depends on sparse mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, E, M, K]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, M, K]
            - is_input_a_sparse=True, is_input_b_sparse=False: [A, E, M, K]
        input_tensor_b: Second input tensor (expert weights). Shape:
            - [1, E, K, N] for all modes
        sparsity: Sparsity mask tensor (bfloat16, ROW_MAJOR). Shape depends on mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, 1, 1, E]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, 1, E]
            - is_input_a_sparse=True, is_input_b_sparse=False: [1, 1, A, E]
        nnz: Number of non-zero elements in sparsity tensor. If None, inferred at runtime.
        is_input_a_sparse: Whether input_tensor_a is sparse.
        is_input_b_sparse: Whether input_tensor_b is sparse.

    Returns:
        Output tensor with sparse results. Shape depends on mode:
            - is_input_a_sparse=True, is_input_b_sparse=True: [1, E, M, N]
            - is_input_a_sparse=False, is_input_b_sparse=True: [A, B, 1, E, M, N]
            - is_input_a_sparse=True, is_input_b_sparse=False: [A, E, M, N]
    """
    device = input_tensor_a.device

    if device.type == "xla":
        # Use einsum instead of stablehlo_custom_call for the XLA path.
        input_b_casted = input_tensor_b.to(input_tensor_a.dtype)
        ib0 = input_b_casted[0]  # [E, K, N]

        if is_input_a_sparse and is_input_b_sparse:
            # [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
            sp_mask = sparsity[0, 0, 0]  # [E]
            output = (
                torch.einsum('emk,ekn->emn', input_tensor_a[0], ib0)
                * sp_mask.unsqueeze(-1).unsqueeze(-1)
            ).unsqueeze(0)

        elif not is_input_a_sparse and is_input_b_sparse:
            # [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
            A, B, M, K = input_tensor_a.shape
            E = input_tensor_b.shape[1]
            sp_mask = sparsity[:, :, 0, :]  # [A, B, E]
            # einsum: [AB, M, K] @ [E, K, N] -> [AB, E, M, N]
            a_flat = input_tensor_a.reshape(A * B, M, K)
            out_flat = torch.einsum('amk,ekn->aemn', a_flat, ib0)
            out_flat = out_flat * sp_mask.reshape(A * B, E, 1, 1)
            output = out_flat.reshape(A, B, E, M, -1).unsqueeze(2)
            # [A, B, 1, E, M, N]

        elif is_input_a_sparse and not is_input_b_sparse:
            # [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
            mask = sparsity[0, 0]  # [A, E]
            output = (
                torch.einsum('aemk,ekn->aemn', input_tensor_a, ib0)
                * mask.unsqueeze(-1).unsqueeze(-1)
            )

        else:
            raise ValueError(
                "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
            )

        return output

    elif device.type == "cpu":
        # CPU fallback: loop over experts to avoid broadcasting weights
        # across large batch dimensions (can exceed 1TB for D=8, E=32).
        input_b_casted = input_tensor_b.to(input_tensor_a.dtype)

        if is_input_a_sparse and is_input_b_sparse:
            # [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
            E = input_tensor_b.shape[1]
            N = input_tensor_b.shape[-1]
            M = input_tensor_a.shape[2]
            output = torch.zeros(1, E, M, N, dtype=input_tensor_a.dtype, device=device)
            for e in range(E):
                if sparsity[0, 0, 0, e] > 0:
                    output[0, e] = torch.matmul(
                        input_tensor_a[0, e], input_b_casted[0, e]
                    )
            return output

        elif not is_input_a_sparse and is_input_b_sparse:
            # [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
            A, B, M, K = input_tensor_a.shape
            E = input_tensor_b.shape[1]
            N = input_tensor_b.shape[-1]
            output = torch.zeros(
                A, B, 1, E, M, N, dtype=input_tensor_a.dtype, device=device
            )
            for e in range(E):
                mask_e = sparsity[:, :, 0, e]  # [A, B]
                if mask_e.any():
                    # [A, B, M, K] @ [K, N] -> [A, B, M, N]
                    out_e = torch.matmul(input_tensor_a, input_b_casted[0, e])
                    output[:, :, 0, e, :, :] = out_e * mask_e.unsqueeze(-1).unsqueeze(
                        -1
                    )
            return output

        elif is_input_a_sparse and not is_input_b_sparse:
            # [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
            A = input_tensor_a.shape[0]
            E = input_tensor_b.shape[1]
            M = input_tensor_a.shape[2]
            N = input_tensor_b.shape[-1]
            output = torch.zeros(A, E, M, N, dtype=input_tensor_a.dtype, device=device)
            for e in range(E):
                mask_e = sparsity[0, 0, :, e]  # [A]
                if mask_e.any():
                    # [A, M, K] @ [K, N] -> [A, M, N]
                    out_e = torch.matmul(input_tensor_a[:, e], input_b_casted[0, e])
                    output[:, e] = out_e * mask_e.unsqueeze(-1).unsqueeze(-1)
            return output

        else:
            raise ValueError(
                "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
            )
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@sparse_matmul.register_fake
def sparse_matmul_fake(
    input_tensor_a: torch.Tensor,
    input_tensor_b: torch.Tensor,
    sparsity: torch.Tensor,
    nnz: int = None,
    is_input_a_sparse: bool = False,
    is_input_b_sparse: bool = True,
) -> torch.Tensor:
    """FakeTensor implementation of sparse_matmul for torch dynamo tracing."""
    if is_input_a_sparse and is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = input_tensor_b.shape[-1]
    elif not is_input_a_sparse and is_input_b_sparse:
        A, B, M, K = input_tensor_a.shape
        E = input_tensor_b.shape[1]
        N = input_tensor_b.shape[-1]
        output_shape = [A, B, 1, E, M, N]
    elif is_input_a_sparse and not is_input_b_sparse:
        output_shape = list(input_tensor_a.shape)
        output_shape[-1] = input_tensor_b.shape[-1]
    else:
        raise ValueError(
            "Invalid sparse mode: both is_input_a_sparse and is_input_b_sparse cannot be False"
        )

    return torch.zeros(
        output_shape, dtype=input_tensor_a.dtype, device=input_tensor_a.device
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

    # Use input_b[0] directly (shape [E, K, N]) to avoid transpose, which produces
    # non-standard XLA memory layout (result_layout=[2,3,1,0]) that the tt.sparse_matmul
    # kernel misreads. XLA dot_general (einsum) is layout-agnostic and handles this correctly.
    ib0 = input_b[0]  # [E, K, N]

    if is_a_sparse and is_b_sparse:
        # Forward: [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
        # sparsity: [1, 1, 1, E]
        sp_mask = sparsity[0, 0, 0]  # [E]

        # grad_a[0, e, m, k] = sp_mask[e] * sum_n grad_output[0,e,m,n] * input_b[0,e,k,n]
        grad_a = (
            torch.einsum('emn,ekn->emk', grad_output[0], ib0)
            * sp_mask.unsqueeze(-1).unsqueeze(-1)
        ).unsqueeze(0)

        # grad_b[0, e, k, n] = sp_mask[e] * sum_m input_a[0,e,m,k] * grad_output[0,e,m,n]
        grad_b = (
            torch.einsum('emk,emn->ekn', input_a[0], grad_output[0])
            * sp_mask.unsqueeze(-1).unsqueeze(-1)
        ).unsqueeze(0)

    elif not is_a_sparse and is_b_sparse:
        # Forward: [A, B, M, K] @ [1, E, K, N] -> [A, B, 1, E, M, N]
        # sparsity: [A, B, 1, E]
        A, B, M, K = input_a.shape
        E = input_b.shape[1]
        N = input_b.shape[-1]
        AB = A * B

        grad_sq = grad_output.squeeze(2)  # [A, B, E, M, N]
        grad_3d = grad_sq.reshape(AB, E, M, N)

        # grad_a[ab, e, m, k] = sp_mask[ab,e] * sum_n grad_3d[ab,e,m,n] * ib0[e,k,n]
        sp_mask = sparsity.reshape(AB, E, 1, 1).to(grad_3d.dtype)
        grad_a_full = torch.einsum('bemn,ekn->bemk', grad_3d, ib0) * sp_mask
        grad_a = grad_a_full.sum(dim=1).reshape(A, B, M, K)

        # grad_b via einsum — avoid transpose which produces non-standard XLA layouts
        a_flat = input_a.reshape(AB, M, K)
        mg = grad_3d * sp_mask
        grad_b = torch.einsum('bmk,bemn->ekn', a_flat, mg).unsqueeze(0)

    elif is_a_sparse and not is_b_sparse:
        # Forward: [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
        # sparsity: [1, 1, A, E]
        mask = sparsity[0, 0]  # [A, E]

        # grad_a[a, e, m, k] = mask[a,e] * sum_n grad_output[a,e,m,n] * ib0[e,k,n]
        grad_a = (
            torch.einsum('aemn,ekn->aemk', grad_output, ib0)
            * mask.unsqueeze(-1).unsqueeze(-1)
        )

        # grad_b via einsum — avoid transpose which produces non-standard XLA layouts
        masked_grad = grad_output * mask.unsqueeze(-1).unsqueeze(-1)
        grad_b = torch.einsum('aemk,aemn->ekn', input_a, masked_grad).unsqueeze(0)

    else:
        raise ValueError("Invalid sparse mode")

    return grad_a, grad_b.to(input_b.dtype), None, None, None, None


sparse_matmul.register_autograd(
    _sparse_matmul_backward,
    setup_context=_sparse_matmul_setup_context,
)


# =============================================================================
# all_to_all_dispatch
# =============================================================================


@torch.library.custom_op(
    "tt::all_to_all_dispatch", mutates_args=[], device_types=["xla", "cpu"]
)
def all_to_all_dispatch(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to devices holding their selected experts.

    Selectively routes tokens based on expert_indices and expert_mapping,
    sending each token only to devices that hold its selected experts.

    Args:
        input_tensor: Input tokens [B, 1, S, H], bfloat16
        expert_indices: Selected expert IDs per token [B, 1, S, K], int64
        expert_mapping: One-hot expert-to-device mapping [1, 1, E, D], int64
        num_devices: Number of devices along dispatch axis (D)
        cluster_axis: Mesh axis to dispatch along (0=rows, 1=cols)

    Returns:
        dispatched_tokens: [1, B*D, S, H] sparsely populated tokens
        expert_metadata: [1, B*D, S, K] all-gathered expert indices
    """
    device = input_tensor.device
    B, _, S, H = input_tensor.shape
    K = expert_indices.shape[-1]

    if device.type == "xla":
        BD = B * num_devices
        output_shapes = [[1, BD, S, H], [1, BD, S, K]]
        output_dtypes = [input_tensor.dtype, expert_indices.dtype]

        frontend_attributes = {
            "num_devices": str(num_devices),
            "cluster_axis": str(cluster_axis),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor, expert_indices, expert_mapping],
            "tt.all_to_all_dispatch",
            output_shapes,
            output_dtypes,
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # CPU fallback: replicate tokens D times (same as TT all-to-all dispatch).
        x = input_tensor.permute(1, 0, 2, 3)  # [1, B, S, H]
        m = expert_indices.permute(1, 0, 2, 3)  # [1, B, S, K]
        if num_devices > 1:
            x = x.repeat(1, num_devices, 1, 1)  # [1, B*D, S, H]
            m = m.repeat(1, num_devices, 1, 1)  # [1, B*D, S, K]
        return x.clone(), m.clone()

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@all_to_all_dispatch.register_fake
def all_to_all_dispatch_fake(
    input_tensor: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, _, S, H = input_tensor.shape
    K = expert_indices.shape[-1]
    BD = B * num_devices

    dispatched = torch.zeros(
        [1, BD, S, H], dtype=input_tensor.dtype, device=input_tensor.device
    )
    metadata = torch.zeros(
        [1, BD, S, K], dtype=expert_indices.dtype, device=expert_indices.device
    )
    return dispatched, metadata


def _all_to_all_dispatch_setup_context(ctx, inputs, output):
    input_tensor, expert_indices, expert_mapping, num_devices, cluster_axis = inputs
    dispatched, metadata = output
    ctx.save_for_backward(expert_mapping, metadata, expert_indices)
    ctx.num_devices = num_devices
    ctx.cluster_axis = cluster_axis
    ctx.input_shape = input_tensor.shape
    ctx.K = expert_indices.shape[-1]


def _all_to_all_dispatch_backward(ctx, grad_dispatched, grad_metadata):
    expert_mapping, metadata, expert_indices = ctx.saved_tensors
    num_devices = ctx.num_devices
    B = ctx.input_shape[0]
    K = ctx.K

    # Dispatch forward replicates tokens D times: [B,1,S,H] → [1,B*D,S,H].
    # Backward: sum the D copies back to [B,1,S,H].
    # Same logic for both CPU and XLA — the sum is a standard op that
    # GSPMD can partition as reduce-scatter along the dispatch axis.
    if num_devices > 1:
        S = grad_dispatched.shape[2]
        H = grad_dispatched.shape[3]
        grad = grad_dispatched.view(1, num_devices, B, S, H).sum(dim=1)
    else:
        grad = grad_dispatched
    grad_input = grad.permute(1, 0, 2, 3)

    return grad_input, None, None, None, None


all_to_all_dispatch.register_autograd(
    _all_to_all_dispatch_backward,
    setup_context=_all_to_all_dispatch_setup_context,
)


# =============================================================================
# all_to_all_combine
# =============================================================================


@torch.library.custom_op(
    "tt::all_to_all_combine", mutates_args=[], device_types=["xla", "cpu"]
)
def all_to_all_combine(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
    num_experts_per_tok: int = 2,
    output_shard_dim: int = 1,
    expert_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combine expert outputs back to original token positions.

    Inverse of dispatch: gathers expert computation results from all devices
    and restores tokens to their original device and order.

    Args:
        input_tensor: Expert outputs, bfloat16. Shape depends on output_shard_dim:
            - output_shard_dim=1: [E_local, B*D, S, H] (default)
            - output_shard_dim=2: [E_local, S, B*D, H] (decode-optimized, avoids tile waste on S=1)
        expert_metadata: Routing metadata from dispatch [1, B*D, S, K], int64
        expert_mapping: One-hot expert-to-device mapping [1, 1, E, D], int64
        num_devices: Number of devices along dispatch axis (D)
        cluster_axis: Mesh axis to combine along (0=rows, 1=cols)
        num_experts_per_tok: Number of selected experts per token (K)
        output_shard_dim: Dimension index for the BD shard dimension (1 or 2).
            Use 2 for decode to place BD on dim -2 and avoid tile padding on S=1.
        expert_indices: Unused, kept for API compatibility.

    Returns:
        combined: Shape depends on output_shard_dim:
            - output_shard_dim=1: [K, B, S, H]
            - output_shard_dim=2: [K, S, B, H]
    """
    device = input_tensor.device
    K = num_experts_per_tok

    if output_shard_dim == 1:
        E_local, BD, S, H = input_tensor.shape
    elif output_shard_dim == 2:
        E_local, S, BD, H = input_tensor.shape
    else:
        raise ValueError(f"output_shard_dim must be 1 or 2, got {output_shard_dim}")

    B = BD // num_devices

    if device.type == "xla":
        if output_shard_dim == 1:
            output_shape = [K, B, S, H]
        else:
            output_shape = [K, S, B, H]

        frontend_attributes = {
            "num_devices": str(num_devices),
            "cluster_axis": str(cluster_axis),
            "num_experts_per_tok": str(K),
            "output_shard_dim": str(output_shard_dim),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor, expert_metadata, expert_mapping],
            "tt.all_to_all_combine",
            [output_shape],
            [input_tensor.dtype],
            frontend_attributes=frontend_attributes,
        )

    elif device.type == "cpu":
        # CPU fallback: gather expert outputs from replicated BD positions.
        metadata_indices = expert_metadata[0]  # [BD, S, K]

        if output_shard_dim == 1:
            combined = torch.zeros(
                K, B, S, H, dtype=input_tensor.dtype, device=device
            )
            for b in range(B):
                for s in range(S):
                    for k in range(K):
                        expert_id = metadata_indices[b, s, k].item()
                        if 0 <= expert_id < E_local:
                            combined[k, b, s, :] = input_tensor[
                                expert_id, b, s, :
                            ]
        else:
            combined = torch.zeros(
                K, S, B, H, dtype=input_tensor.dtype, device=device
            )
            for b in range(B):
                for s in range(S):
                    for k in range(K):
                        expert_id = metadata_indices[b, s, k].item()
                        if 0 <= expert_id < E_local:
                            combined[k, s, b, :] = input_tensor[
                                expert_id, s, b, :
                            ]

        return combined

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


@all_to_all_combine.register_fake
def all_to_all_combine_fake(
    input_tensor: torch.Tensor,
    expert_metadata: torch.Tensor,
    expert_mapping: torch.Tensor,
    num_devices: int = 1,
    cluster_axis: int = 0,
    num_experts_per_tok: int = 2,
    output_shard_dim: int = 1,
    expert_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    K = num_experts_per_tok

    if output_shard_dim == 1:
        _, BD, S, H = input_tensor.shape
        B = BD // num_devices
        return torch.zeros(
            [K, B, S, H], dtype=input_tensor.dtype, device=input_tensor.device
        )
    else:
        _, S, BD, H = input_tensor.shape
        B = BD // num_devices
        return torch.zeros(
            [K, S, B, H], dtype=input_tensor.dtype, device=input_tensor.device
        )


def _all_to_all_combine_setup_context(ctx, inputs, output):
    (
        input_tensor,
        expert_metadata,
        expert_mapping,
        num_devices,
        cluster_axis,
        num_experts_per_tok,
        output_shard_dim,
        expert_indices,
    ) = inputs
    if expert_indices is not None:
        ctx.save_for_backward(expert_metadata, expert_mapping, expert_indices)
        ctx.has_expert_indices = True
    else:
        ctx.save_for_backward(expert_metadata, expert_mapping)
        ctx.has_expert_indices = False
    ctx.num_devices = num_devices
    ctx.cluster_axis = cluster_axis
    ctx.num_experts_per_tok = num_experts_per_tok
    ctx.output_shard_dim = output_shard_dim
    ctx.input_shape = input_tensor.shape


def _all_to_all_combine_backward(ctx, grad_output):
    if ctx.has_expert_indices:
        expert_metadata, expert_mapping, expert_indices = ctx.saved_tensors
    else:
        expert_metadata, expert_mapping = ctx.saved_tensors
        expert_indices = None
    K = ctx.num_experts_per_tok
    num_devices = ctx.num_devices
    output_shard_dim = ctx.output_shard_dim
    E_local = ctx.input_shape[0]

    if output_shard_dim == 1:
        _, BD, S, H = ctx.input_shape
        B = grad_output.shape[1]
    else:
        _, S, BD, H = ctx.input_shape
        B = grad_output.shape[2]

    if grad_output.device.type != "xla" or num_devices <= 1:
        # CPU fallback: mirror TT hardware behavior where each expert's gradient
        # is routed to the row (dispatch batch) where that expert lives.
        # expert_mapping[0,0] is [E, D_total] one-hot; derive each expert's row.
        D_total = expert_mapping.shape[3]
        cols = max(D_total // num_devices, 1)
        device_ids = expert_mapping[0, 0].argmax(dim=-1)  # [E]
        expert_rows = device_ids // cols  # [E] → row index in [0, num_devices)

        meta = expert_metadata[0]  # [BD, S, K]
        if output_shard_dim == 1:
            grad_input = torch.zeros(
                ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device
            )
            flat_grad = grad_input.view(E_local * BD, S, H)  # [E*BD, S, H]
            for k in range(K):
                for b in range(B):
                    e_ids = meta[b, :, k]  # [S]
                    grad_vals = grad_output[k, b]  # [S, H]
                    rows = expert_rows[e_ids]  # [S] → row for each expert
                    bd = b * num_devices + rows  # [S] → BD index
                    flat_idx = e_ids * BD + bd  # [S] in [0, E*BD)
                    flat_grad.scatter_add_(
                        0,
                        flat_idx.view(1, S, 1).expand(1, S, H),
                        grad_vals.unsqueeze(0),
                    )
        else:
            flat_accum = torch.zeros(
                E_local * BD, S, H, dtype=grad_output.dtype, device=grad_output.device
            )
            for k in range(K):
                for b in range(B):
                    e_ids = meta[b, :, k]  # [S]
                    grad_vals = grad_output[k, :, b, :]  # [S, H]
                    rows = expert_rows[e_ids]
                    bd = b * num_devices + rows
                    flat_idx = e_ids * BD + bd
                    flat_accum.scatter_add_(
                        0,
                        flat_idx.view(1, S, 1).expand(1, S, H),
                        grad_vals.unsqueeze(0),
                    )
            grad_input = flat_accum.view(E_local, BD, S, H).permute(0, 2, 1, 3).contiguous()
    else:
        # XLA path: use SEPARATE one_hot for E and BD dimensions.
        # Critical: E must remain a separate dimension (not flattened with BD)
        # so Shardy can partition E across devices.
        #
        # IMPORTANT: Compute expert→row mapping via modulo (e_ids % num_devices)
        # instead of gathering from expert_mapping.argmax(). With compound
        # E-sharding (E_local=1), the gather expert_rows[e_ids] would index a
        # size-1 tensor with global expert IDs (0-31), producing incorrect results.
        # The modulo formula matches build_expert_mapping: row = expert_id % rows.
        meta = expert_metadata[0]  # [BD, S, K]
        grad_input = torch.zeros(
            E_local, BD, S, H, dtype=grad_output.dtype, device=grad_output.device,
        )

        for k in range(K):
            for b in range(B):
                e_ids = meta[b, :, k]  # [S] expert IDs (global)
                if output_shard_dim == 1:
                    grad_vals = grad_output[k, b]  # [S, H]
                else:
                    grad_vals = grad_output[k, :, b, :]  # [S, H]
                # Direct row computation — no gather from E-sharded tensor
                rows = e_ids % num_devices  # [S] → dispatch row
                bd = b * num_devices + rows  # [S] → BD index
                # Separate one-hots keep E and BD as independent dimensions
                # so Shardy can compound-shard E across all mesh axes.
                e_oh = torch.nn.functional.one_hot(
                    e_ids.long(), E_local
                ).to(grad_output.dtype)  # [S, E]
                bd_oh = torch.nn.functional.one_hot(
                    bd.long(), BD
                ).to(grad_output.dtype)  # [S, BD]
                mask = e_oh.unsqueeze(2) * bd_oh.unsqueeze(1)  # [S, E, BD]
                grad_input = grad_input + torch.einsum(
                    "seb,sh->ebsh", mask, grad_vals
                )

        if output_shard_dim != 1:
            grad_input = grad_input.permute(0, 2, 1, 3).contiguous()

    return grad_input, None, None, None, None, None, None, None


all_to_all_combine.register_autograd(
    _all_to_all_combine_backward,
    setup_context=_all_to_all_combine_setup_context,
)


# =============================================================================
# moe_expert_token_remap
# =============================================================================


@torch.library.custom_op(
    "tt::moe_expert_token_remap", mutates_args=[], device_types=["xla", "cpu"]
)
def moe_expert_token_remap(
    topk_tensor: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    reduction_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert global expert routing to local device expert mapping and sparsity.

    Remaps global expert indices from dispatch metadata to local per-device
    expert indices and creates a sparsity pattern for efficient sparse_matmul.

    Args:
        topk_tensor: Routing scores [D, B, S, E], bfloat16
        expert_mapping: Expert-to-device mapping [1, 1, E, D], int64
        expert_metadata: Expert indices from dispatch [D, B, S, K], int64
        reduction_size: Group size for sparsity reduction (default 16)

    Returns:
        mapping: Expert routing weights [1, B, S, E], bfloat16 (compound-sharded to E_local on device)
        reduced: Sparsity pattern [1, 1, ceil(B*S/reduction_size), E], bfloat16 (compound-sharded to E_local on device)
    """
    import math

    device = topk_tensor.device
    D, B, S, E = topk_tensor.shape
    K = expert_metadata.shape[-1]
    num_devices = expert_mapping.shape[-1]
    E_local = E // num_devices

    reduced_seq = math.ceil(B * S / reduction_size)

    if device.type == "xla":
        output_shapes = [
            [1, B, S, E],
            [1, 1, reduced_seq, E],
        ]
        output_dtypes = [topk_tensor.dtype, topk_tensor.dtype]

        frontend_attributes = {
            "reduction_size": str(reduction_size),
        }

        return stablehlo_custom_call.stablehlo_custom_call(
            [topk_tensor, expert_mapping, expert_metadata],
            "tt.moe_expert_token_remap",
            output_shapes,
            output_dtypes,
            frontend_attributes=frontend_attributes,
        )

    # CPU fallback: uses global E shape (compiler shards to E_local on device).
    # Activate selected experts in ALL dispatch batches to match TT SPMD behavior
    # where dispatch replicates tokens across all rows.
    mapping = torch.zeros(1, B, S, E, dtype=topk_tensor.dtype, device=device)
    reduced = torch.zeros(
        1, 1, reduced_seq, E, dtype=topk_tensor.dtype, device=device
    )

    for d in range(D):
        for b in range(B):
            for s in range(S):
                for k in range(K):
                    global_expert = expert_metadata[d, b, s, k].item()
                    if 0 <= global_expert < E:
                        mapping[0, b, s, global_expert] = topk_tensor[
                            d, b, s, global_expert
                        ]
                        chunk_idx = (b * S + s) // reduction_size
                        if chunk_idx < reduced_seq:
                            reduced[0, 0, chunk_idx, global_expert] = 1.0

    return mapping, reduced


@moe_expert_token_remap.register_fake
def moe_expert_token_remap_fake(
    topk_tensor: torch.Tensor,
    expert_mapping: torch.Tensor,
    expert_metadata: torch.Tensor,
    reduction_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    import math

    D, B, S, E = topk_tensor.shape
    num_devices = expert_mapping.shape[-1]
    reduced_seq = math.ceil(B * S / reduction_size)

    mapping = torch.zeros(
        [1, B, S, E], dtype=topk_tensor.dtype, device=topk_tensor.device
    )
    reduced = torch.zeros(
        [1, 1, reduced_seq, E],
        dtype=topk_tensor.dtype,
        device=topk_tensor.device,
    )
    return mapping, reduced


def _moe_expert_token_remap_setup_context(ctx, inputs, output):
    topk_tensor, expert_mapping, expert_metadata, reduction_size = inputs
    ctx.topk_shape = topk_tensor.shape
    ctx.topk_dtype = topk_tensor.dtype


def _moe_expert_token_remap_backward(ctx, grad_mapping, grad_reduced):
    grad_topk = torch.zeros(
        ctx.topk_shape, dtype=ctx.topk_dtype, device=grad_mapping.device
    )
    return grad_topk, None, None, None


moe_expert_token_remap.register_autograd(
    _moe_expert_token_remap_backward,
    setup_context=_moe_expert_token_remap_setup_context,
)


# Allow the torch dynamo to trace the MoE custom operations.
for attr in ("sparse_matmul", "all_to_all_dispatch", "all_to_all_combine", "moe_expert_token_remap"):
    op = getattr(torch.ops.tt, attr)
    if isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        torch.compiler.allow_in_graph(op)
