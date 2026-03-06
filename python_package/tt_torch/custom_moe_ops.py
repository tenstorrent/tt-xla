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
import torch.nn.functional as F
from torch_xla.experimental import stablehlo_custom_call


def _expert_dispatch_slots(e_ids, E, D, cluster_axis):
    """Map global expert IDs to dispatch slot indices (0..D-1).

    Works for both 1D and 2D meshes without reading expert_mapping:
      cluster_axis=0 → experts round-robin across dispatch rows: slot = e % D
      cluster_axis=1 → experts contiguously blocked: slot = e // (E // D)
    """
    if cluster_axis == 0:
        return e_ids % D
    else:
        return e_ids // max(E // D, 1)


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
        frontend_attributes = {
            "is_input_a_sparse": str(is_input_a_sparse),
            "is_input_b_sparse": str(is_input_b_sparse),
        }
        if nnz is not None:
            frontend_attributes["nnz"] = str(nnz)

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

        return stablehlo_custom_call.stablehlo_custom_call(
            [input_tensor_a, input_tensor_b, sparsity],
            "tt.sparse_matmul",
            [output_shape],
            [input_tensor_a.dtype],
            frontend_attributes=frontend_attributes,
        )

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

    input_b_T = input_b.transpose(-2, -1).contiguous()  # [1, E, N, K]

    if is_a_sparse and is_b_sparse:
        # Forward: [1, E, M, K] @ [1, E, K, N] -> [1, E, M, N]
        # sparsity: [1, 1, 1, E]

        # grad_a: [1,E,M,N] @ [1,E,N,K] -> [1,E,M,K] (both_sparse)
        grad_a = sparse_matmul(
            grad_output, input_b_T, sparsity, nnz=0,
            is_input_a_sparse=True, is_input_b_sparse=True,
        )

        # grad_b: [1,E,K,M] @ [1,E,M,N] -> [1,E,K,N] (both_sparse)
        grad_b = sparse_matmul(
            input_a.transpose(-2, -1).contiguous(), grad_output, sparsity, nnz=0,
            is_input_a_sparse=True, is_input_b_sparse=True,
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
            grad_3d, input_b_T, sparsity_a, nnz=0,
            is_input_a_sparse=True, is_input_b_sparse=False,
        )
        # [AB, E, M, K] -> sum over E -> [A, B, M, K]
        grad_a = grad_a_full.sum(dim=1).reshape(A, B, M, K)

        # grad_b: sum_{ab} sp[ab,e] * input_a[ab,M,K]^T @ grad[ab,e,M,N] -> [1,E,K,N]
        # Fold AB into the M contraction dim so a single matmul replaces matmul+reduce.
        # reduce (stablehlo.reduce) may not be zero-initialised on TT.
        sp_mask = sparsity.reshape(AB, E, 1, 1).to(grad_3d.dtype)
        mg = grad_3d * sp_mask                                           # [AB, E, M, N]
        a_flat_T = input_a.reshape(AB, M, K).transpose(-2, -1)          # [AB, K, M]
        a_folded = a_flat_T.permute(1, 0, 2).reshape(K, AB * M)         # [K, AB*M]
        mg_folded = mg.permute(1, 0, 2, 3).reshape(E, AB * M, N)        # [E, AB*M, N]
        grad_b = torch.matmul(a_folded, mg_folded).unsqueeze(0)         # [1, E, K, N]

    elif is_a_sparse and not is_b_sparse:
        # Forward: [A, E, M, K] @ [1, E, K, N] -> [A, E, M, N]
        # sparsity: [1, 1, A, E]

        # grad_a: [A, E, M, N] @ [1, E, N, K] -> [A, E, M, K] (a_sparse)
        grad_a = sparse_matmul(
            grad_output, input_b_T, sparsity, nnz=0,
            is_input_a_sparse=True, is_input_b_sparse=False,
        )

        # grad_b: sum_a mask[a,e] * input_a[a,e,K,M] @ grad[a,e,M,N] -> [1,E,K,N]
        # Fold A into M to avoid separate reduce.
        mask = sparsity[0, 0]  # [A, E]
        masked_grad = grad_output * mask.unsqueeze(-1).unsqueeze(-1)     # [A, E, M, N]
        input_a_T = input_a.transpose(-2, -1)                            # [A, E, K, M]
        A_, E_, K_, M_ = input_a_T.shape
        N_ = masked_grad.shape[3]
        a_T_folded = input_a_T.permute(1, 2, 0, 3).reshape(E_, K_, A_ * M_)  # [E, K, A*M]
        mg_folded = masked_grad.permute(1, 0, 2, 3).reshape(E_, A_ * M_, N_) # [E, A*M, N]
        grad_b = torch.matmul(a_T_folded, mg_folded).unsqueeze(0)            # [1, E, K, N]

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
        D = num_devices
        BD = B * D
        inp = input_tensor[:, 0, :, :]  # [B, S, H]

        # Metadata: replicate expert_indices D times (combine needs full view)
        m = expert_indices.permute(1, 0, 2, 3)  # [1, B, S, K]
        if D > 1:
            m = m.repeat(1, D, 1, 1)  # [1, BD, S, K]

        if D > 1:
            # Sparse dispatch: write tokens only to device slots holding their experts
            E = expert_mapping.shape[2]
            x = torch.zeros(1, BD, S, H, dtype=input_tensor.dtype, device=device)
            e_ids = expert_indices[:, 0, :, :].long()  # [B, S, K]
            slots = _expert_dispatch_slots(e_ids, E, D, cluster_axis)  # [B, S, K]
            slot_oh = F.one_hot(slots, D)  # [B, S, K, D]
            goes_to_device = slot_oh.max(dim=2).values  # [B, S, D]
            for d in range(D):
                mask_d = goes_to_device[:, :, d].unsqueeze(-1).to(inp.dtype)  # [B, S, 1]
                x[0, d * B : (d + 1) * B, :, :] = inp * mask_d
        else:
            x = input_tensor.permute(1, 0, 2, 3).clone()  # [1, B, S, H]

        return x, m.clone()

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
    D = num_devices
    B = input_tensor.shape[0]
    E = expert_mapping.shape[2]

    if D > 1:
        # Precompute dispatch mask: [B, S, D] — which dispatch slots each token writes to
        e_ids = expert_indices[:, 0, :, :].long()  # [B, S, K]
        slots = _expert_dispatch_slots(e_ids, E, D, cluster_axis)  # [B, S, K]
        slot_oh = F.one_hot(slots, D)  # [B, S, K, D]
        goes_to_device = slot_oh.max(dim=2).values.to(
            input_tensor.dtype
        )  # [B, S, D]
        ctx.save_for_backward(goes_to_device)
    else:
        ctx.save_for_backward(expert_mapping)  # placeholder, unused in D==1 path

    ctx.num_devices = D
    ctx.B = B


def _all_to_all_dispatch_backward(ctx, grad_dispatched, _grad_metadata):
    D = ctx.num_devices
    B = ctx.B
    S, H = grad_dispatched.shape[2], grad_dispatched.shape[3]

    # Forward sparsely wrote tokens to device slots: output[b*D+d] = input[b] * mask[b,s,d].
    # Backward: accumulate masked gradient slices back into [1,B,S,H].
    # Avoid .sum() (stablehlo.reduce — may not be zero-init on TT) and
    # scatter_add (stablehlo.scatter — decomposes into O(S*H) sequential ops).
    # Masked slice + add: D-1 additions, each a single stablehlo.add.
    if D > 1:
        (goes_to_device,) = ctx.saved_tensors  # [B, S, D]
        mask_0 = goes_to_device[:, :, 0:1].unsqueeze(0)  # [1, B, S, 1]
        grad = grad_dispatched[:, :B] * mask_0
        for d in range(1, D):
            mask_d = goes_to_device[:, :, d : d + 1].unsqueeze(0)  # [1, B, S, 1]
            grad = grad + grad_dispatched[:, d * B : (d + 1) * B] * mask_d
    else:
        grad = grad_dispatched

    return grad.permute(1, 0, 2, 3), None, None, None, None


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
        # CPU fallback: gather expert outputs from the BD position where
        # dispatch actually wrote the token data (dispatch slot of the expert).
        D = num_devices
        E = expert_mapping.shape[2]
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
                            slot = _expert_dispatch_slots(
                                torch.tensor(expert_id), E, D, cluster_axis
                            ).item()
                            bd = slot * B + b
                            combined[k, b, s, :] = input_tensor[
                                expert_id, bd, s, :
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
                            slot = _expert_dispatch_slots(
                                torch.tensor(expert_id), E, D, cluster_axis
                            ).item()
                            bd = slot * B + b
                            combined[k, s, b, :] = input_tensor[
                                expert_id, s, bd, :
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
    ctx.save_for_backward(expert_metadata, expert_mapping)
    ctx.num_devices = num_devices
    ctx.cluster_axis = cluster_axis
    ctx.num_experts_per_tok = num_experts_per_tok
    ctx.output_shard_dim = output_shard_dim
    ctx.input_shape = input_tensor.shape


def _all_to_all_combine_backward(ctx, grad_output):
    """
    Combine forward gathered input[e, b, s, :] → combined[k, b, s, :].
    Backward scatters grad_output[k, b, s, :] back to grad_input[e, bd, s, :]
    where bd = b * D + device_row(expert_id).

    Vectorized one_hot scatter — single path for CPU and XLA.
    """
    expert_metadata, expert_mapping = ctx.saved_tensors
    D = ctx.num_devices
    K = ctx.num_experts_per_tok
    output_shard_dim = ctx.output_shard_dim
    E_local = ctx.input_shape[0]

    if output_shard_dim == 1:
        _, BD, S, H = ctx.input_shape
        B = grad_output.shape[1]
    else:
        _, S, BD, H = ctx.input_shape
        B = grad_output.shape[2]

    meta = expert_metadata[0][:B]  # [B, S, K]

    if output_shard_dim == 1:
        grad_t = grad_output.permute(1, 2, 0, 3)  # [B, S, K, H]
    else:
        grad_t = grad_output.permute(2, 1, 0, 3)  # [B, S, K, H]

    E_total = expert_mapping.shape[2]
    e_ids = meta.long()                                                    # [B, S, K]
    rows = _expert_dispatch_slots(e_ids, E_total, D, ctx.cluster_axis)     # [B, S, K]
    b_idx = torch.arange(B, device=grad_output.device).view(B, 1, 1)
    bd = rows * B + b_idx                                                   # [B, S, K]

    e_oh = F.one_hot(e_ids.clamp(0, E_local - 1), E_local).to(grad_output.dtype)  # [B,S,K,E]
    bd_oh = F.one_hot(bd.clamp(0, BD - 1), BD).to(grad_output.dtype)              # [B,S,K,BD]

    mask = e_oh.unsqueeze(-1) * bd_oh.unsqueeze(-2)                        # [B,S,K,E,BD]
    grad_input = torch.einsum('bsked,bskh->edsh', mask, grad_t)

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
