# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sparse MLP module for MoE (Mixture of Experts) models.

This module provides utilities to replace dense MLP layers with sparse MLP
implementations that use sparse_matmul for efficient expert computation.

Usage:
    import tt_torch
    from tt_torch.sparse_mlp import enable_sparse_mlp

    model = load_your_moe_model()
    model = enable_sparse_mlp(model)  # Replace MLP layers with SparseMLP
"""

import os
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder

# Activation types for A2aSparseMLP
ACTIVATION_GPT_OSS = "gpt_oss"  # clamp, sigmoid, alpha, glu
ACTIVATION_DEEPSEEK = "deepseek"  # SiLU (swish) for gate * up

# ---------------------------------------------------------------------------
# Fused MoE kernel weight layout constants (Wormhole: 12 DRAM banks)
# ---------------------------------------------------------------------------
_TILE_SIZE = 32
_NUM_DRAM_CORES = 12
_FUSED_FULL_CORES = {0, 1, 4, 5, 8, 9}  # 8 tiles per core
_FUSED_PAD_CORES = {2, 3, 6, 7, 10, 11}  # 7 tiles per core
_FUSED_MAX_TILES_PER_CORE = 8


def _tiles_for_core(ring_pos: int) -> int:
    """Return the number of valid tiles for a core at a given ring position."""
    return 7 if ring_pos in _FUSED_PAD_CORES else 8


def _prepare_w0_w1_tensor(
    torch_w0: torch.Tensor,
    torch_w1: torch.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
) -> torch.Tensor:
    """Interleave, shard, and pad w0/w1 weights for the fused MoE kernel.

    Takes w0 (gate) and w1 (up) weights of shape ``(L, E, K, N)`` and produces
    a tensor of shape ``(12, L, E, 4, K, 128)`` ready for DRAM HEIGHT_SHARDED
    placement.
    """
    num_cores = _NUM_DRAM_CORES
    Nt = N // _TILE_SIZE

    w0_chunks = torch_w0.view(L, E, K, Nt, _TILE_SIZE)
    w1_chunks = torch_w1.view(L, E, K, Nt, _TILE_SIZE)
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)
    interleaved = stacked.view(L, E, K, Nt, 2 * _TILE_SIZE)
    permuted = interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    start_tile = 0
    for ring_pos in range(num_cores):
        num_tiles = _tiles_for_core(ring_pos)
        shard = permuted[:, :, start_tile : start_tile + num_tiles, :, :]
        start_tile += num_tiles

        if num_tiles < _FUSED_MAX_TILES_PER_CORE:
            pad_tiles = _FUSED_MAX_TILES_PER_CORE - num_tiles
            padding = torch.zeros(
                L, E, pad_tiles, K, 2 * _TILE_SIZE, dtype=torch_w0.dtype
            )
            shard = torch.cat([shard, padding], dim=2)

        each_shard.append(shard)

    reordered = torch.cat(each_shard, dim=2)
    groups_per_core = _FUSED_MAX_TILES_PER_CORE // 2  # 4

    all_groups = reordered.view(
        L, E, num_cores, _FUSED_MAX_TILES_PER_CORE, K, 2 * _TILE_SIZE
    )
    all_groups = all_groups.permute(2, 0, 1, 3, 4, 5)

    pair_2_tiles = all_groups.view(
        num_cores, L, E, groups_per_core, 2, K, 2 * _TILE_SIZE
    )
    pair_2_tiles = pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    paired = pair_2_tiles.reshape(
        num_cores, L, E, groups_per_core, K, 4 * _TILE_SIZE
    )
    return paired


def _prepare_w0_b0_w1_b1_tensor(
    torch_w0: torch.Tensor,
    torch_b0: torch.Tensor,
    torch_w1: torch.Tensor,
    torch_b1: torch.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
) -> torch.Tensor:
    """Interleave, shard, and pad w0/w1 weights with bias for the fused MoE kernel.

    Concatenates bias along dim 2 (K_new = K + K_b), then performs the same
    interleave/shard/pad as ``_prepare_w0_w1_tensor``.
    """
    num_cores = _NUM_DRAM_CORES
    torch_w0_b0 = torch.cat([torch_w0, torch_b0], dim=2)
    torch_w1_b1 = torch.cat([torch_w1, torch_b1], dim=2)
    K_new = torch_w0_b0.shape[2]
    Nt = N // _TILE_SIZE

    w0_b0_chunks = torch_w0_b0.view(L, E, K_new, Nt, _TILE_SIZE)
    w1_b1_chunks = torch_w1_b1.view(L, E, K_new, Nt, _TILE_SIZE)
    stacked = torch.stack([w0_b0_chunks, w1_b1_chunks], dim=4)
    interleaved = stacked.view(L, E, K_new, Nt, 2 * _TILE_SIZE)
    permuted = interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    start_tile = 0
    for ring_pos in range(num_cores):
        num_tiles = _tiles_for_core(ring_pos)
        shard = permuted[:, :, start_tile : start_tile + num_tiles, :, :]
        start_tile += num_tiles

        if num_tiles < _FUSED_MAX_TILES_PER_CORE:
            pad_tiles = _FUSED_MAX_TILES_PER_CORE - num_tiles
            padding = torch.zeros(
                L, E, pad_tiles, K_new, 2 * _TILE_SIZE, dtype=torch_w0.dtype
            )
            shard = torch.cat([shard, padding], dim=2)

        each_shard.append(shard)

    reordered = torch.cat(each_shard, dim=2)
    groups_per_core = _FUSED_MAX_TILES_PER_CORE // 2  # 4

    all_groups = reordered.view(
        L, E, num_cores, _FUSED_MAX_TILES_PER_CORE, K_new, 2 * _TILE_SIZE
    )
    all_groups = all_groups.permute(2, 0, 1, 3, 4, 5)

    pair_2_tiles = all_groups.view(
        num_cores, L, E, groups_per_core, 2, K_new, 2 * _TILE_SIZE
    )
    pair_2_tiles = pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    paired = pair_2_tiles.reshape(
        num_cores, L, E, groups_per_core, K_new, 4 * _TILE_SIZE
    )
    return paired


def _prepare_w2_tensor(
    torch_w2: torch.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Shard, pad, and reorder w2 weights for the fused MoE kernel.

    Takes w2 (down) weight of shape ``(L, E, N, K)`` and produces a tensor of
    shape ``(12, L, E, 2, N, 128)`` ready for DRAM HEIGHT_SHARDED placement.
    """
    num_cores = _NUM_DRAM_CORES
    each_shard = []

    start_col = 0
    for ring_pos in range(num_cores):
        pad_flag = 1 if ring_pos in _FUSED_PAD_CORES else 0

        if pad_flag:
            each_shard.append(
                torch_w2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE
            each_shard.append(
                torch_w2[:, :, :, start_col : start_col + 3 * _TILE_SIZE]
            )
            start_col += 3 * _TILE_SIZE
            each_shard.append(
                torch.zeros(L, E, N, 1 * _TILE_SIZE, dtype=torch_w2.dtype)
            )
        else:
            each_shard.append(
                torch_w2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE
            each_shard.append(
                torch_w2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE

    reordered = torch.cat(each_shard, dim=-1)
    all_groups = reordered.view(L, E, N, num_cores, 2, 4 * _TILE_SIZE)
    all_groups = all_groups.permute(3, 0, 1, 4, 2, 5)

    Nt = N // _TILE_SIZE
    N_grouped = all_groups.view(
        num_cores, L, E, 2, Nt, _TILE_SIZE, 4 * _TILE_SIZE
    )

    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)
    chunk_sizes = [_tiles_for_core(i) for i in range(num_cores)]
    chunk_start_positions = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0),
        ]
    )

    each_shard = []
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))
        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(
        num_cores, L, E, 2, -1, 4 * _TILE_SIZE
    )
    return N_reordered


def _prepare_w2_b2_tensor(
    torch_w2: torch.Tensor,
    torch_b2: torch.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
) -> torch.Tensor:
    """Shard, pad, and reorder w2 weights with bias for the fused MoE kernel.

    Concatenates bias along dim 2 (N_new = N + 32), column-shards K, then
    ring-rotates ONLY the weight tile rows. The bias tile row is appended
    at position Nt_weight (the last tile row).
    """
    num_cores = _NUM_DRAM_CORES
    torch_w2_b2 = torch.cat([torch_w2, torch_b2], dim=2)
    N_new = N + _TILE_SIZE  # e.g., 2912

    each_shard = []
    start_col = 0
    for ring_pos in range(num_cores):
        pad_flag = 1 if ring_pos in _FUSED_PAD_CORES else 0

        if pad_flag:
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 3 * _TILE_SIZE]
            )
            start_col += 3 * _TILE_SIZE
            each_shard.append(
                torch.zeros(L, E, N_new, 1 * _TILE_SIZE, dtype=torch_w2.dtype)
            )
        else:
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _TILE_SIZE]
            )
            start_col += 4 * _TILE_SIZE

    reordered = torch.cat(each_shard, dim=-1)
    all_groups = reordered.view(L, E, N_new, num_cores, 2, 4 * _TILE_SIZE)
    all_groups = all_groups.permute(3, 0, 1, 4, 2, 5)

    Nt_all = N_new // _TILE_SIZE
    Nt_weight = N // _TILE_SIZE
    N_grouped = all_groups.view(
        num_cores, L, E, 2, Nt_all, _TILE_SIZE, 4 * _TILE_SIZE
    )

    # Split weight tile rows and bias tile row
    N_weight = N_grouped[:, :, :, :, :Nt_weight, :, :]
    N_bias = N_grouped[:, :, :, :, Nt_weight:, :, :]

    # Ring-rotate ONLY the weight tile rows
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)
    chunk_sizes = [_tiles_for_core(i) for i in range(num_cores)]
    chunk_start_positions = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0),
        ]
    )

    each_shard = []
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_weight[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        # Append bias tile row (not part of rotation)
        each_chunk.append(N_bias[core_id])
        each_shard.append(torch.cat(each_chunk, dim=3))
        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(
        num_cores, L, E, 2, -1, 4 * _TILE_SIZE
    )
    return N_reordered


def preprocess_fused_moe_weights(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_experts: int,
    num_devices: int,
    cluster_axis: int,
    mesh_shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess MoE weights into the fused kernel layout.

    Replicates the pure-PyTorch weight transformation from tt-metal's
    ``create_fused_moe_gpt_config``: de-interleave gate/up, pad biases to
    tile height, tile-interleave w0/w1, core-shard, and ring-rotate w2.

    The output tensors are laid out so that ShardTensor2dMesh(dims=(0, 1))
    (or equivalent XLA sharding on dims 0 and 1) distributes them correctly
    across a (ring_devices x mesh_cols) mesh.

    Args:
        gate_up_proj: ``[num_experts, K, 2*N]`` interleaved gate+up weights.
        gate_up_proj_bias: ``[num_experts, 2*N]`` interleaved gate+up bias.
        down_proj: ``[num_experts, N, K]`` down projection weights.
        down_proj_bias: ``[num_experts, K]`` down projection bias.
        num_experts: Total number of experts.
        num_devices: Total number of devices in mesh.
        cluster_axis: Mesh axis for ring dispatch (0 or 1).
        mesh_shape: Tuple ``(rows, cols)`` of mesh dimensions.

    Returns:
        ``(fused_w0_w1, fused_w2)`` — preprocessed tensors ready for sharding.
        - fused_w0_w1: ``(ring_devices*12, mesh_cols, E_per_dev, 4, K+32, 128)``
        - fused_w2: ``(ring_devices*12, mesh_cols, E_per_dev, 2, N+32, 128)``
    """
    ring_devices = mesh_shape[cluster_axis]
    mesh_cols = num_devices // ring_devices
    experts_per_device = num_experts // num_devices
    experts_per_cluster = num_experts // mesh_cols
    E = experts_per_device
    K = gate_up_proj.shape[1]
    N = gate_up_proj.shape[2] // 2
    L = 1

    # De-interleave gate_up_proj: [g0, u0, g1, u1, ...] -> w0 (gate), w1 (up)
    w0_all = gate_up_proj[..., ::2].contiguous().float()  # [num_experts, K, N]
    w1_all = gate_up_proj[..., 1::2].contiguous().float()

    w2_all = down_proj.contiguous().float()  # [num_experts, N, K]

    # De-interleave biases
    b0_all = gate_up_proj_bias[..., ::2].contiguous().float()  # [num_experts, N]
    b1_all = gate_up_proj_bias[..., 1::2].contiguous().float()
    b2_all = down_proj_bias.contiguous().float()  # [num_experts, K]

    # Pad biases to tile height (32 rows): [num_experts, N] -> [num_experts, 32, N]
    b0_all = b0_all.unsqueeze(1)
    b0_all = F.pad(b0_all, (0, 0, 0, _TILE_SIZE - 1), "constant", 0.0)
    b1_all = b1_all.unsqueeze(1)
    b1_all = F.pad(b1_all, (0, 0, 0, _TILE_SIZE - 1), "constant", 0.0)
    b2_all = b2_all.unsqueeze(1)
    b2_all = F.pad(b2_all, (0, 0, 0, _TILE_SIZE - 1), "constant", 0.0)

    # Per-device weight preprocessing, organized as [rows, cols] for 2D mesh sharding
    w0_w1_rows = []
    w2_rows = []
    for r in range(ring_devices):
        w0_w1_cols = []
        w2_cols = []
        for c in range(mesh_cols):
            start = c * experts_per_cluster + r * E
            w0_d = w0_all[start : start + E].unsqueeze(0)  # [1, E, K, N]
            w1_d = w1_all[start : start + E].unsqueeze(0)
            w2_d = w2_all[start : start + E].unsqueeze(0)  # [1, E, N, K]
            b0_d = b0_all[start : start + E].unsqueeze(0)  # [1, E, 32, N]
            b1_d = b1_all[start : start + E].unsqueeze(0)
            b2_d = b2_all[start : start + E].unsqueeze(0)  # [1, E, 32, K]

            w0_w1_cols.append(
                _prepare_w0_b0_w1_b1_tensor(w0_d, b0_d, w1_d, b1_d, L, E, K, N)
            )
            w2_cols.append(
                _prepare_w2_b2_tensor(w2_d, b2_d, L, E, N, K)
            )
        w0_w1_rows.append(torch.cat(w0_w1_cols, dim=1))  # cat cols on dim 1
        w2_rows.append(torch.cat(w2_cols, dim=1))

    fused_w0_w1 = torch.cat(w0_w1_rows, dim=0)  # cat rows on dim 0
    fused_w2 = torch.cat(w2_rows, dim=0)

    return fused_w0_w1, fused_w2


def _topk_to_sparse_scores(topk_weights, topk_indices, num_experts):
    """Convert topk scores [BS, K] to sparse scores [BS, E].

    Uses one_hot + einsum instead of scatter_ for XLA compatibility.
    """
    one_hot = (
        topk_indices.unsqueeze(-1)
        == torch.arange(num_experts, device=topk_indices.device)
    ).to(
        topk_weights.dtype
    )  # [BS, K, E]
    return torch.einsum("bk,bke->be", topk_weights, one_hot)


def _unpack_router_output(router_out, num_experts):
    """Unpack router output to (scores [BS, E], indices [BS, K]).

    Handles routers returning 2 values (scores, indices) or 3 values
    (logits, scores, indices) like GptOssTopKRouter. Converts topk-only
    scores [BS, K] to sparse scores [BS, E] when needed.
    """
    scores, indices = router_out[-2], router_out[-1]
    if scores.shape[-1] != num_experts:
        scores = _topk_to_sparse_scores(scores, indices, num_experts)
    return scores, indices


def _moe_activation(
    gate_up_out, activation_type, alpha=1.702, limit=7.0, interleaved=True
):
    """Apply gate-up activation for MoE experts.

    Args:
        gate_up_out: Fused gate+up projection output [..., inter*2].
        activation_type: "deepseek" (SiLU) or "gpt_oss" (clamp+sigmoid+glu).
        alpha: Sigmoid scaling factor (gpt_oss only).
        limit: Clamp bound (gpt_oss only).
        interleaved: If True, gate/up are interleaved [g0,u0,g1,u1,...].
                     If False, contiguous [g0,g1,...,u0,u1,...].
    """
    half = gate_up_out.shape[-1] // 2
    if interleaved:
        gate_out = gate_up_out[..., ::2]
        up_out = gate_up_out[..., 1::2]
    else:
        gate_out = gate_up_out[..., :half]
        up_out = gate_up_out[..., half:]

    if activation_type == ACTIVATION_DEEPSEEK:
        return F.silu(gate_out) * up_out
    else:
        gate_out = gate_out.clamp(max=limit)
        up_out = up_out.clamp(-limit, limit)
        glu = gate_out * torch.sigmoid(gate_out * alpha)
        return (up_out + 1) * glu


class _SparseForwardMixin:
    """Mixin that adds sparse_forward() for expert wrapper classes."""

    def sparse_forward(
        self,
        dispatched,
        sparsity_remap,
        activation_type,
        alpha=1.702,
        limit=7.0,
        output_shape=None,
    ):
        return _sparse_expert_forward(
            self,
            dispatched,
            sparsity_remap,
            activation_type,
            alpha,
            limit,
            output_shape,
        )


def _sparse_expert_forward(
    experts,
    dispatched,
    sparsity_remap,
    activation_type,
    alpha=1.702,
    limit=7.0,
    output_shape=None,
):
    """Unified sparse_matmul forward for MoE experts.

    Works with both fused (w3=None, w1=gate_up) and separate (w3=up) expert weights.

    Gate/up output is 5D tiled: [A, B, E, M, N] where A*B*M = BD*S.
    Down input is reshaped to canonical [A*B, E, M, K].
    Down output [A*B, E, M, H] is untiled to [BD, S, E, H].
    """
    E = experts.w2.shape[0]
    w1 = experts.w1.unsqueeze(0)  # [1, E, H, N1]
    w2 = experts.w2.view(1, E, experts.intermediate_size, -1)  # [1, E, inter, H]

    # Gate (or gate+up fused): output [A, B, E, M, N1] (5D tiled)
    w1_out = torch.ops.tt.sparse_matmul(
        dispatched,
        w1,
        sparsity_remap,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )
    if experts.w1_bias is not None:
        w1_out = w1_out + experts.w1_bias.view(1, 1, E, 1, -1)

    if experts.w3 is not None:
        # Separate gate/up: 2 sparse_matmuls
        w3 = experts.w3.unsqueeze(0)  # [1, E, H, inter]
        w3_out = torch.ops.tt.sparse_matmul(
            dispatched,
            w3,
            sparsity_remap,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        if experts.w3_bias is not None:
            w3_out = w3_out + experts.w3_bias.view(1, 1, E, 1, -1)

        if activation_type == ACTIVATION_DEEPSEEK:
            activated = F.silu(w1_out) * w3_out
        else:
            w1_out = w1_out.clamp(max=limit)
            w3_out = w3_out.clamp(-limit, limit)
            glu = w1_out * torch.sigmoid(w1_out * alpha)
            activated = (w3_out + 1) * glu
    else:
        # Fused gate_up: 1 sparse_matmul, split via activation
        activated = _moe_activation(w1_out, activation_type, alpha, limit)

    # Reshape 5D → 4D canonical for down: [A, B, E, M, K] → [A*B, E, M, K]
    A, B = activated.shape[0], activated.shape[1]
    M = activated.shape[3]
    activated = activated.reshape(A * B, E, M, experts.intermediate_size)

    # Down: output [A*B, E, M, H] (canonical)
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        w2,
        sparsity_remap,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )
    if experts.w2_bias is not None:
        down_out = down_out + experts.w2_bias.view(1, E, 1, -1)

    # Untile: [A*B, E, M, H] → [E, BD, S, H]
    # Single permute to move E to front; A*B and M are adjacent so reshape merges them.
    BD, S = output_shape
    H = down_out.shape[-1]
    down_out = down_out.permute(1, 0, 2, 3)  # [E, A*B, M, H]
    return down_out.reshape(E, 1, BD * S, H)


class SparseMLP(nn.Module):
    """
    Sparse MLP implementation that uses sparse_matmul for MoE computation.

    This module wraps an existing MLP and replaces dense expert computation
    with sparse_matmul operations that skip inactive experts.

    Uses separate gate_proj, up_proj, down_proj weights — 3 sparse_matmuls.
    """

    def __init__(
        self,
        original_mlp,
        num_experts: int,
        num_experts_per_tok: int,
        config: Optional[object] = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Copy references to original module's components
        self.router = original_mlp.router
        self.experts = original_mlp.experts

        if hasattr(self.experts, "gate_proj"):
            self.intermediate_size = self.experts.gate_proj.shape[-1]
        elif hasattr(self.experts, "gate_up_proj"):
            self.intermediate_size = self.experts.gate_up_proj.shape[-1] // 2
        else:
            raise ValueError("Expected gate_proj or gate_up_proj in experts module")

        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = self.experts.down_proj.shape[-1]

        # Activation parameters
        self.alpha = getattr(self.experts, "alpha", 1.702)
        self.limit = getattr(self.experts, "limit", 7.0)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. Router — pass 3D for RouterAdapter (handles 3D→2D internally),
        # flatten to 2D for raw routers (e.g. GptOssTopKRouter).
        router_input = hidden_states
        if not hasattr(self.router, "gate"):
            router_input = hidden_states.view(-1, hidden_size)
        router_scores, router_indices = _unpack_router_output(
            self.router(router_input), self.num_experts
        )

        # 2. Sparsity Mask [batch, seq, 1, num_experts] via one-hot
        topk_indices_unsqueezed = router_indices.view(
            batch_size, seq_len, 1, self.num_experts_per_tok
        )
        expert_range = torch.arange(self.num_experts, device=hidden_states.device)
        one_hot = (topk_indices_unsqueezed.unsqueeze(-1) == expert_range).to(
            hidden_states.dtype
        )  # [batch, seq, 1, K, E]
        sparsity = one_hot.sum(dim=-2)  # [batch, seq, 1, E]

        # 3. Input [batch, seq, 1, hidden]
        hidden_4d = hidden_states.view(batch_size, seq_len, 1, hidden_size)

        has_fused = hasattr(self.experts, "gate_up_proj")

        if has_fused:
            # Fused gate_up: 1 sparse_matmul for gate+up
            gate_up_proj = self.experts.gate_up_proj.unsqueeze(0)
            gate_up_out = torch.ops.tt.sparse_matmul(
                hidden_4d,
                gate_up_proj,
                sparsity,
                nnz=0,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
            )
            gate_up_out = gate_up_out.view(
                batch_size, seq_len, self.num_experts, self.intermediate_size * 2
            )
            if self.experts.gate_up_proj_bias is not None:
                gate_up_out = gate_up_out + self.experts.gate_up_proj_bias
            activated = _moe_activation(
                gate_up_out, ACTIVATION_GPT_OSS, self.alpha, self.limit
            )
        else:
            # Separate gate/up: 2 sparse_matmuls
            gate_proj = self.experts.gate_proj.unsqueeze(0)
            gate_out = torch.ops.tt.sparse_matmul(
                hidden_4d,
                gate_proj,
                sparsity,
                nnz=0,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
            )
            gate_out = gate_out.view(
                batch_size, seq_len, self.num_experts, self.intermediate_size
            )
            if self.experts.gate_proj_bias is not None:
                gate_out = gate_out + self.experts.gate_proj_bias

            up_proj = self.experts.up_proj.unsqueeze(0)
            up_out = torch.ops.tt.sparse_matmul(
                hidden_4d,
                up_proj,
                sparsity,
                nnz=0,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
            )
            up_out = up_out.view(
                batch_size, seq_len, self.num_experts, self.intermediate_size
            )
            if self.experts.up_proj_bias is not None:
                up_out = up_out + self.experts.up_proj_bias

            activated = F.silu(gate_out) * up_out

        # 7. Down projection
        activated_reshaped = activated.view(
            batch_size * seq_len, self.num_experts, 1, self.intermediate_size
        )
        sparsity_down = sparsity.view(1, 1, batch_size * seq_len, self.num_experts)
        down_proj = self.experts.down_proj.view(
            1, self.num_experts, self.intermediate_size, hidden_size
        )
        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            down_proj,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        down_out = down_out.squeeze(2)
        if self.experts.down_proj_bias is not None:
            down_out = down_out + self.experts.down_proj_bias

        # 8. Weighted Sum
        output = (down_out * router_scores.unsqueeze(-1)).sum(dim=1)
        output = output.view(batch_size, seq_len, hidden_size)

        return output, router_scores


def build_expert_mapping(num_experts, num_devices, mesh_shape=None):
    """
    Build one-hot expert-to-device mapping tensor.

    Creates a [1, 1, E, D] tensor where mapping[0, 0, i, d] = 1 means
    expert i resides on device d.

    For 1D meshes (mesh_shape=None), experts are sequentially distributed:
    experts 0..E/D-1 on device 0, E/D..2*E/D-1 on device 1, etc.

    For 2D meshes, accounts for GSPMD compound sharding ("axis_0", "axis_1")
    where axis_0 is the inner (fast-varying) dimension:
    expert e -> mesh position (e % rows, e // rows) -> device (e % rows) * cols + (e // rows).

    Args:
        num_experts: Total number of experts (E)
        num_devices: Number of devices along dispatch axis (D)
        mesh_shape: Optional (rows, cols) tuple for 2D compound sharding

    Returns:
        Tensor of shape [1, 1, E, D] with one-hot encoding
    """
    assert (
        num_experts % num_devices == 0
    ), f"num_experts ({num_experts}) must be divisible by num_devices ({num_devices})"
    mapping = torch.zeros(1, 1, num_experts, num_devices, dtype=torch.int64)
    experts_per_device = num_experts // num_devices
    for i in range(num_experts):
        if mesh_shape is not None:
            rows, cols = mesh_shape
            device_id = (i % rows) * cols + (i // rows)
        else:
            device_id = i // experts_per_device
        mapping[0, 0, i, device_id] = 1
    return mapping


class FusedExpertsWrapper(_SparseForwardMixin, nn.Module):
    """Wraps an experts module that has gate_up_proj and adds sparse_forward().

    Original attribute names (gate_up_proj, down_proj, etc.) remain accessible
    for shard specs. w1/w2/w3 aliases are used by _sparse_expert_forward.
    """

    def __init__(self, experts):
        super().__init__()
        self._experts = experts
        self.intermediate_size = experts.gate_up_proj.shape[-1] // 2

    @property
    def w1(self):
        return self._experts.gate_up_proj

    @property
    def w1_bias(self):
        return getattr(self._experts, "gate_up_proj_bias", None)

    @property
    def w2(self):
        return self._experts.down_proj

    @property
    def w2_bias(self):
        return getattr(self._experts, "down_proj_bias", None)

    @property
    def w3(self):
        return None  # fused — no separate up proj

    def forward(self, *args, **kwargs):
        """Delegate to original experts forward for CPU golden path."""
        return self._experts(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_") or name in ("intermediate_size", "training"):
            return super().__getattr__(name)
        return getattr(self._experts, name)


def _scatter_compact_router_scores(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Expand compact [T, K] routing scores into the full sparse [T, E] form."""
    router_scores = torch.zeros(
        topk_scores.shape[0],
        num_experts,
        dtype=topk_scores.dtype,
        device=topk_scores.device,
    )
    router_scores.scatter_(1, topk_indices.long(), topk_scores)
    return router_scores


def _topk_router_gpt_fallback(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback decomposition for the GPT-OSS fused router."""
    router_logits = F.linear(hidden_states, router_weight, router_bias)
    topk_scores, topk_indices = torch.topk(router_logits, k, dim=-1)
    topk_scores = F.softmax(topk_scores, dim=-1, dtype=topk_scores.dtype)
    return topk_indices, topk_scores


def _composite_topk_router_gpt(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    k: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap GPT-OSS router decode into a StableHLO composite."""
    builder = StableHLOCompositeBuilder(
        name="tenstorrent.topk_router_gpt",
        attr={"k": k, "num_experts": num_experts},
    )

    if router_bias is None:
        hidden_states, router_weight = builder.mark_inputs(hidden_states, router_weight)
    else:
        hidden_states, router_weight, router_bias = builder.mark_inputs(
            hidden_states, router_weight, router_bias
        )

    topk_indices, topk_scores = torch.ops.tt.topk_router_gpt(
        hidden_states, router_weight, router_bias, k
    )
    topk_indices, topk_scores = builder.mark_outputs(topk_indices, topk_scores)
    return topk_indices, topk_scores


def _moe_gpt_decode_fallback(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    dispatch_mapping: torch.Tensor,
    moe_gpt_mapping: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_devices: int,
    cluster_axis: int,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    alpha: float,
    limit: float,
    fused_w0_w1: Optional[torch.Tensor] = None,
    fused_w2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode-only GPT-OSS decomposition expressed with SHLO custom ops."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    assert seq_len == 1, f"Expected GPT-OSS decode input, got seq_len={seq_len}"

    x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
    expert_indices = topk_indices.view(batch_size, 1, seq_len, num_experts_per_tok)
    expert_scores = topk_scores.view(batch_size, 1, seq_len, num_experts_per_tok)

    dispatched, metadata_indices, metadata_scores = (
        torch.ops.tt.all_to_all_dispatch_metadata(
            x,
            expert_indices,
            expert_scores,
            dispatch_mapping,
            num_devices=num_devices,
            cluster_axis=cluster_axis,
        )
    )

    # Mirror tt-metal's fused decode sequence:
    # dispatch_metadata -> moe_gpt(metadata bundle) -> selective_reduce_combine.
    moe_gpt_outputs = torch.ops.tt.moe_gpt(
        dispatched,
        metadata_indices,
        metadata_scores,
        moe_gpt_mapping,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
    )

    combined = torch.ops.tt.selective_reduce_combine(
        moe_gpt_outputs[4],
        moe_gpt_outputs[1],
        moe_gpt_outputs[2],
        moe_gpt_outputs[0],
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        num_experts_per_tok=num_experts_per_tok,
        output_shard_dim=2,
    )

    topk_weights = topk_scores.view(batch_size, num_experts_per_tok)
    topk_weights = topk_weights.permute(1, 0).unsqueeze(-1)
    output = (combined.squeeze(1) * topk_weights).sum(dim=0)
    return output.unsqueeze(1)


def _composite_moe_gpt_decode(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    dispatch_mapping: torch.Tensor,
    moe_gpt_mapping: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_devices: int,
    cluster_axis: int,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    alpha: float,
    limit: float,
    fused_w0_w1: Optional[torch.Tensor] = None,
    fused_w2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrap GPT-OSS decode expert flow into one composite.

    TT-MLIR can legalize this composite to a placeholder TTIR op, while the
    embedded StableHLO decomposition exposes the GPT-OSS custom calls used for
    sharding propagation.

    When ``fused_w0_w1`` and ``fused_w2`` are provided they are included as
    additional composite inputs so that tt-MLIR can consume the already-
    preprocessed fused kernel weight layout directly, avoiding a runtime
    transformation.
    """
    has_fused = fused_w0_w1 is not None and fused_w2 is not None
    builder = StableHLOCompositeBuilder(
        name="tenstorrent.moe_gpt_decode",
        attr={
            "num_devices": num_devices,
            "cluster_axis": cluster_axis,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "intermediate_size": intermediate_size,
            "alpha": alpha,
            "limit": limit,
            "has_fused_weights": has_fused,
        },
    )

    inputs = [
        hidden_states,
        topk_indices,
        topk_scores,
        dispatch_mapping,
        moe_gpt_mapping,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
    ]
    if has_fused:
        inputs.extend([fused_w0_w1, fused_w2])

    marked = builder.mark_inputs(*inputs)

    if has_fused:
        (
            hidden_states,
            topk_indices,
            topk_scores,
            dispatch_mapping,
            moe_gpt_mapping,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            fused_w0_w1,
            fused_w2,
        ) = marked
    else:
        (
            hidden_states,
            topk_indices,
            topk_scores,
            dispatch_mapping,
            moe_gpt_mapping,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
        ) = marked

    output = _moe_gpt_decode_fallback(
        hidden_states=hidden_states,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        dispatch_mapping=dispatch_mapping,
        moe_gpt_mapping=moe_gpt_mapping,
        gate_up_proj=gate_up_proj,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj=down_proj,
        down_proj_bias=down_proj_bias,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        intermediate_size=intermediate_size,
        alpha=alpha,
        limit=limit,
    )
    output = builder.mark_outputs(output)
    return output


class A2aSparseMLP(nn.Module):
    """
    Sparse MLP with all-to-all dispatch/combine for multi-device expert parallelism.

    Wraps sparse_matmul expert computation with all_to_all_dispatch (before) and
    all_to_all_combine (after) to selectively route tokens to devices holding
    their selected experts.

    Expert distribution follows the DeepSeek v3 pattern:
    - Experts are compound-sharded across both mesh dims (each device has unique experts)
    - Dispatch/combine operate along cluster_axis only (typically axis 0)
    - BD = B * dispatch_devices (devices along cluster_axis)
    - After combine, reduce-scatter along the other axis aggregates expert results
      from different column devices (handled by the sharding framework via shard_specs)

    On single device (num_devices=1), dispatch/combine are no-ops and this
    produces identical results to SparseMLP.
    """

    def __init__(
        self,
        original_mlp,
        num_experts: int,
        num_experts_per_tok: int,
        num_devices: int = 1,
        cluster_axis: int = 0,
        config: Optional[object] = None,
        activation_type: str = ACTIVATION_GPT_OSS,
        dispatch_devices: Optional[int] = None,
        cpu_forward_module: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.cluster_axis = cluster_axis
        self.activation_type = activation_type

        # num_devices: total mesh devices (for expert_mapping D dimension)
        # dispatch_devices: devices along cluster_axis (for BD = B * dispatch_devices)
        # When dispatch_devices is None, defaults to num_devices (single-axis or flat mesh)
        self.num_devices = num_devices
        self.dispatch_devices = (
            dispatch_devices if dispatch_devices is not None else num_devices
        )
        # When True, uses dense torch.matmul instead of sparse_matmul.
        # Skips remap (no sparsity mask needed). Demo-style approach.
        self.use_dense_matmul = False

        # Keep original MLP for CPU golden path.
        # cpu_forward_module overrides original_mlp when the adapter wrapping
        # doesn't have its own forward (e.g. DeepseekV3MoEToA2AAdapter).
        # Use object.__setattr__ to avoid nn.Module registering it as a submodule,
        # which would cause Dynamo to try tracing it (and fail on numpy ops).
        object.__setattr__(
            self,
            "_original_mlp",
            cpu_forward_module if cpu_forward_module is not None else original_mlp,
        )

        # Copy references to original module's components
        self.router = original_mlp.router
        self.experts = original_mlp.experts
        self.intermediate_size = self.experts.intermediate_size

        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = self.experts.down_proj.shape[-1]

        # GPT-OSS specific activation parameters (used when activation_type=gpt_oss)
        self.alpha = getattr(self.experts, "alpha", 1.702)
        self.limit = getattr(self.experts, "limit", 7.0)

        # Expert-to-device mapping [1, 1, E, D] where D = num_devices (total)
        # Maps each expert to its owning device. When cluster_axis=0, the dispatch
        # kernel derives the target row from the device_id in the mapping.
        mapping = build_expert_mapping(num_experts, num_devices)
        self.register_buffer("expert_mapping", mapping)

    @torch.compiler.disable
    def _cpu_forward(self, hidden_states):
        """CPU golden path: call original MLP forward directly.

        Decorated with @torch.compiler.disable so Dynamo won't trace into it —
        original forward may contain numpy ops or other incompatible constructs.
        """
        result = self._original_mlp(hidden_states)
        if isinstance(result, tuple):
            return result
        return result, None

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok

        # CPU golden path
        if hidden_states.device.type == "cpu":
            return self._cpu_forward(hidden_states)

        # 1. Router — pass 3D for RouterAdapter (handles 3D→2D internally),
        # flatten to 2D for raw routers (e.g. GptOssTopKRouter).
        router_input = hidden_states
        if not hasattr(self.router, "gate"):
            router_input = hidden_states.view(-1, hidden_size)
        router_scores, router_indices = _unpack_router_output(
            self.router(router_input), self.num_experts
        )

        # 2. Dispatch: route tokens to devices along cluster_axis
        # Dispatch accepts [B, S, H] and [B*S, K] directly.
        effective_dispatch = self.dispatch_devices
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            hidden_states,
            router_indices,
            self.expert_mapping,
            num_devices=effective_dispatch,
            cluster_axis=self.cluster_axis,
        )
        # dispatched: [1, BD, S, H],  metadata: [1, BD, S, K]
        # Reshape metadata to [1, 1, BD*S, K] so combine's output_shard_dim=2
        # sees tokens on dim 2 (matching demo layout). Just a reshape, no permute.
        BD = dispatched.shape[1]
        metadata = metadata.reshape(1, 1, BD * seq_len, metadata.shape[-1])
        E = self.num_experts
        if self.use_dense_matmul:
            # Dense matmul with M=32 tiling to avoid large intermediate tensors.
            # torch.matmul doesn't go through custom_ops, so tiling is done here.
            M = 32
            split_seq = seq_len % M == 0 and seq_len >= M
            split_bd = BD % M == 0 and BD >= M
            assert (
                split_seq or split_bd
            ), f"Neither seq_len ({seq_len}) nor BD ({BD}) is divisible by M={M}"
            dim_a = BD if split_seq else BD // M
            dim_b = seq_len // M if split_seq else seq_len

            down_proj = self.experts.down_proj.view(
                1, E, self.intermediate_size, -1
            )  # [1, E, inter, H]
            down_bias = self.experts.down_proj_bias

            # Tile dispatched [1, BD, S, H] → [dim_a, dim_b, M, H]
            if split_seq:
                hidden_4d = dispatched.view(BD, seq_len // M, M, hidden_size)
            else:
                hidden_4d = dispatched.view(BD // M, M, seq_len, hidden_size)
                hidden_4d = hidden_4d.permute(0, 2, 1, 3)

            tokens = hidden_4d.reshape(-1, hidden_size)
            has_fused = hasattr(self.experts, "gate_up_proj")

            if has_fused:
                # Fused gate_up: single matmul
                gate_up_proj = self.experts.gate_up_proj  # [E, H, inter*2]
                gate_up_bias = self.experts.gate_up_proj_bias
                weights_gu_flat = gate_up_proj.permute(1, 0, 2).reshape(
                    hidden_size, E * self.intermediate_size * 2
                )
                gate_up_flat = torch.matmul(tokens, weights_gu_flat)
                gate_up_out = gate_up_flat.view(
                    dim_a, dim_b, M, E, self.intermediate_size * 2
                )
                if gate_up_bias is not None:
                    gate_up_out = gate_up_out + gate_up_bias
                activated = _moe_activation(
                    gate_up_out, self.activation_type, self.alpha, self.limit
                )
            else:
                # Separate gate/up: two matmuls
                gate_proj = self.experts.gate_proj  # [E, H, inter]
                up_proj = self.experts.up_proj  # [E, H, inter]
                gate_bias = self.experts.gate_proj_bias
                up_bias = self.experts.up_proj_bias

                weights_gate_flat = gate_proj.permute(1, 0, 2).reshape(
                    hidden_size, E * self.intermediate_size
                )
                gate_flat = torch.matmul(tokens, weights_gate_flat)
                gate_out = gate_flat.view(dim_a, dim_b, M, E, self.intermediate_size)
                if gate_bias is not None:
                    gate_out = gate_out + gate_bias

                weights_up_flat = up_proj.permute(1, 0, 2).reshape(
                    hidden_size, E * self.intermediate_size
                )
                up_flat = torch.matmul(tokens, weights_up_flat)
                up_out = up_flat.view(dim_a, dim_b, M, E, self.intermediate_size)
                if up_bias is not None:
                    up_out = up_out + up_bias

                if self.activation_type == ACTIVATION_DEEPSEEK:
                    activated = F.silu(gate_out) * up_out
                else:
                    gate_out = gate_out.clamp(max=self.limit)
                    up_out = up_out.clamp(-self.limit, self.limit)
                    glu = gate_out * torch.sigmoid(gate_out * self.alpha)
                    activated = (up_out + 1) * glu

            # Down: bmm over experts — [E, T, inter] @ [E, inter, H] → [E, T, H]
            act_per_expert = activated.permute(0, 1, 3, 2, 4).reshape(
                dim_a * dim_b * M, E, self.intermediate_size
            )
            act_per_expert = act_per_expert.permute(
                1, 0, 2
            )  # [E, dim_a*dim_b*M, inter]
            down_per_expert = down_proj.squeeze(0)  # [E, inter, H]
            down_out = torch.bmm(
                act_per_expert, down_per_expert
            )  # [E, dim_a*dim_b*M, H]
            down_out = down_out.permute(1, 0, 2)  # [dim_a*dim_b*M, E, H]
            down_out = down_out.view(dim_a, dim_b, M, E, hidden_size)

            # Untile → [E, 1, BD*S, H] for combine with output_shard_dim=2
            down_out = down_out.view(dim_a, dim_b, E, M, hidden_size)
            down_out = down_out.permute(0, 1, 3, 2, 4)  # [dim_a, dim_b, M, E, H]
            if down_bias is not None:
                down_out = down_out + down_bias
            # E to front, merge all spatial dims into one token dim
            down_out = down_out.permute(3, 0, 1, 2, 4)  # [E, dim_a, dim_b, M, H]
            down_out = down_out.reshape(E, 1, BD * seq_len, hidden_size)

        else:
            # ===== Fused moe_expert_token_remap path =====
            _, sparsity_remap = torch.ops.tt.moe_expert_token_remap(
                router_scores,
                self.expert_mapping,
                metadata,
                num_devices=effective_dispatch,
            )

            down_out = self.experts.sparse_forward(
                dispatched,
                sparsity_remap,
                self.activation_type,
                self.alpha,
                self.limit,
                output_shape=(BD, seq_len),
            )

        # sparse_forward returns [E, 1, BD*S, H] — combine with output_shard_dim=2.
        combined = torch.ops.tt.all_to_all_combine(
            down_out,
            metadata,
            self.expert_mapping,
            num_devices=effective_dispatch,
            cluster_axis=self.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=2,
        )
        # combined: [K, 1, B*S, H] with output_shard_dim=2

        # Weighted sum
        E = self.num_experts
        expert_range = torch.arange(E, device=router_scores.device)
        one_hot = (router_indices.unsqueeze(-1) == expert_range).to(
            router_scores.dtype
        )  # [T, K, E]
        topk_weights = torch.einsum("te,tke->tk", router_scores, one_hot)  # [T, K]
        if seq_len == 1:
            topk_weights = topk_weights.view(batch_size, K)
            topk_weights = topk_weights.permute(1, 0).unsqueeze(-1)  # [K, B, 1]
            output = (combined.squeeze(1) * topk_weights).sum(dim=0)  # [B, H]
            output = output.unsqueeze(1)  # [B, 1, H]
        else:
            topk_weights = topk_weights.view(batch_size, seq_len, K)
            topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)  # [K, B, S, 1]
            output = (combined * topk_weights).sum(dim=0)  # [B, S, H]

        return output.to(hidden_states.dtype), router_scores


class SparseMOEGPT(A2aSparseMLP):
    """
    GPT-OSS-specific MoE wrapper.

    By default this reuses the existing A2A sparse path directly. Setting
    TT_TORCH_ENABLE_GPT_OSS_COMPOSITE=1 switches decode (S=1) execution to the
    composite path so the new SHLO/TTIR plumbing can be exercised without
    changing the module replacement logic.

    When composite mode is enabled, expert weights are also preprocessed into
    the fused MoE kernel layout (tile-interleaved, core-sharded) matching
    tt-metal's ``_prepare_w0_b0_w1_b1_tensor`` / ``_prepare_w2_b2_tensor``.
    """

    def __init__(self, *args, mesh_shape: Optional[tuple] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_decode_composite = (
            os.environ.get("TT_TORCH_ENABLE_GPT_OSS_COMPOSITE", "0") == "1"
        )
        # tt-metal keeps separate routing tables for dispatch and moe_gpt.
        # They currently carry the same mapping data but participate in
        # different stages of the fused decode flow.
        self.register_buffer("moe_gpt_mapping", self.expert_mapping.clone())

        # Preprocess weights into fused kernel layout when composite mode is on.
        self._has_fused_weights = False
        if self.use_decode_composite and mesh_shape is not None:
            gate_up = self.experts.gate_up_proj
            gate_up_bias = self.experts.gate_up_proj_bias
            down = self.experts.down_proj
            down_bias = self.experts.down_proj_bias

            fused_w0_w1, fused_w2 = preprocess_fused_moe_weights(
                gate_up_proj=gate_up.data,
                gate_up_proj_bias=gate_up_bias.data,
                down_proj=down.data,
                down_proj_bias=down_bias.data,
                num_experts=self.num_experts,
                num_devices=self.num_devices,
                cluster_axis=self.cluster_axis,
                mesh_shape=mesh_shape,
            )

            self.fused_w0_w1 = nn.Parameter(fused_w0_w1)
            self.fused_w2 = nn.Parameter(fused_w2)
            self._has_fused_weights = True

    def forward(self, hidden_states):
        if (
            not self.use_decode_composite
            or hidden_states.device.type != "xla"
            or hidden_states.shape[1] != 1
        ):
            return super().forward(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        router_bias = getattr(self.router, "bias", None)
        topk_indices, topk_scores = _composite_topk_router_gpt(
            hidden_states,
            self.router.weight,
            router_bias,
            self.num_experts_per_tok,
            self.num_experts,
        )

        # Build keyword args for the composite; include preprocessed fused
        # weights when available so tt-MLIR can consume them directly.
        composite_kwargs = dict(
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            dispatch_mapping=self.expert_mapping,
            moe_gpt_mapping=self.moe_gpt_mapping,
            gate_up_proj=self.experts.gate_up_proj,
            gate_up_proj_bias=self.experts.gate_up_proj_bias,
            down_proj=self.experts.down_proj,
            down_proj_bias=self.experts.down_proj_bias,
            num_devices=self.dispatch_devices,
            cluster_axis=self.cluster_axis,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            intermediate_size=self.intermediate_size,
            alpha=self.alpha,
            limit=self.limit,
        )
        if self._has_fused_weights:
            composite_kwargs["fused_w0_w1"] = self.fused_w0_w1
            composite_kwargs["fused_w2"] = self.fused_w2

        output = _composite_moe_gpt_decode(**composite_kwargs)

        router_scores = _scatter_compact_router_scores(
            topk_indices.view(batch_size * seq_len, self.num_experts_per_tok),
            topk_scores.view(batch_size * seq_len, self.num_experts_per_tok),
            self.num_experts,
        )
        return output.to(hidden_states.dtype), router_scores


class A2aSparseStackedMlp(nn.Module):
    """
    Sparse MLP with all-to-all dispatch/combine for multi-device expert parallelism.

    Same as A2aSparseMLP but pre-deinterleaves gate_up_proj weights and biases
    in __init__ so the forward pass uses contiguous splits ([:inter] / [inter:])
    instead of strided slices ([::2] / [1::2]).
    """

    def __init__(
        self,
        original_mlp,
        num_experts: int,
        num_experts_per_tok: int,
        num_devices: int = 1,
        cluster_axis: int = -1,
        config: Optional[object] = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_devices = num_devices
        self.cluster_axis = cluster_axis

        # Copy references to original module's components
        self.router = original_mlp.router
        orig_experts = original_mlp.experts

        if hasattr(orig_experts, "gate_up_proj"):
            self.intermediate_size = orig_experts.gate_up_proj.shape[-1] // 2
        else:
            raise ValueError("Expected fused gate_up_proj in experts module")

        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = orig_experts.down_proj.shape[-1]

        # GPT-OSS specific activation parameters
        self.alpha = getattr(orig_experts, "alpha", 1.702)
        self.limit = getattr(orig_experts, "limit", 7.0)

        # New experts container — preserves layer.mlp.experts.* path for shard_specs
        # (shard_specs is built after replacement)
        self.experts = nn.Module()

        # De-interleave gate_up_proj: [g0, u0, g1, u1, ...] -> [g0, g1, ..., u0, u1, ...]
        # Pre-reshape to [1, E, H, inter*2] for sparse_matmul (no unsqueeze in forward)
        orig_w = orig_experts.gate_up_proj
        gate_w = orig_w[..., ::2].contiguous()
        up_w = orig_w[..., 1::2].contiguous()
        self.experts.gate_up_proj = nn.Parameter(
            torch.cat([gate_w, up_w], dim=-1).unsqueeze(0)
        )

        orig_b = orig_experts.gate_up_proj_bias
        gate_b = orig_b[..., ::2].contiguous()
        up_b = orig_b[..., 1::2].contiguous()
        self.experts.gate_up_proj_bias = nn.Parameter(torch.cat([gate_b, up_b], dim=-1))

        # Down proj / bias — pre-reshape to [1, E, inter, H] for sparse_matmul
        self.experts.down_proj = nn.Parameter(
            orig_experts.down_proj.data.view(
                1, num_experts, self.intermediate_size, hidden_size
            )
        )
        self.experts.down_proj_bias = orig_experts.down_proj_bias

        # Expert-to-device mapping [1, 1, E, D]
        mapping = build_expert_mapping(num_experts, num_devices)
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok

        # 1. Router
        # GptOssTopKRouter returns (router_logits, router_scores, router_indices):
        #   router_out[1]: softmax(top_k_logits) compact probs [B, S, top_k]
        #   router_out[-1]: top-k indices [B, S, top_k]
        router_out = self.router(hidden_states)
        router_scores = router_out[1]  # [B, S, top_k] compact softmax probabilities
        router_indices = router_out[-1]  # [B, S, top_k] compact indices

        # 2. Reshape for dispatch: tt-metal expects [B, 1, S, H] format
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = router_indices.view(batch_size, 1, seq_len, K)

        # 3. Dispatch: route tokens to devices with selected experts
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x,
            expert_indices,
            self.expert_mapping,
            num_devices=self.num_devices,
            cluster_axis=self.cluster_axis,
        )
        # dispatched: [1, B*D, S, H]
        # metadata:   [1, B*D, S, K]

        BD = dispatched.shape[1]

        # 4. Build sparsity mask from metadata
        metadata_indices = metadata.view(BD, seq_len, 1, K)  # [BD, S, 1, K]
        sparsity = torch.zeros(
            BD,
            seq_len,
            1,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        sparsity.scatter_(
            dim=-1,
            index=metadata_indices,
            src=torch.ones_like(metadata_indices, dtype=hidden_states.dtype),
        )

        # 5. Gate+Up projection (stacked layout)
        hidden_4d = dispatched.view(BD, seq_len, 1, hidden_size)
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d,
            self.experts.gate_up_proj,
            sparsity,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        gate_up_out = gate_up_out.view(
            BD, seq_len, self.num_experts, self.intermediate_size * 2
        )
        gate_up_out = gate_up_out + self.experts.gate_up_proj_bias

        # 6. Split & Activation (contiguous stacked layout)
        gate_up_out = gate_up_out.clamp(max=self.limit)
        gate_out = gate_up_out[..., : self.intermediate_size]
        up_out = gate_up_out[..., self.intermediate_size :]

        # gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(min=-self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # 7. Down projection
        activated_reshaped = activated.view(
            BD * seq_len, self.num_experts, 1, self.intermediate_size
        )
        sparsity_down = sparsity.view(1, 1, BD * seq_len, self.num_experts)

        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            self.experts.down_proj,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        down_out = down_out.squeeze(2)
        down_out = down_out + self.experts.down_proj_bias

        # 8. Reshape for combine: [E, BD, S, H]
        down_out = down_out.view(BD, seq_len, self.num_experts, hidden_size)
        down_out = down_out.permute(2, 0, 1, 3)

        # 9. Combine: gather expert outputs back to original positions
        combined = torch.ops.tt.all_to_all_combine(
            down_out,
            metadata,
            self.expert_mapping,
            num_devices=self.num_devices,
            cluster_axis=self.cluster_axis,
            num_experts_per_tok=K,
        )
        # combined: [K, B, S, H]

        # 10. Weighted sum
        # router_scores is compact [B, S, K] softmax probabilities — use directly.
        topk_weights = router_scores.view(batch_size, seq_len, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)

        output = (combined * topk_weights).sum(dim=0)
        output = output.view(batch_size, seq_len, hidden_size)

        return output.to(hidden_states.dtype), router_scores


class DeepseekV3MoEToA2AAdapter(nn.Module):
    """
    Adapter that converts DeepseekV3MoE to A2aSparseMLP-compatible interface.

    DeepseekV3MoE has:
    - gate (MoEGate): returns (topk_idx, topk_weight)
    - experts: ModuleList of DeepseekV3MLP with gate_proj, up_proj, down_proj (separate)

    A2aSparseMLP expects:
    - router: returns (scores, indices)
    - experts: gate_proj [E, H, inter], up_proj [E, H, inter], down_proj [E, inter, H], biases
    """

    class RouterAdapter(nn.Module):
        """Wraps MoEGate to return (scores, indices) for A2aSparseMLP.

        Supports three gate patterns:
        1. Deepseek-style: gate returns (topk_idx, topk_weight)
        2. Other gates: gate returns (topk_weight, topk_idx)
        3. Raw-logits gates (e.g. Glm4MoeTopkRouter): gate returns a single
           router_logits tensor. Requires route_tokens_to_experts_fn to convert
           logits -> (topk_idx, topk_weight).
        """

        def __init__(
            self, gate: nn.Module, n_experts: int, route_tokens_to_experts_fn=None
        ):
            super().__init__()
            self.gate = gate
            self.n_experts = n_experts
            self._route_fn = route_tokens_to_experts_fn
            # Deepseek-style MoEGate returns (topk_idx, topk_weight) and expects
            # 3D [batch, seq, hidden] input, flattening internally. Other gates
            # (e.g. deepseek_v3_2_exp Gate) return (weights, indices) and operate
            # on a flattened 2D [batch * seq, hidden] input.
            self._gate_returns_idx_first = hasattr(gate, "n_routed_experts")

        def forward(self, hidden_states):
            gate_input = hidden_states
            if hidden_states.dim() == 3 and not self._gate_returns_idx_first:
                gate_input = hidden_states.view(-1, hidden_states.shape[-1])

            gate_output = self.gate(gate_input)

            if self._route_fn is not None:
                # Raw-logits gate: use external routing function
                topk_idx, topk_weight = self._route_fn(gate_output)
            elif isinstance(gate_output, (tuple, list)):
                out1, out2 = gate_output
                if self._gate_returns_idx_first:
                    topk_idx, topk_weight = out1, out2
                else:
                    topk_weight, topk_idx = out1, out2
            else:
                raise ValueError(
                    f"Gate returned a single tensor but no route_tokens_to_experts_fn "
                    f"was provided. Gate type: {type(self.gate).__name__}"
                )

            scores = _topk_to_sparse_scores(topk_weight, topk_idx, self.n_experts)
            return scores, topk_idx

    class PreStackedFusedExperts(_SparseForwardMixin, nn.Module):
        """Wraps experts that already have stacked fused weights (e.g. Glm4MoeNaiveMoe).

        Expects gate_up_proj [E, 2*inter, H] and down_proj [E, H, inter] in nn.Linear
        convention (out_features, in_features). Transposes and splits into separate
        gate_proj [E, H, inter], up_proj [E, H, inter], down_proj [E, inter, H]
        stored as actual Parameters for shard spec compatibility.

        Also keeps reference to original experts module for CPU golden path.
        """

        def __init__(self, experts):
            super().__init__()
            self.original_experts = experts
            # gate_up_proj: [E, 2*inter, H] -> transpose -> [E, H, 2*inter] -> split
            gate_up_t = experts.gate_up_proj.data.transpose(1, 2)  # [E, H, 2*inter]
            inter = gate_up_t.shape[-1] // 2
            self.intermediate_size = inter
            self.gate_proj = nn.Parameter(gate_up_t[..., :inter].contiguous())
            self.up_proj = nn.Parameter(gate_up_t[..., inter:].contiguous())
            # down_proj: [E, H, inter] -> transpose -> [E, inter, H]
            self.down_proj = nn.Parameter(
                experts.down_proj.data.transpose(1, 2).contiguous()
            )

        # No bias for pre-stacked fused experts
        gate_proj_bias = None
        up_proj_bias = None
        down_proj_bias = None

        # Aliases for _sparse_expert_forward (w1=gate, w2=down, w3=up)
        w1 = property(lambda self: self.gate_proj)
        w1_bias = None
        w2 = property(lambda self: self.down_proj)
        w2_bias = None
        w3 = property(lambda self: self.up_proj)
        w3_bias = None

    class StackedExperts(_SparseForwardMixin, nn.Module):
        """Stacks expert weights into w1 (gate), w2 (down), w3 (up) format.

        Supports expert layouts with separate projections:
        - DeepseekV3MLP: gate_proj, up_proj, down_proj
        - DeepseekV3-2 Expert: w1 (gate), w3 (up), w2 (down)

        Also keeps references to original expert modules for CPU golden path.
        """

        @staticmethod
        def _get_expert_layers(exp):
            """Return (gate_layer, up_layer, down_layer) from an expert module."""
            if hasattr(exp, "gate_proj"):
                return exp.gate_proj, exp.up_proj, exp.down_proj
            elif hasattr(exp, "w1"):
                return exp.w1, exp.w3, exp.w2
            else:
                raise ValueError(
                    f"Expert {type(exp).__name__} has neither gate_proj/up_proj/down_proj "
                    "nor w1/w3/w2 attributes."
                )

        def __init__(self, expert_list):
            super().__init__()
            experts_list = [e for e in expert_list if e is not None]
            if not experts_list:
                experts_list = list(expert_list)

            # Keep original experts for CPU golden path
            self.original_experts = nn.ModuleList(experts_list)

            first = experts_list[0]
            gate_layer, _, _ = self._get_expert_layers(first)
            inter = gate_layer.out_features
            has_bias = gate_layer.bias is not None

            gate_list, up_list, down_list = [], [], []
            gate_bias_list, up_bias_list, down_bias_list = [], [], []
            for exp in experts_list:
                g, u, d = self._get_expert_layers(exp)
                gate_list.append(g.weight.T)
                up_list.append(u.weight.T)
                down_list.append(d.weight.T)
                if has_bias:
                    gate_bias_list.append(g.bias)
                    up_bias_list.append(u.bias)
                    down_bias_list.append(d.bias)

            self.gate_proj = nn.Parameter(torch.stack(gate_list, dim=0))
            self.up_proj = nn.Parameter(torch.stack(up_list, dim=0))
            self.down_proj = nn.Parameter(torch.stack(down_list, dim=0))
            self.intermediate_size = inter
            if has_bias:
                self.gate_proj_bias = nn.Parameter(torch.stack(gate_bias_list, dim=0))
                self.up_proj_bias = nn.Parameter(torch.stack(up_bias_list, dim=0))
                self.down_proj_bias = nn.Parameter(torch.stack(down_bias_list, dim=0))
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None

        # Aliases for unified _sparse_expert_forward (w1=gate, w2=down, w3=up)
        w1 = property(lambda self: self.gate_proj)
        w1_bias = property(lambda self: self.gate_proj_bias)
        w2 = property(lambda self: self.down_proj)
        w2_bias = property(lambda self: self.down_proj_bias)
        w3 = property(lambda self: self.up_proj)
        w3_bias = property(lambda self: self.up_proj_bias)

    def __init__(self, moe_module):
        super().__init__()
        experts_module = moe_module.experts

        # Detect pre-stacked fused experts (e.g. Glm4MoeNaiveMoe) that have
        # gate_up_proj as a Parameter directly rather than a list of expert modules.
        pre_stacked_fused = (
            hasattr(experts_module, "gate_up_proj")
            and isinstance(experts_module.gate_up_proj, nn.Parameter)
            and not hasattr(experts_module, "__iter__")
        )

        if pre_stacked_fused:
            n_experts = getattr(experts_module, "num_experts", None)
            if n_experts is None:
                n_experts = experts_module.gate_up_proj.shape[0]
        else:
            n_experts = getattr(moe_module.gate, "n_routed_experts", None)
            if n_experts is None:
                n_experts = len([e for e in experts_module if e is not None])
                if n_experts == 0:
                    n_experts = len(list(experts_module))

        # Detect gates that return raw logits (e.g. Glm4MoeTopkRouter) and need
        # a separate routing function to produce (topk_idx, topk_weight).
        route_fn = None
        if hasattr(moe_module, "route_tokens_to_experts"):
            route_fn = self._build_route_fn(moe_module)
        self.router = self.RouterAdapter(moe_module.gate, n_experts, route_fn)

        if pre_stacked_fused:
            self.experts = self.PreStackedFusedExperts(experts_module)
        else:
            experts_list = [e for e in experts_module if e is not None]
            if not experts_list:
                experts_list = list(experts_module)
            if len(experts_list) != n_experts:
                raise ValueError(
                    "DeepseekV3MoEToA2AAdapter requires ep_size=1 (all experts on one process). "
                    f"Got {len(experts_list)} experts, expected {n_experts}."
                )
            self.experts = self.StackedExperts(experts_list)

    @staticmethod
    def _build_route_fn(moe_module):
        """Build a standalone routing function from a MoE module's route_tokens_to_experts.

        Captures config values as constants so the function doesn't depend on the
        original moe_module being alive during tracing.
        """
        gate = moe_module.gate
        n_routed_experts = moe_module.n_routed_experts
        n_group = moe_module.n_group
        topk_group = moe_module.topk_group
        top_k = moe_module.top_k
        norm_topk_prob = moe_module.norm_topk_prob
        routed_scaling_factor = moe_module.routed_scaling_factor

        def route_tokens_to_experts(router_logits):
            router_logits = router_logits.sigmoid()
            router_logits_for_choice = router_logits + gate.e_score_correction_bias
            group_scores = (
                router_logits_for_choice.view(-1, n_group, n_routed_experts // n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, n_group, n_routed_experts // n_group)
                .reshape(-1, n_routed_experts)
            )
            scores_for_choice = router_logits_for_choice.masked_fill(
                ~score_mask.bool(), 0.0
            )
            topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[
                1
            ]
            topk_weights = router_logits.gather(1, topk_indices)
            if norm_topk_prob:
                denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                topk_weights /= denominator
            topk_weights = topk_weights * routed_scaling_factor
            return topk_indices, topk_weights

        return route_tokens_to_experts


class A2aSparseMLPWithSharedExperts(nn.Module):
    """Wraps A2aSparseMLP and adds shared_experts output; returns single tensor for layer compatibility."""

    def __init__(
        self, a2a_mlp: A2aSparseMLP, shared_experts: Optional[nn.Module] = None
    ):
        super().__init__()
        self.mlp = a2a_mlp
        self.shared_experts = shared_experts

    def forward(self, hidden_states):
        out, _ = self.mlp(hidden_states)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


def create_a2a_from_deepseek_v3_moe(
    moe_module,
    config,
    num_devices: int = 8,
    cluster_axis: int = 0,
    dispatch_devices: Optional[int] = None,
) -> A2aSparseMLPWithSharedExperts:
    """
    Create A2aSparseMLP from DeepseekV3MoE.

    Args:
        moe_module: DeepseekV3MoE instance
        config: Model config (DeepseekV3Config)
        num_devices: Total mesh devices (for expert_mapping D dimension)
        cluster_axis: Mesh axis for dispatch/combine (0=rows, 1=cols)
        dispatch_devices: Devices along cluster_axis (for BD = B * dispatch_devices).
            Defaults to num_devices when None (single-axis dispatch).
    """
    adapter = DeepseekV3MoEToA2AAdapter(moe_module)
    num_experts = getattr(config, "n_routed_experts", None) or getattr(
        config, "num_local_experts", len(list(moe_module.experts))
    )
    num_experts_per_tok = getattr(config, "num_experts_per_tok", None) or getattr(
        config, "n_activated_experts", 6
    )
    a2a_mlp = A2aSparseMLP(
        adapter,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        config=config,
        activation_type=ACTIVATION_DEEPSEEK,
        dispatch_devices=dispatch_devices,
        cpu_forward_module=moe_module,
    )
    shared_experts = getattr(moe_module, "shared_experts", None)
    return A2aSparseMLPWithSharedExperts(a2a_mlp, shared_experts)


def _is_moe_mlp(module: nn.Module) -> bool:
    """Check if a module is an MoE MLP that can be replaced with SparseMLP."""
    # Check for common MoE MLP patterns
    module_name = type(module).__name__.lower()

    # Known MoE MLP class names
    moe_patterns = ["gptossmlp", "mixtralmlp", "qwen2moemlp", "deepseekmlp", "deepseek"]

    if any(pattern in module_name for pattern in moe_patterns):
        return True

    # Check if module has router/gate and experts attributes (common MoE pattern)
    has_router = hasattr(module, "router") or hasattr(module, "gate")
    has_experts = hasattr(module, "experts")

    return has_router and has_experts


def _get_moe_config(module: nn.Module) -> Optional[tuple]:
    """Extract MoE configuration from a module."""
    try:
        num_experts = None
        # Try to get from experts
        if hasattr(module, "experts"):
            experts = module.experts
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is None and hasattr(experts, "gate_proj"):
                num_experts = experts.gate_proj.shape[0]
            elif num_experts is None and hasattr(experts, "gate_up_proj"):
                num_experts = experts.gate_up_proj.shape[0]

        # Try to get num_experts_per_tok from router
        if hasattr(module, "router"):
            router = module.router
            num_experts_per_tok = getattr(router, "top_k", None)
            if num_experts_per_tok is None:
                num_experts_per_tok = getattr(router, "num_experts_per_tok", 2)
        else:
            num_experts_per_tok = 2  # Default

        if num_experts is not None:
            return (num_experts, num_experts_per_tok)
    except Exception:
        pass

    return None


def enable_sparse_mlp(
    model: nn.Module,
    mesh: tuple,
    cluster_axis: int = 0,
    target_classes: Optional[List[Type]] = None,
    verbose: bool = False,
    config: Optional[object] = None,
) -> nn.Module:
    """
    Replace MoE MLP layers in a model with A2aSparseMLP implementations.
    """
    replaced_count = 0

    if config is None:
        config = getattr(model, "config", None)

    num_devices = mesh[0] * mesh[1]
    dispatch_devices = mesh[cluster_axis]

    def replace_mlp(parent: nn.Module, name: str, module: nn.Module):
        nonlocal replaced_count

        should_replace = False
        if target_classes:
            should_replace = any(isinstance(module, cls) for cls in target_classes)
        else:
            should_replace = _is_moe_mlp(module)

        if not should_replace:
            return False

        module_type_name = type(module).__name__.lower()

        if (
            "deepseek" in module_type_name
            and hasattr(module, "gate")
            and hasattr(module, "experts")
        ):
            sparse_mlp = create_a2a_from_deepseek_v3_moe(
                moe_module=module,
                config=config,
                num_devices=num_devices,
                cluster_axis=cluster_axis,
                dispatch_devices=dispatch_devices,
            )
            setattr(parent, name, sparse_mlp)
            replaced_count += 1
            if verbose:
                print(
                    f"[SparseMLP] Replaced {name}: {type(module).__name__} -> DeepseekV3MoEToA2AAdapter"
                )
            return True

        moe_config = _get_moe_config(module)
        if moe_config is None and config is not None:
            num_experts = getattr(config, "num_local_experts", None)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
            if num_experts is not None:
                moe_config = (num_experts, num_experts_per_tok)

        if moe_config is None:
            if verbose:
                print(f"[SparseMLP] Skipping {name}: could not determine MoE config")
            return False

        num_experts, num_experts_per_tok = moe_config

        # Wrap fused experts (e.g. GptOssExperts) with FusedExpertsWrapper
        # so they have sparse_forward() like StackedExperts
        if (
            hasattr(module, "experts")
            and hasattr(module.experts, "gate_up_proj")
            and not hasattr(module.experts, "sparse_forward")
        ):
            module.experts = FusedExpertsWrapper(module.experts)

        sparse_mlp_cls = SparseMOEGPT if "gptoss" in module_type_name else A2aSparseMLP
        extra_kwargs = {}
        if sparse_mlp_cls is SparseMOEGPT:
            extra_kwargs["mesh_shape"] = mesh
        sparse_mlp = sparse_mlp_cls(
            module,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_devices=num_devices,
            cluster_axis=cluster_axis,
            config=config,
            dispatch_devices=dispatch_devices,
            cpu_forward_module=module,
            **extra_kwargs,
        )

        setattr(parent, name, sparse_mlp)
        replaced_count += 1

        if verbose:
            print(
                f"[SparseMLP] Replaced {name}: {type(module).__name__} -> {sparse_mlp_cls.__name__} "
                f"(experts={num_experts}, num_devices={num_devices})"
            )
        return True

    # Traverse and replace. Track replaced prefixes to skip their children.
    replaced_prefixes = set()
    for name, module in list(model.named_modules()):
        # Skip children of already-replaced modules
        if any(name.startswith(p + ".") or name == p for p in replaced_prefixes):
            continue

        if "." in name:
            parts = name.rsplit(".", 1)
            parent_name, child_name = parts
            try:
                parent = model.get_submodule(parent_name)
            except AttributeError:
                continue
        else:
            parent = model
            child_name = name

        if replace_mlp(parent, child_name, module):
            replaced_prefixes.add(name)

    if verbose:
        print(f"[SparseMLP] Total layers replaced: {replaced_count}")

    return model


def get_moe_shard_specs(
    model: nn.Module, original_spec_fn, mesh_names: tuple
) -> Dict[str, Any]:
    shard_specs = original_spec_fn(model)
    for layer in model.model.layers:
        if isinstance(layer.mlp, A2aSparseMLP):
            mlp = layer.mlp
            experts = mlp.experts
            compound = (mesh_names[0], mesh_names[1])

            if hasattr(experts, "gate_up_proj"):
                # Fused gate_up (e.g. GPT-OSS via FusedExpertsWrapper)
                shard_specs[experts.gate_up_proj] = (compound, None, None)
                if experts.gate_up_proj_bias is not None:
                    shard_specs[experts.gate_up_proj_bias] = (compound, None)
            else:
                # Separate gate/up (e.g. Deepseek via StackedExperts)
                shard_specs[experts.gate_proj] = (compound, None, None)
                shard_specs[experts.up_proj] = (compound, None, None)
                if experts.gate_proj_bias is not None:
                    shard_specs[experts.gate_proj_bias] = (compound, None)
                if experts.up_proj_bias is not None:
                    shard_specs[experts.up_proj_bias] = (compound, None)

            shard_specs[experts.down_proj] = (compound, None, None)
            if experts.down_proj_bias is not None:
                shard_specs[experts.down_proj_bias] = (compound, None)

            # Fused kernel weight sharding: dim 0 = cluster_axis, dim 1 = other axis
            if isinstance(mlp, SparseMOEGPT) and mlp._has_fused_weights:
                cluster_axis = mlp.cluster_axis
                fused_dim0 = mesh_names[cluster_axis]
                fused_dim1 = mesh_names[1 - cluster_axis]
                shard_specs[mlp.fused_w0_w1] = (
                    fused_dim0, fused_dim1, None, None, None, None
                )
                shard_specs[mlp.fused_w2] = (
                    fused_dim0, fused_dim1, None, None, None, None
                )

    return shard_specs
