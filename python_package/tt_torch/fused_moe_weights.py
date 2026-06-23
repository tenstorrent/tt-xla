# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel weight preprocessing.

``ttir.moe_gpt_decode`` lowers to ``ttnn.moe_gpt``, whose 6D ``fused_w0_w1`` /
``fused_w2`` weight operands follow tt-metal's fused-decode kernel layout. When
the frontend does NOT supply them, the TTIR decomposition substitutes zero
placeholders (``TTIRToTTIRDecomposition.cpp``), so the kernel computes on zero
weights. These helpers replicate tt-metal's ``create_fused_moe_gpt_config``
pure-PyTorch transform to produce the real fused weights from the HF
``gate_up_proj`` / ``down_proj`` (+biases), so the frontend can plumb them
through ``torch.ops.tt.moe_gpt``'s optional fused-weight operands.

The output tensors are laid out so an XLA ``("batch", "model", ...)`` sharding
on dims 0,1 distributes them across a ``(ring_devices x mesh_cols)`` mesh into
the per-device ``[12, 1, E_local, 4|2, K|N+32, 128]`` shards the kernel expects.

Ported verbatim from the GPT-OSS e2e branch
(``odjuricic/wip-gpt-oss-e2e-squashed``: ``tt_torch/sparse_mlp.py``).
"""

from typing import Tuple

import torch
from torch.nn import functional as F

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
    interleave/shard/pad as the bias-free path.
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
    N_grouped = all_groups.view(num_cores, L, E, 2, Nt_all, _TILE_SIZE, 4 * _TILE_SIZE)

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

    N_reordered = torch.stack(each_shard).view(num_cores, L, E, 2, -1, 4 * _TILE_SIZE)
    return N_reordered


def preprocess_fused_moe_weights(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_experts: int,
    num_devices: int,
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess MoE weights into the fused decode kernel layout.

    Replicates the pure-PyTorch weight transformation from tt-metal's
    ``create_fused_moe_gpt_config``: de-interleave gate/up, pad biases to
    tile height, tile-interleave w0/w1, core-shard, and ring-rotate w2.

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
        ``(fused_w0_w1, fused_w2)`` — preprocessed tensors ready for sharding:
        - fused_w0_w1: ``(ring_devices*12, mesh_cols, E_per_dev, 4, K+32, 128)``
        - fused_w2: ``(ring_devices*12, mesh_cols, E_per_dev, 2, N+32, 128)``
    """
    import gc

    ring_devices = mesh_shape[cluster_axis]
    mesh_cols = num_devices // ring_devices
    experts_per_cluster = num_experts // mesh_cols
    E = num_experts // num_devices  # experts per device
    K = gate_up_proj.shape[1]
    N = gate_up_proj.shape[2] // 2
    L = 1
    dtype = gate_up_proj.dtype

    # De-interleave gate_up_proj: [g0, u0, g1, u1, ...] -> w0 (gate), w1 (up)
    w0_all = gate_up_proj[..., ::2].contiguous()  # [num_experts, K, N]
    w1_all = gate_up_proj[..., 1::2].contiguous()
    del gate_up_proj

    w2_all = down_proj.contiguous()  # [num_experts, N, K]
    del down_proj

    # De-interleave biases
    b0_all = gate_up_proj_bias[..., ::2].contiguous()  # [num_experts, N]
    b1_all = gate_up_proj_bias[..., 1::2].contiguous()
    del gate_up_proj_bias
    b2_all = down_proj_bias.contiguous()  # [num_experts, K]
    del down_proj_bias

    # Pad biases to tile height (32 rows): [num_experts, N] -> [num_experts, 32, N]
    b0_all = F.pad(b0_all.unsqueeze(1), (0, 0, 0, _TILE_SIZE - 1))
    b1_all = F.pad(b1_all.unsqueeze(1), (0, 0, 0, _TILE_SIZE - 1))
    b2_all = F.pad(b2_all.unsqueeze(1), (0, 0, 0, _TILE_SIZE - 1))

    gc.collect()

    # Per-device weight preprocessing, organized as [rows, cols] for 2D mesh
    # sharding.
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
            w2_cols.append(_prepare_w2_b2_tensor(w2_d, b2_d, L, E, N, K))
        w0_w1_rows.append(torch.cat(w0_w1_cols, dim=1))  # cat cols on dim 1
        w2_rows.append(torch.cat(w2_cols, dim=1))

    del w0_all, w1_all, w2_all, b0_all, b1_all, b2_all
    gc.collect()

    fused_w0_w1 = torch.cat(w0_w1_rows, dim=0)  # cat rows on dim 0
    fused_w2 = torch.cat(w2_rows, dim=0)
    del w0_w1_rows, w2_rows

    return fused_w0_w1.to(dtype), fused_w2.to(dtype)
