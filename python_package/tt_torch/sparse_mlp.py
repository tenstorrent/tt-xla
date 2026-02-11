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

from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

# Activation types for A2aSparseMLP
ACTIVATION_GPT_OSS = "gpt_oss"  # clamp, sigmoid, alpha, glu
ACTIVATION_DEEPSEEK = "deepseek"  # SiLU (swish) for gate * up


class SparseMLP(nn.Module):
    """
    Sparse MLP implementation that uses sparse_matmul for MoE computation.

    This module wraps an existing MLP and replaces dense expert computation
    with sparse_matmul operations that skip inactive experts.

    Uses INTERLEAVED gate_up_proj layout directly from original model:
    - Weights stored as [g0, u0, g1, u1, ...] (interleaved)
    - Split with [::2]/[1::2] strided slices
    - TP sharding works because UpdateGlobalToLocalShapes pass handles
      strided slices where stride == shard_factor
    """

    def __init__(
        self,
        original_mlp,
        num_experts: int,
        num_experts_per_tok: int,
        config: Optional[object] = None,
    ):
        super().__init__()

        # Note: We intentionally do NOT store original_mlp to avoid memory duplication.
        # Only store references to the components we actually need.
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Copy references to original module's components
        # Keep same structure as original MLP: router + experts
        self.router = original_mlp.router
        self.experts = original_mlp.experts  # Keep same structure for sharding

        # Use INTERLEAVED gate_up_proj directly (no conversion needed)
        # The UpdateGlobalToLocalShapes pass now handles strided slices
        # where stride == shard_factor, making [::2]/[1::2] TP-compatible.
        if hasattr(self.experts, "gate_up_proj"):
            # intermediate_size is half of the last dimension (interleaved)
            self.intermediate_size = self.experts.gate_up_proj.shape[-1] // 2
        else:
            raise ValueError("Expected fused gate_up_proj in experts module")

        # Get hidden_size from config or infer from down_proj shape
        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            # Infer from down_proj shape: [E, inter, hidden]
            hidden_size = self.experts.down_proj.shape[-1]

        # GPT-OSS specific activation parameters
        self.alpha = getattr(self.experts, "alpha", 1.702)
        self.limit = getattr(self.experts, "limit", 7.0)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. Router Execution
        # router_scores: [batch*seq, num_experts] (scattered probabilities)
        # router_indices: [batch*seq, top_k]
        router_scores, router_indices = self.router(hidden_states)

        # 2. Create Sparsity Mask [batch, seq, 1, num_experts]
        sparsity = torch.zeros(
            batch_size,
            seq_len,
            1,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Reshape indices for scatter: [batch, seq, 1, top_k]
        topk_indices_unsqueezed = router_indices.view(
            batch_size, seq_len, 1, self.num_experts_per_tok
        )

        sparsity.scatter_(
            dim=-1,
            index=topk_indices_unsqueezed,
            src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_states.dtype),
        )

        # 3. Reshape Input for sparse_matmul [batch, seq, 1, hidden]
        hidden_4d = hidden_states.view(batch_size, seq_len, 1, hidden_size)

        # 4. Fused Gate+Up Projection
        # gate_up_weight: [1, E, hidden, inter*2]
        gate_up_proj = self.experts.gate_up_proj.unsqueeze(0)
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d,
            gate_up_proj,
            sparsity,
            nnz=0,  # Let runtime calculate
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )

        # Output: [batch, seq, 1, E, M, inter*2] where M=1
        # Reshape to [batch, seq, E, inter*2]
        gate_up_out = gate_up_out.view(
            batch_size, seq_len, self.num_experts, self.intermediate_size * 2
        )
        gate_up_out = gate_up_out + self.experts.gate_up_proj_bias

        # 5. Split & Activation (Interleaved Layout)
        # Slicing works for TP because stride (2) matches shard_factor
        gate_out = gate_up_out[..., ::2]  # Even indices
        up_out = gate_up_out[..., 1::2]  # Odd indices

        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # 6. Down Projection setup
        # activated: [batch*seq, E, 1, inter]
        activated_reshaped = activated.view(
            batch_size * seq_len, self.num_experts, 1, self.intermediate_size
        )

        # sparsity_down: [1, 1, batch*seq, E]
        sparsity_down = sparsity.view(1, 1, batch_size * seq_len, self.num_experts)

        # Reshape down_proj for sparse_matmul
        # down_proj: [1, E, inter, hidden]
        down_proj = self.experts.down_proj.view(
            1, self.num_experts, self.intermediate_size, hidden_size
        )

        # 7. Down Projection (sparse_matmul)
        # down_weight: [1, E, inter, hidden]
        # Output: [batch*seq, E, M, hidden] where M=1
        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            down_proj,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,  # Activations are sparse (only TopK are valid)
            is_input_b_sparse=False,
        )

        # Squeeze M dimension: [batch*seq, E, 1, hidden] -> [batch*seq, E, hidden]
        down_out = down_out.squeeze(2)
        down_out = down_out + self.experts.down_proj_bias

        # 8. Weighted Sum & Final Output
        # down_out: [BS, E, H]
        # router_scores: [BS, E] -> [BS, E, 1]
        # output: [BS, H] (Sum over Experts dim=1)
        output = (down_out * router_scores.unsqueeze(-1)).sum(dim=1)

        # Reshape back to [batch, seq, hidden]
        output = output.view(batch_size, seq_len, hidden_size)

        return output, router_scores


def build_expert_mapping(num_experts, num_devices):
    """
    Build one-hot expert-to-device mapping tensor.

    Creates a [1, 1, E, D] tensor where mapping[0, 0, i, d] = 1 means
    expert i resides on device d. Experts are sequentially distributed:
    experts 0..E/D-1 on device 0, E/D..2*E/D-1 on device 1, etc.

    Args:
        num_experts: Total number of experts (E)
        num_devices: Number of devices along dispatch axis (D)

    Returns:
        Tensor of shape [1, 1, E, D] with one-hot encoding
    """
    assert (
        num_experts % num_devices == 0
    ), f"num_experts ({num_experts}) must be divisible by num_devices ({num_devices})"
    mapping = torch.zeros(1, 1, num_experts, num_devices, dtype=torch.int64)
    experts_per_device = num_experts // num_devices
    for i in range(num_experts):
        device_id = i // experts_per_device
        mapping[0, 0, i, device_id] = 1
    return mapping


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
        flat_device_order: Optional[List[int]] = None,
        activation_type: str = ACTIVATION_GPT_OSS,
        dispatch_devices: Optional[int] = None,
        reduce_axis: Optional[int] = None,
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
        self.dispatch_devices = dispatch_devices if dispatch_devices is not None else num_devices

        # reduce_axis: mesh axis for reduce-scatter after combine (for 2D compound sharding)
        # If cluster_axis=0, reduce_axis=1 means reduce-scatter along axis_1 after combine
        # None means no reduce-scatter (single-axis EP or cluster_axis=-1 flatten)
        self.reduce_axis = reduce_axis

        # Copy references to original module's components
        self.router = original_mlp.router
        self.experts = original_mlp.experts

        if hasattr(self.experts, "gate_up_proj"):
            self.intermediate_size = self.experts.gate_up_proj.shape[-1] // 2
        else:
            raise ValueError("Expected fused gate_up_proj in experts module")

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
        if flat_device_order is not None:
            permuted = torch.zeros_like(mapping)
            for d in range(num_devices):
                permuted[:, :, :, flat_device_order[d]] = mapping[:, :, :, d]
            mapping = permuted
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok

        # 1. Router
        router_scores, router_indices = self.router(hidden_states)
        # router_scores: [B*S, E], router_indices: [B*S, K]

        # 2. Reshape for dispatch: tt-metal expects [B, 1, S, H] format
        x = hidden_states.view(batch_size, 1, seq_len, hidden_size)
        expert_indices = router_indices.view(batch_size, 1, seq_len, K)

        # 3. Dispatch: route tokens to devices along cluster_axis
        # BD = B * dispatch_devices (devices along the dispatch axis)
        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x,
            expert_indices,
            self.expert_mapping,
            num_devices=self.dispatch_devices,
            cluster_axis=self.cluster_axis,
        )
        # dispatched: [1, B*dispatch_devices, S, H]
        # metadata:   [1, B*dispatch_devices, S, K]

        BD = dispatched.shape[1]  # B * dispatch_devices
        M = 32

        # 4. Determine which dimension to split by M
        # Prefer seq_len; fall back to BD; assert if neither works
        split_seq = seq_len % M == 0 and seq_len >= M
        split_bd = BD % M == 0 and BD >= M
        assert split_seq or split_bd, (
            f"Neither seq_len ({seq_len}) nor BD ({BD}) is divisible by M={M}"
        )
        if split_seq:
            dim_a, dim_b = BD, seq_len // M
        else:
            dim_a, dim_b = BD // M, seq_len

        # 5. Build sparsity mask from metadata (M grouping)
        # metadata[0]: [B*D, S, K] — expert indices for dispatched tokens
        # Groups of M tokens share the same routing; take first of each group
        if split_seq:
            metadata_indices = metadata[0].view(BD, seq_len // M, M, K)[
                :, :, 0
            ]  # [BD, S//M, K]
        else:
            metadata_indices = metadata[0].view(BD // M, M, seq_len, K)[
                :, 0
            ]  # [BD//M, S, K]
        sparsity = torch.zeros(
            dim_a,
            dim_b,
            1,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        topk_indices_unsqueezed = metadata_indices.unsqueeze(2)
        sparsity.scatter_(
            dim=-1,
            index=topk_indices_unsqueezed,
            src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_states.dtype),
        )

        # 6. Reshape weights for sparse_matmul: [E, H, inter*2] → [1, E, H, inter*2]
        gate_up_proj = self.experts.gate_up_proj.unsqueeze(0)
        down_proj = self.experts.down_proj.view(
            1, self.num_experts, self.intermediate_size, -1
        )
        gate_up_bias = self.experts.gate_up_proj_bias
        down_bias = self.experts.down_proj_bias

        # 7. sparse_matmul(gate_up) with M grouping
        # input_a: [dim_a, dim_b, M, H], sparsity: [dim_a, dim_b, 1, E]
        if split_seq:
            hidden_4d = dispatched.view(BD, seq_len // M, M, hidden_size)
        elif dim_b == 1:
            # decode (S=1): view directly — same layout as permute when S=1
            hidden_4d = dispatched.view(BD // M, 1, M, hidden_size)
        else:
            # dispatched is [1, BD, S, H]; need [BD//M, S, M, H]
            hidden_4d = dispatched.view(BD // M, M, seq_len, hidden_size)
            hidden_4d = hidden_4d.permute(0, 2, 1, 3)  # [BD//M, S, M, H]
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d,
            gate_up_proj,
            sparsity,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        # Permute E and M so bias [E, inter*2] broadcasts naturally from the right
        gate_up_out = gate_up_out.squeeze(2)  # [dim_a, dim_b, E, M, inter*2]
        gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [dim_a, dim_b, M, E, inter*2]
        gate_up_out = gate_up_out + gate_up_bias

        # 8. Split & Activation (interleaved layout, all ops work on 5D)
        gate_out = gate_up_out[..., ::2]  # [dim_a, dim_b, M, E, inter]
        up_out = gate_up_out[..., 1::2]

        if self.activation_type == ACTIVATION_DEEPSEEK:
            activated = F.silu(gate_out) * up_out
        else:
            gate_out = gate_out.clamp(max=self.limit)
            up_out = up_out.clamp(-self.limit, self.limit)
            glu = gate_out * torch.sigmoid(gate_out * self.alpha)
            activated = (up_out + 1) * glu

        # 9. sparse_matmul(down) with M grouping
        # activated: [dim_a, dim_b, M, E, inter] → [dim_a*dim_b, E, M, inter]
        activated_reshaped = activated.permute(0, 1, 3, 2, 4).contiguous().view(
            dim_a * dim_b, self.num_experts, M, self.intermediate_size
        )
        sparsity_down = sparsity.view(1, 1, dim_a * dim_b, self.num_experts)
        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            down_proj,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )
        # [dim_a*dim_b, E, M, H] → [dim_a, dim_b, M, E, H] (swap E,M for bias broadcast)
        down_out = down_out.view(dim_a, dim_b, self.num_experts, M, hidden_size)
        down_out = down_out.permute(0, 1, 3, 2, 4)  # [dim_a, dim_b, M, E, H]
        down_out = down_out + down_bias  # [E, H] broadcasts naturally
        if split_seq:
            # [BD, S//M, M, E, H] → [E, BD, S//M, M, H] → [E, BD, S, H]
            down_out = down_out.permute(3, 0, 1, 2, 4).contiguous()
            down_out = down_out.view(self.num_experts, BD, seq_len, hidden_size)
        elif dim_b == 1:
            # decode (S=1): produce [E, 1, BD, H] to avoid tile waste on S=1 dim
            # [BD//M, 1, M, E, H] → [BD//M, M, E, H] → [E, BD//M, M, H] → [E, BD, H] → [E, 1, BD, H]
            down_out = down_out.squeeze(1)
            down_out = down_out.permute(2, 0, 1, 3).contiguous()
            down_out = down_out.view(self.num_experts, BD, hidden_size).unsqueeze(1)
        else:
            # [BD//M, S, M, E, H] → [E, BD//M, M, S, H] → [E, BD, S, H]
            down_out = down_out.permute(3, 0, 2, 1, 4).contiguous()
            down_out = down_out.view(self.num_experts, BD, seq_len, hidden_size)

        # 9. Combine: gather expert outputs back along cluster_axis
        # Use output_shard_dim=2 for decode to place BD on dim -2 (avoids tile waste on S=1)
        # After combine, each device has results from its LOCAL experts only.
        # Reduce-scatter along the other axis (handled by shard framework) aggregates
        # expert results across column devices to get the full expert output sum.
        decode_mode = dim_b == 1 and not split_seq
        combined = torch.ops.tt.all_to_all_combine(
            down_out,
            metadata,
            self.expert_mapping,
            num_devices=self.dispatch_devices,
            cluster_axis=self.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=2 if decode_mode else 1,
        )
        # combined: [K, B, S, H] (default) or [K, 1, B, H] (decode)

        # 10. All-reduce for 2D compound sharding is handled by the C++ sharding
        # rule for all_to_all_combine. The H dimension is split into two factors:
        # H_reduce (kReduction, operand→kNullDim) triggers all-reduce on the TP
        # axis, and H_out (kPassThrough, kNullDim→result) creates the output H.

        # 11. Weighted sum
        # Extract top-k weights from router_scores
        topk_weights = torch.gather(
            router_scores, dim=-1, index=router_indices
        )  # [B*S, K]
        if seq_len == 1:
            # decode: combined is [K, 1, B, H] — squeeze the S=1 dim at position 1
            topk_weights = topk_weights.view(batch_size, K)
            topk_weights = topk_weights.permute(1, 0).unsqueeze(-1)  # [K, B, 1]
            output = (combined.squeeze(1) * topk_weights).sum(dim=0)  # [B, H]
            output = output.unsqueeze(1)  # [B, 1, H]
        else:
            topk_weights = topk_weights.view(batch_size, seq_len, K)
            topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)  # [K, B, S, 1]
            output = (combined * topk_weights).sum(dim=0)  # [B, S, H]

        return output, router_scores


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
        flat_device_order: Optional[List[int]] = None,
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
        self.experts.gate_up_proj_bias = nn.Parameter(
            torch.cat([gate_b, up_b], dim=-1)
        )

        # Down proj / bias — pre-reshape to [1, E, inter, H] for sparse_matmul
        self.experts.down_proj = nn.Parameter(
            orig_experts.down_proj.data.view(
                1, num_experts, self.intermediate_size, hidden_size
            )
        )
        self.experts.down_proj_bias = orig_experts.down_proj_bias

        # Expert-to-device mapping [1, 1, E, D]
        mapping = build_expert_mapping(num_experts, num_devices)
        if flat_device_order is not None:
            permuted = torch.zeros_like(mapping)
            for d in range(num_devices):
                permuted[:, :, :, flat_device_order[d]] = mapping[:, :, :, d]
            mapping = permuted
        self.register_buffer("expert_mapping", mapping)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok

        # 1. Router
        router_scores, router_indices = self.router(hidden_states)

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
        topk_weights = torch.gather(
            router_scores, dim=-1, index=router_indices
        )
        topk_weights = topk_weights.view(batch_size, seq_len, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)

        output = (combined * topk_weights).sum(dim=0)
        output = output.view(batch_size, seq_len, hidden_size)

        return output, router_scores


class DeepseekV3MoEToA2AAdapter(nn.Module):
    """
    Adapter that converts DeepseekV3MoE to A2aSparseMLP-compatible interface.

    DeepseekV3MoE has:
    - gate (MoEGate): returns (topk_idx, topk_weight)
    - experts: ModuleList of DeepseekV3MLP with gate_proj, up_proj, down_proj (separate)

    A2aSparseMLP expects:
    - router: returns (scores, indices)
    - experts: gate_up_proj [E, H, inter*2], down_proj [E, inter, H], biases
    """

    class RouterAdapter(nn.Module):
        """Wraps MoEGate to return (scores, indices) for A2aSparseMLP."""

        def __init__(self, gate: nn.Module, n_experts: int):
            super().__init__()
            self.gate = gate
            self.n_experts = n_experts

        def forward(self, hidden_states):
            topk_idx, topk_weight = self.gate(hidden_states)
            bsz_seq = topk_idx.shape[0]
            # Build sparse scores so gather(scores, indices) gives topk_weight
            scores = torch.zeros(
                bsz_seq,
                self.n_experts,
                dtype=topk_weight.dtype,
                device=topk_weight.device,
            )
            scores.scatter_(1, topk_idx, topk_weight)
            return scores, topk_idx

    class StackedExperts(nn.Module):
        """Stacks DeepseekV3MLP experts into gate_up_proj, down_proj format."""

        def __init__(self, expert_list):
            super().__init__()
            experts_list = [e for e in expert_list if e is not None]
            if not experts_list:
                experts_list = list(expert_list)
            first = experts_list[0]
            hidden_size = first.gate_proj.in_features
            inter = first.gate_proj.out_features

            gate_up_list = []
            down_list = []
            for exp in experts_list:
                # gate_proj.weight [inter, H], up_proj.weight [inter, H] -> interleave [H, inter*2]
                gate_up = torch.empty(
                    hidden_size,
                    inter * 2,
                    dtype=exp.gate_proj.weight.dtype,
                    device=exp.gate_proj.weight.device,
                )
                gate_up[:, 0::2] = exp.gate_proj.weight.T
                gate_up[:, 1::2] = exp.up_proj.weight.T
                gate_up_list.append(gate_up)
                down_list.append(exp.down_proj.weight.T)

            num_experts = len(gate_up_list)
            gate_up_proj = torch.stack(gate_up_list, dim=0)
            down_proj = torch.stack(down_list, dim=0)
            self.gate_up_proj = nn.Parameter(gate_up_proj)
            self.down_proj = nn.Parameter(down_proj)
            self.gate_up_proj_bias = nn.Parameter(
                torch.zeros(num_experts, inter * 2, dtype=gate_up_proj.dtype, device=gate_up_proj.device)
            )
            self.down_proj_bias = nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=down_proj.dtype, device=down_proj.device)
            )

    def __init__(self, moe_module):
        super().__init__()
        self.router = self.RouterAdapter(moe_module.gate, moe_module.gate.n_routed_experts)
        experts_list = [e for e in moe_module.experts if e is not None]
        if not experts_list:
            experts_list = list(moe_module.experts)
        if len(experts_list) != moe_module.gate.n_routed_experts:
            raise ValueError(
                "DeepseekV3MoEToA2AAdapter requires ep_size=1 (all experts on one process). "
                f"Got {len(experts_list)} experts, expected {moe_module.gate.n_routed_experts}."
            )
        self.experts = self.StackedExperts(experts_list)


class A2aSparseMLPWithSharedExperts(nn.Module):
    """Wraps A2aSparseMLP and adds shared_experts output; returns single tensor for layer compatibility."""

    def __init__(self, a2a_mlp: A2aSparseMLP, shared_experts: Optional[nn.Module] = None):
        super().__init__()
        self.a2a_mlp = a2a_mlp
        self.shared_experts = shared_experts

    def forward(self, hidden_states):
        out, _ = self.a2a_mlp(hidden_states)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


def create_a2a_from_deepseek_v3_moe(
    moe_module,
    config,
    num_devices: int = 8,
    cluster_axis: int = 0,
    flat_device_order: Optional[List[int]] = None,
    dispatch_devices: Optional[int] = None,
    reduce_axis: Optional[int] = None,
) -> A2aSparseMLPWithSharedExperts:
    """
    Create A2aSparseMLP from DeepseekV3MoE.

    Args:
        moe_module: DeepseekV3MoE instance
        config: Model config (DeepseekV3Config)
        num_devices: Total mesh devices (for expert_mapping D dimension)
        cluster_axis: Mesh axis for dispatch/combine (0=rows, 1=cols)
        flat_device_order: Snake order for T3K, e.g. [0,1,2,3,7,6,5,4]
        dispatch_devices: Devices along cluster_axis (for BD = B * dispatch_devices).
            Defaults to num_devices when None (single-axis dispatch).
        reduce_axis: Mesh axis for reduce-scatter after combine (for 2D compound sharding).
            If cluster_axis=0, typically reduce_axis=1 to aggregate across columns.
            None means no reduce-scatter (single-axis EP).
    """
    adapter = DeepseekV3MoEToA2AAdapter(moe_module)
    a2a_mlp = A2aSparseMLP(
        adapter,
        num_experts=config.n_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        config=config,
        flat_device_order=flat_device_order,
        activation_type=ACTIVATION_DEEPSEEK,
        dispatch_devices=dispatch_devices,
        reduce_axis=reduce_axis,
    )
    shared_experts = getattr(moe_module, "shared_experts", None)
    return A2aSparseMLPWithSharedExperts(a2a_mlp, shared_experts)


def _is_moe_mlp(module: nn.Module) -> bool:
    """Check if a module is an MoE MLP that can be replaced with SparseMLP."""
    # Check for common MoE MLP patterns
    module_name = type(module).__name__.lower()

    # Known MoE MLP class names
    moe_patterns = ["gptossmlp", "mixtralmlp", "qwen2moemlp", "deepseekmlp"]

    if any(pattern in module_name for pattern in moe_patterns):
        return True

    # Check if module has router and experts attributes (common MoE pattern)
    has_router = hasattr(module, "router")
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
            # Try fused gate_up_proj (expected input format)
            if num_experts is None and hasattr(experts, "gate_up_proj"):
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
    target_classes: Optional[List[Type]] = None,
    verbose: bool = False,
    config: Optional[object] = None,
) -> nn.Module:
    """
    Replace MoE MLP layers in a model with SparseMLP implementations.

    This function traverses the model and replaces compatible MLP layers
    with SparseMLP, which uses sparse_matmul for efficient expert computation.

    Args:
        model: The model to transform
        target_classes: Optional list of specific MLP classes to replace.
                       If None, auto-detects MoE MLPs.
        verbose: If True, print information about replaced layers
        config: Optional model config for extracting MoE parameters

    Returns:
        The transformed model with SparseMLP layers

    Example:
        >>> import tt_torch
        >>> from tt_torch.sparse_mlp import enable_sparse_mlp
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
        >>> model = enable_sparse_mlp(model, verbose=True)
    """
    replaced_count = 0

    # Try to get config from model if not provided
    if config is None:
        config = getattr(model, "config", None)

    def replace_mlp(parent: nn.Module, name: str, module: nn.Module):
        nonlocal replaced_count

        # Check if this is a target MLP
        should_replace = False
        if target_classes:
            should_replace = any(isinstance(module, cls) for cls in target_classes)
        else:
            should_replace = _is_moe_mlp(module)

        if not should_replace:
            return False

        # Get MoE configuration from module first, then fall back to config
        moe_config = _get_moe_config(module)
        if moe_config is None and config is not None:
            # Try to get from model config
            num_experts = getattr(config, "num_local_experts", None)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
            if num_experts is not None:
                moe_config = (num_experts, num_experts_per_tok)

        if moe_config is None:
            if verbose:
                print(f"[SparseMLP] Skipping {name}: could not determine MoE config")
            return False

        num_experts, num_experts_per_tok = moe_config

        # Create SparseMLP wrapper
        sparse_mlp = SparseMLP(module, num_experts, num_experts_per_tok, config)

        # Replace the module
        setattr(parent, name, sparse_mlp)
        replaced_count += 1

        if verbose:
            print(
                f"[SparseMLP] Replaced {name}: {type(module).__name__} -> SparseMLP "
                f"(experts={num_experts}, top_k={num_experts_per_tok})"
            )

        return True

    # Traverse and replace
    for name, module in list(model.named_modules()):
        if "." in name:
            # Get parent module
            parts = name.rsplit(".", 1)
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        replace_mlp(parent, child_name, module)

    if verbose:
        print(f"[SparseMLP] Total layers replaced: {replaced_count}")

    return model
