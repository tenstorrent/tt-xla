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

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

# Activation types for A2aSparseMLP
ACTIVATION_GPT_OSS = "gpt_oss"  # clamp, sigmoid, alpha, glu
ACTIVATION_DEEPSEEK = "deepseek"  # SiLU (swish) for gate * up


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
        mesh_axis_names: Optional[Tuple[str, ...]] = None,
        use_dense_matmul: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.cluster_axis = cluster_axis
        self.activation_type = activation_type
        # When set, used to emit an sdy.sharding_constraint on combine's input
        # forcing E sharded only on cluster_axis (drops compound sharding so
        # the all_to_all_combine kernel sees E_local = E / mesh[cluster_axis]).
        self.mesh_axis_names = mesh_axis_names

        # num_devices: total mesh devices (for expert_mapping D dimension)
        # dispatch_devices: devices along cluster_axis (for BD = B * dispatch_devices)
        # When dispatch_devices is None, defaults to num_devices (single-axis or flat mesh)
        self.num_devices = num_devices
        self.dispatch_devices = (
            dispatch_devices if dispatch_devices is not None else num_devices
        )
        # When True, uses dense torch.matmul instead of sparse_matmul.
        # Skips remap (no sparsity mask needed). Dense path also supports
        # autograd (sparse_matmul doesn't), so training requires dense.
        self.use_dense_matmul = use_dense_matmul
        # When True, uses the corrected down-proj reshape (permute E to front
        # directly from `activated` instead of the buggy view-swap dance).
        # Old path scrambles the (E, M) axes via `permute(0,1,3,2,4).reshape(
        # dim_a*dim_b*M, E, I)` which doesn't preserve semantics. New tests
        # set this True; existing callers keep the legacy behaviour.
        self.correct_down_proj_reshape = False

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
        # Under dense-matmul path, split fused gate_up_proj into separate
        # gate_proj/up_proj so the forward can use two bmms instead of the
        # strided slice `gate_up_proj[..., ::2]`. The strided-slice backward
        # miscompiles on TT XLA (PCC cratered to ~0.4 on expert gradients
        # without this). Sparse path (use_dense_matmul=False) keeps the fused
        # layout since sparse_matmul is expected to consume it.
        if (
            self.use_dense_matmul
            and hasattr(self.experts, "gate_up_proj")
            and not hasattr(self.experts, "gate_proj")
        ):
            self._deinterleave_fused_experts(self.experts)
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

    @staticmethod
    def _deinterleave_fused_experts(experts: nn.Module) -> None:
        """Split experts.gate_up_proj [E, H, 2*inter] into separate
        experts.gate_proj [E, H, inter] and experts.up_proj [E, H, inter]
        by de-interleaving the last dim ([..., ::2] / [..., 1::2]). Same for
        gate_up_proj_bias. Deletes the fused buffers afterward.

        This mirrors tt_forge_models/gpt_oss/pytorch/overrides.py:_deinterleave_expert_weights
        but is invoked unconditionally here so A2aSparseMLP always sees the
        layout its forward/backward path was tuned for.
        """
        with torch.no_grad():
            gate_data = experts.gate_up_proj.data[..., ::2].contiguous()
            up_data = experts.gate_up_proj.data[..., 1::2].contiguous()
            gate_bias_data = None
            up_bias_data = None
            if getattr(experts, "gate_up_proj_bias", None) is not None:
                gate_bias_data = experts.gate_up_proj_bias.data[..., ::2].contiguous()
                up_bias_data = experts.gate_up_proj_bias.data[..., 1::2].contiguous()

        del experts.gate_up_proj
        if hasattr(experts, "gate_up_proj_bias"):
            del experts.gate_up_proj_bias

        experts.gate_proj = nn.Parameter(gate_data)
        experts.up_proj = nn.Parameter(up_data)
        if gate_bias_data is not None:
            experts.gate_proj_bias = nn.Parameter(gate_bias_data)
        if up_bias_data is not None:
            experts.up_proj_bias = nn.Parameter(up_bias_data)

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
            if self.correct_down_proj_reshape:
                # Put E to front first, then flatten (dim_a, dim_b, M) to T.
                # activated: [dim_a, dim_b, M, E, inter] → [E, T, inter]
                act_per_expert = activated.permute(3, 0, 1, 2, 4).reshape(
                    E, dim_a * dim_b * M, self.intermediate_size
                )
                down_per_expert = down_proj.squeeze(0)  # [E, inter, H]
                down_out = torch.bmm(
                    act_per_expert, down_per_expert
                )  # [E, T, H]
                if down_bias is not None:
                    down_out = down_out + down_bias[:, None, :]
                down_out = down_out.view(E, 1, BD * seq_len, hidden_size)
            else:
                # Legacy path — keeps the prior (buggy) E↔M reshape behaviour
                # so existing demo-style tests continue to match.
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
        )  # [B*S, K, E]
        topk_weights = torch.einsum("nke,ne->nk", one_hot, router_scores)  # [B*S, K]
        topk_weights = (
            topk_weights.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        )  # [K, 1, B*S, 1]
        output = (combined * topk_weights).sum(dim=0)  # [1, B*S, H]
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
    mesh_axis_names: Optional[Tuple[str, ...]] = None,
    use_dense_matmul: bool = False,
) -> nn.Module:
    """
    Replace MoE MLP layers in a model with A2aSparseMLP implementations.

    mesh_axis_names: optional tuple of axis names matching `mesh`. If provided
    and experts are compound-sharded across both mesh axes, A2aSparseMLP will
    emit an sdy.sharding_constraint before all_to_all_combine forcing E onto
    cluster_axis only — required for the combine kernel's input[0] assert.

    use_dense_matmul: when True, A2aSparseMLP uses dense torch.bmm (autograd
    supported) instead of sparse_matmul (no autograd). Also triggers de-
    interleaving of fused gate_up_proj into separate gate_proj/up_proj in
    A2aSparseMLP.__init__ to avoid the strided-slice backward miscompile.
    Required for training.
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

        if hasattr(module, "gate") and hasattr(module, "experts"):
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
        # so they have sparse_forward() like StackedExperts. Skip when the
        # dense-matmul path is requested — A2aSparseMLP de-interleaves the
        # fused weight in its __init__, which would invalidate the wrapper's
        # gate_up_proj-based properties.
        if (
            not use_dense_matmul
            and hasattr(module, "experts")
            and hasattr(module.experts, "gate_up_proj")
            and not hasattr(module.experts, "sparse_forward")
        ):
            module.experts = FusedExpertsWrapper(module.experts)

        sparse_mlp = A2aSparseMLP(
            module,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_devices=num_devices,
            cluster_axis=cluster_axis,
            config=config,
            dispatch_devices=dispatch_devices,
            cpu_forward_module=module,
            mesh_axis_names=mesh_axis_names,
            use_dense_matmul=use_dense_matmul,
        )

        setattr(parent, name, sparse_mlp)
        replaced_count += 1

        if verbose:
            print(
                f"[SparseMLP] Replaced {name}: {type(module).__name__} -> A2aSparseMLP "
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
    """Wrap the loader's shard_spec with compound sharding for A2aSparseMLP
    expert weights. Only fills in entries the loader didn't specify — loaders
    that set their own expert sharding (matching their cluster_axis choice)
    take precedence."""
    shard_specs = original_spec_fn(model)
    for layer in model.model.layers:
        if isinstance(layer.mlp, A2aSparseMLP):
            experts = layer.mlp.experts
            compound = (mesh_names[0], mesh_names[1])

            def _set_default(param, spec):
                if param is not None and param not in shard_specs:
                    shard_specs[param] = spec

            if hasattr(experts, "gate_up_proj"):
                # Fused gate_up (e.g. GPT-OSS via FusedExpertsWrapper)
                _set_default(experts.gate_up_proj, (compound, None, None))
                _set_default(experts.gate_up_proj_bias, (compound, None))
            else:
                # Separate gate/up (e.g. Deepseek via StackedExperts)
                _set_default(experts.gate_proj, (compound, None, None))
                _set_default(experts.up_proj, (compound, None, None))
                _set_default(experts.gate_proj_bias, (compound, None))
                _set_default(experts.up_proj_bias, (compound, None))

            _set_default(experts.down_proj, (compound, None, None))
            _set_default(experts.down_proj_bias, (compound, None))

    return shard_specs
