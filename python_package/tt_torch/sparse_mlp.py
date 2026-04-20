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

from typing import Any, Dict, List, Optional, Type

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
        output = (up_out + 1) * glu
        # print(f"MoE activation: gate_out={gate_out.shape}, up_out={up_out.shape}, glu={glu.shape}, output={output.shape}", flush=True)
        return output


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
    w1 = experts.w1.unsqueeze(0)  # [1, E, H, N1] [1, 32, 1440, 5760]
    w2 = experts.w2.view(1, E, experts.intermediate_size, -1)  # [1, E, inter, H]
    # w2 = experts.w2.view(1, E, -1, experts.intermediate_size) #, -1)  # [1, E, inter, H]
    # print(f"w2.shape: {w2.shape}", flush=True)
    # print(f"experts.w3 shape: {getattr(experts, 'w3', None).shape}", flush=True)

    # Gate (or gate+up fused): output [A, B, E, M, N1] (5D tiled)
    w1_out = torch.ops.tt.sparse_matmul(
        dispatched,
        w1,
        sparsity_remap,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )
    # print(f"Sparse expert forward: dispatched={dispatched.shape}", flush=True)
    # print(f"Sparse expert forward: sparsity_remap={sparsity_remap.shape}", flush=True)
    # print(f"experts.w1: {experts.w1}", flush=True)
    # print(f"experts.w2: {experts.w2}", flush=True)
    # print(f"experts.w1 shape: {w1.shape}", flush=True)
    # print(f"experts.w2 shape: {w2.shape}", flush=True)
    if experts.w1_bias is not None:
        w1_out = w1_out + experts.w1_bias.view(1, 1, E, 1, -1)

    # print(f"w1_out shape: {w1_out.shape}", flush=True)
    # print(f"experts.w3 shape: {getattr(experts, 'w3', None).shape}", flush=True)
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

    # print(f"experts.w3 shape: {getattr(experts, 'w3', None).shape}", flush=True)
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
    # print(f"Sparse expert forward: activated={activated.shape}, w2={w2.shape}, down_out={down_out.shape}", flush=True)
    # print(f"experts.w3 shape: {getattr(experts, 'w3', None).shape}", flush=True)
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


class FusedMoEWrapper(_SparseForwardMixin, nn.Module):
    # class FusedMoEWrapper(SparseMLP, nn.Module):
    """Wraps vLLM's FusedMoE to work with A2aSparseMLP.

    Extracts internal weight matrices from FusedMoE's quant_method and exposes them
    as separate gate_proj, up_proj, down_proj attributes for sparse operations.
    """

    def __init__(self, fused_moe):
        super().__init__()
        self._fused_moe = fused_moe

        # Extract weights from FusedMoE's quant_method
        print(f"[FusedMoEWrapper] Initializing wrapper for {type(fused_moe)}")
        self._extract_weights()
        print(
            f"[FusedMoEWrapper] After extraction: gate_up_proj={getattr(self, 'gate_up_proj', None) is not None}, down_proj={getattr(self, 'down_proj', None) is not None}"
        )

        # Extract intermediate_size from FusedMoE attributes
        print(
            f"[FusedMoEWrapper] Extracting intermediate_size from FusedMoE {hasattr(fused_moe, 'intermediate_size')} or {hasattr(fused_moe, 'intermediate_size_per_partition')}"
        )
        if hasattr(fused_moe, "intermediate_size_per_partition"):
            self.intermediate_size = fused_moe.intermediate_size_per_partition
        elif hasattr(fused_moe, "intermediate_size"):
            self.intermediate_size = fused_moe.intermediate_size
        else:
            # Default fallback if we can't determine it
            self.intermediate_size = 2880  # Common size for MoE models
        print(f"[FusedMoEWrapper] Set intermediate_size: {self.intermediate_size}")

    def _extract_weights(self):
        """Extract weight matrices from FusedMoE's internal structure."""
        # Set defaults
        self.gate_up_proj = None
        self.down_proj = None
        self.gate_up_proj_bias = None
        self.down_proj_bias = None

        try:
            # Access FusedMoE's quant_method which holds the actual weights
            quant_method = self._fused_moe.quant_method
            print(f"FusedMoE quant_method type: {type(quant_method)}")
            print(
                f"quant_method attributes: {[attr for attr in dir(quant_method) if not attr.startswith('_')]}"
            )

            # Handle TTMxfp4MoEMethod specifically
            if hasattr(quant_method, "w13_weight"):
                print(f"[FusedMoEWrapper] Found TTMxfp4MoEMethod with w13_weight")
                self.gate_up_proj = quant_method.w13_weight
                print(
                    f"[FusedMoEWrapper] Found w13_weight (gate_up_proj): {self.gate_up_proj.shape}"
                )
                if hasattr(quant_method, "w13_bias"):
                    self.gate_up_proj_bias = quant_method.w13_bias
            elif hasattr(quant_method, "gate_up_weight"):
                self.gate_up_proj = quant_method.gate_up_weight
                print(
                    f"[FusedMoEWrapper] Found gate_up_weight: {self.gate_up_proj.shape}"
                )

            if hasattr(quant_method, "w2_weight"):
                print(f"[FusedMoEWrapper] Found TTMxfp4MoEMethod with w2_weight")
                self.down_proj = quant_method.w2_weight
                print(
                    f"[FusedMoEWrapper] Found w2_weight (down_proj): {self.down_proj.shape}"
                )
                if hasattr(quant_method, "w2_bias"):
                    self.down_proj_bias = quant_method.w2_bias
            elif hasattr(quant_method, "down_weight"):
                self.down_proj = quant_method.down_weight
                print(f"[FusedMoEWrapper] Found down_weight: {self.down_proj.shape}")

            # If direct attribute access fails, try named_parameters fallback for other methods
            if self.gate_up_proj is None or self.down_proj is None:
                if hasattr(quant_method, "named_parameters"):
                    print("Trying named_parameters fallback...")
                    all_params = dict(quant_method.named_parameters())
                    print(f"Available parameters: {list(all_params.keys())}")

                    for name, param in all_params.items():
                        if "w13_weight" in name or "gate_up_proj" in name:
                            self.gate_up_proj = param
                            print(f"Found gate_up_proj: {name} {param.shape}")
                        elif "w2_weight" in name or "down_proj" in name:
                            self.down_proj = param
                            print(f"Found down_proj: {name} {param.shape}")
                else:
                    print("No named_parameters method available, trying fallback...")
                    self._fallback_weight_extraction()

        except Exception as e:
            print(f"[FusedMoEWrapper] Failed to extract weights from FusedMoE: {e}")
            print(
                f"[FusedMoEWrapper] This is expected for TTMxfp4MoEMethod - dense fallback will be used"
            )
            # Mark all weights as unavailable to force dense fallback
            self.gate_up_proj = None
            self.down_proj = None
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

    def _unpack_mxfp4(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack MXFP4 4-bit values using proper E2M1 format (matches vLLM).

        Args:
            packed: uint8 tensor [..., packed_dim] with 2 4-bit values per byte

        Returns:
            float32 tensor [..., packed_dim * 2] with proper E2M1 values
        """
        assert packed.dtype == torch.uint8, f"Expected uint8, got {packed.dtype}"

        # E2M1 lookup table from vLLM (proper MXFP4 format)
        E2M1_TABLE = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=packed.device,
        )

        # Vectorized nibble processing (following vLLM approach)
        original_shape = packed.shape
        a_flat = packed.flatten()

        # Extract nibbles - note vLLM uses (low, high) order
        high = (a_flat & 0xF0) >> 4  # Upper nibbles
        low = a_flat & 0x0F  # Lower nibbles

        # Combine nibbles for batch processing (low first, then high - vLLM order)
        combined = torch.stack((low, high), dim=1).flatten()

        # Vectorized sign and magnitude extraction (E2M1 format)
        signs = (combined & 0x08).to(torch.bool)  # Sign bits (bit 3)
        abs_vals = (combined & 0x07).to(torch.long)  # Magnitude (bits 0-2)

        # Device-aware lookup and sign application (vLLM approach)
        kE2M1 = E2M1_TABLE.to(device=packed.device)
        values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

        # Reshape to match original packed dimensions × 2
        new_shape = original_shape[:-1] + (original_shape[-1] * 2,)
        return values.reshape(new_shape).to(torch.float32)

    def _apply_block_scale(
        self, values: torch.Tensor, scale: torch.Tensor, block_size: int = 16
    ) -> torch.Tensor:
        """Apply block-wise scaling to MXFP4 values using proper E8M0 conversion.

        Args:
            values: (E, dim1, H) unpacked float32 E2M1 values
            scale: (E, dim1, num_blocks) scale factors (uint8 E8M0 format)
            block_size: Number of elements per scale factor (default 16)

        Returns:
            Scaled float tensor
        """
        E, dim1, H = values.shape
        _, _, num_blocks = scale.shape
        print(
            f"[FusedMoEWrapper] Applying block scale: values={values.shape}, scale={scale.shape}, block_size={block_size}"
        )

        # For unpacked MXFP4 weights, H = packed_dim * 2, and num_blocks = packed_dim // block_size
        expected_H = num_blocks * block_size * 2
        assert H == expected_H, f"H mismatch: {H} vs {expected_H}"

        # Decode E8M0 scale: value = 2^(exp - 127)
        scale_f32 = torch.pow(2.0, scale.to(torch.float32) - 127.0)

        # Reshape for block-wise multiply
        values_reshaped = values.view(E, dim1, num_blocks, block_size * 2)
        scale_expanded = scale_f32.unsqueeze(-1)

        output = values_reshaped * scale_expanded
        return output.view(E, dim1, H)

    def _decode_mxfp4(
        self, weight: torch.Tensor, scale: torch.Tensor, block_size: int = 16
    ) -> torch.Tensor:
        """Decode MXFP4 weight tensor using proper E2M1 unpacking and E8M0 scaling.

        Args:
            weight: (E, dim1, packed_dim) uint8 packed MXFP4 weights
            scale: (E, dim1, num_blocks) scale factors (uint8 E8M0 format)
            block_size: Elements per scale factor (default 16)

        Returns:
            (E, dim1, H) decoded bfloat16 tensor where H = packed_dim * 2
        """
        print(
            f"[FusedMoEWrapper] Decoding MXFP4 weight: {weight.shape}, scale: {scale.shape}"
        )

        # Step 1: Unpack 4-bit values to E2M1 float32 (following vLLM)
        unpacked = self._unpack_mxfp4(weight)  # (E, dim1, H) where H = packed_dim * 2
        print(
            f"[FusedMoEWrapper] Unpacked shape: {unpacked.shape}, dtype: {unpacked.dtype}"
        )

        # Step 2: Apply block-wise scaling with proper E8M0 scale handling
        decoded = self._apply_block_scale(unpacked, scale, block_size)

        # Step 3: Convert to target dtype
        decoded = decoded.to(torch.bfloat16)

        print(
            f"[FusedMoEWrapper] Decoded weight shape: {decoded.shape}, dtype: {decoded.dtype}"
        )
        return decoded

    def _convert_gate_up_proj(
        self, w13_weight: torch.Tensor, w13_scale: torch.Tensor, D: int
    ) -> torch.Tensor:
        """Convert gate_up_proj weights from MXFP4 to HuggingFace layout.

        Args:
            w13_weight: (E, D, packed_dim) MXFP4 weights
            w13_scale: (E, D, num_blocks) scale factors
            D: Dimension size (typically 2880)

        Returns:
            (E, 2*D, H) tensor in HF format where H = packed_dim * 2
        """
        E, D_, packed_dim = w13_weight.shape
        assert D_ == D, f"Expected D={D}, got D_={D_}"

        # Decode MXFP4 → (E, D, H)
        w = self._decode_mxfp4(w13_weight, w13_scale)
        print(f"[FusedMoEWrapper] Decoded gate_up_proj shape: {w.shape}")

        # vLLM stores fused gate+up as (E, 2*D, H)
        # Reshape to fused format
        w = w.view(
            E, D, -1
        )  # (E, 2*Intermediate, H) # D is already 2*Intermediate for gate_up_proj

        return w

    def _convert_down_proj(
        self, w2_weight: torch.Tensor, w2_scale: torch.Tensor, D: int
    ) -> torch.Tensor:
        """Convert down_proj weights from MXFP4 to HuggingFace layout.

        Args:
            w2_weight: (E, D, packed_dim) MXFP4 weights
            w2_scale: (E, D, num_blocks) scale factors
            D: Dimension size (typically 2880)

        Returns:
            (E, H, D) tensor in HF format where H = packed_dim * 2
        """
        # Decode MXFP4 → (E, D, H)
        w = self._decode_mxfp4(w2_weight, w2_scale)

        # Convert to HF layout: (E, H, D)
        w = w.transpose(1, 2).contiguous()
        print(f"[FusedMoEWrapper] Decoded down_proj shape: {w.shape}, dtype: {w.dtype}")
        return w

    def _decode_mxfp4_weight(self, weight, scale, target_dim=None):
        """Legacy wrapper for MXFP4 decoding - delegates to proper implementation."""
        try:
            # Use proper MXFP4 decoding if weight is uint8 (packed format)
            if weight.dtype == torch.uint8:
                print(
                    f"[FusedMoEWrapper] Using proper MXFP4 decoding for uint8 weights"
                )
                return self._decode_mxfp4(weight, scale)
            else:
                # Fallback for non-uint8 weights - use simple approach
                print(
                    f"[FusedMoEWrapper] Using fallback decoding for {weight.dtype} weights"
                )
                E, dim1, packed_dim = weight.shape
                scale_dim = scale.shape[-1]

                if target_dim is None:
                    target_dim = packed_dim * 2

                elements_per_scale = target_dim // scale_dim
                expanded_scale = scale.repeat_interleave(elements_per_scale, dim=-1)

                # Simple expansion for non-uint8 data
                if target_dim > packed_dim:
                    unpack_factor = target_dim // packed_dim
                    weight = weight.unsqueeze(-1).expand(
                        E, dim1, packed_dim, unpack_factor
                    )
                    weight = weight.reshape(E, dim1, target_dim)

                decoded_weight = weight.float() * expanded_scale.float()
                return decoded_weight.to(torch.bfloat16)

        except Exception as e:
            print(f"[FusedMoEWrapper] MXFP4 decoding failed: {e}")
            return weight.to(torch.bfloat16)

    def _fallback_weight_extraction(self):
        """Fallback weight extraction from FusedMoE named_parameters with MXFP4 decoding."""
        try:
            # Try to get weights directly from the module
            all_params = dict(self._fused_moe.named_parameters())
            print(
                f"[FusedMoEWrapper] Fallback extraction with {len(all_params)} parameters:"
            )
            for name, param in all_params.items():
                print(f"[FusedMoEWrapper]  - {name}: {param.shape}")

            # Common weight names to check (including scale tensors for MXFP4)
            weight_patterns = {
                "gate_up": ["w13_weight", "gate_up_proj", "gate_up.weight"],
                "gate_up_scale": ["w13_weight_scale", "gate_up_proj_scale"],
                "gate_up_bias": ["w13_bias", "gate_up_proj_bias", "gate_up.bias"],
                "down": ["w2_weight", "down_proj", "down.weight"],
                "down_scale": ["w2_weight_scale", "down_proj_scale"],
                "down_bias": ["w2_bias", "down_proj_bias", "down.bias"],
                "gate": ["w1_weight", "gate_proj", "gate.weight"],
                "up": ["w3_weight", "up_proj", "up.weight"],
            }

            found = {}
            for param_name, param in all_params.items():
                for weight_type, patterns in weight_patterns.items():
                    if param_name in patterns:
                        found[weight_type] = param
                        print(
                            f"[FusedMoEWrapper] Found {weight_type}: {param_name} {param.shape}"
                        )
                        break

            # Set up the weights based on what we found
            if "gate_up" in found:
                gate_up_weight = found["gate_up"]
                print(
                    f"[FusedMoEWrapper] Found gate_up weight: {gate_up_weight.shape}, dtype: {gate_up_weight.dtype}"
                )

                # Check for MXFP4 scale tensor and decode if found
                if "gate_up_scale" in found:
                    print(
                        f"[FusedMoEWrapper] Found MXFP4 scale for gate_up: {found['gate_up_scale'].shape} -- {found['gate_up_scale'].dtype}"
                    )
                    # Use proper MXFP4 conversion for gate_up_proj
                    # Assumes D = 2880 (hidden_size), will be inferred from shapes
                    D = gate_up_weight.shape[1]  # Should be 2880
                    gate_up_weight = self._convert_gate_up_proj(
                        gate_up_weight, found["gate_up_scale"], D
                    )
                else:
                    # No scale found, use legacy approach
                    target_dim = 2880  # hidden_size (unpacked)
                    gate_up_weight = self._decode_mxfp4_weight(
                        gate_up_weight, None, target_dim
                    )

                self.gate_up_proj = torch.nn.Parameter(gate_up_weight.transpose(1, 2))
                print(
                    f"[FusedMoEWrapper] Set gate_up_proj: {self.gate_up_proj.shape}, dtype: {self.gate_up_proj.dtype}"
                )
                print(
                    f"[FusedMoEWrapper] gate_up_proj: {self.gate_up_proj}", flush=True
                )

            elif "gate" in found and "up" in found:
                gate_weight = found["gate"].to(torch.bfloat16)
                up_weight = found["up"].to(torch.bfloat16)
                self.gate_up_proj = nn.Parameter(
                    torch.cat([gate_weight, up_weight], dim=-1)
                )
                print(
                    f"[FusedMoEWrapper] Set gate_up_proj from cat: {self.gate_up_proj.shape}"
                )
            else:
                self.gate_up_proj = None
                print(f"[FusedMoEWrapper] No gate_up_proj found")

            if "gate_up_bias" in found:
                self.gate_up_proj_bias = found["gate_up_bias"]
                print(
                    f"[FusedMoEWrapper] Set gate_up_proj_bias: {self.gate_up_proj_bias.shape} -- {self.gate_up_proj_bias}",
                    flush=True,
                )

            # Handle down projection weights with MXFP4 decoding
            if "down" in found:
                down_weight = found["down"]
                print(
                    f"[FusedMoEWrapper] Found down weight: {down_weight.shape}, dtype: {down_weight.dtype}"
                )

                # Check for MXFP4 scale tensor and decode if found
                if "down_scale" in found:
                    print(
                        f"[FusedMoEWrapper] Found MXFP4 scale for down: {found['down_scale'].shape} -- {found['down_scale'].dtype}"
                    )
                    # Use proper MXFP4 conversion for down_proj
                    # Assumes D = 2880 (hidden_size), will be inferred from shapes
                    D = down_weight.shape[1]  # Should be 2880
                    down_weight = self._convert_down_proj(
                        down_weight, found["down_scale"], D
                    )
                else:
                    # No scale found, use legacy approach
                    target_dim = 2880  # hidden_size
                    down_weight = self._decode_mxfp4_weight(
                        down_weight, None, target_dim
                    )

                self.down_proj = torch.nn.Parameter(down_weight)
                print(
                    f"[FusedMoEWrapper] Set down_proj: {self.down_proj.shape}, dtype: {self.down_proj.dtype} -- {self.down_proj}",
                    flush=True,
                )
            else:
                self.down_proj = None
                print(f"[FusedMoEWrapper] No down_proj found")

            if "down_bias" in found:
                self.down_proj_bias = found["down_bias"]
                print(
                    f"[FusedMoEWrapper] Set down_proj_bias: {self.down_proj_bias.shape} -- {self.down_proj_bias}",
                    flush=True,
                )

        except Exception as e:
            print(f"Fallback weight extraction failed: {e}")
            self.gate_up_proj = None
            self.down_proj = None
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

    def forward(self, *args, **kwargs):
        """Delegate to original FusedMoE forward for dense computation."""
        print(f"[FusedMoEWrapper] Using dense forward via original FusedMoE")
        return self._fused_moe(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to original FusedMoE."""
        if name.startswith("_") or name in (
            "intermediate_size",
            "training",
            "gate_up_proj",
            "down_proj",
            "gate_up_proj_bias",
            "down_proj_bias",
        ):
            return super().__getattr__(name)
        return getattr(self._fused_moe, name)

    def sparse_forward(
        self,
        dispatched,
        sparsity_remap,
        activation_type,
        alpha=1.702,
        limit=7.0,
        output_shape=None,
    ):
        """
        Sparse forward using extracted weights.
        """
        if self.gate_up_proj is None or self.down_proj is None:
            # Weights not available, raise to force dense fallback
            raise NotImplementedError(
                "FusedMoE sparse_forward: weights not extracted - use dense_matmul=True"
            )

        # Use the sparse implementation with extracted weights
        return super().sparse_forward(
            dispatched, sparsity_remap, activation_type, alpha, limit, output_shape
        )

    @property
    def w1(self):
        return self.gate_up_proj
        # For FusedMoE, weights may not be directly accessible
        if hasattr(self._fused_moe, "gate_up_proj"):
            return self._fused_moe.gate_up_proj
        elif hasattr(self._fused_moe, "w1"):
            return self._fused_moe.w1
        else:
            # Return None to indicate weights not accessible
            return None

    @property
    def w1_bias(self):
        return self.gate_up_proj_bias
        return getattr(self._fused_moe, "gate_up_proj_bias", None)

    @property
    def w2(self):
        return self.down_proj
        if hasattr(self._fused_moe, "down_proj"):
            return self._fused_moe.down_proj
        elif hasattr(self._fused_moe, "w2"):
            return self._fused_moe.w2
        else:
            return None

    @property
    def w2_bias(self):
        return self.down_proj_bias
        return getattr(self._fused_moe, "down_proj_bias", None)

    @property
    def w3(self):
        # For FusedMoE, w3 is typically None (fused with w1)
        return None

    @property
    def w3_bias(self):
        # For FusedMoE, w3 is typically None (fused with w1)
        return None


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
        # print(f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}, K={K}, {hasattr(self.router, 'gate')}", flush=True)
        # print(f"Router type: {type(self.router)}, router: {self.router}", flush=True)

        # CPU golden path
        if hidden_states.device.type == "cpu":
            return self._cpu_forward(hidden_states)

        # 1. Router — pass 3D for RouterAdapter (handles 3D→2D internally),
        # flatten to 2D for raw routers (e.g. GptOssTopKRouter).
        router_input = hidden_states
        if not hasattr(self.router, "gate"):
            router_input = hidden_states.view(-1, hidden_size)
        router_output = self.router(router_input)
        # print(f"Router output shape: {router_output.shape}", flush=True)

        # Handle raw logits router (single tensor) vs tuple-based routers
        if isinstance(router_output, torch.Tensor) and router_output.dim() == 2:
            # Raw logits from Linear router: [BS, num_experts] → convert to (scores, indices)
            topk_weights, topk_indices = torch.topk(router_output, k=K, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
            router_scores = _topk_to_sparse_scores(
                topk_weights, topk_indices, self.num_experts
            )
            router_indices = topk_indices
        else:
            # Tuple-based router output
            router_scores, router_indices = _unpack_router_output(
                router_output, self.num_experts
            )
        # print(f"Router scores shape: {router_scores.shape}, indices shape: {router_indices.shape}, dispatch_devices: {self.dispatch_devices}", flush=True)
        # print(f"self.expert_mapping shape: {self.expert_mapping.shape} -- {self.expert_mapping} -- {self.use_dense_matmul}", flush=True)

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
        # print(f"Dispatched shape: {dispatched.shape}, metadata shape: {metadata.shape}", flush=True)
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
        )  # [B*S, K, E]
        topk_weights = torch.einsum("nke,ne->nk", one_hot, router_scores)  # [B*S, K]
        topk_weights = (
            topk_weights.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        )  # [K, 1, B*S, 1]
        output = (combined * topk_weights).sum(dim=0)  # [1, B*S, H]
        # print(f"Combined shape: {combined.shape}, topk_weights shape: {topk_weights.shape}, output shape: {output.shape}", flush=True)
        output = output.view(batch_size, seq_len, hidden_size)

        return output.to(hidden_states.dtype)  # , router_scores


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
        print(f"Attempting to extract MoE config from module {module}")
        num_experts = None
        # Try to get from experts
        if hasattr(module, "experts"):
            experts = module.experts
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is None:
                num_experts = getattr(experts, "global_num_experts", None)
            print(f"Extracted num_experts from experts: {num_experts}")
            print(f"hasattr(experts, 'gate_proj'): {hasattr(experts, 'gate_proj')}")
            print(
                f"hasattr(experts, 'gate_up_proj'): {hasattr(experts, 'gate_up_proj')}"
            )
            print(
                f"gate_up_proj shape: {hasattr(experts, 'w13_weight') and experts.w13_weight.shape if hasattr(experts, 'w13_weight') else 'N/A'}"
            )
            print(
                f"check-1: {hasattr(experts, 'w13_weight')} -- {hasattr(experts, '.w13_weigth')}"
            )
            if num_experts is None and hasattr(experts, "gate_proj"):
                num_experts = experts.gate_proj.shape[0]
            elif num_experts is None and hasattr(experts, "gate_up_proj"):
                num_experts = experts.gate_up_proj.shape[0]

        print(f"Extracted num_experts: {num_experts}")
        print(f"Attempting to extract num_experts_per_tok from router/gate")
        print(f"hasattr(module, 'router'): {hasattr(module, 'router')}")
        print(f"expert.topk: {hasattr(experts, 'top_k')}")
        # Try to get num_experts_per_tok from router
        if hasattr(module, "router"):
            router = module.router
            num_experts_per_tok = getattr(router, "top_k", None)
            if hasattr(experts, "top_k"):
                print(
                    f"Extracting num_experts_per_tok from experts.top_k: {experts.top_k}"
                )
                num_experts_per_tok = experts.top_k
            elif num_experts_per_tok is None:
                num_experts_per_tok = getattr(router, "num_experts_per_tok", 2)
        else:
            num_experts_per_tok = 2  # Default

        if num_experts is not None:
            print(
                f"Successfully extracted MoE config: num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}"
            )
            return (num_experts, num_experts_per_tok)
    except Exception:
        pass
    sys.exit(0)

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
    print(
        f"dispatch_devices: {dispatch_devices}, num_devices: {num_devices}, cluster_axis: {cluster_axis}",
        flush=True,
    )

    def replace_mlp(parent: nn.Module, name: str, module: nn.Module):
        nonlocal replaced_count

        should_replace = False
        if target_classes:
            should_replace = any(isinstance(module, cls) for cls in target_classes)
            print(f"module {module}")
            print(
                f"Checking {name} against target classes: {should_replace}", flush=True
            )
        else:
            should_replace = _is_moe_mlp(module)
            print(f"Checking {name} for MoE MLP patterns: {should_replace}", flush=True)

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

        # Wrap fused experts (e.g. GptOssExperts or FusedMoE) with appropriate wrapper
        # so they have sparse_forward() like StackedExperts
        print(f'hasattr(module, "experts")={hasattr(module, "experts")}')
        print(
            f'hasattr(module.experts, "gate_up_proj")={hasattr(module.experts, "gate_up_proj")}'
        )
        print(
            f'hasattr(module.experts, "sparse_forward")={hasattr(module.experts, "sparse_forward")}'
        )
        print(f"type(module.experts).__name__={type(module.experts).__name__}")
        if (
            hasattr(module, "experts")
            and hasattr(module.experts, "gate_up_proj")
            and not hasattr(module.experts, "sparse_forward")
        ):
            print(f"Applying FusedExpertsWrapper to {name} with gate_up_proj")
            module.experts = FusedExpertsWrapper(module.experts)
        elif (
            hasattr(module, "experts")
            and type(module.experts).__name__ == "FusedMoE"
            and not hasattr(module.experts, "sparse_forward")
        ):
            print(f"Applying FusedMoEWrapper to {name} with FusedMoE experts")
            # Handle vLLM's FusedMoE specifically
            module.experts = FusedMoEWrapper(module.experts)
            print(f"Wrapped FusedMoE experts for {name}")

        sparse_mlp = A2aSparseMLP(
            module,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_devices=num_devices,
            cluster_axis=cluster_axis,
            config=config,
            dispatch_devices=dispatch_devices,
            cpu_forward_module=module,
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
    shard_specs = original_spec_fn(model)
    for layer in model.model.layers:
        if isinstance(layer.mlp, A2aSparseMLP):
            experts = layer.mlp.experts
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

    return shard_specs
