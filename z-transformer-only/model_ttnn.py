# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Clean TTNN implementation of the Z-Image transformer.

Self-contained -- no imports from sibling directories.
Extracted from the 35K-line codegen main.py and organized into clean
LightweightModule classes matching the PyTorch reference model_pt.py.

Architecture:
    ZImageTransformerTTNN
    +-- TimestepEmbedder: freq table -> cos/sin -> MLP(silu) -> MLP
    +-- x_embedder: Linear(64, 3840) patch embedding
    +-- cap_embedder: RMSNorm(2560) + Linear(2560, 3840)
    +-- noise_refiner: 2x TransformerBlock (with adaLN modulation)
    +-- context_refiner: 2x TransformerBlock (without modulation)
    +-- layers: 30x TransformerBlock (with adaLN modulation)
    +-- final_layer: FinalLayer (LayerNorm + adaLN + Linear + un-patchify)
"""

import ttnn
from consteval import run_const_evals
from parameters import load_params_from_pytorch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
RMS_EPS = 9.9999997473787516e-06
LAYERNORM_EPS = 9.9999999747524271e-07  # 1e-6 for FinalLayer LayerNorm
SDPA_SCALE = 0.088388338685035706  # 1/sqrt(128)

# Model dimensions
DIM = 3840
N_HEADS = 30
HEAD_DIM = 128
FFN_HIDDEN = 10240
CAP_FEAT_DIM = 2560
ADALN_DIM = 256
IN_CHANNELS = 16
PATCH_SIZE = 2

# Sequence lengths
IMAGE_ORI_LEN = 3600   # 80 * 45
IMAGE_PADDED = 3616     # 3600 + 16 padding
CAP_ORI_LEN = 18
CAP_PADDED = 32         # 18 + 14 padding
TOTAL_SEQ = 3648        # 3616 + 32


# ---------------------------------------------------------------------------
# LightweightModule base class
# ---------------------------------------------------------------------------

class LightweightModule:
    """Base class for TTNN modules. Stores weights as attributes, no nn.Module."""

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(LightweightModule):
    """SwiGLU feed-forward network.

    Codegen pattern:
        matmul(x, w1, transpose_b=True, activation="silu") -> silu_out
        matmul(x, w3, transpose_b=True) -> gate_out
        multiply(silu_out, gate_out) -> gated
        matmul(gated, w2, transpose_b=True) -> output

    All matmuls use dtype=BFLOAT16 and COMPUTE_CONFIG.
    """

    def __init__(self, w1, w2, w3):
        """
        Args:
            w1: [10240, 3840] weight for gate+silu projection (on device, BFLOAT16)
            w2: [3840, 10240] weight for down projection (on device, BFLOAT16)
            w3: [10240, 3840] weight for value projection (on device, BFLOAT16)
        """
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, x):
        """
        Args:
            x: [seq, 3840] in BFLOAT16 (already flattened from [1, seq, 3840])

        Returns:
            [seq, 3840] in BFLOAT16
        """
        # silu(x @ w1.T) -- fused activation
        silu_out = ttnn.matmul(
            x, self.w1,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation="silu",
            compute_kernel_config=COMPUTE_CONFIG,
        )
        # x @ w3.T
        gate_out = ttnn.matmul(
            x, self.w3,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        # element-wise gate
        gated = ttnn.multiply(silu_out, gate_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(gate_out, False)
        ttnn.deallocate(silu_out, False)
        # down projection
        out = ttnn.matmul(
            gated, self.w2,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(gated, False)
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(LightweightModule):
    """Self-attention with QKNorm and real-valued RoPE.

    Codegen pattern:
        Q projection: matmul(x, to_q, transpose_b=True) -> reshape [1,seq,30,128]
        Q norm: rms_norm(q, norm_q_weight) -> typecast FLOAT32
        RoPE on Q: reshape [1,seq,30,64,2] -> slice real/imag ->
                   real*cos - imag*sin, real*sin + imag*cos -> concat -> reshape back
        K projection + norm + RoPE: same pattern as Q
        V projection: matmul(x, to_v, transpose_b=True) -> typecast FLOAT32 ->
                      reshape [1,seq,30,128] -> permute [0,2,1,3]

        Permute Q,K,V to [1, 30, seq, 128] (heads-first)
        Typecast Q,K,V to BFLOAT16
        SDPA: scaled_dot_product_attention(Q, K, V, attn_mask, scale=1/sqrt(128))
        concatenate_heads -> reshape [seq, 3840]
        Output projection: matmul(out, to_out, transpose_b=True)
    """

    def __init__(self, to_q, to_k, to_v, to_out, norm_q_weight, norm_k_weight):
        """
        Args:
            to_q, to_k, to_v, to_out: [3840, 3840] weights (on device, BFLOAT16 TILE)
            norm_q_weight, norm_k_weight: [128] RMS norm weights (on device, BFLOAT16 TILE)
        """
        super().__init__()
        self.to_q = to_q
        self.to_k = to_k
        self.to_v = to_v
        self.to_out = to_out
        self.norm_q_weight = norm_q_weight
        self.norm_k_weight = norm_k_weight

    def _apply_rope(self, x_normed, rope_cos, rope_sin, seq):
        """Apply real-valued RoPE to a normed Q or K tensor.

        Args:
            x_normed: [1, seq, 30, 128] after QK-norm, in FLOAT32
            rope_cos: [1, seq, 1, 64, 1] FLOAT32
            rope_sin: [1, seq, 1, 64, 1] FLOAT32
            seq: sequence length (used for reshape dimensions)

        Returns:
            [1, seq, 30, 128] with RoPE applied, in FLOAT32
        """
        # Split into real/imag pairs: [1, seq, 30, 64, 2]
        x_ri = ttnn.reshape(x_normed, [1, seq, N_HEADS, HEAD_DIM // 2, 2], memory_config=DRAM)
        ttnn.deallocate(x_normed, False)

        # Slice real and imaginary parts
        x_real = ttnn.slice(
            x_ri, [0, 0, 0, 0, 0], [1, seq, N_HEADS, HEAD_DIM // 2, 1],
            [1, 1, 1, 1, 1], memory_config=DRAM,
        )
        x_imag = ttnn.slice(
            x_ri, [0, 0, 0, 0, 1], [1, seq, N_HEADS, HEAD_DIM // 2, 2],
            [1, 1, 1, 1, 1], memory_config=DRAM,
        )
        ttnn.deallocate(x_ri, False)

        # out_real = real * cos - imag * sin
        rc = ttnn.multiply(x_real, rope_cos, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        is_ = ttnn.multiply(x_imag, rope_sin, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        out_real = ttnn.subtract(rc, is_, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(is_, False)
        ttnn.deallocate(rc, False)

        # out_imag = real * sin + imag * cos
        rs = ttnn.multiply(x_real, rope_sin, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x_real, False)
        ic = ttnn.multiply(x_imag, rope_cos, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x_imag, False)
        out_imag = ttnn.add(rs, ic, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(ic, False)
        ttnn.deallocate(rs, False)

        # Concatenate [out_real, out_imag] along last dim -> [1, seq, 30, 64, 2]
        result = ttnn.concat([out_real, out_imag], 4, memory_config=DRAM)
        ttnn.deallocate(out_real, False)
        ttnn.deallocate(out_imag, False)

        # Reshape back to [1, seq, 30, 128]
        result = ttnn.reshape(result, [1, seq, N_HEADS, HEAD_DIM], memory_config=DRAM)
        return result

    def __call__(self, x_2d, attn_mask, rope_cos, rope_sin):
        """
        Args:
            x_2d: [seq, dim] in BFLOAT16 (flattened for matmul)
            attn_mask: precomputed attention mask (0 or -inf)
            rope_cos: [1, seq, 1, 64, 1] FLOAT32
            rope_sin: [1, seq, 1, 64, 1] FLOAT32

        Returns:
            [1, seq, dim] in BFLOAT16
        """
        seq = x_2d.shape[0]

        # --- Q projection ---
        q = ttnn.matmul(
            x_2d, self.to_q,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        q = ttnn.reshape(q, [1, seq, N_HEADS, HEAD_DIM], memory_config=DRAM)

        # Q norm (per-head RMSNorm)
        q = ttnn.rms_norm(
            q, epsilon=RMS_EPS, weight=self.norm_q_weight, bias=None,
            residual_input_tensor=None, memory_config=DRAM,
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )
        # Cast to FLOAT32 for RoPE math
        q = ttnn.typecast(q, ttnn.DataType.FLOAT32, memory_config=DRAM)

        # Q RoPE
        q = self._apply_rope(q, rope_cos, rope_sin, seq)

        # Permute to heads-first: [1, 30, seq, 128]
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)

        # --- K projection ---
        k = ttnn.matmul(
            x_2d, self.to_k,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        k = ttnn.reshape(k, [1, seq, N_HEADS, HEAD_DIM], memory_config=DRAM)

        # K norm
        k = ttnn.rms_norm(
            k, epsilon=RMS_EPS, weight=self.norm_k_weight, bias=None,
            residual_input_tensor=None, memory_config=DRAM,
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )
        k = ttnn.typecast(k, ttnn.DataType.FLOAT32, memory_config=DRAM)

        # K RoPE
        k = self._apply_rope(k, rope_cos, rope_sin, seq)

        # Permute to heads-first
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)

        # --- V projection (no norm, no RoPE) ---
        v = ttnn.matmul(
            x_2d, self.to_v,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        v = ttnn.typecast(v, ttnn.DataType.FLOAT32, memory_config=DRAM)
        v = ttnn.reshape(v, [1, seq, N_HEADS, HEAD_DIM], memory_config=DRAM)
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)

        # --- Typecast Q, K, V to BFLOAT16 for SDPA ---
        v = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        q = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        k = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # --- SDPA ---
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=SDPA_SCALE,
            sliding_window_size=None,
            memory_config=DRAM,
        )
        ttnn.deallocate(k, False)
        ttnn.deallocate(q, False)
        ttnn.deallocate(v, False)

        # --- Concatenate heads and output projection ---
        attn_out = ttnn.transformer.concatenate_heads(attn_out, memory_config=DRAM)
        attn_out = ttnn.reshape(attn_out, [seq, DIM], memory_config=DRAM)

        out = ttnn.matmul(
            attn_out, self.to_out,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(attn_out, False)

        # Reshape to [1, seq, dim]
        out = ttnn.reshape(out, [1, seq, DIM], memory_config=DRAM)
        return out


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(LightweightModule):
    """Transformer block with optional adaLN modulation.

    With modulation (noise_refiner, layers):
        adaLN: linear(adaln_input) -> 4 chunks of DIM each
            chunk 0 [0:3840] = attn_scale (add 1.0)
            chunk 1 [3840:7680] = attn_gate (tanh)
            chunk 2 [7680:11520] = ffn_scale (add 1.0)
            chunk 3 [11520:15360] = ffn_gate (tanh)

        Attention path:
            flatten [seq, dim] -> rms_norm(x, attention_norm1)
            -> multiply(normed, attn_scale) -> attention
            -> rms_norm(attn_out, attention_norm2)
            -> multiply(attn_gate, normed_attn) -> residual add

        FFN path:
            flatten [seq, dim] -> rms_norm(x, ffn_norm1)
            -> multiply(normed, ffn_scale) -> feed_forward
            -> reshape [1, seq, dim] -> rms_norm(ffn_out, ffn_norm2)
            -> multiply(ffn_gate, normed_ffn) -> residual add

    Without modulation (context_refiner):
        Same but no adaLN, no scale/gate. Direct rms_norm -> attention -> residual.
    """

    def __init__(
        self,
        attention,
        feed_forward,
        attention_norm1_weight,
        attention_norm2_weight,
        ffn_norm1_weight,
        ffn_norm2_weight,
        adaln_weight=None,
        adaln_bias=None,
        modulation=True,
    ):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.attention_norm1_weight = attention_norm1_weight
        self.attention_norm2_weight = attention_norm2_weight
        self.ffn_norm1_weight = ffn_norm1_weight
        self.ffn_norm2_weight = ffn_norm2_weight
        self.adaln_weight = adaln_weight
        self.adaln_bias = adaln_bias
        self.modulation = modulation

    def __call__(self, x, attn_mask, rope_cos, rope_sin, adaln_input=None, scalar_one=None):
        """
        Args:
            x: [1, seq, dim] in BFLOAT16
            attn_mask: precomputed attention mask
            rope_cos: [1, seq, 1, 64, 1] FLOAT32
            rope_sin: [1, seq, 1, 64, 1] FLOAT32
            adaln_input: [1, 256] FLOAT32 (timestep embedding, only for modulation)
            scalar_one: [1, 1] BFLOAT16 scalar 1.0 (for 1 + scale)

        Returns:
            [1, seq, dim] in BFLOAT16
        """
        seq = x.shape[1]

        if self.modulation:
            # --- adaLN modulation ---
            # linear(adaln_input) -> [1, 4*dim=15360] in FLOAT32
            adaln_out_f32 = ttnn.linear(
                adaln_input, self.adaln_weight, bias=self.adaln_bias,
                transpose_a=False, transpose_b=False,
                memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
                program_config=None, activation=None,
                compute_kernel_config=COMPUTE_CONFIG,
            )
            adaln_out = ttnn.typecast(adaln_out_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(adaln_out_f32, False)

            # Reshape to [1, 1, 15360] for 3D slicing (gate slices)
            adaln_3d = ttnn.reshape(adaln_out, [1, 1, 4 * DIM], memory_config=DRAM)

            # attn_gate = tanh(chunk[1]) from 3D view: [0,0,DIM] to [1,1,2*DIM]
            attn_gate_raw = ttnn.slice(adaln_3d, [0, 0, DIM], [1, 1, 2 * DIM], [1, 1, 1], memory_config=DRAM)
            attn_gate = ttnn.tanh(attn_gate_raw, fast_and_approximate_mode=False, memory_config=DRAM)
            ttnn.deallocate(attn_gate_raw, False)

            # attn_scale = 1 + chunk[0] from 2D view: [0,0] to [1,DIM]
            attn_scale_raw = ttnn.slice(adaln_out, [0, 0], [1, DIM], [1, 1], memory_config=DRAM)
            attn_scale = ttnn.add(attn_scale_raw, scalar_one, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(attn_scale_raw, False)

            # ffn_gate = tanh(chunk[3]) from 3D view
            ffn_gate_raw = ttnn.slice(adaln_3d, [0, 0, 3 * DIM], [1, 1, 4 * DIM], [1, 1, 1], memory_config=DRAM)
            ttnn.deallocate(adaln_3d, False)
            ffn_gate = ttnn.tanh(ffn_gate_raw, fast_and_approximate_mode=False, memory_config=DRAM)
            ttnn.deallocate(ffn_gate_raw, False)

            # ffn_scale = 1 + chunk[2] from 2D view
            ffn_scale_raw = ttnn.slice(adaln_out, [0, 2 * DIM], [1, 3 * DIM], [1, 1], memory_config=DRAM)
            ttnn.deallocate(adaln_out, False)
            ffn_scale = ttnn.add(ffn_scale_raw, scalar_one, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(ffn_scale_raw, False)

            # --- Attention path ---
            # Flatten [1, seq, dim] -> [seq, dim] for RMSNorm
            x_flat = ttnn.reshape(x, [seq, DIM], memory_config=DRAM)
            normed = ttnn.rms_norm(
                x_flat, epsilon=RMS_EPS, weight=self.attention_norm1_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(x_flat, False)

            # Apply adaLN scale: normed * (1 + attn_scale)
            # normed is [seq, DIM], attn_scale is [1, DIM] -> broadcasts
            normed_scaled = ttnn.multiply(normed, attn_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(attn_scale, False)
            ttnn.deallocate(normed, False)

            # Attention (input is [seq, dim] 2D)
            attn_out = self.attention(normed_scaled, attn_mask, rope_cos, rope_sin)
            ttnn.deallocate(normed_scaled, False)

            # Post-attention norm + gate + residual
            attn_normed = ttnn.rms_norm(
                attn_out, epsilon=RMS_EPS, weight=self.attention_norm2_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(attn_out, False)
            gated_attn = ttnn.multiply(attn_gate, attn_normed, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(attn_normed, False)
            ttnn.deallocate(attn_gate, False)
            x = ttnn.add(x, gated_attn, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(gated_attn, False)

            # --- FFN path ---
            # Flatten for FFN: [1, seq, dim] -> [seq, dim]
            x_flat2 = ttnn.reshape(x, [seq, DIM], memory_config=DRAM)
            ffn_normed = ttnn.rms_norm(
                x_flat2, epsilon=RMS_EPS, weight=self.ffn_norm1_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(x_flat2, False)

            # Apply adaLN scale
            ffn_scaled = ttnn.multiply(ffn_normed, ffn_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(ffn_scale, False)
            ttnn.deallocate(ffn_normed, False)

            # FFN
            ffn_out = self.feed_forward(ffn_scaled)
            ttnn.deallocate(ffn_scaled, False)

            # Post-FFN norm + gate + residual
            ffn_out_3d = ttnn.reshape(ffn_out, [1, seq, DIM], memory_config=DRAM)
            ttnn.deallocate(ffn_out, False)
            ffn_normed2 = ttnn.rms_norm(
                ffn_out_3d, epsilon=RMS_EPS, weight=self.ffn_norm2_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(ffn_out_3d, False)
            gated_ffn = ttnn.multiply(ffn_gate, ffn_normed2, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(ffn_normed2, False)
            ttnn.deallocate(ffn_gate, False)
            x = ttnn.add(x, gated_ffn, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(gated_ffn, False)

        else:
            # --- No modulation (context_refiner) ---
            # Save 3D view for residual
            x_3d = ttnn.reshape(x, [1, seq, DIM], memory_config=DRAM)

            # Flatten to [seq, dim] for norms and matmuls
            x_flat = ttnn.reshape(x, [seq, DIM], memory_config=DRAM)

            # Attention path: norm -> attention -> norm -> residual
            normed = ttnn.rms_norm(
                x_flat, epsilon=RMS_EPS, weight=self.attention_norm1_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(x_flat, False)

            # Attention (input is [seq, dim] 2D, output is [1, seq, dim])
            attn_out = self.attention(normed, attn_mask, rope_cos, rope_sin)
            ttnn.deallocate(normed, False)

            attn_normed = ttnn.rms_norm(
                attn_out, epsilon=RMS_EPS, weight=self.attention_norm2_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(attn_out, False)

            # Plain residual add (no gate)
            x = ttnn.add(x_3d, attn_normed, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(attn_normed, False)
            ttnn.deallocate(x_3d, False)

            # FFN path: flatten -> norm -> ffn -> reshape -> norm -> residual
            x_flat2 = ttnn.reshape(x, [seq, DIM], memory_config=DRAM)
            ffn_normed = ttnn.rms_norm(
                x_flat2, epsilon=RMS_EPS, weight=self.ffn_norm1_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(x_flat2, False)

            ffn_out = self.feed_forward(ffn_normed)
            ttnn.deallocate(ffn_normed, False)

            ffn_out_3d = ttnn.reshape(ffn_out, [1, seq, DIM], memory_config=DRAM)
            ttnn.deallocate(ffn_out, False)
            ffn_normed2 = ttnn.rms_norm(
                ffn_out_3d, epsilon=RMS_EPS, weight=self.ffn_norm2_weight, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            ttnn.deallocate(ffn_out_3d, False)

            # Plain residual add (no gate)
            x = ttnn.add(x, ffn_normed2, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(ffn_normed2, False)

        return x


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(LightweightModule):
    """Timestep embedding via frequency table -> cos/sin -> MLP.

    Codegen pattern:
        t [1] -> to_layout(TILE) -> reshape [1, 1]
        -> multiply(t, freqs)           # freqs is [1, 128], result [1, 128]
        -> cos -> sin -> concat [1, 256]
        -> linear(w0, b0, silu)         # [1, 256] -> [1, 1024]
        -> linear(w2, b2)               # [1, 1024] -> [1, 256]

    The [1, 256] output is the adaln_input used by all modulated transformer blocks.
    """

    def __init__(self, freqs, w0, b0, w2, b2):
        """
        Args:
            freqs: [1, 128] FLOAT32 prepared frequency table
            w0: pre-transposed weight for first linear (silu), FLOAT32
            b0: [1024] FLOAT32 bias
            w2: pre-transposed weight for second linear, FLOAT32
            b2: [256] FLOAT32 bias
        """
        super().__init__()
        self.freqs = freqs
        self.w0 = w0
        self.b0 = b0
        self.w2 = w2
        self.b2 = b2

    def __call__(self, t):
        """
        Args:
            t: [1] scalar timestep, FLOAT32 on device

        Returns:
            [1, 256] FLOAT32 timestep embedding (adaln_input for all blocks)
        """
        # Reshape to [1, 1]
        t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
        t = ttnn.reshape(t, [1, 1], memory_config=DRAM)

        # Multiply with freq table: [1, 1] * [1, 128] -> [1, 128]
        freq_args = ttnn.multiply(t, self.freqs, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(t, False)

        # cos and sin
        cos_t = ttnn.cos(freq_args, memory_config=DRAM)
        sin_t = ttnn.sin(freq_args, memory_config=DRAM)
        ttnn.deallocate(freq_args, False)

        # Concat [cos, sin] -> [1, 256]
        t_freq = ttnn.concat([cos_t, sin_t], 1, memory_config=DRAM)
        ttnn.deallocate(sin_t, False)
        ttnn.deallocate(cos_t, False)

        # MLP layer 0: linear with silu activation -> [1, 1024]
        h = ttnn.linear(
            t_freq, self.w0, bias=self.b0,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation="silu",
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(t_freq, False)

        # MLP layer 2: linear -> [1, 256]
        out = ttnn.linear(
            h, self.w2, bias=self.b2,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(h, False)

        # Return [1, 256] FLOAT32 -- this is the adaln_input used everywhere
        return out


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

class FinalLayer(LightweightModule):
    """Final layer: manual LayerNorm + adaLN scale + Linear + un-patchify.

    Codegen pattern:
        1. Manual LayerNorm (elementwise_affine=False, eps=1e-6):
           typecast FLOAT32 -> mean -> subtract -> square -> mean -> add(eps) -> rsqrt
           -> reshape -> multiply(centered, rsqrt) -> typecast BFLOAT16

        2. adaLN modulation:
           silu(adaln_input) -> linear(w, b) -> typecast BFLOAT16
           -> add(1.0) -> multiply(layernorm_out, scale)

        3. Output projection:
           matmul(scaled, linear_weight, transpose_b=True) -> add(bias_broadcast)

        4. Un-patchify:
           slice [0:3600] -> reshape [1,80,45,1,2,2,16]
           -> permute [6,0,3,1,4,2,5] -> reshape [16,1,160,90]
    """

    def __init__(self, adaln_weight, adaln_bias, linear_weight, linear_bias_broadcast,
                 scalar_one, layernorm_eps):
        """
        Args:
            adaln_weight: pre-transposed weight for final adaLN linear, FLOAT32
            adaln_bias: bias for final adaLN linear, FLOAT32
            linear_weight: [64, 3840] BFLOAT16 (used with transpose_b=True)
            linear_bias_broadcast: [3648, 64] FLOAT32 broadcast bias
            scalar_one: [1, 1] BFLOAT16 scalar 1.0
            layernorm_eps: [1, 1, 1] FLOAT32 scalar ~1e-6
        """
        super().__init__()
        self.adaln_weight = adaln_weight
        self.adaln_bias = adaln_bias
        self.linear_weight = linear_weight
        self.linear_bias_broadcast = linear_bias_broadcast
        self.scalar_one = scalar_one
        self.layernorm_eps = layernorm_eps

    def __call__(self, x, adaln_input):
        """
        Args:
            x: [1, 3648, 3840] in BFLOAT16 (unified sequence after all layers)
            adaln_input: [1, 256] FLOAT32 timestep embedding

        Returns:
            [16, 1, 160, 90] in BFLOAT16 (un-patchified output)
        """
        # --- Step 1: Manual LayerNorm (no learnable parameters, eps=1e-6) ---
        x_f32 = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x, False)

        # mean along last dim (hidden dimension)
        x_mean = ttnn.mean(x_f32, [2], True, memory_config=DRAM, compute_kernel_config=COMPUTE_CONFIG)
        x_centered_neg = ttnn.neg(x_mean, memory_config=DRAM)
        ttnn.deallocate(x_mean, False)
        x_centered = ttnn.add(x_f32, x_centered_neg, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x_centered_neg, False)
        ttnn.deallocate(x_f32, False)

        # variance = mean(centered^2)
        x_sq = ttnn.multiply(x_centered, x_centered, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        x_var = ttnn.mean(x_sq, [2], True, memory_config=DRAM, compute_kernel_config=COMPUTE_CONFIG)
        ttnn.deallocate(x_sq, False)

        # rsqrt(variance + eps)
        x_var_eps = ttnn.add(x_var, self.layernorm_eps, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x_var, False)
        x_rstd = ttnn.rsqrt(x_var_eps, fast_and_approximate_mode=False, memory_config=DRAM)
        ttnn.deallocate(x_var_eps, False)

        # Reshape for broadcast multiply: rstd -> [seq, 1], centered -> [seq, dim]
        x_rstd_2d = ttnn.reshape(x_rstd, [TOTAL_SEQ, 1], memory_config=DRAM)
        ttnn.deallocate(x_rstd, False)
        x_centered_2d = ttnn.reshape(x_centered, [TOTAL_SEQ, DIM], memory_config=DRAM)
        ttnn.deallocate(x_centered, False)

        # Normalize
        x_normed_f32 = ttnn.multiply(x_centered_2d, x_rstd_2d, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(x_centered_2d, False)
        ttnn.deallocate(x_rstd_2d, False)
        x_normed = ttnn.typecast(x_normed_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(x_normed_f32, False)

        # --- Step 2: adaLN modulation ---
        # silu(adaln_input) -> linear -> (1 + scale)
        adaln_silu = ttnn.silu(adaln_input, memory_config=DRAM)

        scale_f32 = ttnn.linear(
            adaln_silu, self.adaln_weight, bias=self.adaln_bias,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(adaln_silu, False)
        scale_bf16 = ttnn.typecast(scale_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(scale_f32, False)
        scale = ttnn.add(scale_bf16, self.scalar_one, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(scale_bf16, False)

        # Apply scale to normalized output
        x_modulated = ttnn.multiply(x_normed, scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(scale, False)
        ttnn.deallocate(x_normed, False)

        # --- Step 3: Output projection ---
        x_proj = ttnn.matmul(
            x_modulated, self.linear_weight,
            transpose_a=False, transpose_b=True,
            memory_config=DRAM, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(x_modulated, False)

        # Add broadcast bias
        x_biased = ttnn.add(x_proj, self.linear_bias_broadcast, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(x_proj, False)

        # --- Step 4: Un-patchify ---
        # Slice to image tokens only [0:3600] (drop 16 padding + 32 caption tokens)
        # x_biased is [3648, 64] (2D from the LayerNorm reshape path)
        img_tokens = ttnn.slice(x_biased, [0, 0], [IMAGE_ORI_LEN, 64], [1, 1], memory_config=DRAM)
        ttnn.deallocate(x_biased, False)

        # Reshape: [3600, 64] -> [1, 80, 45, 1, 2, 2, 16]
        img_blocks = ttnn.reshape(img_tokens, [1, 80, 45, 1, PATCH_SIZE, PATCH_SIZE, IN_CHANNELS], memory_config=DRAM)
        ttnn.deallocate(img_tokens, False)

        # Permute: [6, 0, 3, 1, 4, 2, 5] -> [16, 1, 1, 80, 2, 45, 2]
        img_permuted = ttnn.permute(img_blocks, [6, 0, 3, 1, 4, 2, 5], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(img_blocks, False)

        # Final reshape: [16, 1, 160, 90]
        output = ttnn.reshape(img_permuted, [IN_CHANNELS, 1, 160, 90], memory_config=DRAM)
        ttnn.deallocate(img_permuted, False)

        return output


# ---------------------------------------------------------------------------
# ZImageTransformerTTNN
# ---------------------------------------------------------------------------

class ZImageTransformerTTNN(LightweightModule):
    """Complete Z-Image transformer in TTNN.

    Constructor takes a device and a PyTorch state_dict, converts all weights
    to TTNN format, runs const-evals, and builds all submodules.
    """

    def __init__(self, device, state_dict):
        """
        Args:
            device: ttnn device handle.
            state_dict: dict from model_pt.py's ZImageTransformerPT.state_dict_for_ttnn().
        """
        super().__init__()
        self.device = device

        # Convert PyTorch state_dict to TTNN tensors on host
        host_params = load_params_from_pytorch(state_dict, device)

        # Run const-evals to prepare weights for device
        self.ce = run_const_evals(host_params, device)

        # --- Build submodules ---

        # TimestepEmbedder
        self.t_embedder = TimestepEmbedder(
            freqs=self.ce["freqs_prepared"],
            w0=self.ce["t_embedder.mlp.0.weight"],
            b0=self.ce["t_embedder.mlp.0.bias"],
            w2=self.ce["t_embedder.mlp.2.weight"],
            b2=self.ce["t_embedder.mlp.2.bias"],
        )

        # x_embedder weight (pre-transposed for matmul without transpose_b)
        self.x_embedder_weight = self.ce["x_embedder.weight"]
        self.x_embedder_bias_broadcast = self.ce["x_embedder.bias_broadcast"]

        # cap_embedder weight (pre-transposed)
        self.cap_embedder_weight = self.ce["cap_embedder.1.weight"]
        self.cap_embedder_bias = self.ce["cap_embedder.1.bias"]

        # Pad tokens and masks (on device)
        self.x_pad_token = host_params["x_pad_token"]
        self.cap_pad_token_broadcast = self.ce.get("cap_pad_token_broadcast")
        self.image_pad_mask = self.ce["image_pad_mask"]
        self.cap_pad_mask = self.ce.get("cap_pad_mask")

        # Attention masks (precomputed)
        self.x_attn_mask = self.ce.get("x_attn_mask")
        self.cap_attn_mask = self.ce.get("cap_attn_mask")
        self.unified_attn_mask = self.ce.get("unified_attn_mask")

        # RoPE embeddings (precomputed)
        self.image_rope_sin = self.ce.get("image_rope_sin")
        self.image_rope_cos = self.ce.get("image_rope_cos")
        self.cap_rope_sin = self.ce.get("cap_rope_sin")
        self.cap_rope_cos = self.ce.get("cap_rope_cos")
        self.unified_rope_sin = self.ce.get("unified_rope_sin")
        self.unified_rope_cos = self.ce.get("unified_rope_cos")

        # Scalar one for adaLN (1 + scale)
        self.scalar_one = self.ce["scalar_one"]

        # Cap embedder norm weight (used directly for rms_norm)
        self.cap_embedder_norm_weight = host_params["cap_embedder.0.weight"]

        # --- Build transformer blocks ---

        def _build_block(prefix, modulation):
            """Build a TransformerBlock from weights with the given prefix."""
            attn = Attention(
                to_q=host_params[f"{prefix}.attention.to_q.weight"],
                to_k=host_params[f"{prefix}.attention.to_k.weight"],
                to_v=host_params[f"{prefix}.attention.to_v.weight"],
                to_out=host_params[f"{prefix}.attention.to_out.weight"],
                norm_q_weight=host_params[f"{prefix}.attention.norm_q.weight"],
                norm_k_weight=host_params[f"{prefix}.attention.norm_k.weight"],
            )
            ff = FeedForward(
                w1=host_params[f"{prefix}.feed_forward.w1.weight"],
                w2=host_params[f"{prefix}.feed_forward.w2.weight"],
                w3=host_params[f"{prefix}.feed_forward.w3.weight"],
            )
            kwargs = dict(
                attention=attn,
                feed_forward=ff,
                attention_norm1_weight=host_params[f"{prefix}.attention_norm1.weight"],
                attention_norm2_weight=host_params[f"{prefix}.attention_norm2.weight"],
                ffn_norm1_weight=host_params[f"{prefix}.ffn_norm1.weight"],
                ffn_norm2_weight=host_params[f"{prefix}.ffn_norm2.weight"],
                modulation=modulation,
            )
            if modulation:
                kwargs["adaln_weight"] = self.ce[f"{prefix}.adaLN_modulation.0.weight"]
                kwargs["adaln_bias"] = self.ce[f"{prefix}.adaLN_modulation.0.bias"]
            return TransformerBlock(**kwargs)

        # Noise refiner: 2 blocks with modulation (image-only attention)
        self.noise_refiner = [_build_block(f"noise_refiner.{i}", modulation=True) for i in range(2)]

        # Context refiner: 2 blocks without modulation (caption-only attention)
        self.context_refiner = [_build_block(f"context_refiner.{i}", modulation=False) for i in range(2)]

        # Main layers: 30 blocks with modulation (unified attention)
        self.layers = [_build_block(f"layers.{i}", modulation=True) for i in range(30)]

        # Final layer
        self.final_layer = FinalLayer(
            adaln_weight=self.ce["final_layer.adaLN_modulation.1.weight"],
            adaln_bias=self.ce["final_layer.adaLN_modulation.1.bias"],
            linear_weight=host_params["final_layer.linear.weight"],
            linear_bias_broadcast=self.ce["final_layer.linear.bias_broadcast"],
            scalar_one=self.scalar_one,
            layernorm_eps=self.ce["layernorm_eps"],
        )

    def forward(self, latent_input, timestep, cap_feat):
        """Full forward pass of the Z-Image transformer.

        Args:
            latent_input: [16, 1, 160, 90] BFLOAT16 latent tensor (on device)
            timestep: [1] FLOAT32 scalar timestep (on device)
            cap_feat: [18, 2560] BFLOAT16 caption features (on device)

        Returns:
            [16, 1, 160, 90] BFLOAT16 denoised output
        """
        # ===================================================================
        # 0. Move inputs to device
        # ===================================================================
        latent_input = ttnn.to_device(latent_input, device=self.device, memory_config=DRAM)
        timestep = ttnn.to_device(timestep, device=self.device, memory_config=DRAM)
        cap_feat = ttnn.to_device(cap_feat, device=self.device, memory_config=DRAM)

        # ===================================================================
        # 1. Timestep embedding -> [1, 256] FLOAT32
        # ===================================================================
        adaln_input = self.t_embedder(timestep)

        # ===================================================================
        # 2. Patch embedding: patchify + linear + pad mask
        # ===================================================================
        # latent [16,1,160,90] -> [16,1,1,80,2,45,2] -> [1,80,45,1,2,2,16] -> [3600,64]
        img = ttnn.to_layout(latent_input, ttnn.Layout.TILE, None, memory_config=None)
        img = ttnn.reshape(img, [IN_CHANNELS, 1, 1, 80, PATCH_SIZE, 45, PATCH_SIZE], memory_config=DRAM)
        img = ttnn.permute(img, [1, 3, 5, 2, 4, 6, 0], memory_config=DRAM, pad_value=0.0)
        img = ttnn.reshape(img, [IMAGE_ORI_LEN, 64], memory_config=DRAM)

        # Pad: take last row, repeat 16 times, concat -> [3616, 64]
        last_row = ttnn.slice(img, [IMAGE_ORI_LEN - 1, 0], [IMAGE_ORI_LEN, 64], [1, 1], memory_config=DRAM)
        pad_rows = ttnn.repeat(last_row, ttnn.Shape([IMAGE_PADDED - IMAGE_ORI_LEN, 1]), memory_config=DRAM)
        ttnn.deallocate(last_row, False)
        img = ttnn.concat([img, pad_rows], 0, memory_config=DRAM)
        ttnn.deallocate(pad_rows, False)

        # x_embedder: typecast -> matmul with pre-transposed weight + bias
        img = ttnn.typecast(img, ttnn.DataType.FLOAT32, memory_config=DRAM)
        img = ttnn.matmul(
            img, self.x_embedder_weight,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        img = ttnn.add(img, self.x_embedder_bias_broadcast, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        img = ttnn.typecast(img, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Apply pad mask: where(pad_mask, pad_token, embedded)
        img = ttnn.where(self.image_pad_mask, self.x_pad_token, img, memory_config=DRAM)

        # Reshape to [1, 3616, 3840]
        x = ttnn.reshape(img, [1, IMAGE_PADDED, DIM], memory_config=DRAM)

        # ===================================================================
        # 3. Caption embedding: pad + RMSNorm + linear + pad mask
        # ===================================================================
        cap = ttnn.to_layout(cap_feat, ttnn.Layout.TILE, None, memory_config=None)
        cap_last = ttnn.slice(cap, [CAP_ORI_LEN - 1, 0], [CAP_ORI_LEN, CAP_FEAT_DIM], [1, 1], memory_config=DRAM)
        cap_pad = ttnn.repeat(cap_last, ttnn.Shape([CAP_PADDED - CAP_ORI_LEN, 1]), memory_config=DRAM)
        ttnn.deallocate(cap_last, False)
        cap = ttnn.concat([cap, cap_pad], 0, memory_config=DRAM)
        ttnn.deallocate(cap_pad, False)

        # RMSNorm with cap_embedder.0.weight
        cap = ttnn.rms_norm(
            cap, epsilon=RMS_EPS, weight=self.cap_embedder_norm_weight, bias=None,
            residual_input_tensor=None, memory_config=DRAM,
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )

        # Linear: typecast -> matmul with pre-transposed weight + bias
        cap = ttnn.typecast(cap, ttnn.DataType.FLOAT32, memory_config=DRAM)
        cap = ttnn.matmul(
            cap, self.cap_embedder_weight,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        cap = ttnn.add(cap, self.cap_embedder_bias, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        cap = ttnn.typecast(cap, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Apply pad mask
        cap = ttnn.where(self.cap_pad_mask, self.cap_pad_token_broadcast, cap, memory_config=DRAM)

        # Reshape to [1, 32, 3840]
        cap = ttnn.reshape(cap, [1, CAP_PADDED, DIM], memory_config=DRAM)

        # ===================================================================
        # 4. Noise refiner (2 blocks on image only, with modulation)
        # Uses image RoPE [1, 3616, 1, 64, 1] and x_attn_mask
        # ===================================================================
        for block in self.noise_refiner:
            x = block(
                x, self.x_attn_mask,
                self.image_rope_cos, self.image_rope_sin,
                adaln_input=adaln_input, scalar_one=self.scalar_one,
            )

        # ===================================================================
        # 5. Context refiner (2 blocks on caption only, without modulation)
        # Uses caption RoPE [1, 32, 1, 64, 1] and cap_attn_mask
        # ===================================================================
        for block in self.context_refiner:
            cap = block(
                cap, self.cap_attn_mask,
                self.cap_rope_cos, self.cap_rope_sin,
            )

        # ===================================================================
        # 6. Concatenate image + caption for unified layers
        # [1, 3616, 3840] + [1, 32, 3840] -> [1, 3648, 3840]
        # ===================================================================
        unified = ttnn.concat([x, cap], 1, memory_config=DRAM)
        ttnn.deallocate(x, False)
        ttnn.deallocate(cap, False)

        # ===================================================================
        # 7. Main transformer layers (30 blocks, with modulation)
        # Uses unified RoPE [1, 3648, 1, 64, 1] and unified_attn_mask
        # ===================================================================
        for block in self.layers:
            unified = block(
                unified, self.unified_attn_mask,
                self.unified_rope_cos, self.unified_rope_sin,
                adaln_input=adaln_input, scalar_one=self.scalar_one,
            )

        # ===================================================================
        # 8. Final layer + un-patchify
        # ===================================================================
        out = self.final_layer(unified, adaln_input)
        ttnn.deallocate(unified, False)
        ttnn.deallocate(adaln_input, False)

        return out

