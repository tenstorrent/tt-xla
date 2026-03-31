# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Clean TTNN implementation of the Z-Image text encoder (Qwen3-based).

Self-contained -- no imports from sibling directories.
Extracted from the 22K-line codegen main.py and organized into clean
LightweightModule classes matching the PyTorch reference model_pt.py.

Architecture:
    TextEncoderTTNN
    +-- embed_tokens: embedding lookup [151936, 2560]
    +-- rotary_emb: precomputed cos/sin from inv_freq
    +-- 35 DecoderLayers (0-34), each:
        +-- input_layernorm: RMSNorm [2560]
        +-- Attention: GQA (32 Q heads, 8 KV heads, head_dim=128)
            +-- q_proj, k_proj, v_proj, o_proj
            +-- q_norm, k_norm (per-head RMSNorm)
            +-- RoPE via ttnn.experimental.rotary_embedding
        +-- post_attention_layernorm: RMSNorm [2560]
        +-- FeedForward: SwiGLU (gate_proj with fused silu, up_proj, down_proj)
    Output: last layer hidden states [1, 512, 2560] (no final norm)
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
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
RMS_EPS = 9.9999999747524271e-07  # 1e-6
SDPA_SCALE = 0.088388338685035706  # 1/sqrt(128)

# Model dimensions
HIDDEN_DIM = 2560
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
MLP_INTER = 9728
SEQ_LEN = 512
NUM_LAYERS = 35

# ---------------------------------------------------------------------------
# Sharded memory configs dictionary
# ---------------------------------------------------------------------------

MEMORY_CONFIGS = {
    "DRAM": DRAM,
    # Token IDs typecast: 8 cores in single row, shard [32, 64]
    "L1_WIDTH_8x1_32x64": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]
            ),
            [32, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
    # Q projection output: 10x8 grid, shard [64, 416]
    "L1_BLOCK_10x8_64x416": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))]
            ),
            [64, 416],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
    # K/V projection output: 8x8 grid, shard [64, 128]
    "L1_BLOCK_8x8_64x128": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
            ),
            [64, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
    # O projection + residuals: 10x8 grid, shard [64, 256]
    "L1_BLOCK_10x8_64x256": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))]
            ),
            [64, 256],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
    # Gate/Up projection output: 10x8 grid, shard [64, 992]
    "L1_BLOCK_10x8_64x992": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))]
            ),
            [64, 992],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
    # SwiGLU multiply output: 9x8 grid, shard [64, 1088]
    "L1_BLOCK_9x8_64x1088": ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 7))]
            ),
            [64, 1088],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    ),
}

# ---------------------------------------------------------------------------
# Matmul program configs dictionary
# ---------------------------------------------------------------------------

MATMUL_CONFIGS = {
    # Q projection: [512, 2560] x [2560, 4096] -> [512, 4096]
    "q_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=13,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    ),
    # K/V projection: [512, 2560] x [2560, 1024] -> [512, 1024]
    "kv_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=2,
        per_core_N=4,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    ),
    # O projection + down projection: [512, 4096] x [4096, 2560] -> [512, 2560]
    "o_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=2,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    ),
    # Gate projection: [512, 2560] x [2560, 9728] -> [512, 9728] (fused SiLU)
    "gate_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=31,
        transpose_mcast=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        fuse_batch=True,
    ),
    # Up projection: [512, 2560] x [2560, 9728] -> [512, 9728] (no activation)
    "up_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=31,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    ),
    # Down projection: [512, 9728] x [9728, 2560] -> [512, 2560]
    # Same config as o_proj
    "down_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=2,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    ),
}


# ---------------------------------------------------------------------------
# LightweightModule base class
# ---------------------------------------------------------------------------

class LightweightModule:
    """Base class for TTNN modules. Stores weights as attributes, no nn.Module."""

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU MLP)
# ---------------------------------------------------------------------------

class FeedForward(LightweightModule):
    """SwiGLU feed-forward network.

    gate_proj with fused SiLU activation, up_proj, element-wise multiply,
    then down_proj.
    """

    def __init__(self, gate_proj_weight, up_proj_weight, down_proj_weight):
        super().__init__()
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight

    def __call__(self, x):
        """
        Args:
            x: [512, 2560] in BFLOAT16

        Returns:
            [512, 2560] in BFLOAT16, BLOCK_SHARDED L1
        """
        # Gate projection with fused SiLU: [512, 2560] x [2560, 9728] -> [512, 9728]
        gate = ttnn.matmul(
            x, self.gate_proj_weight,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x992"],
            dtype=None,
            program_config=MATMUL_CONFIGS["gate_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        gate = ttnn.to_memory_config(gate, MEMORY_CONFIGS["DRAM"])

        # Up projection: [512, 2560] x [2560, 9728] -> [512, 9728]
        up = ttnn.matmul(
            x, self.up_proj_weight,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x992"],
            dtype=None,
            program_config=MATMUL_CONFIGS["up_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        up = ttnn.to_memory_config(up, MEMORY_CONFIGS["DRAM"])

        # SwiGLU element-wise multiply: silu(gate) * up
        hidden = ttnn.multiply(
            gate, up,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_9x8_64x1088"],
        )
        ttnn.deallocate(gate, False)
        ttnn.deallocate(up, False)

        # Down projection: [512, 9728] x [9728, 2560] -> [512, 2560]
        out = ttnn.matmul(
            hidden, self.down_proj_weight,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x256"],
            dtype=None,
            program_config=MATMUL_CONFIGS["down_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(hidden, False)
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(LightweightModule):
    """GQA self-attention with QK-norm and RoPE.

    32 Q heads, 8 KV heads, head_dim=128.
    RoPE applied via ttnn.experimental.rotary_embedding after QK norm.
    """

    def __init__(self, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.q_norm = q_norm
        self.k_norm = k_norm

    def __call__(self, x_2d, attn_mask, rope_cos, rope_sin):
        """
        Args:
            x_2d: [512, 2560] BFLOAT16 (already flattened for matmul)
            attn_mask: [1, 1, 512, 512] BF16 additive attention mask
            rope_cos: [1, 1, 512, 128] BF16
            rope_sin: [1, 1, 512, 128] BF16

        Returns:
            [512, 2560] BFLOAT16, BLOCK_SHARDED L1 (o_proj output)
        """
        # --- Q projection ---
        q = ttnn.matmul(
            x_2d, self.q_proj,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x416"],
            dtype=None,
            program_config=MATMUL_CONFIGS["q_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        q = ttnn.to_memory_config(q, MEMORY_CONFIGS["DRAM"])
        q = ttnn.reshape(q, [1, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM], memory_config=MEMORY_CONFIGS["DRAM"])

        # Q norm (per-head RMSNorm)
        q = ttnn.rms_norm(
            q, epsilon=RMS_EPS, weight=self.q_norm, bias=None,
            residual_input_tensor=None, memory_config=MEMORY_CONFIGS["DRAM"],
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )

        # Permute to heads-first: [1, 32, 512, 128]
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=MEMORY_CONFIGS["DRAM"], pad_value=0.0)

        # Q RoPE
        q = ttnn.experimental.rotary_embedding(
            q, rope_cos, rope_sin, None,
            memory_config=MEMORY_CONFIGS["DRAM"],
            compute_kernel_config=COMPUTE_CONFIG,
        )

        # --- K projection ---
        k = ttnn.matmul(
            x_2d, self.k_proj,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_8x8_64x128"],
            dtype=None,
            program_config=MATMUL_CONFIGS["kv_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        k = ttnn.to_memory_config(k, MEMORY_CONFIGS["DRAM"])
        k = ttnn.reshape(k, [1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM], memory_config=MEMORY_CONFIGS["DRAM"])

        # K norm (per-head RMSNorm)
        k = ttnn.rms_norm(
            k, epsilon=RMS_EPS, weight=self.k_norm, bias=None,
            residual_input_tensor=None, memory_config=MEMORY_CONFIGS["DRAM"],
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )

        # Permute to heads-first: [1, 8, 512, 128]
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=MEMORY_CONFIGS["DRAM"], pad_value=0.0)

        # K RoPE
        k = ttnn.experimental.rotary_embedding(
            k, rope_cos, rope_sin, None,
            memory_config=MEMORY_CONFIGS["DRAM"],
            compute_kernel_config=COMPUTE_CONFIG,
        )

        # --- V projection (no norm, no RoPE) ---
        v = ttnn.matmul(
            x_2d, self.v_proj,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_8x8_64x128"],
            dtype=None,
            program_config=MATMUL_CONFIGS["kv_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        v = ttnn.to_memory_config(v, MEMORY_CONFIGS["DRAM"])
        v = ttnn.reshape(v, [1, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM], memory_config=MEMORY_CONFIGS["DRAM"])

        # Permute to heads-first: [1, 8, 512, 128]
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=MEMORY_CONFIGS["DRAM"], pad_value=0.0)

        # --- Scaled dot-product attention (GQA: 32 Q / 8 KV) ---
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=SDPA_SCALE,
            sliding_window_size=None,
            memory_config=MEMORY_CONFIGS["DRAM"],
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(q, False)
        ttnn.deallocate(k, False)
        ttnn.deallocate(v, False)

        # Concatenate heads: [1, 32, 512, 128] -> [1, 512, 4096]
        attn_out = ttnn.transformer.concatenate_heads(
            attn_out, memory_config=MEMORY_CONFIGS["DRAM"]
        )

        # O projection: [512, 4096] x [4096, 2560] -> [512, 2560]
        o_out = ttnn.matmul(
            attn_out, self.o_proj,
            transpose_a=False, transpose_b=True,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x256"],
            dtype=None,
            program_config=MATMUL_CONFIGS["o_proj"],
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(attn_out, False)
        return o_out


# ---------------------------------------------------------------------------
# DecoderLayer
# ---------------------------------------------------------------------------

class DecoderLayer(LightweightModule):
    """Single Qwen3 decoder layer.

    Pattern: norm -> attention -> residual -> norm -> MLP -> residual
    """

    def __init__(self, attention, feed_forward, input_layernorm, post_attn_layernorm):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm

    def __call__(self, hidden_states, attn_mask, rope_cos, rope_sin):
        """
        Args:
            hidden_states: [1, 512, 2560] BFLOAT16 DRAM (or [512, 2560] for first layer)
            attn_mask: [1, 1, 512, 512] BF16 additive attention mask
            rope_cos: [1, 1, 512, 128] BF16
            rope_sin: [1, 1, 512, 128] BF16

        Returns:
            [1, 512, 2560] BFLOAT16 DRAM
        """
        residual = hidden_states

        # Flatten to [512, 2560] for RMSNorm
        x_flat = ttnn.reshape(
            hidden_states, [SEQ_LEN, HIDDEN_DIM],
            memory_config=MEMORY_CONFIGS["DRAM"],
        )

        # Input layernorm
        normed = ttnn.rms_norm(
            x_flat, epsilon=RMS_EPS, weight=self.input_layernorm, bias=None,
            residual_input_tensor=None, memory_config=MEMORY_CONFIGS["DRAM"],
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )
        ttnn.deallocate(x_flat, False)

        # Self-attention (returns [512, 2560] BLOCK_SHARDED L1)
        attn_out = self.attention(normed, attn_mask, rope_cos, rope_sin)
        ttnn.deallocate(normed, False)

        # Attention residual add
        attn_residual = ttnn.add(
            residual, attn_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x256"],
        )
        ttnn.deallocate(attn_out, False)
        ttnn.deallocate(residual, False)

        attn_residual = ttnn.to_memory_config(
            attn_residual, MEMORY_CONFIGS["DRAM"]
        )

        # Flatten for post-attention layernorm
        attn_flat = ttnn.reshape(
            attn_residual, [SEQ_LEN, HIDDEN_DIM],
            memory_config=MEMORY_CONFIGS["DRAM"],
        )

        # Post-attention layernorm
        mlp_normed = ttnn.rms_norm(
            attn_flat, epsilon=RMS_EPS, weight=self.post_attn_layernorm, bias=None,
            residual_input_tensor=None, memory_config=MEMORY_CONFIGS["DRAM"],
            program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
        )
        ttnn.deallocate(attn_flat, False)

        # MLP (returns [512, 2560] BLOCK_SHARDED L1)
        mlp_out = self.feed_forward(mlp_normed)
        ttnn.deallocate(mlp_normed, False)

        # MLP residual add
        mlp_residual = ttnn.add(
            attn_residual, mlp_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=MEMORY_CONFIGS["L1_BLOCK_10x8_64x256"],
        )
        ttnn.deallocate(mlp_out, False)
        ttnn.deallocate(attn_residual, False)

        # Move to DRAM for next layer
        output = ttnn.to_memory_config(
            mlp_residual, MEMORY_CONFIGS["DRAM"]
        )
        ttnn.deallocate(mlp_residual, False)

        return output


# ---------------------------------------------------------------------------
# TextEncoderTTNN (top-level model)
# ---------------------------------------------------------------------------

class TextEncoderTTNN(LightweightModule):
    """TTNN implementation of the Z-Image text encoder (Qwen3 decoder).

    embed_tokens -> 35 decoder layers -> output hidden states [1, 512, 2560]
    No final RMSNorm (returns raw last-layer output).
    """

    def __init__(self, device, state_dict):
        """
        Args:
            device: ttnn device handle
            state_dict: PyTorch state_dict from TextEncoderModule
        """
        super().__init__()
        self.device = device

        # Load parameters
        print("Loading parameters...")
        self.params = load_params_from_pytorch(state_dict, device)

        # Run const-evals
        print("Running const-evals...")
        self.const_tensors = run_const_evals(self.params, device)

        # Build decoder layers
        self.layers = []
        for i in range(NUM_LAYERS):
            p = f"layers.{i}"
            attention = Attention(
                q_proj=self.params[f"{p}.self_attn.q_proj.weight"],
                k_proj=self.params[f"{p}.self_attn.k_proj.weight"],
                v_proj=self.params[f"{p}.self_attn.v_proj.weight"],
                o_proj=self.params[f"{p}.self_attn.o_proj.weight"],
                q_norm=self.params[f"{p}.self_attn.q_norm.weight"],
                k_norm=self.params[f"{p}.self_attn.k_norm.weight"],
            )
            feed_forward = FeedForward(
                gate_proj_weight=self.params[f"{p}.mlp.gate_proj.weight"],
                up_proj_weight=self.params[f"{p}.mlp.up_proj.weight"],
                down_proj_weight=self.params[f"{p}.mlp.down_proj.weight"],
            )
            layer = DecoderLayer(
                attention=attention,
                feed_forward=feed_forward,
                input_layernorm=self.params[f"{p}.input_layernorm.weight"],
                post_attn_layernorm=self.params[f"{p}.post_attention_layernorm.weight"],
            )
            self.layers.append(layer)

    def _build_attention_mask(self, attention_mask):
        """Build the additive attention mask from causal mask and padding mask.

        Args:
            attention_mask: [1, 512] BF16 DRAM (1=attend, 0=pad)

        Returns:
            [1, 1, 512, 512] BF16 additive mask (0 or -inf)
        """
        # Reshape attention_mask to [1, 1, 1, 512] for broadcast
        attn_mask_tiled = ttnn.to_layout(
            attention_mask, ttnn.Layout.TILE, None, memory_config=None
        )
        attn_mask_4d = ttnn.reshape(
            attn_mask_tiled, [1, 1, 1, SEQ_LEN],
            memory_config=MEMORY_CONFIGS["DRAM"],
        )

        # Combine causal mask with attention mask
        # causal_mask: [1, 1, 512, 512] BF16 (1=attend, 0=mask)
        # attn_mask_4d: [1, 1, 1, 512] BF16 (broadcasts across rows)
        combined = ttnn.logical_and(
            self.const_tensors["causal_mask"], attn_mask_4d,
            memory_config=MEMORY_CONFIGS["DRAM"],
        )
        ttnn.deallocate(attn_mask_4d, False)

        # Convert bool mask to float: where(mask, 0.0, -inf)
        combined_f32 = ttnn.typecast(
            combined, ttnn.DataType.FLOAT32,
            memory_config=MEMORY_CONFIGS["DRAM"],
        )
        ttnn.deallocate(combined, False)

        mask_float = ttnn.where(
            combined_f32,
            self.const_tensors["zeros_mask"],
            self.const_tensors["neg_inf_mask"],
            memory_config=MEMORY_CONFIGS["DRAM"],
        )
        ttnn.deallocate(combined_f32, False)

        # Cast to BF16
        mask_bf16 = ttnn.typecast(
            mask_float, ttnn.DataType.BFLOAT16,
            memory_config=MEMORY_CONFIGS["DRAM"],
        )
        ttnn.deallocate(mask_float, False)

        return mask_bf16

    def _process_input_ids(self, input_ids):
        """Process input_ids for embedding lookup.

        Args:
            input_ids: host TTNN tensor, ROW_MAJOR INT32 [1, 512]

        Returns:
            token embeddings [1, 512, 2560] BF16 TILE DRAM
        """
        # To TILE layout on DRAM
        ids_tiled = ttnn.to_layout(
            input_ids, ttnn.Layout.TILE, None,
            memory_config=MEMORY_CONFIGS["DRAM"],
        )

        # Move to L1 WIDTH_SHARDED for typecast
        ids_sharded = ttnn.to_memory_config(
            ids_tiled, MEMORY_CONFIGS["L1_WIDTH_8x1_32x64"]
        )

        # Typecast INT32 -> UINT32
        ids_uint32 = ttnn.typecast(
            ids_sharded, ttnn.DataType.UINT32,
            memory_config=MEMORY_CONFIGS["L1_WIDTH_8x1_32x64"],
        )

        # Back to ROW_MAJOR DRAM for embedding lookup
        ids_rm = ttnn.to_layout(ids_uint32, ttnn.Layout.ROW_MAJOR, None)
        ids_dram = ttnn.to_memory_config(ids_rm, MEMORY_CONFIGS["DRAM"])

        # Embedding lookup
        embeddings = ttnn.embedding(
            ids_dram,
            self.const_tensors["embed_table"],
            layout=ttnn.Layout.TILE,
        )
        ttnn.deallocate(ids_dram, False)

        return embeddings

    def forward(self, input_ids, attention_mask):
        """Run the text encoder forward pass.

        Args:
            input_ids: host TTNN tensor, ROW_MAJOR INT32 [1, 512]
            attention_mask: host TTNN tensor, ROW_MAJOR BF16 [1, 512]

        Returns:
            TTNN tensor [1, 512, 2560] BF16 DRAM — last decoder layer output
        """
        # Build attention mask
        attn_mask = self._build_attention_mask(attention_mask)

        # Get RoPE cos/sin
        rope_cos = self.const_tensors["rope_cos"]
        rope_sin = self.const_tensors["rope_sin"]

        # Embedding lookup
        hidden_states = self._process_input_ids(input_ids)
        # hidden_states: [1, 512, 2560] BF16 TILE DRAM

        # Run through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask, rope_cos, rope_sin)

        ttnn.deallocate(attn_mask, False)

        return hidden_states
