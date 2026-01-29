# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLIPEncoderLayer implementation in TTNN."""

import ttnn
from models.common.lightweightmodule import LightweightModule


class CLIPEncoderLayerTTNN(LightweightModule):
    """
    A single CLIP encoder layer implemented in TTNN.

    Matches the PyTorch CLIPEncoderLayer structure:
    - layer_norm1 -> self_attn -> residual_add
    - layer_norm2 -> mlp -> residual_add

    For layers 0-29: outputs (residual, layer_norm1_next_output)
    For layer 30 (last): outputs (residual, None) - no layer_norm1_next
    """

    def __init__(
        self,
        layer_idx: int,
        layer_weights: dict,
        layer_cer: dict,
        device,
        is_last_layer: bool = False,
    ):
        """
        Initialize the encoder layer.

        Args:
            layer_idx: Layer index (0-30)
            layer_weights: Dictionary mapping semantic names to weight tensors
            layer_cer: Dictionary mapping semantic names to const-eval result tensors
            device: TTNN device
            is_last_layer: If True, skip layer_norm1_next (layer 30)
        """
        self.layer_idx = layer_idx
        self.device = device
        self.is_first_layer = layer_idx == 0
        self.is_last_layer = is_last_layer

        # Layer 0 only: initial layer_norm1 weights
        if self.is_first_layer:
            self.layer_norm1_bias = layer_weights["layer_norm1_bias"]
            self.layer_norm1_weight = layer_weights["layer_norm1_weight"]

        # Attention weights
        self.qkv_weight = layer_cer[
            "qkv_weight"
        ]  # Fused QKV projection (from const eval)
        self.qkv_bias = layer_cer["qkv_bias"]  # Fused QKV bias (from const eval)
        self.out_proj_weight = layer_weights["out_proj_weight"]
        self.out_proj_bias = layer_cer["out_proj_bias"]

        # LayerNorm2 weights
        self.layer_norm2_bias = layer_weights["layer_norm2_bias"]
        self.layer_norm2_weight = layer_weights["layer_norm2_weight"]

        # MLP weights
        self.fc1_weight = layer_weights["fc1_weight"]
        self.fc1_bias = layer_cer["fc1_bias"]
        self.fc2_weight = layer_weights["fc2_weight"]
        self.fc2_bias = layer_cer["fc2_bias"]

        # LayerNorm1 for NEXT layer (not present on last layer)
        if not self.is_last_layer:
            self.layer_norm1_next_weight = layer_weights["layer_norm1_next_weight"]
            self.layer_norm1_next_bias = layer_weights["layer_norm1_next_bias"]

    def forward(self, hidden_states, residual):
        """
        Forward pass through the encoder layer.

        Args:
            hidden_states: Already normalized input (from previous layer's layer_norm1_next,
                          or for layer 0, the raw pre_layernorm output)
            residual: Residual tensor (for layer 0, same as hidden_states after layer_norm1)

        Returns:
            tuple: (new_residual, normalized_for_next)
                - new_residual: Output after final residual add
                - normalized_for_next: LayerNorm1 output for next layer (None for last layer)
        """
        # Layer 0 only: apply initial layer_norm1
        if self.is_first_layer:
            hidden_states = self._layer_norm1(hidden_states)

        # Self-attention
        attn_output = self._attention(hidden_states)

        # First residual add + layer_norm2
        residual = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        mlp_input = ttnn.layer_norm(
            residual,
            epsilon=9.9999997473787516e-06,
            weight=self.layer_norm2_weight,
            bias=self.layer_norm2_bias,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )

        # MLP
        mlp_output = self._mlp(mlp_input)

        # Second residual add
        new_residual = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Layer norm for next layer (skip on last layer)
        if self.is_last_layer:
            return new_residual, None
        else:
            normalized_for_next = ttnn.layer_norm(
                new_residual,
                epsilon=9.9999997473787516e-06,
                weight=self.layer_norm1_next_weight,
                bias=self.layer_norm1_next_bias,
                residual_input_tensor=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                program_config=None,
            )
            return new_residual, normalized_for_next

    def _layer_norm1(self, hidden_states):
        """Apply initial layer_norm1 (layer 0 only)."""
        return ttnn.layer_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.layer_norm1_weight,
            bias=self.layer_norm1_bias,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )

    def _attention(self, hidden_states):
        """
        Self-attention computation.

        Operations:
        1. Reshape input to [257, 1280]
        2. Fused QKV projection: matmul + bias -> [257, 3840]
        3. Slice into Q, K, V
        4. Reshape each to [1, 257, 16, 80] (multi-head)
        5. Permute to [1, 16, 257, 80]
        6. Pad to [1, 16, 257, 96] for hardware alignment
        7. Scaled dot-product attention
        8. Slice back to [1, 16, 257, 80]
        9. Permute to [1, 257, 16, 80]
        10. Reshape to [257, 1280]
        11. Output projection: matmul + bias
        """
        # Reshape to 2D for matmul
        x = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Fused QKV projection
        qkv = ttnn.matmul(
            x,
            self.qkv_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        qkv = ttnn.add(
            qkv,
            self.qkv_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Slice into V, K, Q (order from slicing pattern in original code)
        v = ttnn.slice(
            qkv,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        k = ttnn.slice(
            qkv,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        q = ttnn.slice(
            qkv,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Reshape to multi-head format [1, 257, 16, 80]
        v = ttnn.reshape(
            v,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        k = ttnn.reshape(
            k,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        q = ttnn.reshape(
            q,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Permute to [1, 16, 257, 80] (batch, heads, seq, head_dim)
        k = ttnn.permute(
            k,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        q = ttnn.permute(
            q,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v = ttnn.permute(
            v,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )

        # Pad to 96 for hardware alignment
        k = ttnn.pad(
            k,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        q = ttnn.pad(
            q,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v = ttnn.pad(
            v,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=0.11180340498685837,  # 1/sqrt(80)
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Slice back to [1, 16, 257, 80]
        attn_output = ttnn.slice(
            attn_output,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Permute back to [1, 257, 16, 80]
        attn_output = ttnn.permute(
            attn_output,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )

        # Reshape to [257, 1280]
        attn_output = ttnn.reshape(
            attn_output,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Output projection
        attn_output = ttnn.matmul(
            attn_output,
            self.out_proj_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        attn_output = ttnn.add(
            attn_output,
            self.out_proj_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        return attn_output

    def _mlp(self, hidden_states):
        """
        MLP (feed-forward) computation.

        Operations:
        1. Reshape to [257, 1280]
        2. FC1: matmul + bias -> [257, 5120]
        3. GELU activation
        4. Reshape to [257, 5120]
        5. FC2: matmul + bias -> [257, 1280]
        """
        # Reshape to 2D
        x = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # FC1
        x = ttnn.matmul(
            x,
            self.fc1_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            self.fc1_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # GELU
        x = ttnn.gelu(
            x,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Reshape (may be redundant but keeping for exact match)
        x = ttnn.reshape(
            x,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # FC2
        x = ttnn.matmul(
            x,
            self.fc2_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            self.fc2_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        return x
