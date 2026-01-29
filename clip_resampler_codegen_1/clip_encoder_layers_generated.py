# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generated CLIP encoder layer classes."""

import ttnn
from models.common.lightweightmodule import LightweightModule


class CLIPEncoderLayerTTNN_0(LightweightModule):
    """CLIP Encoder Layer 0."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w384 = weights[
            "image_encoder.vision_model.encoder.layers.0.layer_norm1.weight"
        ]
        self.w383 = weights[
            "image_encoder.vision_model.encoder.layers.0.layer_norm1.bias"
        ]
        self.w380 = weights[
            "image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.weight"
        ]
        self.w378 = weights[
            "image_encoder.vision_model.encoder.layers.0.layer_norm2.weight"
        ]
        self.w377 = weights[
            "image_encoder.vision_model.encoder.layers.0.layer_norm2.bias"
        ]
        self.w376 = weights[
            "image_encoder.vision_model.encoder.layers.0.mlp.fc1.weight"
        ]
        self.w374 = weights[
            "image_encoder.vision_model.encoder.layers.0.mlp.fc2.weight"
        ]
        self.w372 = weights[
            "image_encoder.vision_model.encoder.layers.1.layer_norm1.weight"
        ]
        self.w371 = weights[
            "image_encoder.vision_model.encoder.layers.1.layer_norm1.bias"
        ]
        self.cer_124_0 = cer["utils_constEvalFuncWrapper_124_0"]
        self.cer_42_0 = cer["utils_constEvalFuncWrapper_42_0"]
        self.cer_47_0 = cer["utils_constEvalFuncWrapper_47_0"]
        self.cer_70_0 = cer["utils_constEvalFuncWrapper_70_0"]
        self.cer_73_0 = cer["utils_constEvalFuncWrapper_73_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # layer_norm1
        hidden_states = self._layer_norm1(hidden_states)
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _layer_norm1(self, hidden_states):

        ttnn_layer_norm_2 = ttnn.layer_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.w384,
            bias=self.w383,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_2

    def _attention(self, hidden_states):

        ttnn_reshape_195 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_reshape_195,
            self.cer_70_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_1 = ttnn.add(
            ttnn_matmul_1,
            self.cer_47_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_0 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_1 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_2 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_196 = ttnn.reshape(
            ttnn_slice_0,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_197 = ttnn.reshape(
            ttnn_slice_1,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_198 = ttnn.reshape(
            ttnn_slice_2,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_6 = ttnn.permute(
            ttnn_reshape_197,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_7 = ttnn.permute(
            ttnn_reshape_198,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_8 = ttnn.permute(
            ttnn_reshape_196,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_0 = ttnn.pad(
            ttnn_permute_6,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_1 = ttnn.pad(
            ttnn_permute_7,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_2 = ttnn.pad(
            ttnn_permute_8,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_0 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_1,
                ttnn_pad_0,
                ttnn_pad_2,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_3 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_0,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_9 = ttnn.permute(
            ttnn_slice_3,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_199 = ttnn.reshape(
            ttnn_permute_9,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_reshape_199,
            self.w380,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_2 = ttnn.add(
            ttnn_matmul_2,
            self.cer_124_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_2

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_3 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_3 = ttnn.layer_norm(
            ttnn_add_3,
            epsilon=9.9999997473787516e-06,
            weight=self.w378,
            bias=self.w377,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_3, ttnn_add_3

    def _mlp(self, hidden_states):

        ttnn_reshape_200 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_reshape_200,
            self.w376,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_4 = ttnn.add(
            ttnn_matmul_3,
            self.cer_42_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_0 = ttnn.gelu(
            ttnn_add_4,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_201 = ttnn.reshape(
            ttnn_gelu_0,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_reshape_201,
            self.w374,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_5 = ttnn.add(
            ttnn_matmul_4,
            self.cer_73_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_5

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_6 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_4 = ttnn.layer_norm(
            ttnn_add_6,
            epsilon=9.9999997473787516e-06,
            weight=self.w372,
            bias=self.w371,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_6, ttnn_layer_norm_4


class CLIPEncoderLayerTTNN_1(LightweightModule):
    """CLIP Encoder Layer 1."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w368 = weights[
            "image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.weight"
        ]
        self.w366 = weights[
            "image_encoder.vision_model.encoder.layers.1.layer_norm2.weight"
        ]
        self.w365 = weights[
            "image_encoder.vision_model.encoder.layers.1.layer_norm2.bias"
        ]
        self.w364 = weights[
            "image_encoder.vision_model.encoder.layers.1.mlp.fc1.weight"
        ]
        self.w362 = weights[
            "image_encoder.vision_model.encoder.layers.1.mlp.fc2.weight"
        ]
        self.w360 = weights[
            "image_encoder.vision_model.encoder.layers.2.layer_norm1.weight"
        ]
        self.w359 = weights[
            "image_encoder.vision_model.encoder.layers.2.layer_norm1.bias"
        ]
        self.cer_13_0 = cer["utils_constEvalFuncWrapper_13_0"]
        self.cer_157_0 = cer["utils_constEvalFuncWrapper_157_0"]
        self.cer_21_0 = cer["utils_constEvalFuncWrapper_21_0"]
        self.cer_55_0 = cer["utils_constEvalFuncWrapper_55_0"]
        self.cer_62_0 = cer["utils_constEvalFuncWrapper_62_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_202 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_reshape_202,
            self.cer_157_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_7 = ttnn.add(
            ttnn_matmul_5,
            self.cer_62_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_4 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_5 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_6 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_203 = ttnn.reshape(
            ttnn_slice_4,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_204 = ttnn.reshape(
            ttnn_slice_5,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_205 = ttnn.reshape(
            ttnn_slice_6,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_10 = ttnn.permute(
            ttnn_reshape_204,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_11 = ttnn.permute(
            ttnn_reshape_205,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_12 = ttnn.permute(
            ttnn_reshape_203,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_3 = ttnn.pad(
            ttnn_permute_10,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_4 = ttnn.pad(
            ttnn_permute_11,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_5 = ttnn.pad(
            ttnn_permute_12,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_1 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_4,
                ttnn_pad_3,
                ttnn_pad_5,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_7 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_1,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_13 = ttnn.permute(
            ttnn_slice_7,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_206 = ttnn.reshape(
            ttnn_permute_13,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_6 = ttnn.matmul(
            ttnn_reshape_206,
            self.w368,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_8 = ttnn.add(
            ttnn_matmul_6,
            self.cer_55_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_8

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_9 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_5 = ttnn.layer_norm(
            ttnn_add_9,
            epsilon=9.9999997473787516e-06,
            weight=self.w366,
            bias=self.w365,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_5, ttnn_add_9

    def _mlp(self, hidden_states):

        ttnn_reshape_207 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_7 = ttnn.matmul(
            ttnn_reshape_207,
            self.w364,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_10 = ttnn.add(
            ttnn_matmul_7,
            self.cer_13_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_1 = ttnn.gelu(
            ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_208 = ttnn.reshape(
            ttnn_gelu_1,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_reshape_208,
            self.w362,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_11 = ttnn.add(
            ttnn_matmul_8,
            self.cer_21_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_11

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_12 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_6 = ttnn.layer_norm(
            ttnn_add_12,
            epsilon=9.9999997473787516e-06,
            weight=self.w360,
            bias=self.w359,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_12, ttnn_layer_norm_6


class CLIPEncoderLayerTTNN_2(LightweightModule):
    """CLIP Encoder Layer 2."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w356 = weights[
            "image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.weight"
        ]
        self.w354 = weights[
            "image_encoder.vision_model.encoder.layers.2.layer_norm2.weight"
        ]
        self.w353 = weights[
            "image_encoder.vision_model.encoder.layers.2.layer_norm2.bias"
        ]
        self.w352 = weights[
            "image_encoder.vision_model.encoder.layers.2.mlp.fc1.weight"
        ]
        self.w350 = weights[
            "image_encoder.vision_model.encoder.layers.2.mlp.fc2.weight"
        ]
        self.w348 = weights[
            "image_encoder.vision_model.encoder.layers.3.layer_norm1.weight"
        ]
        self.w347 = weights[
            "image_encoder.vision_model.encoder.layers.3.layer_norm1.bias"
        ]
        self.cer_122_0 = cer["utils_constEvalFuncWrapper_122_0"]
        self.cer_146_0 = cer["utils_constEvalFuncWrapper_146_0"]
        self.cer_25_0 = cer["utils_constEvalFuncWrapper_25_0"]
        self.cer_80_0 = cer["utils_constEvalFuncWrapper_80_0"]
        self.cer_81_0 = cer["utils_constEvalFuncWrapper_81_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_209 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_reshape_209,
            self.cer_25_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_13 = ttnn.add(
            ttnn_matmul_9,
            self.cer_80_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_8 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_9 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_10 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_210 = ttnn.reshape(
            ttnn_slice_8,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_211 = ttnn.reshape(
            ttnn_slice_9,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_212 = ttnn.reshape(
            ttnn_slice_10,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_14 = ttnn.permute(
            ttnn_reshape_211,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_15 = ttnn.permute(
            ttnn_reshape_212,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_16 = ttnn.permute(
            ttnn_reshape_210,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_6 = ttnn.pad(
            ttnn_permute_14,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_7 = ttnn.pad(
            ttnn_permute_15,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_8 = ttnn.pad(
            ttnn_permute_16,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_2 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_7,
                ttnn_pad_6,
                ttnn_pad_8,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_11 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_2,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_17 = ttnn.permute(
            ttnn_slice_11,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_213 = ttnn.reshape(
            ttnn_permute_17,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_reshape_213,
            self.w356,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_14 = ttnn.add(
            ttnn_matmul_10,
            self.cer_122_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_14

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_15 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_7 = ttnn.layer_norm(
            ttnn_add_15,
            epsilon=9.9999997473787516e-06,
            weight=self.w354,
            bias=self.w353,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_7, ttnn_add_15

    def _mlp(self, hidden_states):

        ttnn_reshape_214 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_11 = ttnn.matmul(
            ttnn_reshape_214,
            self.w352,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_16 = ttnn.add(
            ttnn_matmul_11,
            self.cer_81_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_2 = ttnn.gelu(
            ttnn_add_16,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_215 = ttnn.reshape(
            ttnn_gelu_2,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_12 = ttnn.matmul(
            ttnn_reshape_215,
            self.w350,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_17 = ttnn.add(
            ttnn_matmul_12,
            self.cer_146_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_17

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_18 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_8 = ttnn.layer_norm(
            ttnn_add_18,
            epsilon=9.9999997473787516e-06,
            weight=self.w348,
            bias=self.w347,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_18, ttnn_layer_norm_8


class CLIPEncoderLayerTTNN_3(LightweightModule):
    """CLIP Encoder Layer 3."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w344 = weights[
            "image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.weight"
        ]
        self.w342 = weights[
            "image_encoder.vision_model.encoder.layers.3.layer_norm2.weight"
        ]
        self.w341 = weights[
            "image_encoder.vision_model.encoder.layers.3.layer_norm2.bias"
        ]
        self.w340 = weights[
            "image_encoder.vision_model.encoder.layers.3.mlp.fc1.weight"
        ]
        self.w338 = weights[
            "image_encoder.vision_model.encoder.layers.3.mlp.fc2.weight"
        ]
        self.w336 = weights[
            "image_encoder.vision_model.encoder.layers.4.layer_norm1.weight"
        ]
        self.w335 = weights[
            "image_encoder.vision_model.encoder.layers.4.layer_norm1.bias"
        ]
        self.cer_10_0 = cer["utils_constEvalFuncWrapper_10_0"]
        self.cer_132_0 = cer["utils_constEvalFuncWrapper_132_0"]
        self.cer_145_0 = cer["utils_constEvalFuncWrapper_145_0"]
        self.cer_26_0 = cer["utils_constEvalFuncWrapper_26_0"]
        self.cer_90_0 = cer["utils_constEvalFuncWrapper_90_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_216 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_13 = ttnn.matmul(
            ttnn_reshape_216,
            self.cer_26_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_19 = ttnn.add(
            ttnn_matmul_13,
            self.cer_90_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_12 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_13 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_14 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_217 = ttnn.reshape(
            ttnn_slice_12,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_218 = ttnn.reshape(
            ttnn_slice_13,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_219 = ttnn.reshape(
            ttnn_slice_14,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_18 = ttnn.permute(
            ttnn_reshape_218,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_19 = ttnn.permute(
            ttnn_reshape_219,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_20 = ttnn.permute(
            ttnn_reshape_217,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_9 = ttnn.pad(
            ttnn_permute_18,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_10 = ttnn.pad(
            ttnn_permute_19,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_11 = ttnn.pad(
            ttnn_permute_20,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_3 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_10,
                ttnn_pad_9,
                ttnn_pad_11,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_15 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_3,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_21 = ttnn.permute(
            ttnn_slice_15,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_220 = ttnn.reshape(
            ttnn_permute_21,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_14 = ttnn.matmul(
            ttnn_reshape_220,
            self.w344,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_20 = ttnn.add(
            ttnn_matmul_14,
            self.cer_132_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_20

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_21 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_9 = ttnn.layer_norm(
            ttnn_add_21,
            epsilon=9.9999997473787516e-06,
            weight=self.w342,
            bias=self.w341,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_9, ttnn_add_21

    def _mlp(self, hidden_states):

        ttnn_reshape_221 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_15 = ttnn.matmul(
            ttnn_reshape_221,
            self.w340,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_22 = ttnn.add(
            ttnn_matmul_15,
            self.cer_145_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_3 = ttnn.gelu(
            ttnn_add_22,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_222 = ttnn.reshape(
            ttnn_gelu_3,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_16 = ttnn.matmul(
            ttnn_reshape_222,
            self.w338,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_23 = ttnn.add(
            ttnn_matmul_16,
            self.cer_10_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_23

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_24 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_10 = ttnn.layer_norm(
            ttnn_add_24,
            epsilon=9.9999997473787516e-06,
            weight=self.w336,
            bias=self.w335,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_24, ttnn_layer_norm_10


class CLIPEncoderLayerTTNN_4(LightweightModule):
    """CLIP Encoder Layer 4."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w332 = weights[
            "image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.weight"
        ]
        self.w330 = weights[
            "image_encoder.vision_model.encoder.layers.4.layer_norm2.weight"
        ]
        self.w329 = weights[
            "image_encoder.vision_model.encoder.layers.4.layer_norm2.bias"
        ]
        self.w328 = weights[
            "image_encoder.vision_model.encoder.layers.4.mlp.fc1.weight"
        ]
        self.w326 = weights[
            "image_encoder.vision_model.encoder.layers.4.mlp.fc2.weight"
        ]
        self.w324 = weights[
            "image_encoder.vision_model.encoder.layers.5.layer_norm1.weight"
        ]
        self.w323 = weights[
            "image_encoder.vision_model.encoder.layers.5.layer_norm1.bias"
        ]
        self.cer_127_0 = cer["utils_constEvalFuncWrapper_127_0"]
        self.cer_149_0 = cer["utils_constEvalFuncWrapper_149_0"]
        self.cer_150_0 = cer["utils_constEvalFuncWrapper_150_0"]
        self.cer_43_0 = cer["utils_constEvalFuncWrapper_43_0"]
        self.cer_97_0 = cer["utils_constEvalFuncWrapper_97_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_223 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_17 = ttnn.matmul(
            ttnn_reshape_223,
            self.cer_127_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_25 = ttnn.add(
            ttnn_matmul_17,
            self.cer_43_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_16 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_17 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_18 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_224 = ttnn.reshape(
            ttnn_slice_16,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_225 = ttnn.reshape(
            ttnn_slice_17,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_226 = ttnn.reshape(
            ttnn_slice_18,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_22 = ttnn.permute(
            ttnn_reshape_225,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_23 = ttnn.permute(
            ttnn_reshape_226,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_24 = ttnn.permute(
            ttnn_reshape_224,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_12 = ttnn.pad(
            ttnn_permute_22,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_13 = ttnn.pad(
            ttnn_permute_23,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_14 = ttnn.pad(
            ttnn_permute_24,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_4 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_13,
                ttnn_pad_12,
                ttnn_pad_14,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_19 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_4,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_25 = ttnn.permute(
            ttnn_slice_19,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_227 = ttnn.reshape(
            ttnn_permute_25,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_18 = ttnn.matmul(
            ttnn_reshape_227,
            self.w332,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_26 = ttnn.add(
            ttnn_matmul_18,
            self.cer_97_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_26

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_27 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_11 = ttnn.layer_norm(
            ttnn_add_27,
            epsilon=9.9999997473787516e-06,
            weight=self.w330,
            bias=self.w329,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_11, ttnn_add_27

    def _mlp(self, hidden_states):

        ttnn_reshape_228 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_19 = ttnn.matmul(
            ttnn_reshape_228,
            self.w328,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_28 = ttnn.add(
            ttnn_matmul_19,
            self.cer_150_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_4 = ttnn.gelu(
            ttnn_add_28,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_229 = ttnn.reshape(
            ttnn_gelu_4,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_20 = ttnn.matmul(
            ttnn_reshape_229,
            self.w326,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_29 = ttnn.add(
            ttnn_matmul_20,
            self.cer_149_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_29

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_30 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_12 = ttnn.layer_norm(
            ttnn_add_30,
            epsilon=9.9999997473787516e-06,
            weight=self.w324,
            bias=self.w323,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_30, ttnn_layer_norm_12


class CLIPEncoderLayerTTNN_5(LightweightModule):
    """CLIP Encoder Layer 5."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w320 = weights[
            "image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.weight"
        ]
        self.w318 = weights[
            "image_encoder.vision_model.encoder.layers.5.layer_norm2.weight"
        ]
        self.w317 = weights[
            "image_encoder.vision_model.encoder.layers.5.layer_norm2.bias"
        ]
        self.w316 = weights[
            "image_encoder.vision_model.encoder.layers.5.mlp.fc1.weight"
        ]
        self.w314 = weights[
            "image_encoder.vision_model.encoder.layers.5.mlp.fc2.weight"
        ]
        self.w312 = weights[
            "image_encoder.vision_model.encoder.layers.6.layer_norm1.weight"
        ]
        self.w311 = weights[
            "image_encoder.vision_model.encoder.layers.6.layer_norm1.bias"
        ]
        self.cer_106_0 = cer["utils_constEvalFuncWrapper_106_0"]
        self.cer_158_0 = cer["utils_constEvalFuncWrapper_158_0"]
        self.cer_69_0 = cer["utils_constEvalFuncWrapper_69_0"]
        self.cer_91_0 = cer["utils_constEvalFuncWrapper_91_0"]
        self.cer_96_0 = cer["utils_constEvalFuncWrapper_96_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_230 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_21 = ttnn.matmul(
            ttnn_reshape_230,
            self.cer_96_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_31 = ttnn.add(
            ttnn_matmul_21,
            self.cer_158_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_20 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_21 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_22 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_231 = ttnn.reshape(
            ttnn_slice_20,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_232 = ttnn.reshape(
            ttnn_slice_21,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_233 = ttnn.reshape(
            ttnn_slice_22,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_26 = ttnn.permute(
            ttnn_reshape_232,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_27 = ttnn.permute(
            ttnn_reshape_233,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_28 = ttnn.permute(
            ttnn_reshape_231,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_15 = ttnn.pad(
            ttnn_permute_26,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_16 = ttnn.pad(
            ttnn_permute_27,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_17 = ttnn.pad(
            ttnn_permute_28,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_5 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_16,
                ttnn_pad_15,
                ttnn_pad_17,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_23 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_5,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_29 = ttnn.permute(
            ttnn_slice_23,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_234 = ttnn.reshape(
            ttnn_permute_29,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_22 = ttnn.matmul(
            ttnn_reshape_234,
            self.w320,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_32 = ttnn.add(
            ttnn_matmul_22,
            self.cer_69_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_32

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_33 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_13 = ttnn.layer_norm(
            ttnn_add_33,
            epsilon=9.9999997473787516e-06,
            weight=self.w318,
            bias=self.w317,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_13, ttnn_add_33

    def _mlp(self, hidden_states):

        ttnn_reshape_235 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_23 = ttnn.matmul(
            ttnn_reshape_235,
            self.w316,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_34 = ttnn.add(
            ttnn_matmul_23,
            self.cer_91_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_5 = ttnn.gelu(
            ttnn_add_34,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_236 = ttnn.reshape(
            ttnn_gelu_5,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_24 = ttnn.matmul(
            ttnn_reshape_236,
            self.w314,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_35 = ttnn.add(
            ttnn_matmul_24,
            self.cer_106_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_35

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_36 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_14 = ttnn.layer_norm(
            ttnn_add_36,
            epsilon=9.9999997473787516e-06,
            weight=self.w312,
            bias=self.w311,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_36, ttnn_layer_norm_14


class CLIPEncoderLayerTTNN_6(LightweightModule):
    """CLIP Encoder Layer 6."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w308 = weights[
            "image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.weight"
        ]
        self.w306 = weights[
            "image_encoder.vision_model.encoder.layers.6.layer_norm2.weight"
        ]
        self.w305 = weights[
            "image_encoder.vision_model.encoder.layers.6.layer_norm2.bias"
        ]
        self.w304 = weights[
            "image_encoder.vision_model.encoder.layers.6.mlp.fc1.weight"
        ]
        self.w302 = weights[
            "image_encoder.vision_model.encoder.layers.6.mlp.fc2.weight"
        ]
        self.w300 = weights[
            "image_encoder.vision_model.encoder.layers.7.layer_norm1.weight"
        ]
        self.w299 = weights[
            "image_encoder.vision_model.encoder.layers.7.layer_norm1.bias"
        ]
        self.cer_103_0 = cer["utils_constEvalFuncWrapper_103_0"]
        self.cer_128_0 = cer["utils_constEvalFuncWrapper_128_0"]
        self.cer_46_0 = cer["utils_constEvalFuncWrapper_46_0"]
        self.cer_53_0 = cer["utils_constEvalFuncWrapper_53_0"]
        self.cer_99_0 = cer["utils_constEvalFuncWrapper_99_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_237 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_25 = ttnn.matmul(
            ttnn_reshape_237,
            self.cer_128_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_37 = ttnn.add(
            ttnn_matmul_25,
            self.cer_99_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_24 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_25 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_26 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_238 = ttnn.reshape(
            ttnn_slice_24,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_239 = ttnn.reshape(
            ttnn_slice_25,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_240 = ttnn.reshape(
            ttnn_slice_26,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_30 = ttnn.permute(
            ttnn_reshape_239,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_31 = ttnn.permute(
            ttnn_reshape_240,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_32 = ttnn.permute(
            ttnn_reshape_238,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_18 = ttnn.pad(
            ttnn_permute_30,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_19 = ttnn.pad(
            ttnn_permute_31,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_20 = ttnn.pad(
            ttnn_permute_32,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_6 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_19,
                ttnn_pad_18,
                ttnn_pad_20,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_27 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_6,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_33 = ttnn.permute(
            ttnn_slice_27,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_241 = ttnn.reshape(
            ttnn_permute_33,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_26 = ttnn.matmul(
            ttnn_reshape_241,
            self.w308,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_38 = ttnn.add(
            ttnn_matmul_26,
            self.cer_46_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_38

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_39 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_15 = ttnn.layer_norm(
            ttnn_add_39,
            epsilon=9.9999997473787516e-06,
            weight=self.w306,
            bias=self.w305,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_15, ttnn_add_39

    def _mlp(self, hidden_states):

        ttnn_reshape_242 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_27 = ttnn.matmul(
            ttnn_reshape_242,
            self.w304,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_40 = ttnn.add(
            ttnn_matmul_27,
            self.cer_53_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_6 = ttnn.gelu(
            ttnn_add_40,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_243 = ttnn.reshape(
            ttnn_gelu_6,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_28 = ttnn.matmul(
            ttnn_reshape_243,
            self.w302,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_41 = ttnn.add(
            ttnn_matmul_28,
            self.cer_103_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_41

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_42 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_16 = ttnn.layer_norm(
            ttnn_add_42,
            epsilon=9.9999997473787516e-06,
            weight=self.w300,
            bias=self.w299,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_42, ttnn_layer_norm_16


class CLIPEncoderLayerTTNN_7(LightweightModule):
    """CLIP Encoder Layer 7."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w296 = weights[
            "image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.weight"
        ]
        self.w294 = weights[
            "image_encoder.vision_model.encoder.layers.7.layer_norm2.weight"
        ]
        self.w293 = weights[
            "image_encoder.vision_model.encoder.layers.7.layer_norm2.bias"
        ]
        self.w292 = weights[
            "image_encoder.vision_model.encoder.layers.7.mlp.fc1.weight"
        ]
        self.w290 = weights[
            "image_encoder.vision_model.encoder.layers.7.mlp.fc2.weight"
        ]
        self.w288 = weights[
            "image_encoder.vision_model.encoder.layers.8.layer_norm1.weight"
        ]
        self.w287 = weights[
            "image_encoder.vision_model.encoder.layers.8.layer_norm1.bias"
        ]
        self.cer_120_0 = cer["utils_constEvalFuncWrapper_120_0"]
        self.cer_153_0 = cer["utils_constEvalFuncWrapper_153_0"]
        self.cer_27_0 = cer["utils_constEvalFuncWrapper_27_0"]
        self.cer_49_0 = cer["utils_constEvalFuncWrapper_49_0"]
        self.cer_84_0 = cer["utils_constEvalFuncWrapper_84_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_244 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_29 = ttnn.matmul(
            ttnn_reshape_244,
            self.cer_49_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_43 = ttnn.add(
            ttnn_matmul_29,
            self.cer_84_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_28 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_29 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_30 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_245 = ttnn.reshape(
            ttnn_slice_28,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_246 = ttnn.reshape(
            ttnn_slice_29,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_247 = ttnn.reshape(
            ttnn_slice_30,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_34 = ttnn.permute(
            ttnn_reshape_246,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_35 = ttnn.permute(
            ttnn_reshape_247,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_36 = ttnn.permute(
            ttnn_reshape_245,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_21 = ttnn.pad(
            ttnn_permute_34,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_22 = ttnn.pad(
            ttnn_permute_35,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_23 = ttnn.pad(
            ttnn_permute_36,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_7 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_22,
                ttnn_pad_21,
                ttnn_pad_23,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_31 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_7,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_37 = ttnn.permute(
            ttnn_slice_31,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_248 = ttnn.reshape(
            ttnn_permute_37,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_30 = ttnn.matmul(
            ttnn_reshape_248,
            self.w296,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_44 = ttnn.add(
            ttnn_matmul_30,
            self.cer_120_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_44

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_45 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_17 = ttnn.layer_norm(
            ttnn_add_45,
            epsilon=9.9999997473787516e-06,
            weight=self.w294,
            bias=self.w293,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_17, ttnn_add_45

    def _mlp(self, hidden_states):

        ttnn_reshape_249 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_31 = ttnn.matmul(
            ttnn_reshape_249,
            self.w292,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_46 = ttnn.add(
            ttnn_matmul_31,
            self.cer_27_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_7 = ttnn.gelu(
            ttnn_add_46,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_250 = ttnn.reshape(
            ttnn_gelu_7,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_32 = ttnn.matmul(
            ttnn_reshape_250,
            self.w290,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_47 = ttnn.add(
            ttnn_matmul_32,
            self.cer_153_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_47

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_48 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_18 = ttnn.layer_norm(
            ttnn_add_48,
            epsilon=9.9999997473787516e-06,
            weight=self.w288,
            bias=self.w287,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_48, ttnn_layer_norm_18


class CLIPEncoderLayerTTNN_8(LightweightModule):
    """CLIP Encoder Layer 8."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w284 = weights[
            "image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.weight"
        ]
        self.w282 = weights[
            "image_encoder.vision_model.encoder.layers.8.layer_norm2.weight"
        ]
        self.w281 = weights[
            "image_encoder.vision_model.encoder.layers.8.layer_norm2.bias"
        ]
        self.w280 = weights[
            "image_encoder.vision_model.encoder.layers.8.mlp.fc1.weight"
        ]
        self.w278 = weights[
            "image_encoder.vision_model.encoder.layers.8.mlp.fc2.weight"
        ]
        self.w276 = weights[
            "image_encoder.vision_model.encoder.layers.9.layer_norm1.weight"
        ]
        self.w275 = weights[
            "image_encoder.vision_model.encoder.layers.9.layer_norm1.bias"
        ]
        self.cer_24_0 = cer["utils_constEvalFuncWrapper_24_0"]
        self.cer_29_0 = cer["utils_constEvalFuncWrapper_29_0"]
        self.cer_40_0 = cer["utils_constEvalFuncWrapper_40_0"]
        self.cer_74_0 = cer["utils_constEvalFuncWrapper_74_0"]
        self.cer_93_0 = cer["utils_constEvalFuncWrapper_93_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_251 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_33 = ttnn.matmul(
            ttnn_reshape_251,
            self.cer_29_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_49 = ttnn.add(
            ttnn_matmul_33,
            self.cer_40_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_32 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_33 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_34 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_252 = ttnn.reshape(
            ttnn_slice_32,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_253 = ttnn.reshape(
            ttnn_slice_33,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_254 = ttnn.reshape(
            ttnn_slice_34,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_38 = ttnn.permute(
            ttnn_reshape_253,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_39 = ttnn.permute(
            ttnn_reshape_254,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_40 = ttnn.permute(
            ttnn_reshape_252,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_24 = ttnn.pad(
            ttnn_permute_38,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_25 = ttnn.pad(
            ttnn_permute_39,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_26 = ttnn.pad(
            ttnn_permute_40,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_8 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_25,
                ttnn_pad_24,
                ttnn_pad_26,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_35 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_8,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_41 = ttnn.permute(
            ttnn_slice_35,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_255 = ttnn.reshape(
            ttnn_permute_41,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_34 = ttnn.matmul(
            ttnn_reshape_255,
            self.w284,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_50 = ttnn.add(
            ttnn_matmul_34,
            self.cer_74_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_50

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_51 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_19 = ttnn.layer_norm(
            ttnn_add_51,
            epsilon=9.9999997473787516e-06,
            weight=self.w282,
            bias=self.w281,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_19, ttnn_add_51

    def _mlp(self, hidden_states):

        ttnn_reshape_256 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_35 = ttnn.matmul(
            ttnn_reshape_256,
            self.w280,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_52 = ttnn.add(
            ttnn_matmul_35,
            self.cer_24_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_8 = ttnn.gelu(
            ttnn_add_52,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_257 = ttnn.reshape(
            ttnn_gelu_8,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_36 = ttnn.matmul(
            ttnn_reshape_257,
            self.w278,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_53 = ttnn.add(
            ttnn_matmul_36,
            self.cer_93_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_53

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_54 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_20 = ttnn.layer_norm(
            ttnn_add_54,
            epsilon=9.9999997473787516e-06,
            weight=self.w276,
            bias=self.w275,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_54, ttnn_layer_norm_20


class CLIPEncoderLayerTTNN_9(LightweightModule):
    """CLIP Encoder Layer 9."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w272 = weights[
            "image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.weight"
        ]
        self.w270 = weights[
            "image_encoder.vision_model.encoder.layers.9.layer_norm2.weight"
        ]
        self.w269 = weights[
            "image_encoder.vision_model.encoder.layers.9.layer_norm2.bias"
        ]
        self.w268 = weights[
            "image_encoder.vision_model.encoder.layers.9.mlp.fc1.weight"
        ]
        self.w266 = weights[
            "image_encoder.vision_model.encoder.layers.9.mlp.fc2.weight"
        ]
        self.w264 = weights[
            "image_encoder.vision_model.encoder.layers.10.layer_norm1.weight"
        ]
        self.w263 = weights[
            "image_encoder.vision_model.encoder.layers.10.layer_norm1.bias"
        ]
        self.cer_113_0 = cer["utils_constEvalFuncWrapper_113_0"]
        self.cer_119_0 = cer["utils_constEvalFuncWrapper_119_0"]
        self.cer_133_0 = cer["utils_constEvalFuncWrapper_133_0"]
        self.cer_155_0 = cer["utils_constEvalFuncWrapper_155_0"]
        self.cer_2_0 = cer["utils_constEvalFuncWrapper_2_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_258 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_37 = ttnn.matmul(
            ttnn_reshape_258,
            self.cer_119_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_55 = ttnn.add(
            ttnn_matmul_37,
            self.cer_133_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_36 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_37 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_38 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_259 = ttnn.reshape(
            ttnn_slice_36,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_260 = ttnn.reshape(
            ttnn_slice_37,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_261 = ttnn.reshape(
            ttnn_slice_38,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_42 = ttnn.permute(
            ttnn_reshape_260,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_43 = ttnn.permute(
            ttnn_reshape_261,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_44 = ttnn.permute(
            ttnn_reshape_259,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_27 = ttnn.pad(
            ttnn_permute_42,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_28 = ttnn.pad(
            ttnn_permute_43,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_29 = ttnn.pad(
            ttnn_permute_44,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_9 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_28,
                ttnn_pad_27,
                ttnn_pad_29,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_39 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_9,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_45 = ttnn.permute(
            ttnn_slice_39,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_262 = ttnn.reshape(
            ttnn_permute_45,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_38 = ttnn.matmul(
            ttnn_reshape_262,
            self.w272,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_56 = ttnn.add(
            ttnn_matmul_38,
            self.cer_113_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_56

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_57 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_21 = ttnn.layer_norm(
            ttnn_add_57,
            epsilon=9.9999997473787516e-06,
            weight=self.w270,
            bias=self.w269,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_21, ttnn_add_57

    def _mlp(self, hidden_states):

        ttnn_reshape_263 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_39 = ttnn.matmul(
            ttnn_reshape_263,
            self.w268,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_58 = ttnn.add(
            ttnn_matmul_39,
            self.cer_155_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_9 = ttnn.gelu(
            ttnn_add_58,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_264 = ttnn.reshape(
            ttnn_gelu_9,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_40 = ttnn.matmul(
            ttnn_reshape_264,
            self.w266,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_59 = ttnn.add(
            ttnn_matmul_40,
            self.cer_2_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_59

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_60 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_22 = ttnn.layer_norm(
            ttnn_add_60,
            epsilon=9.9999997473787516e-06,
            weight=self.w264,
            bias=self.w263,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_60, ttnn_layer_norm_22


class CLIPEncoderLayerTTNN_10(LightweightModule):
    """CLIP Encoder Layer 10."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w260 = weights[
            "image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.weight"
        ]
        self.w258 = weights[
            "image_encoder.vision_model.encoder.layers.10.layer_norm2.weight"
        ]
        self.w257 = weights[
            "image_encoder.vision_model.encoder.layers.10.layer_norm2.bias"
        ]
        self.w256 = weights[
            "image_encoder.vision_model.encoder.layers.10.mlp.fc1.weight"
        ]
        self.w254 = weights[
            "image_encoder.vision_model.encoder.layers.10.mlp.fc2.weight"
        ]
        self.w252 = weights[
            "image_encoder.vision_model.encoder.layers.11.layer_norm1.weight"
        ]
        self.w251 = weights[
            "image_encoder.vision_model.encoder.layers.11.layer_norm1.bias"
        ]
        self.cer_152_0 = cer["utils_constEvalFuncWrapper_152_0"]
        self.cer_64_0 = cer["utils_constEvalFuncWrapper_64_0"]
        self.cer_71_0 = cer["utils_constEvalFuncWrapper_71_0"]
        self.cer_85_0 = cer["utils_constEvalFuncWrapper_85_0"]
        self.cer_95_0 = cer["utils_constEvalFuncWrapper_95_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_265 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_41 = ttnn.matmul(
            ttnn_reshape_265,
            self.cer_152_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_61 = ttnn.add(
            ttnn_matmul_41,
            self.cer_71_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_40 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_41 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_42 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_266 = ttnn.reshape(
            ttnn_slice_40,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_267 = ttnn.reshape(
            ttnn_slice_41,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_268 = ttnn.reshape(
            ttnn_slice_42,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_46 = ttnn.permute(
            ttnn_reshape_267,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_47 = ttnn.permute(
            ttnn_reshape_268,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_48 = ttnn.permute(
            ttnn_reshape_266,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_30 = ttnn.pad(
            ttnn_permute_46,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_31 = ttnn.pad(
            ttnn_permute_47,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_32 = ttnn.pad(
            ttnn_permute_48,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_10 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_31,
                ttnn_pad_30,
                ttnn_pad_32,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_43 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_10,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_49 = ttnn.permute(
            ttnn_slice_43,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_269 = ttnn.reshape(
            ttnn_permute_49,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_42 = ttnn.matmul(
            ttnn_reshape_269,
            self.w260,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_62 = ttnn.add(
            ttnn_matmul_42,
            self.cer_64_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_62

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_63 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_23 = ttnn.layer_norm(
            ttnn_add_63,
            epsilon=9.9999997473787516e-06,
            weight=self.w258,
            bias=self.w257,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_23, ttnn_add_63

    def _mlp(self, hidden_states):

        ttnn_reshape_270 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_43 = ttnn.matmul(
            ttnn_reshape_270,
            self.w256,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_64 = ttnn.add(
            ttnn_matmul_43,
            self.cer_95_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_10 = ttnn.gelu(
            ttnn_add_64,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_271 = ttnn.reshape(
            ttnn_gelu_10,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_44 = ttnn.matmul(
            ttnn_reshape_271,
            self.w254,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_65 = ttnn.add(
            ttnn_matmul_44,
            self.cer_85_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_65

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_66 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_24 = ttnn.layer_norm(
            ttnn_add_66,
            epsilon=9.9999997473787516e-06,
            weight=self.w252,
            bias=self.w251,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_66, ttnn_layer_norm_24


class CLIPEncoderLayerTTNN_11(LightweightModule):
    """CLIP Encoder Layer 11."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w248 = weights[
            "image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.weight"
        ]
        self.w246 = weights[
            "image_encoder.vision_model.encoder.layers.11.layer_norm2.weight"
        ]
        self.w245 = weights[
            "image_encoder.vision_model.encoder.layers.11.layer_norm2.bias"
        ]
        self.w244 = weights[
            "image_encoder.vision_model.encoder.layers.11.mlp.fc1.weight"
        ]
        self.w242 = weights[
            "image_encoder.vision_model.encoder.layers.11.mlp.fc2.weight"
        ]
        self.w240 = weights[
            "image_encoder.vision_model.encoder.layers.12.layer_norm1.weight"
        ]
        self.w239 = weights[
            "image_encoder.vision_model.encoder.layers.12.layer_norm1.bias"
        ]
        self.cer_116_0 = cer["utils_constEvalFuncWrapper_116_0"]
        self.cer_140_0 = cer["utils_constEvalFuncWrapper_140_0"]
        self.cer_151_0 = cer["utils_constEvalFuncWrapper_151_0"]
        self.cer_156_0 = cer["utils_constEvalFuncWrapper_156_0"]
        self.cer_67_0 = cer["utils_constEvalFuncWrapper_67_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_272 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_45 = ttnn.matmul(
            ttnn_reshape_272,
            self.cer_67_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_67 = ttnn.add(
            ttnn_matmul_45,
            self.cer_116_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_44 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_45 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_46 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_273 = ttnn.reshape(
            ttnn_slice_44,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_274 = ttnn.reshape(
            ttnn_slice_45,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_275 = ttnn.reshape(
            ttnn_slice_46,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_50 = ttnn.permute(
            ttnn_reshape_274,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_51 = ttnn.permute(
            ttnn_reshape_275,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_52 = ttnn.permute(
            ttnn_reshape_273,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_33 = ttnn.pad(
            ttnn_permute_50,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_34 = ttnn.pad(
            ttnn_permute_51,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_35 = ttnn.pad(
            ttnn_permute_52,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_11 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_34,
                ttnn_pad_33,
                ttnn_pad_35,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_47 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_11,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_53 = ttnn.permute(
            ttnn_slice_47,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_276 = ttnn.reshape(
            ttnn_permute_53,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_46 = ttnn.matmul(
            ttnn_reshape_276,
            self.w248,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_68 = ttnn.add(
            ttnn_matmul_46,
            self.cer_140_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_68

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_69 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_25 = ttnn.layer_norm(
            ttnn_add_69,
            epsilon=9.9999997473787516e-06,
            weight=self.w246,
            bias=self.w245,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_25, ttnn_add_69

    def _mlp(self, hidden_states):

        ttnn_reshape_277 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_47 = ttnn.matmul(
            ttnn_reshape_277,
            self.w244,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_70 = ttnn.add(
            ttnn_matmul_47,
            self.cer_156_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_11 = ttnn.gelu(
            ttnn_add_70,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_278 = ttnn.reshape(
            ttnn_gelu_11,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_48 = ttnn.matmul(
            ttnn_reshape_278,
            self.w242,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_71 = ttnn.add(
            ttnn_matmul_48,
            self.cer_151_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_71

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_72 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_26 = ttnn.layer_norm(
            ttnn_add_72,
            epsilon=9.9999997473787516e-06,
            weight=self.w240,
            bias=self.w239,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_72, ttnn_layer_norm_26


class CLIPEncoderLayerTTNN_12(LightweightModule):
    """CLIP Encoder Layer 12."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w236 = weights[
            "image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.weight"
        ]
        self.w234 = weights[
            "image_encoder.vision_model.encoder.layers.12.layer_norm2.weight"
        ]
        self.w233 = weights[
            "image_encoder.vision_model.encoder.layers.12.layer_norm2.bias"
        ]
        self.w232 = weights[
            "image_encoder.vision_model.encoder.layers.12.mlp.fc1.weight"
        ]
        self.w230 = weights[
            "image_encoder.vision_model.encoder.layers.12.mlp.fc2.weight"
        ]
        self.w228 = weights[
            "image_encoder.vision_model.encoder.layers.13.layer_norm1.weight"
        ]
        self.w227 = weights[
            "image_encoder.vision_model.encoder.layers.13.layer_norm1.bias"
        ]
        self.cer_136_0 = cer["utils_constEvalFuncWrapper_136_0"]
        self.cer_15_0 = cer["utils_constEvalFuncWrapper_15_0"]
        self.cer_5_0 = cer["utils_constEvalFuncWrapper_5_0"]
        self.cer_68_0 = cer["utils_constEvalFuncWrapper_68_0"]
        self.cer_87_0 = cer["utils_constEvalFuncWrapper_87_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_279 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_49 = ttnn.matmul(
            ttnn_reshape_279,
            self.cer_87_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_73 = ttnn.add(
            ttnn_matmul_49,
            self.cer_136_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_48 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_49 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_50 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_280 = ttnn.reshape(
            ttnn_slice_48,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_281 = ttnn.reshape(
            ttnn_slice_49,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_282 = ttnn.reshape(
            ttnn_slice_50,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_54 = ttnn.permute(
            ttnn_reshape_281,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_55 = ttnn.permute(
            ttnn_reshape_282,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_56 = ttnn.permute(
            ttnn_reshape_280,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_36 = ttnn.pad(
            ttnn_permute_54,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_37 = ttnn.pad(
            ttnn_permute_55,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_38 = ttnn.pad(
            ttnn_permute_56,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_12 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_37,
                ttnn_pad_36,
                ttnn_pad_38,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_51 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_12,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_57 = ttnn.permute(
            ttnn_slice_51,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_283 = ttnn.reshape(
            ttnn_permute_57,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_50 = ttnn.matmul(
            ttnn_reshape_283,
            self.w236,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_74 = ttnn.add(
            ttnn_matmul_50,
            self.cer_68_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_74

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_75 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_27 = ttnn.layer_norm(
            ttnn_add_75,
            epsilon=9.9999997473787516e-06,
            weight=self.w234,
            bias=self.w233,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_27, ttnn_add_75

    def _mlp(self, hidden_states):

        ttnn_reshape_284 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_51 = ttnn.matmul(
            ttnn_reshape_284,
            self.w232,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_76 = ttnn.add(
            ttnn_matmul_51,
            self.cer_5_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_12 = ttnn.gelu(
            ttnn_add_76,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_285 = ttnn.reshape(
            ttnn_gelu_12,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_52 = ttnn.matmul(
            ttnn_reshape_285,
            self.w230,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_77 = ttnn.add(
            ttnn_matmul_52,
            self.cer_15_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_77

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_78 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_28 = ttnn.layer_norm(
            ttnn_add_78,
            epsilon=9.9999997473787516e-06,
            weight=self.w228,
            bias=self.w227,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_78, ttnn_layer_norm_28


class CLIPEncoderLayerTTNN_13(LightweightModule):
    """CLIP Encoder Layer 13."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w224 = weights[
            "image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.weight"
        ]
        self.w222 = weights[
            "image_encoder.vision_model.encoder.layers.13.layer_norm2.weight"
        ]
        self.w221 = weights[
            "image_encoder.vision_model.encoder.layers.13.layer_norm2.bias"
        ]
        self.w220 = weights[
            "image_encoder.vision_model.encoder.layers.13.mlp.fc1.weight"
        ]
        self.w218 = weights[
            "image_encoder.vision_model.encoder.layers.13.mlp.fc2.weight"
        ]
        self.w216 = weights[
            "image_encoder.vision_model.encoder.layers.14.layer_norm1.weight"
        ]
        self.w215 = weights[
            "image_encoder.vision_model.encoder.layers.14.layer_norm1.bias"
        ]
        self.cer_102_0 = cer["utils_constEvalFuncWrapper_102_0"]
        self.cer_109_0 = cer["utils_constEvalFuncWrapper_109_0"]
        self.cer_126_0 = cer["utils_constEvalFuncWrapper_126_0"]
        self.cer_1_0 = cer["utils_constEvalFuncWrapper_1_0"]
        self.cer_92_0 = cer["utils_constEvalFuncWrapper_92_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_286 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_53 = ttnn.matmul(
            ttnn_reshape_286,
            self.cer_102_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_79 = ttnn.add(
            ttnn_matmul_53,
            self.cer_1_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_52 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_53 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_54 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_287 = ttnn.reshape(
            ttnn_slice_52,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_288 = ttnn.reshape(
            ttnn_slice_53,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_289 = ttnn.reshape(
            ttnn_slice_54,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_58 = ttnn.permute(
            ttnn_reshape_288,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_59 = ttnn.permute(
            ttnn_reshape_289,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_60 = ttnn.permute(
            ttnn_reshape_287,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_39 = ttnn.pad(
            ttnn_permute_58,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_40 = ttnn.pad(
            ttnn_permute_59,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_41 = ttnn.pad(
            ttnn_permute_60,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_13 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_40,
                ttnn_pad_39,
                ttnn_pad_41,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_55 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_13,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_61 = ttnn.permute(
            ttnn_slice_55,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_290 = ttnn.reshape(
            ttnn_permute_61,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_54 = ttnn.matmul(
            ttnn_reshape_290,
            self.w224,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_80 = ttnn.add(
            ttnn_matmul_54,
            self.cer_126_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_80

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_81 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_29 = ttnn.layer_norm(
            ttnn_add_81,
            epsilon=9.9999997473787516e-06,
            weight=self.w222,
            bias=self.w221,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_29, ttnn_add_81

    def _mlp(self, hidden_states):

        ttnn_reshape_291 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_55 = ttnn.matmul(
            ttnn_reshape_291,
            self.w220,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_82 = ttnn.add(
            ttnn_matmul_55,
            self.cer_92_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_13 = ttnn.gelu(
            ttnn_add_82,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_292 = ttnn.reshape(
            ttnn_gelu_13,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_56 = ttnn.matmul(
            ttnn_reshape_292,
            self.w218,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_83 = ttnn.add(
            ttnn_matmul_56,
            self.cer_109_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_83

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_84 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_30 = ttnn.layer_norm(
            ttnn_add_84,
            epsilon=9.9999997473787516e-06,
            weight=self.w216,
            bias=self.w215,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_84, ttnn_layer_norm_30


class CLIPEncoderLayerTTNN_14(LightweightModule):
    """CLIP Encoder Layer 14."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w212 = weights[
            "image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.weight"
        ]
        self.w210 = weights[
            "image_encoder.vision_model.encoder.layers.14.layer_norm2.weight"
        ]
        self.w209 = weights[
            "image_encoder.vision_model.encoder.layers.14.layer_norm2.bias"
        ]
        self.w208 = weights[
            "image_encoder.vision_model.encoder.layers.14.mlp.fc1.weight"
        ]
        self.w206 = weights[
            "image_encoder.vision_model.encoder.layers.14.mlp.fc2.weight"
        ]
        self.w204 = weights[
            "image_encoder.vision_model.encoder.layers.15.layer_norm1.weight"
        ]
        self.w203 = weights[
            "image_encoder.vision_model.encoder.layers.15.layer_norm1.bias"
        ]
        self.cer_101_0 = cer["utils_constEvalFuncWrapper_101_0"]
        self.cer_11_0 = cer["utils_constEvalFuncWrapper_11_0"]
        self.cer_141_0 = cer["utils_constEvalFuncWrapper_141_0"]
        self.cer_18_0 = cer["utils_constEvalFuncWrapper_18_0"]
        self.cer_86_0 = cer["utils_constEvalFuncWrapper_86_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_293 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_57 = ttnn.matmul(
            ttnn_reshape_293,
            self.cer_86_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_85 = ttnn.add(
            ttnn_matmul_57,
            self.cer_101_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_56 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_57 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_58 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_294 = ttnn.reshape(
            ttnn_slice_56,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_295 = ttnn.reshape(
            ttnn_slice_57,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_296 = ttnn.reshape(
            ttnn_slice_58,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_62 = ttnn.permute(
            ttnn_reshape_295,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_63 = ttnn.permute(
            ttnn_reshape_296,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_64 = ttnn.permute(
            ttnn_reshape_294,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_42 = ttnn.pad(
            ttnn_permute_62,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_43 = ttnn.pad(
            ttnn_permute_63,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_44 = ttnn.pad(
            ttnn_permute_64,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_14 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_43,
                ttnn_pad_42,
                ttnn_pad_44,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_59 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_14,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_65 = ttnn.permute(
            ttnn_slice_59,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_297 = ttnn.reshape(
            ttnn_permute_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_58 = ttnn.matmul(
            ttnn_reshape_297,
            self.w212,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_86 = ttnn.add(
            ttnn_matmul_58,
            self.cer_11_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_86

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_87 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_31 = ttnn.layer_norm(
            ttnn_add_87,
            epsilon=9.9999997473787516e-06,
            weight=self.w210,
            bias=self.w209,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_31, ttnn_add_87

    def _mlp(self, hidden_states):

        ttnn_reshape_298 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_59 = ttnn.matmul(
            ttnn_reshape_298,
            self.w208,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_88 = ttnn.add(
            ttnn_matmul_59,
            self.cer_141_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_14 = ttnn.gelu(
            ttnn_add_88,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_299 = ttnn.reshape(
            ttnn_gelu_14,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_60 = ttnn.matmul(
            ttnn_reshape_299,
            self.w206,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_89 = ttnn.add(
            ttnn_matmul_60,
            self.cer_18_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_89

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_90 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_32 = ttnn.layer_norm(
            ttnn_add_90,
            epsilon=9.9999997473787516e-06,
            weight=self.w204,
            bias=self.w203,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_90, ttnn_layer_norm_32


class CLIPEncoderLayerTTNN_15(LightweightModule):
    """CLIP Encoder Layer 15."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w200 = weights[
            "image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.weight"
        ]
        self.w198 = weights[
            "image_encoder.vision_model.encoder.layers.15.layer_norm2.weight"
        ]
        self.w197 = weights[
            "image_encoder.vision_model.encoder.layers.15.layer_norm2.bias"
        ]
        self.w196 = weights[
            "image_encoder.vision_model.encoder.layers.15.mlp.fc1.weight"
        ]
        self.w194 = weights[
            "image_encoder.vision_model.encoder.layers.15.mlp.fc2.weight"
        ]
        self.w192 = weights[
            "image_encoder.vision_model.encoder.layers.16.layer_norm1.weight"
        ]
        self.w191 = weights[
            "image_encoder.vision_model.encoder.layers.16.layer_norm1.bias"
        ]
        self.cer_114_0 = cer["utils_constEvalFuncWrapper_114_0"]
        self.cer_154_0 = cer["utils_constEvalFuncWrapper_154_0"]
        self.cer_23_0 = cer["utils_constEvalFuncWrapper_23_0"]
        self.cer_72_0 = cer["utils_constEvalFuncWrapper_72_0"]
        self.cer_83_0 = cer["utils_constEvalFuncWrapper_83_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_300 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_61 = ttnn.matmul(
            ttnn_reshape_300,
            self.cer_23_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_91 = ttnn.add(
            ttnn_matmul_61,
            self.cer_72_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_60 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_61 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_62 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_301 = ttnn.reshape(
            ttnn_slice_60,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_302 = ttnn.reshape(
            ttnn_slice_61,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_303 = ttnn.reshape(
            ttnn_slice_62,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_66 = ttnn.permute(
            ttnn_reshape_302,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_67 = ttnn.permute(
            ttnn_reshape_303,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_68 = ttnn.permute(
            ttnn_reshape_301,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_45 = ttnn.pad(
            ttnn_permute_66,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_46 = ttnn.pad(
            ttnn_permute_67,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_47 = ttnn.pad(
            ttnn_permute_68,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_15 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_46,
                ttnn_pad_45,
                ttnn_pad_47,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_63 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_15,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_69 = ttnn.permute(
            ttnn_slice_63,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_304 = ttnn.reshape(
            ttnn_permute_69,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_62 = ttnn.matmul(
            ttnn_reshape_304,
            self.w200,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_92 = ttnn.add(
            ttnn_matmul_62,
            self.cer_114_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_92

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_93 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_33 = ttnn.layer_norm(
            ttnn_add_93,
            epsilon=9.9999997473787516e-06,
            weight=self.w198,
            bias=self.w197,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_33, ttnn_add_93

    def _mlp(self, hidden_states):

        ttnn_reshape_305 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_63 = ttnn.matmul(
            ttnn_reshape_305,
            self.w196,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_94 = ttnn.add(
            ttnn_matmul_63,
            self.cer_154_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_15 = ttnn.gelu(
            ttnn_add_94,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_306 = ttnn.reshape(
            ttnn_gelu_15,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_64 = ttnn.matmul(
            ttnn_reshape_306,
            self.w194,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_95 = ttnn.add(
            ttnn_matmul_64,
            self.cer_83_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_95

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_96 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_34 = ttnn.layer_norm(
            ttnn_add_96,
            epsilon=9.9999997473787516e-06,
            weight=self.w192,
            bias=self.w191,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_96, ttnn_layer_norm_34


class CLIPEncoderLayerTTNN_16(LightweightModule):
    """CLIP Encoder Layer 16."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w188 = weights[
            "image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.weight"
        ]
        self.w186 = weights[
            "image_encoder.vision_model.encoder.layers.16.layer_norm2.weight"
        ]
        self.w185 = weights[
            "image_encoder.vision_model.encoder.layers.16.layer_norm2.bias"
        ]
        self.w184 = weights[
            "image_encoder.vision_model.encoder.layers.16.mlp.fc1.weight"
        ]
        self.w182 = weights[
            "image_encoder.vision_model.encoder.layers.16.mlp.fc2.weight"
        ]
        self.w180 = weights[
            "image_encoder.vision_model.encoder.layers.17.layer_norm1.weight"
        ]
        self.w179 = weights[
            "image_encoder.vision_model.encoder.layers.17.layer_norm1.bias"
        ]
        self.cer_104_0 = cer["utils_constEvalFuncWrapper_104_0"]
        self.cer_118_0 = cer["utils_constEvalFuncWrapper_118_0"]
        self.cer_130_0 = cer["utils_constEvalFuncWrapper_130_0"]
        self.cer_63_0 = cer["utils_constEvalFuncWrapper_63_0"]
        self.cer_89_0 = cer["utils_constEvalFuncWrapper_89_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_307 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_65 = ttnn.matmul(
            ttnn_reshape_307,
            self.cer_89_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_97 = ttnn.add(
            ttnn_matmul_65,
            self.cer_118_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_64 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_65 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_66 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_308 = ttnn.reshape(
            ttnn_slice_64,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_309 = ttnn.reshape(
            ttnn_slice_65,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_310 = ttnn.reshape(
            ttnn_slice_66,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_70 = ttnn.permute(
            ttnn_reshape_309,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_71 = ttnn.permute(
            ttnn_reshape_310,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_72 = ttnn.permute(
            ttnn_reshape_308,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_48 = ttnn.pad(
            ttnn_permute_70,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_49 = ttnn.pad(
            ttnn_permute_71,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_50 = ttnn.pad(
            ttnn_permute_72,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_16 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_49,
                ttnn_pad_48,
                ttnn_pad_50,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_67 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_16,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_73 = ttnn.permute(
            ttnn_slice_67,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_311 = ttnn.reshape(
            ttnn_permute_73,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_66 = ttnn.matmul(
            ttnn_reshape_311,
            self.w188,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_98 = ttnn.add(
            ttnn_matmul_66,
            self.cer_63_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_98

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_99 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_35 = ttnn.layer_norm(
            ttnn_add_99,
            epsilon=9.9999997473787516e-06,
            weight=self.w186,
            bias=self.w185,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_35, ttnn_add_99

    def _mlp(self, hidden_states):

        ttnn_reshape_312 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_67 = ttnn.matmul(
            ttnn_reshape_312,
            self.w184,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_100 = ttnn.add(
            ttnn_matmul_67,
            self.cer_130_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_16 = ttnn.gelu(
            ttnn_add_100,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_313 = ttnn.reshape(
            ttnn_gelu_16,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_68 = ttnn.matmul(
            ttnn_reshape_313,
            self.w182,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_101 = ttnn.add(
            ttnn_matmul_68,
            self.cer_104_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_101

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_102 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_36 = ttnn.layer_norm(
            ttnn_add_102,
            epsilon=9.9999997473787516e-06,
            weight=self.w180,
            bias=self.w179,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_102, ttnn_layer_norm_36


class CLIPEncoderLayerTTNN_17(LightweightModule):
    """CLIP Encoder Layer 17."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w176 = weights[
            "image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.weight"
        ]
        self.w174 = weights[
            "image_encoder.vision_model.encoder.layers.17.layer_norm2.weight"
        ]
        self.w173 = weights[
            "image_encoder.vision_model.encoder.layers.17.layer_norm2.bias"
        ]
        self.w172 = weights[
            "image_encoder.vision_model.encoder.layers.17.mlp.fc1.weight"
        ]
        self.w170 = weights[
            "image_encoder.vision_model.encoder.layers.17.mlp.fc2.weight"
        ]
        self.w168 = weights[
            "image_encoder.vision_model.encoder.layers.18.layer_norm1.weight"
        ]
        self.w167 = weights[
            "image_encoder.vision_model.encoder.layers.18.layer_norm1.bias"
        ]
        self.cer_108_0 = cer["utils_constEvalFuncWrapper_108_0"]
        self.cer_17_0 = cer["utils_constEvalFuncWrapper_17_0"]
        self.cer_19_0 = cer["utils_constEvalFuncWrapper_19_0"]
        self.cer_34_0 = cer["utils_constEvalFuncWrapper_34_0"]
        self.cer_7_0 = cer["utils_constEvalFuncWrapper_7_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_314 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_69 = ttnn.matmul(
            ttnn_reshape_314,
            self.cer_34_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_103 = ttnn.add(
            ttnn_matmul_69,
            self.cer_17_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_68 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_69 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_70 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_315 = ttnn.reshape(
            ttnn_slice_68,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_316 = ttnn.reshape(
            ttnn_slice_69,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_317 = ttnn.reshape(
            ttnn_slice_70,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_74 = ttnn.permute(
            ttnn_reshape_316,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_75 = ttnn.permute(
            ttnn_reshape_317,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_76 = ttnn.permute(
            ttnn_reshape_315,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_51 = ttnn.pad(
            ttnn_permute_74,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_52 = ttnn.pad(
            ttnn_permute_75,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_53 = ttnn.pad(
            ttnn_permute_76,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_17 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_52,
                ttnn_pad_51,
                ttnn_pad_53,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_71 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_17,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_77 = ttnn.permute(
            ttnn_slice_71,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_318 = ttnn.reshape(
            ttnn_permute_77,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_70 = ttnn.matmul(
            ttnn_reshape_318,
            self.w176,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_104 = ttnn.add(
            ttnn_matmul_70,
            self.cer_7_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_104

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_105 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_37 = ttnn.layer_norm(
            ttnn_add_105,
            epsilon=9.9999997473787516e-06,
            weight=self.w174,
            bias=self.w173,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_37, ttnn_add_105

    def _mlp(self, hidden_states):

        ttnn_reshape_319 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_71 = ttnn.matmul(
            ttnn_reshape_319,
            self.w172,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_106 = ttnn.add(
            ttnn_matmul_71,
            self.cer_108_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_17 = ttnn.gelu(
            ttnn_add_106,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_320 = ttnn.reshape(
            ttnn_gelu_17,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_72 = ttnn.matmul(
            ttnn_reshape_320,
            self.w170,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_107 = ttnn.add(
            ttnn_matmul_72,
            self.cer_19_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_107

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_108 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_38 = ttnn.layer_norm(
            ttnn_add_108,
            epsilon=9.9999997473787516e-06,
            weight=self.w168,
            bias=self.w167,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_108, ttnn_layer_norm_38


class CLIPEncoderLayerTTNN_18(LightweightModule):
    """CLIP Encoder Layer 18."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w164 = weights[
            "image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.weight"
        ]
        self.w162 = weights[
            "image_encoder.vision_model.encoder.layers.18.layer_norm2.weight"
        ]
        self.w161 = weights[
            "image_encoder.vision_model.encoder.layers.18.layer_norm2.bias"
        ]
        self.w160 = weights[
            "image_encoder.vision_model.encoder.layers.18.mlp.fc1.weight"
        ]
        self.w158 = weights[
            "image_encoder.vision_model.encoder.layers.18.mlp.fc2.weight"
        ]
        self.w156 = weights[
            "image_encoder.vision_model.encoder.layers.19.layer_norm1.weight"
        ]
        self.w155 = weights[
            "image_encoder.vision_model.encoder.layers.19.layer_norm1.bias"
        ]
        self.cer_100_0 = cer["utils_constEvalFuncWrapper_100_0"]
        self.cer_112_0 = cer["utils_constEvalFuncWrapper_112_0"]
        self.cer_134_0 = cer["utils_constEvalFuncWrapper_134_0"]
        self.cer_147_0 = cer["utils_constEvalFuncWrapper_147_0"]
        self.cer_94_0 = cer["utils_constEvalFuncWrapper_94_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_321 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_73 = ttnn.matmul(
            ttnn_reshape_321,
            self.cer_112_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_109 = ttnn.add(
            ttnn_matmul_73,
            self.cer_134_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_72 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_73 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_74 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_322 = ttnn.reshape(
            ttnn_slice_72,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_323 = ttnn.reshape(
            ttnn_slice_73,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_324 = ttnn.reshape(
            ttnn_slice_74,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_78 = ttnn.permute(
            ttnn_reshape_323,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_79 = ttnn.permute(
            ttnn_reshape_324,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_80 = ttnn.permute(
            ttnn_reshape_322,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_54 = ttnn.pad(
            ttnn_permute_78,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_55 = ttnn.pad(
            ttnn_permute_79,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_56 = ttnn.pad(
            ttnn_permute_80,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_18 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_55,
                ttnn_pad_54,
                ttnn_pad_56,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_75 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_18,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_81 = ttnn.permute(
            ttnn_slice_75,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_325 = ttnn.reshape(
            ttnn_permute_81,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_74 = ttnn.matmul(
            ttnn_reshape_325,
            self.w164,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_110 = ttnn.add(
            ttnn_matmul_74,
            self.cer_100_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_110

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_111 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_39 = ttnn.layer_norm(
            ttnn_add_111,
            epsilon=9.9999997473787516e-06,
            weight=self.w162,
            bias=self.w161,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_39, ttnn_add_111

    def _mlp(self, hidden_states):

        ttnn_reshape_326 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_75 = ttnn.matmul(
            ttnn_reshape_326,
            self.w160,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_112 = ttnn.add(
            ttnn_matmul_75,
            self.cer_94_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_18 = ttnn.gelu(
            ttnn_add_112,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_327 = ttnn.reshape(
            ttnn_gelu_18,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_76 = ttnn.matmul(
            ttnn_reshape_327,
            self.w158,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_113 = ttnn.add(
            ttnn_matmul_76,
            self.cer_147_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_113

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_114 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_40 = ttnn.layer_norm(
            ttnn_add_114,
            epsilon=9.9999997473787516e-06,
            weight=self.w156,
            bias=self.w155,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_114, ttnn_layer_norm_40


class CLIPEncoderLayerTTNN_19(LightweightModule):
    """CLIP Encoder Layer 19."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w152 = weights[
            "image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.weight"
        ]
        self.w150 = weights[
            "image_encoder.vision_model.encoder.layers.19.layer_norm2.weight"
        ]
        self.w149 = weights[
            "image_encoder.vision_model.encoder.layers.19.layer_norm2.bias"
        ]
        self.w148 = weights[
            "image_encoder.vision_model.encoder.layers.19.mlp.fc1.weight"
        ]
        self.w146 = weights[
            "image_encoder.vision_model.encoder.layers.19.mlp.fc2.weight"
        ]
        self.w144 = weights[
            "image_encoder.vision_model.encoder.layers.20.layer_norm1.weight"
        ]
        self.w143 = weights[
            "image_encoder.vision_model.encoder.layers.20.layer_norm1.bias"
        ]
        self.cer_12_0 = cer["utils_constEvalFuncWrapper_12_0"]
        self.cer_28_0 = cer["utils_constEvalFuncWrapper_28_0"]
        self.cer_44_0 = cer["utils_constEvalFuncWrapper_44_0"]
        self.cer_50_0 = cer["utils_constEvalFuncWrapper_50_0"]
        self.cer_52_0 = cer["utils_constEvalFuncWrapper_52_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_328 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_77 = ttnn.matmul(
            ttnn_reshape_328,
            self.cer_12_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_115 = ttnn.add(
            ttnn_matmul_77,
            self.cer_50_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_76 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_77 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_78 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_329 = ttnn.reshape(
            ttnn_slice_76,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_330 = ttnn.reshape(
            ttnn_slice_77,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_331 = ttnn.reshape(
            ttnn_slice_78,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_82 = ttnn.permute(
            ttnn_reshape_330,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_83 = ttnn.permute(
            ttnn_reshape_331,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_84 = ttnn.permute(
            ttnn_reshape_329,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_57 = ttnn.pad(
            ttnn_permute_82,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_58 = ttnn.pad(
            ttnn_permute_83,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_59 = ttnn.pad(
            ttnn_permute_84,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_19 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_58,
                ttnn_pad_57,
                ttnn_pad_59,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_79 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_19,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_85 = ttnn.permute(
            ttnn_slice_79,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_332 = ttnn.reshape(
            ttnn_permute_85,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_78 = ttnn.matmul(
            ttnn_reshape_332,
            self.w152,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_116 = ttnn.add(
            ttnn_matmul_78,
            self.cer_52_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_116

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_117 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_41 = ttnn.layer_norm(
            ttnn_add_117,
            epsilon=9.9999997473787516e-06,
            weight=self.w150,
            bias=self.w149,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_41, ttnn_add_117

    def _mlp(self, hidden_states):

        ttnn_reshape_333 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_79 = ttnn.matmul(
            ttnn_reshape_333,
            self.w148,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_118 = ttnn.add(
            ttnn_matmul_79,
            self.cer_44_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_19 = ttnn.gelu(
            ttnn_add_118,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_334 = ttnn.reshape(
            ttnn_gelu_19,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_80 = ttnn.matmul(
            ttnn_reshape_334,
            self.w146,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_119 = ttnn.add(
            ttnn_matmul_80,
            self.cer_28_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_119

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_120 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_42 = ttnn.layer_norm(
            ttnn_add_120,
            epsilon=9.9999997473787516e-06,
            weight=self.w144,
            bias=self.w143,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_120, ttnn_layer_norm_42


class CLIPEncoderLayerTTNN_20(LightweightModule):
    """CLIP Encoder Layer 20."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w140 = weights[
            "image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.weight"
        ]
        self.w138 = weights[
            "image_encoder.vision_model.encoder.layers.20.layer_norm2.weight"
        ]
        self.w137 = weights[
            "image_encoder.vision_model.encoder.layers.20.layer_norm2.bias"
        ]
        self.w136 = weights[
            "image_encoder.vision_model.encoder.layers.20.mlp.fc1.weight"
        ]
        self.w134 = weights[
            "image_encoder.vision_model.encoder.layers.20.mlp.fc2.weight"
        ]
        self.w132 = weights[
            "image_encoder.vision_model.encoder.layers.21.layer_norm1.weight"
        ]
        self.w131 = weights[
            "image_encoder.vision_model.encoder.layers.21.layer_norm1.bias"
        ]
        self.cer_107_0 = cer["utils_constEvalFuncWrapper_107_0"]
        self.cer_60_0 = cer["utils_constEvalFuncWrapper_60_0"]
        self.cer_65_0 = cer["utils_constEvalFuncWrapper_65_0"]
        self.cer_78_0 = cer["utils_constEvalFuncWrapper_78_0"]
        self.cer_82_0 = cer["utils_constEvalFuncWrapper_82_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_335 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_81 = ttnn.matmul(
            ttnn_reshape_335,
            self.cer_65_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_121 = ttnn.add(
            ttnn_matmul_81,
            self.cer_60_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_80 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_81 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_82 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_336 = ttnn.reshape(
            ttnn_slice_80,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_337 = ttnn.reshape(
            ttnn_slice_81,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_338 = ttnn.reshape(
            ttnn_slice_82,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_86 = ttnn.permute(
            ttnn_reshape_337,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_87 = ttnn.permute(
            ttnn_reshape_338,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_88 = ttnn.permute(
            ttnn_reshape_336,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_60 = ttnn.pad(
            ttnn_permute_86,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_61 = ttnn.pad(
            ttnn_permute_87,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_62 = ttnn.pad(
            ttnn_permute_88,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_20 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_61,
                ttnn_pad_60,
                ttnn_pad_62,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_83 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_20,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_89 = ttnn.permute(
            ttnn_slice_83,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_339 = ttnn.reshape(
            ttnn_permute_89,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_82 = ttnn.matmul(
            ttnn_reshape_339,
            self.w140,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_122 = ttnn.add(
            ttnn_matmul_82,
            self.cer_78_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_122

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_123 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_43 = ttnn.layer_norm(
            ttnn_add_123,
            epsilon=9.9999997473787516e-06,
            weight=self.w138,
            bias=self.w137,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_43, ttnn_add_123

    def _mlp(self, hidden_states):

        ttnn_reshape_340 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_83 = ttnn.matmul(
            ttnn_reshape_340,
            self.w136,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_124 = ttnn.add(
            ttnn_matmul_83,
            self.cer_107_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_20 = ttnn.gelu(
            ttnn_add_124,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_341 = ttnn.reshape(
            ttnn_gelu_20,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_84 = ttnn.matmul(
            ttnn_reshape_341,
            self.w134,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_125 = ttnn.add(
            ttnn_matmul_84,
            self.cer_82_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_125

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_126 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_44 = ttnn.layer_norm(
            ttnn_add_126,
            epsilon=9.9999997473787516e-06,
            weight=self.w132,
            bias=self.w131,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_126, ttnn_layer_norm_44


class CLIPEncoderLayerTTNN_21(LightweightModule):
    """CLIP Encoder Layer 21."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w128 = weights[
            "image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.weight"
        ]
        self.w126 = weights[
            "image_encoder.vision_model.encoder.layers.21.layer_norm2.weight"
        ]
        self.w125 = weights[
            "image_encoder.vision_model.encoder.layers.21.layer_norm2.bias"
        ]
        self.w124 = weights[
            "image_encoder.vision_model.encoder.layers.21.mlp.fc1.weight"
        ]
        self.w122 = weights[
            "image_encoder.vision_model.encoder.layers.21.mlp.fc2.weight"
        ]
        self.w120 = weights[
            "image_encoder.vision_model.encoder.layers.22.layer_norm1.weight"
        ]
        self.w119 = weights[
            "image_encoder.vision_model.encoder.layers.22.layer_norm1.bias"
        ]
        self.cer_110_0 = cer["utils_constEvalFuncWrapper_110_0"]
        self.cer_111_0 = cer["utils_constEvalFuncWrapper_111_0"]
        self.cer_160_0 = cer["utils_constEvalFuncWrapper_160_0"]
        self.cer_20_0 = cer["utils_constEvalFuncWrapper_20_0"]
        self.cer_37_0 = cer["utils_constEvalFuncWrapper_37_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_342 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_85 = ttnn.matmul(
            ttnn_reshape_342,
            self.cer_37_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_127 = ttnn.add(
            ttnn_matmul_85,
            self.cer_111_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_84 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_85 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_86 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_343 = ttnn.reshape(
            ttnn_slice_84,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_344 = ttnn.reshape(
            ttnn_slice_85,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_345 = ttnn.reshape(
            ttnn_slice_86,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_90 = ttnn.permute(
            ttnn_reshape_344,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_91 = ttnn.permute(
            ttnn_reshape_345,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_92 = ttnn.permute(
            ttnn_reshape_343,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_63 = ttnn.pad(
            ttnn_permute_90,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_64 = ttnn.pad(
            ttnn_permute_91,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_65 = ttnn.pad(
            ttnn_permute_92,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_21 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_64,
                ttnn_pad_63,
                ttnn_pad_65,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_87 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_21,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_93 = ttnn.permute(
            ttnn_slice_87,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_346 = ttnn.reshape(
            ttnn_permute_93,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_86 = ttnn.matmul(
            ttnn_reshape_346,
            self.w128,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_128 = ttnn.add(
            ttnn_matmul_86,
            self.cer_20_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_128

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_129 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_45 = ttnn.layer_norm(
            ttnn_add_129,
            epsilon=9.9999997473787516e-06,
            weight=self.w126,
            bias=self.w125,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_45, ttnn_add_129

    def _mlp(self, hidden_states):

        ttnn_reshape_347 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_87 = ttnn.matmul(
            ttnn_reshape_347,
            self.w124,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_130 = ttnn.add(
            ttnn_matmul_87,
            self.cer_110_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_21 = ttnn.gelu(
            ttnn_add_130,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_348 = ttnn.reshape(
            ttnn_gelu_21,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_88 = ttnn.matmul(
            ttnn_reshape_348,
            self.w122,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_131 = ttnn.add(
            ttnn_matmul_88,
            self.cer_160_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_131

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_132 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_46 = ttnn.layer_norm(
            ttnn_add_132,
            epsilon=9.9999997473787516e-06,
            weight=self.w120,
            bias=self.w119,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_132, ttnn_layer_norm_46


class CLIPEncoderLayerTTNN_22(LightweightModule):
    """CLIP Encoder Layer 22."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w116 = weights[
            "image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.weight"
        ]
        self.w114 = weights[
            "image_encoder.vision_model.encoder.layers.22.layer_norm2.weight"
        ]
        self.w113 = weights[
            "image_encoder.vision_model.encoder.layers.22.layer_norm2.bias"
        ]
        self.w112 = weights[
            "image_encoder.vision_model.encoder.layers.22.mlp.fc1.weight"
        ]
        self.w110 = weights[
            "image_encoder.vision_model.encoder.layers.22.mlp.fc2.weight"
        ]
        self.w108 = weights[
            "image_encoder.vision_model.encoder.layers.23.layer_norm1.weight"
        ]
        self.w107 = weights[
            "image_encoder.vision_model.encoder.layers.23.layer_norm1.bias"
        ]
        self.cer_125_0 = cer["utils_constEvalFuncWrapper_125_0"]
        self.cer_148_0 = cer["utils_constEvalFuncWrapper_148_0"]
        self.cer_33_0 = cer["utils_constEvalFuncWrapper_33_0"]
        self.cer_4_0 = cer["utils_constEvalFuncWrapper_4_0"]
        self.cer_57_0 = cer["utils_constEvalFuncWrapper_57_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_349 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_89 = ttnn.matmul(
            ttnn_reshape_349,
            self.cer_148_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_133 = ttnn.add(
            ttnn_matmul_89,
            self.cer_57_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_88 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_89 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_90 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_350 = ttnn.reshape(
            ttnn_slice_88,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_351 = ttnn.reshape(
            ttnn_slice_89,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_352 = ttnn.reshape(
            ttnn_slice_90,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_94 = ttnn.permute(
            ttnn_reshape_351,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_95 = ttnn.permute(
            ttnn_reshape_352,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_96 = ttnn.permute(
            ttnn_reshape_350,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_66 = ttnn.pad(
            ttnn_permute_94,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_67 = ttnn.pad(
            ttnn_permute_95,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_68 = ttnn.pad(
            ttnn_permute_96,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_22 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_67,
                ttnn_pad_66,
                ttnn_pad_68,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_91 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_22,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_97 = ttnn.permute(
            ttnn_slice_91,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_353 = ttnn.reshape(
            ttnn_permute_97,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_90 = ttnn.matmul(
            ttnn_reshape_353,
            self.w116,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_134 = ttnn.add(
            ttnn_matmul_90,
            self.cer_33_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_134

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_135 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_47 = ttnn.layer_norm(
            ttnn_add_135,
            epsilon=9.9999997473787516e-06,
            weight=self.w114,
            bias=self.w113,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_47, ttnn_add_135

    def _mlp(self, hidden_states):

        ttnn_reshape_354 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_91 = ttnn.matmul(
            ttnn_reshape_354,
            self.w112,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_136 = ttnn.add(
            ttnn_matmul_91,
            self.cer_4_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_22 = ttnn.gelu(
            ttnn_add_136,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_355 = ttnn.reshape(
            ttnn_gelu_22,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_92 = ttnn.matmul(
            ttnn_reshape_355,
            self.w110,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_137 = ttnn.add(
            ttnn_matmul_92,
            self.cer_125_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_137

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_138 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_48 = ttnn.layer_norm(
            ttnn_add_138,
            epsilon=9.9999997473787516e-06,
            weight=self.w108,
            bias=self.w107,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_138, ttnn_layer_norm_48


class CLIPEncoderLayerTTNN_23(LightweightModule):
    """CLIP Encoder Layer 23."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w104 = weights[
            "image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.weight"
        ]
        self.w102 = weights[
            "image_encoder.vision_model.encoder.layers.23.layer_norm2.weight"
        ]
        self.w101 = weights[
            "image_encoder.vision_model.encoder.layers.23.layer_norm2.bias"
        ]
        self.w100 = weights[
            "image_encoder.vision_model.encoder.layers.23.mlp.fc1.weight"
        ]
        self.w98 = weights[
            "image_encoder.vision_model.encoder.layers.23.mlp.fc2.weight"
        ]
        self.w96 = weights[
            "image_encoder.vision_model.encoder.layers.24.layer_norm1.weight"
        ]
        self.w95 = weights[
            "image_encoder.vision_model.encoder.layers.24.layer_norm1.bias"
        ]
        self.cer_0_0 = cer["utils_constEvalFuncWrapper_0_0"]
        self.cer_22_0 = cer["utils_constEvalFuncWrapper_22_0"]
        self.cer_32_0 = cer["utils_constEvalFuncWrapper_32_0"]
        self.cer_36_0 = cer["utils_constEvalFuncWrapper_36_0"]
        self.cer_51_0 = cer["utils_constEvalFuncWrapper_51_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_356 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_93 = ttnn.matmul(
            ttnn_reshape_356,
            self.cer_36_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_139 = ttnn.add(
            ttnn_matmul_93,
            self.cer_32_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_92 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_93 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_94 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_357 = ttnn.reshape(
            ttnn_slice_92,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_358 = ttnn.reshape(
            ttnn_slice_93,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_359 = ttnn.reshape(
            ttnn_slice_94,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_98 = ttnn.permute(
            ttnn_reshape_358,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_99 = ttnn.permute(
            ttnn_reshape_359,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_100 = ttnn.permute(
            ttnn_reshape_357,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_69 = ttnn.pad(
            ttnn_permute_98,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_70 = ttnn.pad(
            ttnn_permute_99,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_71 = ttnn.pad(
            ttnn_permute_100,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_23 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_70,
                ttnn_pad_69,
                ttnn_pad_71,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_95 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_23,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_101 = ttnn.permute(
            ttnn_slice_95,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_360 = ttnn.reshape(
            ttnn_permute_101,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_94 = ttnn.matmul(
            ttnn_reshape_360,
            self.w104,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_140 = ttnn.add(
            ttnn_matmul_94,
            self.cer_51_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_140

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_141 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_49 = ttnn.layer_norm(
            ttnn_add_141,
            epsilon=9.9999997473787516e-06,
            weight=self.w102,
            bias=self.w101,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_49, ttnn_add_141

    def _mlp(self, hidden_states):

        ttnn_reshape_361 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_95 = ttnn.matmul(
            ttnn_reshape_361,
            self.w100,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_142 = ttnn.add(
            ttnn_matmul_95,
            self.cer_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_23 = ttnn.gelu(
            ttnn_add_142,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_362 = ttnn.reshape(
            ttnn_gelu_23,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_96 = ttnn.matmul(
            ttnn_reshape_362,
            self.w98,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_143 = ttnn.add(
            ttnn_matmul_96,
            self.cer_22_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_143

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_144 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_50 = ttnn.layer_norm(
            ttnn_add_144,
            epsilon=9.9999997473787516e-06,
            weight=self.w96,
            bias=self.w95,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_144, ttnn_layer_norm_50


class CLIPEncoderLayerTTNN_24(LightweightModule):
    """CLIP Encoder Layer 24."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w92 = weights[
            "image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.weight"
        ]
        self.w90 = weights[
            "image_encoder.vision_model.encoder.layers.24.layer_norm2.weight"
        ]
        self.w89 = weights[
            "image_encoder.vision_model.encoder.layers.24.layer_norm2.bias"
        ]
        self.w88 = weights[
            "image_encoder.vision_model.encoder.layers.24.mlp.fc1.weight"
        ]
        self.w86 = weights[
            "image_encoder.vision_model.encoder.layers.24.mlp.fc2.weight"
        ]
        self.w84 = weights[
            "image_encoder.vision_model.encoder.layers.25.layer_norm1.weight"
        ]
        self.w83 = weights[
            "image_encoder.vision_model.encoder.layers.25.layer_norm1.bias"
        ]
        self.cer_139_0 = cer["utils_constEvalFuncWrapper_139_0"]
        self.cer_143_0 = cer["utils_constEvalFuncWrapper_143_0"]
        self.cer_144_0 = cer["utils_constEvalFuncWrapper_144_0"]
        self.cer_59_0 = cer["utils_constEvalFuncWrapper_59_0"]
        self.cer_76_0 = cer["utils_constEvalFuncWrapper_76_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_363 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_97 = ttnn.matmul(
            ttnn_reshape_363,
            self.cer_76_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_145 = ttnn.add(
            ttnn_matmul_97,
            self.cer_59_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_96 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_97 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_98 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_364 = ttnn.reshape(
            ttnn_slice_96,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_365 = ttnn.reshape(
            ttnn_slice_97,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_366 = ttnn.reshape(
            ttnn_slice_98,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_102 = ttnn.permute(
            ttnn_reshape_365,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_103 = ttnn.permute(
            ttnn_reshape_366,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_104 = ttnn.permute(
            ttnn_reshape_364,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_72 = ttnn.pad(
            ttnn_permute_102,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_73 = ttnn.pad(
            ttnn_permute_103,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_74 = ttnn.pad(
            ttnn_permute_104,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_24 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_73,
                ttnn_pad_72,
                ttnn_pad_74,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_99 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_24,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_105 = ttnn.permute(
            ttnn_slice_99,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_367 = ttnn.reshape(
            ttnn_permute_105,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_98 = ttnn.matmul(
            ttnn_reshape_367,
            self.w92,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_146 = ttnn.add(
            ttnn_matmul_98,
            self.cer_143_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_146

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_147 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_51 = ttnn.layer_norm(
            ttnn_add_147,
            epsilon=9.9999997473787516e-06,
            weight=self.w90,
            bias=self.w89,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_51, ttnn_add_147

    def _mlp(self, hidden_states):

        ttnn_reshape_368 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_99 = ttnn.matmul(
            ttnn_reshape_368,
            self.w88,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_148 = ttnn.add(
            ttnn_matmul_99,
            self.cer_139_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_24 = ttnn.gelu(
            ttnn_add_148,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_369 = ttnn.reshape(
            ttnn_gelu_24,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_100 = ttnn.matmul(
            ttnn_reshape_369,
            self.w86,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_149 = ttnn.add(
            ttnn_matmul_100,
            self.cer_144_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_149

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_150 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_52 = ttnn.layer_norm(
            ttnn_add_150,
            epsilon=9.9999997473787516e-06,
            weight=self.w84,
            bias=self.w83,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_150, ttnn_layer_norm_52


class CLIPEncoderLayerTTNN_25(LightweightModule):
    """CLIP Encoder Layer 25."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w80 = weights[
            "image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.weight"
        ]
        self.w78 = weights[
            "image_encoder.vision_model.encoder.layers.25.layer_norm2.weight"
        ]
        self.w77 = weights[
            "image_encoder.vision_model.encoder.layers.25.layer_norm2.bias"
        ]
        self.w76 = weights[
            "image_encoder.vision_model.encoder.layers.25.mlp.fc1.weight"
        ]
        self.w74 = weights[
            "image_encoder.vision_model.encoder.layers.25.mlp.fc2.weight"
        ]
        self.w72 = weights[
            "image_encoder.vision_model.encoder.layers.26.layer_norm1.weight"
        ]
        self.w71 = weights[
            "image_encoder.vision_model.encoder.layers.26.layer_norm1.bias"
        ]
        self.cer_117_0 = cer["utils_constEvalFuncWrapper_117_0"]
        self.cer_31_0 = cer["utils_constEvalFuncWrapper_31_0"]
        self.cer_39_0 = cer["utils_constEvalFuncWrapper_39_0"]
        self.cer_58_0 = cer["utils_constEvalFuncWrapper_58_0"]
        self.cer_61_0 = cer["utils_constEvalFuncWrapper_61_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_370 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_101 = ttnn.matmul(
            ttnn_reshape_370,
            self.cer_61_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_151 = ttnn.add(
            ttnn_matmul_101,
            self.cer_58_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_100 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_101 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_102 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_371 = ttnn.reshape(
            ttnn_slice_100,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_372 = ttnn.reshape(
            ttnn_slice_101,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_373 = ttnn.reshape(
            ttnn_slice_102,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_106 = ttnn.permute(
            ttnn_reshape_372,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_107 = ttnn.permute(
            ttnn_reshape_373,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_108 = ttnn.permute(
            ttnn_reshape_371,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_75 = ttnn.pad(
            ttnn_permute_106,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_76 = ttnn.pad(
            ttnn_permute_107,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_77 = ttnn.pad(
            ttnn_permute_108,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_25 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_76,
                ttnn_pad_75,
                ttnn_pad_77,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_103 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_25,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_109 = ttnn.permute(
            ttnn_slice_103,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_374 = ttnn.reshape(
            ttnn_permute_109,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_102 = ttnn.matmul(
            ttnn_reshape_374,
            self.w80,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_152 = ttnn.add(
            ttnn_matmul_102,
            self.cer_31_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_152

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_153 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_53 = ttnn.layer_norm(
            ttnn_add_153,
            epsilon=9.9999997473787516e-06,
            weight=self.w78,
            bias=self.w77,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_53, ttnn_add_153

    def _mlp(self, hidden_states):

        ttnn_reshape_375 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_103 = ttnn.matmul(
            ttnn_reshape_375,
            self.w76,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_154 = ttnn.add(
            ttnn_matmul_103,
            self.cer_117_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_25 = ttnn.gelu(
            ttnn_add_154,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_376 = ttnn.reshape(
            ttnn_gelu_25,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_104 = ttnn.matmul(
            ttnn_reshape_376,
            self.w74,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_155 = ttnn.add(
            ttnn_matmul_104,
            self.cer_39_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_155

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_156 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_54 = ttnn.layer_norm(
            ttnn_add_156,
            epsilon=9.9999997473787516e-06,
            weight=self.w72,
            bias=self.w71,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_156, ttnn_layer_norm_54


class CLIPEncoderLayerTTNN_26(LightweightModule):
    """CLIP Encoder Layer 26."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w68 = weights[
            "image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.weight"
        ]
        self.w66 = weights[
            "image_encoder.vision_model.encoder.layers.26.layer_norm2.weight"
        ]
        self.w65 = weights[
            "image_encoder.vision_model.encoder.layers.26.layer_norm2.bias"
        ]
        self.w64 = weights[
            "image_encoder.vision_model.encoder.layers.26.mlp.fc1.weight"
        ]
        self.w62 = weights[
            "image_encoder.vision_model.encoder.layers.26.mlp.fc2.weight"
        ]
        self.w60 = weights[
            "image_encoder.vision_model.encoder.layers.27.layer_norm1.weight"
        ]
        self.w59 = weights[
            "image_encoder.vision_model.encoder.layers.27.layer_norm1.bias"
        ]
        self.cer_105_0 = cer["utils_constEvalFuncWrapper_105_0"]
        self.cer_123_0 = cer["utils_constEvalFuncWrapper_123_0"]
        self.cer_77_0 = cer["utils_constEvalFuncWrapper_77_0"]
        self.cer_98_0 = cer["utils_constEvalFuncWrapper_98_0"]
        self.cer_9_0 = cer["utils_constEvalFuncWrapper_9_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_377 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_105 = ttnn.matmul(
            ttnn_reshape_377,
            self.cer_9_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_157 = ttnn.add(
            ttnn_matmul_105,
            self.cer_77_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_104 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_105 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_106 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_378 = ttnn.reshape(
            ttnn_slice_104,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_379 = ttnn.reshape(
            ttnn_slice_105,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_380 = ttnn.reshape(
            ttnn_slice_106,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_110 = ttnn.permute(
            ttnn_reshape_379,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_111 = ttnn.permute(
            ttnn_reshape_380,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_112 = ttnn.permute(
            ttnn_reshape_378,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_78 = ttnn.pad(
            ttnn_permute_110,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_79 = ttnn.pad(
            ttnn_permute_111,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_80 = ttnn.pad(
            ttnn_permute_112,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_26 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_79,
                ttnn_pad_78,
                ttnn_pad_80,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_107 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_26,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_113 = ttnn.permute(
            ttnn_slice_107,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_381 = ttnn.reshape(
            ttnn_permute_113,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_106 = ttnn.matmul(
            ttnn_reshape_381,
            self.w68,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_158 = ttnn.add(
            ttnn_matmul_106,
            self.cer_105_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_158

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_159 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_55 = ttnn.layer_norm(
            ttnn_add_159,
            epsilon=9.9999997473787516e-06,
            weight=self.w66,
            bias=self.w65,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_55, ttnn_add_159

    def _mlp(self, hidden_states):

        ttnn_reshape_382 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_107 = ttnn.matmul(
            ttnn_reshape_382,
            self.w64,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_160 = ttnn.add(
            ttnn_matmul_107,
            self.cer_98_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_26 = ttnn.gelu(
            ttnn_add_160,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_383 = ttnn.reshape(
            ttnn_gelu_26,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_108 = ttnn.matmul(
            ttnn_reshape_383,
            self.w62,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_161 = ttnn.add(
            ttnn_matmul_108,
            self.cer_123_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_161

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_162 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_56 = ttnn.layer_norm(
            ttnn_add_162,
            epsilon=9.9999997473787516e-06,
            weight=self.w60,
            bias=self.w59,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_162, ttnn_layer_norm_56


class CLIPEncoderLayerTTNN_27(LightweightModule):
    """CLIP Encoder Layer 27."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w56 = weights[
            "image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.weight"
        ]
        self.w54 = weights[
            "image_encoder.vision_model.encoder.layers.27.layer_norm2.weight"
        ]
        self.w53 = weights[
            "image_encoder.vision_model.encoder.layers.27.layer_norm2.bias"
        ]
        self.w52 = weights[
            "image_encoder.vision_model.encoder.layers.27.mlp.fc1.weight"
        ]
        self.w50 = weights[
            "image_encoder.vision_model.encoder.layers.27.mlp.fc2.weight"
        ]
        self.w48 = weights[
            "image_encoder.vision_model.encoder.layers.28.layer_norm1.weight"
        ]
        self.w47 = weights[
            "image_encoder.vision_model.encoder.layers.28.layer_norm1.bias"
        ]
        self.cer_115_0 = cer["utils_constEvalFuncWrapper_115_0"]
        self.cer_129_0 = cer["utils_constEvalFuncWrapper_129_0"]
        self.cer_159_0 = cer["utils_constEvalFuncWrapper_159_0"]
        self.cer_41_0 = cer["utils_constEvalFuncWrapper_41_0"]
        self.cer_8_0 = cer["utils_constEvalFuncWrapper_8_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_384 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_109 = ttnn.matmul(
            ttnn_reshape_384,
            self.cer_159_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_163 = ttnn.add(
            ttnn_matmul_109,
            self.cer_41_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_108 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_109 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_110 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_385 = ttnn.reshape(
            ttnn_slice_108,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_386 = ttnn.reshape(
            ttnn_slice_109,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_387 = ttnn.reshape(
            ttnn_slice_110,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_114 = ttnn.permute(
            ttnn_reshape_386,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_115 = ttnn.permute(
            ttnn_reshape_387,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_116 = ttnn.permute(
            ttnn_reshape_385,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_81 = ttnn.pad(
            ttnn_permute_114,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_82 = ttnn.pad(
            ttnn_permute_115,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_83 = ttnn.pad(
            ttnn_permute_116,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_27 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_82,
                ttnn_pad_81,
                ttnn_pad_83,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_111 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_27,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_117 = ttnn.permute(
            ttnn_slice_111,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_388 = ttnn.reshape(
            ttnn_permute_117,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_110 = ttnn.matmul(
            ttnn_reshape_388,
            self.w56,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_164 = ttnn.add(
            ttnn_matmul_110,
            self.cer_8_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_164

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_165 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_57 = ttnn.layer_norm(
            ttnn_add_165,
            epsilon=9.9999997473787516e-06,
            weight=self.w54,
            bias=self.w53,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_57, ttnn_add_165

    def _mlp(self, hidden_states):

        ttnn_reshape_389 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_111 = ttnn.matmul(
            ttnn_reshape_389,
            self.w52,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_166 = ttnn.add(
            ttnn_matmul_111,
            self.cer_115_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_27 = ttnn.gelu(
            ttnn_add_166,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_390 = ttnn.reshape(
            ttnn_gelu_27,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_112 = ttnn.matmul(
            ttnn_reshape_390,
            self.w50,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_167 = ttnn.add(
            ttnn_matmul_112,
            self.cer_129_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_167

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_168 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_58 = ttnn.layer_norm(
            ttnn_add_168,
            epsilon=9.9999997473787516e-06,
            weight=self.w48,
            bias=self.w47,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_168, ttnn_layer_norm_58


class CLIPEncoderLayerTTNN_28(LightweightModule):
    """CLIP Encoder Layer 28."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w44 = weights[
            "image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.weight"
        ]
        self.w42 = weights[
            "image_encoder.vision_model.encoder.layers.28.layer_norm2.weight"
        ]
        self.w41 = weights[
            "image_encoder.vision_model.encoder.layers.28.layer_norm2.bias"
        ]
        self.w40 = weights[
            "image_encoder.vision_model.encoder.layers.28.mlp.fc1.weight"
        ]
        self.w38 = weights[
            "image_encoder.vision_model.encoder.layers.28.mlp.fc2.weight"
        ]
        self.w36 = weights[
            "image_encoder.vision_model.encoder.layers.29.layer_norm1.weight"
        ]
        self.w35 = weights[
            "image_encoder.vision_model.encoder.layers.29.layer_norm1.bias"
        ]
        self.cer_121_0 = cer["utils_constEvalFuncWrapper_121_0"]
        self.cer_14_0 = cer["utils_constEvalFuncWrapper_14_0"]
        self.cer_16_0 = cer["utils_constEvalFuncWrapper_16_0"]
        self.cer_3_0 = cer["utils_constEvalFuncWrapper_3_0"]
        self.cer_56_0 = cer["utils_constEvalFuncWrapper_56_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_391 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_113 = ttnn.matmul(
            ttnn_reshape_391,
            self.cer_3_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_169 = ttnn.add(
            ttnn_matmul_113,
            self.cer_16_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_112 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_113 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_114 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_392 = ttnn.reshape(
            ttnn_slice_112,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_393 = ttnn.reshape(
            ttnn_slice_113,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_394 = ttnn.reshape(
            ttnn_slice_114,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_118 = ttnn.permute(
            ttnn_reshape_393,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_119 = ttnn.permute(
            ttnn_reshape_394,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_120 = ttnn.permute(
            ttnn_reshape_392,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_84 = ttnn.pad(
            ttnn_permute_118,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_85 = ttnn.pad(
            ttnn_permute_119,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_86 = ttnn.pad(
            ttnn_permute_120,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_28 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_85,
                ttnn_pad_84,
                ttnn_pad_86,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_115 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_28,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_121 = ttnn.permute(
            ttnn_slice_115,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_395 = ttnn.reshape(
            ttnn_permute_121,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_114 = ttnn.matmul(
            ttnn_reshape_395,
            self.w44,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_170 = ttnn.add(
            ttnn_matmul_114,
            self.cer_121_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_170

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_171 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_59 = ttnn.layer_norm(
            ttnn_add_171,
            epsilon=9.9999997473787516e-06,
            weight=self.w42,
            bias=self.w41,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_59, ttnn_add_171

    def _mlp(self, hidden_states):

        ttnn_reshape_396 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_115 = ttnn.matmul(
            ttnn_reshape_396,
            self.w40,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_172 = ttnn.add(
            ttnn_matmul_115,
            self.cer_14_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_28 = ttnn.gelu(
            ttnn_add_172,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_397 = ttnn.reshape(
            ttnn_gelu_28,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_116 = ttnn.matmul(
            ttnn_reshape_397,
            self.w38,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_173 = ttnn.add(
            ttnn_matmul_116,
            self.cer_56_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_173

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_174 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_60 = ttnn.layer_norm(
            ttnn_add_174,
            epsilon=9.9999997473787516e-06,
            weight=self.w36,
            bias=self.w35,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_174, ttnn_layer_norm_60


class CLIPEncoderLayerTTNN_29(LightweightModule):
    """CLIP Encoder Layer 29."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w32 = weights[
            "image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.weight"
        ]
        self.w30 = weights[
            "image_encoder.vision_model.encoder.layers.29.layer_norm2.weight"
        ]
        self.w29 = weights[
            "image_encoder.vision_model.encoder.layers.29.layer_norm2.bias"
        ]
        self.w28 = weights[
            "image_encoder.vision_model.encoder.layers.29.mlp.fc1.weight"
        ]
        self.w26 = weights[
            "image_encoder.vision_model.encoder.layers.29.mlp.fc2.weight"
        ]
        self.w24 = weights[
            "image_encoder.vision_model.encoder.layers.30.layer_norm1.weight"
        ]
        self.w23 = weights[
            "image_encoder.vision_model.encoder.layers.30.layer_norm1.bias"
        ]
        self.cer_35_0 = cer["utils_constEvalFuncWrapper_35_0"]
        self.cer_38_0 = cer["utils_constEvalFuncWrapper_38_0"]
        self.cer_45_0 = cer["utils_constEvalFuncWrapper_45_0"]
        self.cer_75_0 = cer["utils_constEvalFuncWrapper_75_0"]
        self.cer_79_0 = cer["utils_constEvalFuncWrapper_79_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # residual + layer_norm1_next
        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)
        return new_residual, normalized

    def _attention(self, hidden_states):

        ttnn_reshape_398 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_117 = ttnn.matmul(
            ttnn_reshape_398,
            self.cer_75_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_175 = ttnn.add(
            ttnn_matmul_117,
            self.cer_45_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_116 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_117 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_118 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_399 = ttnn.reshape(
            ttnn_slice_116,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_400 = ttnn.reshape(
            ttnn_slice_117,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_401 = ttnn.reshape(
            ttnn_slice_118,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_122 = ttnn.permute(
            ttnn_reshape_400,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_123 = ttnn.permute(
            ttnn_reshape_401,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_124 = ttnn.permute(
            ttnn_reshape_399,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_87 = ttnn.pad(
            ttnn_permute_122,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_88 = ttnn.pad(
            ttnn_permute_123,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_89 = ttnn.pad(
            ttnn_permute_124,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_29 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_88,
                ttnn_pad_87,
                ttnn_pad_89,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_119 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_29,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_125 = ttnn.permute(
            ttnn_slice_119,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_402 = ttnn.reshape(
            ttnn_permute_125,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_118 = ttnn.matmul(
            ttnn_reshape_402,
            self.w32,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_176 = ttnn.add(
            ttnn_matmul_118,
            self.cer_79_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_176

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_177 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_61 = ttnn.layer_norm(
            ttnn_add_177,
            epsilon=9.9999997473787516e-06,
            weight=self.w30,
            bias=self.w29,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_61, ttnn_add_177

    def _mlp(self, hidden_states):

        ttnn_reshape_403 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_119 = ttnn.matmul(
            ttnn_reshape_403,
            self.w28,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_178 = ttnn.add(
            ttnn_matmul_119,
            self.cer_38_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_29 = ttnn.gelu(
            ttnn_add_178,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_404 = ttnn.reshape(
            ttnn_gelu_29,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_120 = ttnn.matmul(
            ttnn_reshape_404,
            self.w26,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_179 = ttnn.add(
            ttnn_matmul_120,
            self.cer_35_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_179

    def _layer_norm1_next(self, residual, mlp_output):

        ttnn_add_180 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_62 = ttnn.layer_norm(
            ttnn_add_180,
            epsilon=9.9999997473787516e-06,
            weight=self.w24,
            bias=self.w23,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_180, ttnn_layer_norm_62


class CLIPEncoderLayerTTNN_30(LightweightModule):
    """CLIP Encoder Layer 30."""

    def __init__(self, weights, cer):
        """Store layer weights and cer values."""
        self.w20 = weights[
            "image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.weight"
        ]
        self.w18 = weights[
            "image_encoder.vision_model.encoder.layers.30.layer_norm2.weight"
        ]
        self.w17 = weights[
            "image_encoder.vision_model.encoder.layers.30.layer_norm2.bias"
        ]
        self.w16 = weights[
            "image_encoder.vision_model.encoder.layers.30.mlp.fc1.weight"
        ]
        self.w14 = weights[
            "image_encoder.vision_model.encoder.layers.30.mlp.fc2.weight"
        ]
        self.cer_131_0 = cer["utils_constEvalFuncWrapper_131_0"]
        self.cer_135_0 = cer["utils_constEvalFuncWrapper_135_0"]
        self.cer_138_0 = cer["utils_constEvalFuncWrapper_138_0"]
        self.cer_48_0 = cer["utils_constEvalFuncWrapper_48_0"]
        self.cer_54_0 = cer["utils_constEvalFuncWrapper_54_0"]

    def forward(self, hidden_states, residual):
        """Forward pass."""
        # attention
        attn_output = self._attention(hidden_states)
        # residual + layer_norm2
        mlp_input, residual = self._layer_norm2_add(residual, attn_output)
        # mlp
        mlp_output = self._mlp(mlp_input)
        # final residual
        output = self._final_residual(residual, mlp_output)
        return output, None

    def _attention(self, hidden_states):

        ttnn_reshape_405 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_121 = ttnn.matmul(
            ttnn_reshape_405,
            self.cer_138_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_181 = ttnn.add(
            ttnn_matmul_121,
            self.cer_131_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_120 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_121 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_122 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_406 = ttnn.reshape(
            ttnn_slice_120,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_407 = ttnn.reshape(
            ttnn_slice_121,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_408 = ttnn.reshape(
            ttnn_slice_122,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_126 = ttnn.permute(
            ttnn_reshape_407,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_127 = ttnn.permute(
            ttnn_reshape_408,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_128 = ttnn.permute(
            ttnn_reshape_406,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_90 = ttnn.pad(
            ttnn_permute_126,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_91 = ttnn.pad(
            ttnn_permute_127,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_92 = ttnn.pad(
            ttnn_permute_128,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_30 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_91,
                ttnn_pad_90,
                ttnn_pad_92,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_123 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_30,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_129 = ttnn.permute(
            ttnn_slice_123,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_409 = ttnn.reshape(
            ttnn_permute_129,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_122 = ttnn.matmul(
            ttnn_reshape_409,
            self.w20,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_182 = ttnn.add(
            ttnn_matmul_122,
            self.cer_48_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_182

    def _layer_norm2_add(self, residual, attn_output):

        ttnn_add_183 = ttnn.add(
            residual,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_63 = ttnn.layer_norm(
            ttnn_add_183,
            epsilon=9.9999997473787516e-06,
            weight=self.w18,
            bias=self.w17,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_63, ttnn_add_183

    def _mlp(self, hidden_states):

        ttnn_reshape_410 = ttnn.reshape(
            hidden_states,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_123 = ttnn.matmul(
            ttnn_reshape_410,
            self.w16,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_184 = ttnn.add(
            ttnn_matmul_123,
            self.cer_135_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_30 = ttnn.gelu(
            ttnn_add_184,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_411 = ttnn.reshape(
            ttnn_gelu_30,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_124 = ttnn.matmul(
            ttnn_reshape_411,
            self.w14,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_185 = ttnn.add(
            ttnn_matmul_124,
            self.cer_54_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_185

    def _final_residual(self, residual, mlp_output):

        ttnn_add_186 = ttnn.add(
            residual,
            mlp_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_412 = ttnn.reshape(
            ttnn_add_186,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_412
