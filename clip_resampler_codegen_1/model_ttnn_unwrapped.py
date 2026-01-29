# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model with all ops inlined into forward()."""

import ttnn
import utils
from consteval import run_const_evals
from load_weights_from_pytorch import load_weights_from_pytorch
from models.common.lightweightmodule import LightweightModule


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    def __init__(self, device, torch_weights, cache):
        self.device = device
        self.weights = load_weights_from_pytorch(torch_weights, device)
        self.cer = run_const_evals(self.weights, cache)

    def forward(self, pixel_values):
        # Move input to device
        assert pixel_values.device() is None, "pixel_values must be on host"
        pixel_values = ttnn.to_device(
            pixel_values,
            self.device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_to_layout_287 = ttnn.to_layout(
            pixel_values,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_utils_DeviceGetter_get_device_162 = utils.DeviceGetter.get_device((1, 1))
        v0_ttnn_permute_3 = ttnn.permute(
            v0_ttnn_to_layout_287,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v0_ttnn_reshape_192 = ttnn.reshape(
            v0_ttnn_permute_3,
            [1, 1, 50176, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_conv2d_0 = ttnn.conv2d(
            input_tensor=v0_ttnn_reshape_192,
            weight_tensor=self.cer["utils_constEvalFuncWrapper_66_0"],
            device=v0_utils_DeviceGetter_get_device_162,
            in_channels=3,
            out_channels=1280,
            batch_size=1,
            input_height=224,
            input_width=224,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=None,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dL1Full, num_slices=0
            ),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_reshape_193 = ttnn.reshape(
            v0_ttnn_conv2d_0,
            [1, 16, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_permute_4 = ttnn.permute(
            v0_ttnn_reshape_193,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v0_ttnn_reshape_194 = ttnn.reshape(
            v0_ttnn_permute_4,
            [1, 1280, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_util_create_list_386 = [
            self.cer["utils_constEvalFuncWrapper_142_0"],
            v0_ttnn_reshape_194,
        ]
        v0_ttnn_concat_62 = ttnn.concat(
            v0_util_create_list_386,
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_add_0 = ttnn.add(
            v0_ttnn_concat_62,
            self.cer["utils_constEvalFuncWrapper_88_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v0_ttnn_permute_5 = ttnn.permute(
            v0_ttnn_add_0,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        CLIPVisionEmbeddings_0_0_0 = v0_ttnn_permute_5
        v1_ttnn_layer_norm_1 = ttnn.layer_norm(
            CLIPVisionEmbeddings_0_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["image_encoder.vision_model.pre_layrnorm.weight"],
            bias=self.weights["image_encoder.vision_model.pre_layrnorm.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        LayerNorm_1_0_0 = v1_ttnn_layer_norm_1
        v2_ttnn_layer_norm_2 = ttnn.layer_norm(
            LayerNorm_1_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.0.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.0.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        CLIPEncoderLayer_2_0_0 = v2_ttnn_layer_norm_2
        v3_ttnn_reshape_195 = ttnn.reshape(
            CLIPEncoderLayer_2_0_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_matmul_1 = ttnn.matmul(
            v3_ttnn_reshape_195,
            self.cer["utils_constEvalFuncWrapper_70_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v3_ttnn_add_1 = ttnn.add(
            v3_ttnn_matmul_1,
            self.cer["utils_constEvalFuncWrapper_47_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_slice_0 = ttnn.slice(
            v3_ttnn_add_1,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_slice_1 = ttnn.slice(
            v3_ttnn_add_1,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_slice_2 = ttnn.slice(
            v3_ttnn_add_1,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_reshape_196 = ttnn.reshape(
            v3_ttnn_slice_0,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_reshape_197 = ttnn.reshape(
            v3_ttnn_slice_1,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_reshape_198 = ttnn.reshape(
            v3_ttnn_slice_2,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_permute_6 = ttnn.permute(
            v3_ttnn_reshape_197,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v3_ttnn_permute_7 = ttnn.permute(
            v3_ttnn_reshape_198,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v3_ttnn_permute_8 = ttnn.permute(
            v3_ttnn_reshape_196,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v3_ttnn_pad_0 = ttnn.pad(
            v3_ttnn_permute_6,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_pad_1 = ttnn.pad(
            v3_ttnn_permute_7,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_pad_2 = ttnn.pad(
            v3_ttnn_permute_8,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_transformer_scaled_dot_product_attention_0 = (
            ttnn.transformer.scaled_dot_product_attention(
                v3_ttnn_pad_1,
                v3_ttnn_pad_0,
                v3_ttnn_pad_2,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v3_ttnn_slice_3 = ttnn.slice(
            v3_ttnn_transformer_scaled_dot_product_attention_0,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_permute_9 = ttnn.permute(
            v3_ttnn_slice_3,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v3_ttnn_reshape_199 = ttnn.reshape(
            v3_ttnn_permute_9,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v3_ttnn_matmul_2 = ttnn.matmul(
            v3_ttnn_reshape_199,
            self.weights[
                "image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v3_ttnn_add_2 = ttnn.add(
            v3_ttnn_matmul_2,
            self.cer["utils_constEvalFuncWrapper_124_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_3_0_0 = v3_ttnn_add_2
        v4_ttnn_add_3 = ttnn.add(
            LayerNorm_1_0_0,
            CLIPAttention_3_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v4_ttnn_layer_norm_3 = ttnn.layer_norm(
            v4_ttnn_add_3,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.0.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.0.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_163, v_164 = v4_ttnn_layer_norm_3, v4_ttnn_add_3
        v5_ttnn_reshape_200 = ttnn.reshape(
            v_163,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v5_ttnn_matmul_3 = ttnn.matmul(
            v5_ttnn_reshape_200,
            self.weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v5_ttnn_add_4 = ttnn.add(
            v5_ttnn_matmul_3,
            self.cer["utils_constEvalFuncWrapper_42_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v5_ttnn_gelu_0 = ttnn.gelu(
            v5_ttnn_add_4,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v5_ttnn_reshape_201 = ttnn.reshape(
            v5_ttnn_gelu_0,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v5_ttnn_matmul_4 = ttnn.matmul(
            v5_ttnn_reshape_201,
            self.weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v5_ttnn_add_5 = ttnn.add(
            v5_ttnn_matmul_4,
            self.cer["utils_constEvalFuncWrapper_73_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_5_0_0 = v5_ttnn_add_5
        v6_ttnn_add_6 = ttnn.add(
            v_164,
            CLIPMLP_5_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v6_ttnn_layer_norm_4 = ttnn.layer_norm(
            v6_ttnn_add_6,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.1.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.1.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_165, v_166 = v6_ttnn_add_6, v6_ttnn_layer_norm_4
        v7_ttnn_reshape_202 = ttnn.reshape(
            v_166,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_matmul_5 = ttnn.matmul(
            v7_ttnn_reshape_202,
            self.cer["utils_constEvalFuncWrapper_157_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v7_ttnn_add_7 = ttnn.add(
            v7_ttnn_matmul_5,
            self.cer["utils_constEvalFuncWrapper_62_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_slice_4 = ttnn.slice(
            v7_ttnn_add_7,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_slice_5 = ttnn.slice(
            v7_ttnn_add_7,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_slice_6 = ttnn.slice(
            v7_ttnn_add_7,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_reshape_203 = ttnn.reshape(
            v7_ttnn_slice_4,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_reshape_204 = ttnn.reshape(
            v7_ttnn_slice_5,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_reshape_205 = ttnn.reshape(
            v7_ttnn_slice_6,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_permute_10 = ttnn.permute(
            v7_ttnn_reshape_204,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v7_ttnn_permute_11 = ttnn.permute(
            v7_ttnn_reshape_205,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v7_ttnn_permute_12 = ttnn.permute(
            v7_ttnn_reshape_203,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v7_ttnn_pad_3 = ttnn.pad(
            v7_ttnn_permute_10,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_pad_4 = ttnn.pad(
            v7_ttnn_permute_11,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_pad_5 = ttnn.pad(
            v7_ttnn_permute_12,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_transformer_scaled_dot_product_attention_1 = (
            ttnn.transformer.scaled_dot_product_attention(
                v7_ttnn_pad_4,
                v7_ttnn_pad_3,
                v7_ttnn_pad_5,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v7_ttnn_slice_7 = ttnn.slice(
            v7_ttnn_transformer_scaled_dot_product_attention_1,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_permute_13 = ttnn.permute(
            v7_ttnn_slice_7,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v7_ttnn_reshape_206 = ttnn.reshape(
            v7_ttnn_permute_13,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v7_ttnn_matmul_6 = ttnn.matmul(
            v7_ttnn_reshape_206,
            self.weights[
                "image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v7_ttnn_add_8 = ttnn.add(
            v7_ttnn_matmul_6,
            self.cer["utils_constEvalFuncWrapper_55_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_7_0_0 = v7_ttnn_add_8
        v8_ttnn_add_9 = ttnn.add(
            v_165,
            CLIPAttention_7_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v8_ttnn_layer_norm_5 = ttnn.layer_norm(
            v8_ttnn_add_9,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.1.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.1.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_167, v_168 = v8_ttnn_add_9, v8_ttnn_layer_norm_5
        v9_ttnn_reshape_207 = ttnn.reshape(
            v_168,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v9_ttnn_matmul_7 = ttnn.matmul(
            v9_ttnn_reshape_207,
            self.weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v9_ttnn_add_10 = ttnn.add(
            v9_ttnn_matmul_7,
            self.cer["utils_constEvalFuncWrapper_13_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v9_ttnn_gelu_1 = ttnn.gelu(
            v9_ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v9_ttnn_reshape_208 = ttnn.reshape(
            v9_ttnn_gelu_1,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v9_ttnn_matmul_8 = ttnn.matmul(
            v9_ttnn_reshape_208,
            self.weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v9_ttnn_add_11 = ttnn.add(
            v9_ttnn_matmul_8,
            self.cer["utils_constEvalFuncWrapper_21_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_9_0_0 = v9_ttnn_add_11
        v10_ttnn_add_12 = ttnn.add(
            v_167,
            CLIPMLP_9_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v10_ttnn_layer_norm_6 = ttnn.layer_norm(
            v10_ttnn_add_12,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.2.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.2.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_169, v_170 = v10_ttnn_add_12, v10_ttnn_layer_norm_6
        v11_ttnn_reshape_209 = ttnn.reshape(
            v_170,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_matmul_9 = ttnn.matmul(
            v11_ttnn_reshape_209,
            self.cer["utils_constEvalFuncWrapper_25_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v11_ttnn_add_13 = ttnn.add(
            v11_ttnn_matmul_9,
            self.cer["utils_constEvalFuncWrapper_80_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_slice_8 = ttnn.slice(
            v11_ttnn_add_13,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_slice_9 = ttnn.slice(
            v11_ttnn_add_13,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_slice_10 = ttnn.slice(
            v11_ttnn_add_13,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_reshape_210 = ttnn.reshape(
            v11_ttnn_slice_8,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_reshape_211 = ttnn.reshape(
            v11_ttnn_slice_9,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_reshape_212 = ttnn.reshape(
            v11_ttnn_slice_10,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_permute_14 = ttnn.permute(
            v11_ttnn_reshape_211,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v11_ttnn_permute_15 = ttnn.permute(
            v11_ttnn_reshape_212,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v11_ttnn_permute_16 = ttnn.permute(
            v11_ttnn_reshape_210,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v11_ttnn_pad_6 = ttnn.pad(
            v11_ttnn_permute_14,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_pad_7 = ttnn.pad(
            v11_ttnn_permute_15,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_pad_8 = ttnn.pad(
            v11_ttnn_permute_16,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_transformer_scaled_dot_product_attention_2 = (
            ttnn.transformer.scaled_dot_product_attention(
                v11_ttnn_pad_7,
                v11_ttnn_pad_6,
                v11_ttnn_pad_8,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v11_ttnn_slice_11 = ttnn.slice(
            v11_ttnn_transformer_scaled_dot_product_attention_2,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_permute_17 = ttnn.permute(
            v11_ttnn_slice_11,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v11_ttnn_reshape_213 = ttnn.reshape(
            v11_ttnn_permute_17,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v11_ttnn_matmul_10 = ttnn.matmul(
            v11_ttnn_reshape_213,
            self.weights[
                "image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v11_ttnn_add_14 = ttnn.add(
            v11_ttnn_matmul_10,
            self.cer["utils_constEvalFuncWrapper_122_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_11_0_0 = v11_ttnn_add_14
        v12_ttnn_add_15 = ttnn.add(
            v_169,
            CLIPAttention_11_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v12_ttnn_layer_norm_7 = ttnn.layer_norm(
            v12_ttnn_add_15,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.2.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.2.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_171, v_172 = v12_ttnn_add_15, v12_ttnn_layer_norm_7
        v13_ttnn_reshape_214 = ttnn.reshape(
            v_172,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v13_ttnn_matmul_11 = ttnn.matmul(
            v13_ttnn_reshape_214,
            self.weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v13_ttnn_add_16 = ttnn.add(
            v13_ttnn_matmul_11,
            self.cer["utils_constEvalFuncWrapper_81_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v13_ttnn_gelu_2 = ttnn.gelu(
            v13_ttnn_add_16,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v13_ttnn_reshape_215 = ttnn.reshape(
            v13_ttnn_gelu_2,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v13_ttnn_matmul_12 = ttnn.matmul(
            v13_ttnn_reshape_215,
            self.weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v13_ttnn_add_17 = ttnn.add(
            v13_ttnn_matmul_12,
            self.cer["utils_constEvalFuncWrapper_146_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_13_0_0 = v13_ttnn_add_17
        v14_ttnn_add_18 = ttnn.add(
            v_171,
            CLIPMLP_13_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v14_ttnn_layer_norm_8 = ttnn.layer_norm(
            v14_ttnn_add_18,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.3.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.3.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_173, v_174 = v14_ttnn_add_18, v14_ttnn_layer_norm_8
        v15_ttnn_reshape_216 = ttnn.reshape(
            v_174,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_matmul_13 = ttnn.matmul(
            v15_ttnn_reshape_216,
            self.cer["utils_constEvalFuncWrapper_26_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v15_ttnn_add_19 = ttnn.add(
            v15_ttnn_matmul_13,
            self.cer["utils_constEvalFuncWrapper_90_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_slice_12 = ttnn.slice(
            v15_ttnn_add_19,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_slice_13 = ttnn.slice(
            v15_ttnn_add_19,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_slice_14 = ttnn.slice(
            v15_ttnn_add_19,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_reshape_217 = ttnn.reshape(
            v15_ttnn_slice_12,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_reshape_218 = ttnn.reshape(
            v15_ttnn_slice_13,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_reshape_219 = ttnn.reshape(
            v15_ttnn_slice_14,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_permute_18 = ttnn.permute(
            v15_ttnn_reshape_218,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v15_ttnn_permute_19 = ttnn.permute(
            v15_ttnn_reshape_219,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v15_ttnn_permute_20 = ttnn.permute(
            v15_ttnn_reshape_217,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v15_ttnn_pad_9 = ttnn.pad(
            v15_ttnn_permute_18,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_pad_10 = ttnn.pad(
            v15_ttnn_permute_19,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_pad_11 = ttnn.pad(
            v15_ttnn_permute_20,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_transformer_scaled_dot_product_attention_3 = (
            ttnn.transformer.scaled_dot_product_attention(
                v15_ttnn_pad_10,
                v15_ttnn_pad_9,
                v15_ttnn_pad_11,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v15_ttnn_slice_15 = ttnn.slice(
            v15_ttnn_transformer_scaled_dot_product_attention_3,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_permute_21 = ttnn.permute(
            v15_ttnn_slice_15,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v15_ttnn_reshape_220 = ttnn.reshape(
            v15_ttnn_permute_21,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v15_ttnn_matmul_14 = ttnn.matmul(
            v15_ttnn_reshape_220,
            self.weights[
                "image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v15_ttnn_add_20 = ttnn.add(
            v15_ttnn_matmul_14,
            self.cer["utils_constEvalFuncWrapper_132_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_15_0_0 = v15_ttnn_add_20
        v16_ttnn_add_21 = ttnn.add(
            v_173,
            CLIPAttention_15_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v16_ttnn_layer_norm_9 = ttnn.layer_norm(
            v16_ttnn_add_21,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.3.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.3.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_175, v_176 = v16_ttnn_layer_norm_9, v16_ttnn_add_21
        v17_ttnn_reshape_221 = ttnn.reshape(
            v_175,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v17_ttnn_matmul_15 = ttnn.matmul(
            v17_ttnn_reshape_221,
            self.weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v17_ttnn_add_22 = ttnn.add(
            v17_ttnn_matmul_15,
            self.cer["utils_constEvalFuncWrapper_145_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v17_ttnn_gelu_3 = ttnn.gelu(
            v17_ttnn_add_22,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v17_ttnn_reshape_222 = ttnn.reshape(
            v17_ttnn_gelu_3,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v17_ttnn_matmul_16 = ttnn.matmul(
            v17_ttnn_reshape_222,
            self.weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v17_ttnn_add_23 = ttnn.add(
            v17_ttnn_matmul_16,
            self.cer["utils_constEvalFuncWrapper_10_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_17_0_0 = v17_ttnn_add_23
        v18_ttnn_add_24 = ttnn.add(
            v_176,
            CLIPMLP_17_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v18_ttnn_layer_norm_10 = ttnn.layer_norm(
            v18_ttnn_add_24,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.4.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.4.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_177, v_178 = v18_ttnn_layer_norm_10, v18_ttnn_add_24
        v19_ttnn_reshape_223 = ttnn.reshape(
            v_177,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_matmul_17 = ttnn.matmul(
            v19_ttnn_reshape_223,
            self.cer["utils_constEvalFuncWrapper_127_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v19_ttnn_add_25 = ttnn.add(
            v19_ttnn_matmul_17,
            self.cer["utils_constEvalFuncWrapper_43_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_slice_16 = ttnn.slice(
            v19_ttnn_add_25,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_slice_17 = ttnn.slice(
            v19_ttnn_add_25,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_slice_18 = ttnn.slice(
            v19_ttnn_add_25,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_reshape_224 = ttnn.reshape(
            v19_ttnn_slice_16,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_reshape_225 = ttnn.reshape(
            v19_ttnn_slice_17,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_reshape_226 = ttnn.reshape(
            v19_ttnn_slice_18,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_permute_22 = ttnn.permute(
            v19_ttnn_reshape_225,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v19_ttnn_permute_23 = ttnn.permute(
            v19_ttnn_reshape_226,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v19_ttnn_permute_24 = ttnn.permute(
            v19_ttnn_reshape_224,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v19_ttnn_pad_12 = ttnn.pad(
            v19_ttnn_permute_22,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_pad_13 = ttnn.pad(
            v19_ttnn_permute_23,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_pad_14 = ttnn.pad(
            v19_ttnn_permute_24,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_transformer_scaled_dot_product_attention_4 = (
            ttnn.transformer.scaled_dot_product_attention(
                v19_ttnn_pad_13,
                v19_ttnn_pad_12,
                v19_ttnn_pad_14,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v19_ttnn_slice_19 = ttnn.slice(
            v19_ttnn_transformer_scaled_dot_product_attention_4,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_permute_25 = ttnn.permute(
            v19_ttnn_slice_19,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v19_ttnn_reshape_227 = ttnn.reshape(
            v19_ttnn_permute_25,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v19_ttnn_matmul_18 = ttnn.matmul(
            v19_ttnn_reshape_227,
            self.weights[
                "image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v19_ttnn_add_26 = ttnn.add(
            v19_ttnn_matmul_18,
            self.cer["utils_constEvalFuncWrapper_97_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_19_0_0 = v19_ttnn_add_26
        v20_ttnn_add_27 = ttnn.add(
            v_178,
            CLIPAttention_19_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v20_ttnn_layer_norm_11 = ttnn.layer_norm(
            v20_ttnn_add_27,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.4.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.4.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_179, v_180 = v20_ttnn_layer_norm_11, v20_ttnn_add_27
        v21_ttnn_reshape_228 = ttnn.reshape(
            v_179,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v21_ttnn_matmul_19 = ttnn.matmul(
            v21_ttnn_reshape_228,
            self.weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v21_ttnn_add_28 = ttnn.add(
            v21_ttnn_matmul_19,
            self.cer["utils_constEvalFuncWrapper_150_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v21_ttnn_gelu_4 = ttnn.gelu(
            v21_ttnn_add_28,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v21_ttnn_reshape_229 = ttnn.reshape(
            v21_ttnn_gelu_4,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v21_ttnn_matmul_20 = ttnn.matmul(
            v21_ttnn_reshape_229,
            self.weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v21_ttnn_add_29 = ttnn.add(
            v21_ttnn_matmul_20,
            self.cer["utils_constEvalFuncWrapper_149_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_21_0_0 = v21_ttnn_add_29
        v22_ttnn_add_30 = ttnn.add(
            v_180,
            CLIPMLP_21_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v22_ttnn_layer_norm_12 = ttnn.layer_norm(
            v22_ttnn_add_30,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.5.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.5.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_181, v_182 = v22_ttnn_add_30, v22_ttnn_layer_norm_12
        v23_ttnn_reshape_230 = ttnn.reshape(
            v_182,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_matmul_21 = ttnn.matmul(
            v23_ttnn_reshape_230,
            self.cer["utils_constEvalFuncWrapper_96_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v23_ttnn_add_31 = ttnn.add(
            v23_ttnn_matmul_21,
            self.cer["utils_constEvalFuncWrapper_158_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_slice_20 = ttnn.slice(
            v23_ttnn_add_31,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_slice_21 = ttnn.slice(
            v23_ttnn_add_31,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_slice_22 = ttnn.slice(
            v23_ttnn_add_31,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_reshape_231 = ttnn.reshape(
            v23_ttnn_slice_20,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_reshape_232 = ttnn.reshape(
            v23_ttnn_slice_21,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_reshape_233 = ttnn.reshape(
            v23_ttnn_slice_22,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_permute_26 = ttnn.permute(
            v23_ttnn_reshape_232,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v23_ttnn_permute_27 = ttnn.permute(
            v23_ttnn_reshape_233,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v23_ttnn_permute_28 = ttnn.permute(
            v23_ttnn_reshape_231,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v23_ttnn_pad_15 = ttnn.pad(
            v23_ttnn_permute_26,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_pad_16 = ttnn.pad(
            v23_ttnn_permute_27,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_pad_17 = ttnn.pad(
            v23_ttnn_permute_28,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_transformer_scaled_dot_product_attention_5 = (
            ttnn.transformer.scaled_dot_product_attention(
                v23_ttnn_pad_16,
                v23_ttnn_pad_15,
                v23_ttnn_pad_17,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v23_ttnn_slice_23 = ttnn.slice(
            v23_ttnn_transformer_scaled_dot_product_attention_5,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_permute_29 = ttnn.permute(
            v23_ttnn_slice_23,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v23_ttnn_reshape_234 = ttnn.reshape(
            v23_ttnn_permute_29,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v23_ttnn_matmul_22 = ttnn.matmul(
            v23_ttnn_reshape_234,
            self.weights[
                "image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v23_ttnn_add_32 = ttnn.add(
            v23_ttnn_matmul_22,
            self.cer["utils_constEvalFuncWrapper_69_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_23_0_0 = v23_ttnn_add_32
        v24_ttnn_add_33 = ttnn.add(
            v_181,
            CLIPAttention_23_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v24_ttnn_layer_norm_13 = ttnn.layer_norm(
            v24_ttnn_add_33,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.5.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.5.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_183, v_184 = v24_ttnn_layer_norm_13, v24_ttnn_add_33
        v25_ttnn_reshape_235 = ttnn.reshape(
            v_183,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v25_ttnn_matmul_23 = ttnn.matmul(
            v25_ttnn_reshape_235,
            self.weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v25_ttnn_add_34 = ttnn.add(
            v25_ttnn_matmul_23,
            self.cer["utils_constEvalFuncWrapper_91_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v25_ttnn_gelu_5 = ttnn.gelu(
            v25_ttnn_add_34,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v25_ttnn_reshape_236 = ttnn.reshape(
            v25_ttnn_gelu_5,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v25_ttnn_matmul_24 = ttnn.matmul(
            v25_ttnn_reshape_236,
            self.weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v25_ttnn_add_35 = ttnn.add(
            v25_ttnn_matmul_24,
            self.cer["utils_constEvalFuncWrapper_106_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_25_0_0 = v25_ttnn_add_35
        v26_ttnn_add_36 = ttnn.add(
            v_184,
            CLIPMLP_25_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v26_ttnn_layer_norm_14 = ttnn.layer_norm(
            v26_ttnn_add_36,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.6.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.6.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_185, v_186 = v26_ttnn_layer_norm_14, v26_ttnn_add_36
        v27_ttnn_reshape_237 = ttnn.reshape(
            v_185,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_matmul_25 = ttnn.matmul(
            v27_ttnn_reshape_237,
            self.cer["utils_constEvalFuncWrapper_128_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v27_ttnn_add_37 = ttnn.add(
            v27_ttnn_matmul_25,
            self.cer["utils_constEvalFuncWrapper_99_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_slice_24 = ttnn.slice(
            v27_ttnn_add_37,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_slice_25 = ttnn.slice(
            v27_ttnn_add_37,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_slice_26 = ttnn.slice(
            v27_ttnn_add_37,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_reshape_238 = ttnn.reshape(
            v27_ttnn_slice_24,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_reshape_239 = ttnn.reshape(
            v27_ttnn_slice_25,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_reshape_240 = ttnn.reshape(
            v27_ttnn_slice_26,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_permute_30 = ttnn.permute(
            v27_ttnn_reshape_239,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v27_ttnn_permute_31 = ttnn.permute(
            v27_ttnn_reshape_240,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v27_ttnn_permute_32 = ttnn.permute(
            v27_ttnn_reshape_238,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v27_ttnn_pad_18 = ttnn.pad(
            v27_ttnn_permute_30,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_pad_19 = ttnn.pad(
            v27_ttnn_permute_31,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_pad_20 = ttnn.pad(
            v27_ttnn_permute_32,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_transformer_scaled_dot_product_attention_6 = (
            ttnn.transformer.scaled_dot_product_attention(
                v27_ttnn_pad_19,
                v27_ttnn_pad_18,
                v27_ttnn_pad_20,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v27_ttnn_slice_27 = ttnn.slice(
            v27_ttnn_transformer_scaled_dot_product_attention_6,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_permute_33 = ttnn.permute(
            v27_ttnn_slice_27,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v27_ttnn_reshape_241 = ttnn.reshape(
            v27_ttnn_permute_33,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v27_ttnn_matmul_26 = ttnn.matmul(
            v27_ttnn_reshape_241,
            self.weights[
                "image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v27_ttnn_add_38 = ttnn.add(
            v27_ttnn_matmul_26,
            self.cer["utils_constEvalFuncWrapper_46_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_27_0_0 = v27_ttnn_add_38
        v28_ttnn_add_39 = ttnn.add(
            v_186,
            CLIPAttention_27_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v28_ttnn_layer_norm_15 = ttnn.layer_norm(
            v28_ttnn_add_39,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.6.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.6.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_187, v_188 = v28_ttnn_add_39, v28_ttnn_layer_norm_15
        v29_ttnn_reshape_242 = ttnn.reshape(
            v_188,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v29_ttnn_matmul_27 = ttnn.matmul(
            v29_ttnn_reshape_242,
            self.weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v29_ttnn_add_40 = ttnn.add(
            v29_ttnn_matmul_27,
            self.cer["utils_constEvalFuncWrapper_53_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v29_ttnn_gelu_6 = ttnn.gelu(
            v29_ttnn_add_40,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v29_ttnn_reshape_243 = ttnn.reshape(
            v29_ttnn_gelu_6,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v29_ttnn_matmul_28 = ttnn.matmul(
            v29_ttnn_reshape_243,
            self.weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v29_ttnn_add_41 = ttnn.add(
            v29_ttnn_matmul_28,
            self.cer["utils_constEvalFuncWrapper_103_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_29_0_0 = v29_ttnn_add_41
        v30_ttnn_add_42 = ttnn.add(
            v_187,
            CLIPMLP_29_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v30_ttnn_layer_norm_16 = ttnn.layer_norm(
            v30_ttnn_add_42,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.7.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.7.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_189, v_190 = v30_ttnn_layer_norm_16, v30_ttnn_add_42
        v31_ttnn_reshape_244 = ttnn.reshape(
            v_189,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_matmul_29 = ttnn.matmul(
            v31_ttnn_reshape_244,
            self.cer["utils_constEvalFuncWrapper_49_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v31_ttnn_add_43 = ttnn.add(
            v31_ttnn_matmul_29,
            self.cer["utils_constEvalFuncWrapper_84_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_slice_28 = ttnn.slice(
            v31_ttnn_add_43,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_slice_29 = ttnn.slice(
            v31_ttnn_add_43,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_slice_30 = ttnn.slice(
            v31_ttnn_add_43,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_reshape_245 = ttnn.reshape(
            v31_ttnn_slice_28,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_reshape_246 = ttnn.reshape(
            v31_ttnn_slice_29,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_reshape_247 = ttnn.reshape(
            v31_ttnn_slice_30,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_permute_34 = ttnn.permute(
            v31_ttnn_reshape_246,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v31_ttnn_permute_35 = ttnn.permute(
            v31_ttnn_reshape_247,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v31_ttnn_permute_36 = ttnn.permute(
            v31_ttnn_reshape_245,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v31_ttnn_pad_21 = ttnn.pad(
            v31_ttnn_permute_34,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_pad_22 = ttnn.pad(
            v31_ttnn_permute_35,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_pad_23 = ttnn.pad(
            v31_ttnn_permute_36,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_transformer_scaled_dot_product_attention_7 = (
            ttnn.transformer.scaled_dot_product_attention(
                v31_ttnn_pad_22,
                v31_ttnn_pad_21,
                v31_ttnn_pad_23,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v31_ttnn_slice_31 = ttnn.slice(
            v31_ttnn_transformer_scaled_dot_product_attention_7,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_permute_37 = ttnn.permute(
            v31_ttnn_slice_31,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v31_ttnn_reshape_248 = ttnn.reshape(
            v31_ttnn_permute_37,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v31_ttnn_matmul_30 = ttnn.matmul(
            v31_ttnn_reshape_248,
            self.weights[
                "image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v31_ttnn_add_44 = ttnn.add(
            v31_ttnn_matmul_30,
            self.cer["utils_constEvalFuncWrapper_120_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_31_0_0 = v31_ttnn_add_44
        v32_ttnn_add_45 = ttnn.add(
            v_190,
            CLIPAttention_31_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v32_ttnn_layer_norm_17 = ttnn.layer_norm(
            v32_ttnn_add_45,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.7.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.7.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_191, v_192 = v32_ttnn_add_45, v32_ttnn_layer_norm_17
        v33_ttnn_reshape_249 = ttnn.reshape(
            v_192,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v33_ttnn_matmul_31 = ttnn.matmul(
            v33_ttnn_reshape_249,
            self.weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v33_ttnn_add_46 = ttnn.add(
            v33_ttnn_matmul_31,
            self.cer["utils_constEvalFuncWrapper_27_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v33_ttnn_gelu_7 = ttnn.gelu(
            v33_ttnn_add_46,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v33_ttnn_reshape_250 = ttnn.reshape(
            v33_ttnn_gelu_7,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v33_ttnn_matmul_32 = ttnn.matmul(
            v33_ttnn_reshape_250,
            self.weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v33_ttnn_add_47 = ttnn.add(
            v33_ttnn_matmul_32,
            self.cer["utils_constEvalFuncWrapper_153_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_33_0_0 = v33_ttnn_add_47
        v34_ttnn_add_48 = ttnn.add(
            v_191,
            CLIPMLP_33_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v34_ttnn_layer_norm_18 = ttnn.layer_norm(
            v34_ttnn_add_48,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.8.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.8.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_193, v_194 = v34_ttnn_layer_norm_18, v34_ttnn_add_48
        v35_ttnn_reshape_251 = ttnn.reshape(
            v_193,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_matmul_33 = ttnn.matmul(
            v35_ttnn_reshape_251,
            self.cer["utils_constEvalFuncWrapper_29_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v35_ttnn_add_49 = ttnn.add(
            v35_ttnn_matmul_33,
            self.cer["utils_constEvalFuncWrapper_40_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_slice_32 = ttnn.slice(
            v35_ttnn_add_49,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_slice_33 = ttnn.slice(
            v35_ttnn_add_49,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_slice_34 = ttnn.slice(
            v35_ttnn_add_49,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_reshape_252 = ttnn.reshape(
            v35_ttnn_slice_32,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_reshape_253 = ttnn.reshape(
            v35_ttnn_slice_33,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_reshape_254 = ttnn.reshape(
            v35_ttnn_slice_34,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_permute_38 = ttnn.permute(
            v35_ttnn_reshape_253,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v35_ttnn_permute_39 = ttnn.permute(
            v35_ttnn_reshape_254,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v35_ttnn_permute_40 = ttnn.permute(
            v35_ttnn_reshape_252,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v35_ttnn_pad_24 = ttnn.pad(
            v35_ttnn_permute_38,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_pad_25 = ttnn.pad(
            v35_ttnn_permute_39,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_pad_26 = ttnn.pad(
            v35_ttnn_permute_40,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_transformer_scaled_dot_product_attention_8 = (
            ttnn.transformer.scaled_dot_product_attention(
                v35_ttnn_pad_25,
                v35_ttnn_pad_24,
                v35_ttnn_pad_26,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v35_ttnn_slice_35 = ttnn.slice(
            v35_ttnn_transformer_scaled_dot_product_attention_8,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_permute_41 = ttnn.permute(
            v35_ttnn_slice_35,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v35_ttnn_reshape_255 = ttnn.reshape(
            v35_ttnn_permute_41,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v35_ttnn_matmul_34 = ttnn.matmul(
            v35_ttnn_reshape_255,
            self.weights[
                "image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v35_ttnn_add_50 = ttnn.add(
            v35_ttnn_matmul_34,
            self.cer["utils_constEvalFuncWrapper_74_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_35_0_0 = v35_ttnn_add_50
        v36_ttnn_add_51 = ttnn.add(
            v_194,
            CLIPAttention_35_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v36_ttnn_layer_norm_19 = ttnn.layer_norm(
            v36_ttnn_add_51,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.8.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.8.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_195, v_196 = v36_ttnn_layer_norm_19, v36_ttnn_add_51
        v37_ttnn_reshape_256 = ttnn.reshape(
            v_195,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v37_ttnn_matmul_35 = ttnn.matmul(
            v37_ttnn_reshape_256,
            self.weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v37_ttnn_add_52 = ttnn.add(
            v37_ttnn_matmul_35,
            self.cer["utils_constEvalFuncWrapper_24_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v37_ttnn_gelu_8 = ttnn.gelu(
            v37_ttnn_add_52,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v37_ttnn_reshape_257 = ttnn.reshape(
            v37_ttnn_gelu_8,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v37_ttnn_matmul_36 = ttnn.matmul(
            v37_ttnn_reshape_257,
            self.weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v37_ttnn_add_53 = ttnn.add(
            v37_ttnn_matmul_36,
            self.cer["utils_constEvalFuncWrapper_93_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_37_0_0 = v37_ttnn_add_53
        v38_ttnn_add_54 = ttnn.add(
            v_196,
            CLIPMLP_37_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v38_ttnn_layer_norm_20 = ttnn.layer_norm(
            v38_ttnn_add_54,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.9.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.9.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_197, v_198 = v38_ttnn_layer_norm_20, v38_ttnn_add_54
        v39_ttnn_reshape_258 = ttnn.reshape(
            v_197,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_matmul_37 = ttnn.matmul(
            v39_ttnn_reshape_258,
            self.cer["utils_constEvalFuncWrapper_119_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v39_ttnn_add_55 = ttnn.add(
            v39_ttnn_matmul_37,
            self.cer["utils_constEvalFuncWrapper_133_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_slice_36 = ttnn.slice(
            v39_ttnn_add_55,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_slice_37 = ttnn.slice(
            v39_ttnn_add_55,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_slice_38 = ttnn.slice(
            v39_ttnn_add_55,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_reshape_259 = ttnn.reshape(
            v39_ttnn_slice_36,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_reshape_260 = ttnn.reshape(
            v39_ttnn_slice_37,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_reshape_261 = ttnn.reshape(
            v39_ttnn_slice_38,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_permute_42 = ttnn.permute(
            v39_ttnn_reshape_260,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v39_ttnn_permute_43 = ttnn.permute(
            v39_ttnn_reshape_261,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v39_ttnn_permute_44 = ttnn.permute(
            v39_ttnn_reshape_259,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v39_ttnn_pad_27 = ttnn.pad(
            v39_ttnn_permute_42,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_pad_28 = ttnn.pad(
            v39_ttnn_permute_43,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_pad_29 = ttnn.pad(
            v39_ttnn_permute_44,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_transformer_scaled_dot_product_attention_9 = (
            ttnn.transformer.scaled_dot_product_attention(
                v39_ttnn_pad_28,
                v39_ttnn_pad_27,
                v39_ttnn_pad_29,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v39_ttnn_slice_39 = ttnn.slice(
            v39_ttnn_transformer_scaled_dot_product_attention_9,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_permute_45 = ttnn.permute(
            v39_ttnn_slice_39,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v39_ttnn_reshape_262 = ttnn.reshape(
            v39_ttnn_permute_45,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v39_ttnn_matmul_38 = ttnn.matmul(
            v39_ttnn_reshape_262,
            self.weights[
                "image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v39_ttnn_add_56 = ttnn.add(
            v39_ttnn_matmul_38,
            self.cer["utils_constEvalFuncWrapper_113_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_39_0_0 = v39_ttnn_add_56
        v40_ttnn_add_57 = ttnn.add(
            v_198,
            CLIPAttention_39_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v40_ttnn_layer_norm_21 = ttnn.layer_norm(
            v40_ttnn_add_57,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.9.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.9.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_199, v_200 = v40_ttnn_add_57, v40_ttnn_layer_norm_21
        v41_ttnn_reshape_263 = ttnn.reshape(
            v_200,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v41_ttnn_matmul_39 = ttnn.matmul(
            v41_ttnn_reshape_263,
            self.weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v41_ttnn_add_58 = ttnn.add(
            v41_ttnn_matmul_39,
            self.cer["utils_constEvalFuncWrapper_155_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v41_ttnn_gelu_9 = ttnn.gelu(
            v41_ttnn_add_58,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v41_ttnn_reshape_264 = ttnn.reshape(
            v41_ttnn_gelu_9,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v41_ttnn_matmul_40 = ttnn.matmul(
            v41_ttnn_reshape_264,
            self.weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v41_ttnn_add_59 = ttnn.add(
            v41_ttnn_matmul_40,
            self.cer["utils_constEvalFuncWrapper_2_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_41_0_0 = v41_ttnn_add_59
        v42_ttnn_add_60 = ttnn.add(
            v_199,
            CLIPMLP_41_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v42_ttnn_layer_norm_22 = ttnn.layer_norm(
            v42_ttnn_add_60,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.10.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.10.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_201, v_202 = v42_ttnn_add_60, v42_ttnn_layer_norm_22
        v43_ttnn_reshape_265 = ttnn.reshape(
            v_202,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_matmul_41 = ttnn.matmul(
            v43_ttnn_reshape_265,
            self.cer["utils_constEvalFuncWrapper_152_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v43_ttnn_add_61 = ttnn.add(
            v43_ttnn_matmul_41,
            self.cer["utils_constEvalFuncWrapper_71_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_slice_40 = ttnn.slice(
            v43_ttnn_add_61,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_slice_41 = ttnn.slice(
            v43_ttnn_add_61,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_slice_42 = ttnn.slice(
            v43_ttnn_add_61,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_reshape_266 = ttnn.reshape(
            v43_ttnn_slice_40,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_reshape_267 = ttnn.reshape(
            v43_ttnn_slice_41,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_reshape_268 = ttnn.reshape(
            v43_ttnn_slice_42,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_permute_46 = ttnn.permute(
            v43_ttnn_reshape_267,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v43_ttnn_permute_47 = ttnn.permute(
            v43_ttnn_reshape_268,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v43_ttnn_permute_48 = ttnn.permute(
            v43_ttnn_reshape_266,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v43_ttnn_pad_30 = ttnn.pad(
            v43_ttnn_permute_46,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_pad_31 = ttnn.pad(
            v43_ttnn_permute_47,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_pad_32 = ttnn.pad(
            v43_ttnn_permute_48,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_transformer_scaled_dot_product_attention_10 = (
            ttnn.transformer.scaled_dot_product_attention(
                v43_ttnn_pad_31,
                v43_ttnn_pad_30,
                v43_ttnn_pad_32,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v43_ttnn_slice_43 = ttnn.slice(
            v43_ttnn_transformer_scaled_dot_product_attention_10,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_permute_49 = ttnn.permute(
            v43_ttnn_slice_43,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v43_ttnn_reshape_269 = ttnn.reshape(
            v43_ttnn_permute_49,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v43_ttnn_matmul_42 = ttnn.matmul(
            v43_ttnn_reshape_269,
            self.weights[
                "image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v43_ttnn_add_62 = ttnn.add(
            v43_ttnn_matmul_42,
            self.cer["utils_constEvalFuncWrapper_64_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_43_0_0 = v43_ttnn_add_62
        v44_ttnn_add_63 = ttnn.add(
            v_201,
            CLIPAttention_43_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v44_ttnn_layer_norm_23 = ttnn.layer_norm(
            v44_ttnn_add_63,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.10.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.10.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_203, v_204 = v44_ttnn_add_63, v44_ttnn_layer_norm_23
        v45_ttnn_reshape_270 = ttnn.reshape(
            v_204,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v45_ttnn_matmul_43 = ttnn.matmul(
            v45_ttnn_reshape_270,
            self.weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v45_ttnn_add_64 = ttnn.add(
            v45_ttnn_matmul_43,
            self.cer["utils_constEvalFuncWrapper_95_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v45_ttnn_gelu_10 = ttnn.gelu(
            v45_ttnn_add_64,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v45_ttnn_reshape_271 = ttnn.reshape(
            v45_ttnn_gelu_10,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v45_ttnn_matmul_44 = ttnn.matmul(
            v45_ttnn_reshape_271,
            self.weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v45_ttnn_add_65 = ttnn.add(
            v45_ttnn_matmul_44,
            self.cer["utils_constEvalFuncWrapper_85_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_45_0_0 = v45_ttnn_add_65
        v46_ttnn_add_66 = ttnn.add(
            v_203,
            CLIPMLP_45_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v46_ttnn_layer_norm_24 = ttnn.layer_norm(
            v46_ttnn_add_66,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.11.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.11.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_205, v_206 = v46_ttnn_layer_norm_24, v46_ttnn_add_66
        v47_ttnn_reshape_272 = ttnn.reshape(
            v_205,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_matmul_45 = ttnn.matmul(
            v47_ttnn_reshape_272,
            self.cer["utils_constEvalFuncWrapper_67_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v47_ttnn_add_67 = ttnn.add(
            v47_ttnn_matmul_45,
            self.cer["utils_constEvalFuncWrapper_116_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_slice_44 = ttnn.slice(
            v47_ttnn_add_67,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_slice_45 = ttnn.slice(
            v47_ttnn_add_67,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_slice_46 = ttnn.slice(
            v47_ttnn_add_67,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_reshape_273 = ttnn.reshape(
            v47_ttnn_slice_44,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_reshape_274 = ttnn.reshape(
            v47_ttnn_slice_45,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_reshape_275 = ttnn.reshape(
            v47_ttnn_slice_46,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_permute_50 = ttnn.permute(
            v47_ttnn_reshape_274,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v47_ttnn_permute_51 = ttnn.permute(
            v47_ttnn_reshape_275,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v47_ttnn_permute_52 = ttnn.permute(
            v47_ttnn_reshape_273,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v47_ttnn_pad_33 = ttnn.pad(
            v47_ttnn_permute_50,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_pad_34 = ttnn.pad(
            v47_ttnn_permute_51,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_pad_35 = ttnn.pad(
            v47_ttnn_permute_52,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_transformer_scaled_dot_product_attention_11 = (
            ttnn.transformer.scaled_dot_product_attention(
                v47_ttnn_pad_34,
                v47_ttnn_pad_33,
                v47_ttnn_pad_35,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v47_ttnn_slice_47 = ttnn.slice(
            v47_ttnn_transformer_scaled_dot_product_attention_11,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_permute_53 = ttnn.permute(
            v47_ttnn_slice_47,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v47_ttnn_reshape_276 = ttnn.reshape(
            v47_ttnn_permute_53,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v47_ttnn_matmul_46 = ttnn.matmul(
            v47_ttnn_reshape_276,
            self.weights[
                "image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v47_ttnn_add_68 = ttnn.add(
            v47_ttnn_matmul_46,
            self.cer["utils_constEvalFuncWrapper_140_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_47_0_0 = v47_ttnn_add_68
        v48_ttnn_add_69 = ttnn.add(
            v_206,
            CLIPAttention_47_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v48_ttnn_layer_norm_25 = ttnn.layer_norm(
            v48_ttnn_add_69,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.11.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.11.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_207, v_208 = v48_ttnn_layer_norm_25, v48_ttnn_add_69
        v49_ttnn_reshape_277 = ttnn.reshape(
            v_207,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v49_ttnn_matmul_47 = ttnn.matmul(
            v49_ttnn_reshape_277,
            self.weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v49_ttnn_add_70 = ttnn.add(
            v49_ttnn_matmul_47,
            self.cer["utils_constEvalFuncWrapper_156_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v49_ttnn_gelu_11 = ttnn.gelu(
            v49_ttnn_add_70,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v49_ttnn_reshape_278 = ttnn.reshape(
            v49_ttnn_gelu_11,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v49_ttnn_matmul_48 = ttnn.matmul(
            v49_ttnn_reshape_278,
            self.weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v49_ttnn_add_71 = ttnn.add(
            v49_ttnn_matmul_48,
            self.cer["utils_constEvalFuncWrapper_151_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_49_0_0 = v49_ttnn_add_71
        v50_ttnn_add_72 = ttnn.add(
            v_208,
            CLIPMLP_49_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v50_ttnn_layer_norm_26 = ttnn.layer_norm(
            v50_ttnn_add_72,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.12.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.12.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_209, v_210 = v50_ttnn_layer_norm_26, v50_ttnn_add_72
        v51_ttnn_reshape_279 = ttnn.reshape(
            v_209,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_matmul_49 = ttnn.matmul(
            v51_ttnn_reshape_279,
            self.cer["utils_constEvalFuncWrapper_87_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v51_ttnn_add_73 = ttnn.add(
            v51_ttnn_matmul_49,
            self.cer["utils_constEvalFuncWrapper_136_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_slice_48 = ttnn.slice(
            v51_ttnn_add_73,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_slice_49 = ttnn.slice(
            v51_ttnn_add_73,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_slice_50 = ttnn.slice(
            v51_ttnn_add_73,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_reshape_280 = ttnn.reshape(
            v51_ttnn_slice_48,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_reshape_281 = ttnn.reshape(
            v51_ttnn_slice_49,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_reshape_282 = ttnn.reshape(
            v51_ttnn_slice_50,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_permute_54 = ttnn.permute(
            v51_ttnn_reshape_281,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v51_ttnn_permute_55 = ttnn.permute(
            v51_ttnn_reshape_282,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v51_ttnn_permute_56 = ttnn.permute(
            v51_ttnn_reshape_280,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v51_ttnn_pad_36 = ttnn.pad(
            v51_ttnn_permute_54,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_pad_37 = ttnn.pad(
            v51_ttnn_permute_55,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_pad_38 = ttnn.pad(
            v51_ttnn_permute_56,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_transformer_scaled_dot_product_attention_12 = (
            ttnn.transformer.scaled_dot_product_attention(
                v51_ttnn_pad_37,
                v51_ttnn_pad_36,
                v51_ttnn_pad_38,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v51_ttnn_slice_51 = ttnn.slice(
            v51_ttnn_transformer_scaled_dot_product_attention_12,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_permute_57 = ttnn.permute(
            v51_ttnn_slice_51,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v51_ttnn_reshape_283 = ttnn.reshape(
            v51_ttnn_permute_57,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v51_ttnn_matmul_50 = ttnn.matmul(
            v51_ttnn_reshape_283,
            self.weights[
                "image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v51_ttnn_add_74 = ttnn.add(
            v51_ttnn_matmul_50,
            self.cer["utils_constEvalFuncWrapper_68_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_51_0_0 = v51_ttnn_add_74
        v52_ttnn_add_75 = ttnn.add(
            v_210,
            CLIPAttention_51_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v52_ttnn_layer_norm_27 = ttnn.layer_norm(
            v52_ttnn_add_75,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.12.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.12.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_211, v_212 = v52_ttnn_layer_norm_27, v52_ttnn_add_75
        v53_ttnn_reshape_284 = ttnn.reshape(
            v_211,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v53_ttnn_matmul_51 = ttnn.matmul(
            v53_ttnn_reshape_284,
            self.weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v53_ttnn_add_76 = ttnn.add(
            v53_ttnn_matmul_51,
            self.cer["utils_constEvalFuncWrapper_5_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v53_ttnn_gelu_12 = ttnn.gelu(
            v53_ttnn_add_76,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v53_ttnn_reshape_285 = ttnn.reshape(
            v53_ttnn_gelu_12,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v53_ttnn_matmul_52 = ttnn.matmul(
            v53_ttnn_reshape_285,
            self.weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v53_ttnn_add_77 = ttnn.add(
            v53_ttnn_matmul_52,
            self.cer["utils_constEvalFuncWrapper_15_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_53_0_0 = v53_ttnn_add_77
        v54_ttnn_add_78 = ttnn.add(
            v_212,
            CLIPMLP_53_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v54_ttnn_layer_norm_28 = ttnn.layer_norm(
            v54_ttnn_add_78,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.13.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.13.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_213, v_214 = v54_ttnn_add_78, v54_ttnn_layer_norm_28
        v55_ttnn_reshape_286 = ttnn.reshape(
            v_214,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_matmul_53 = ttnn.matmul(
            v55_ttnn_reshape_286,
            self.cer["utils_constEvalFuncWrapper_102_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v55_ttnn_add_79 = ttnn.add(
            v55_ttnn_matmul_53,
            self.cer["utils_constEvalFuncWrapper_1_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_slice_52 = ttnn.slice(
            v55_ttnn_add_79,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_slice_53 = ttnn.slice(
            v55_ttnn_add_79,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_slice_54 = ttnn.slice(
            v55_ttnn_add_79,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_reshape_287 = ttnn.reshape(
            v55_ttnn_slice_52,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_reshape_288 = ttnn.reshape(
            v55_ttnn_slice_53,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_reshape_289 = ttnn.reshape(
            v55_ttnn_slice_54,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_permute_58 = ttnn.permute(
            v55_ttnn_reshape_288,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v55_ttnn_permute_59 = ttnn.permute(
            v55_ttnn_reshape_289,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v55_ttnn_permute_60 = ttnn.permute(
            v55_ttnn_reshape_287,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v55_ttnn_pad_39 = ttnn.pad(
            v55_ttnn_permute_58,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_pad_40 = ttnn.pad(
            v55_ttnn_permute_59,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_pad_41 = ttnn.pad(
            v55_ttnn_permute_60,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_transformer_scaled_dot_product_attention_13 = (
            ttnn.transformer.scaled_dot_product_attention(
                v55_ttnn_pad_40,
                v55_ttnn_pad_39,
                v55_ttnn_pad_41,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v55_ttnn_slice_55 = ttnn.slice(
            v55_ttnn_transformer_scaled_dot_product_attention_13,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_permute_61 = ttnn.permute(
            v55_ttnn_slice_55,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v55_ttnn_reshape_290 = ttnn.reshape(
            v55_ttnn_permute_61,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v55_ttnn_matmul_54 = ttnn.matmul(
            v55_ttnn_reshape_290,
            self.weights[
                "image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v55_ttnn_add_80 = ttnn.add(
            v55_ttnn_matmul_54,
            self.cer["utils_constEvalFuncWrapper_126_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_55_0_0 = v55_ttnn_add_80
        v56_ttnn_add_81 = ttnn.add(
            v_213,
            CLIPAttention_55_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v56_ttnn_layer_norm_29 = ttnn.layer_norm(
            v56_ttnn_add_81,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.13.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.13.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_215, v_216 = v56_ttnn_add_81, v56_ttnn_layer_norm_29
        v57_ttnn_reshape_291 = ttnn.reshape(
            v_216,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v57_ttnn_matmul_55 = ttnn.matmul(
            v57_ttnn_reshape_291,
            self.weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v57_ttnn_add_82 = ttnn.add(
            v57_ttnn_matmul_55,
            self.cer["utils_constEvalFuncWrapper_92_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v57_ttnn_gelu_13 = ttnn.gelu(
            v57_ttnn_add_82,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v57_ttnn_reshape_292 = ttnn.reshape(
            v57_ttnn_gelu_13,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v57_ttnn_matmul_56 = ttnn.matmul(
            v57_ttnn_reshape_292,
            self.weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v57_ttnn_add_83 = ttnn.add(
            v57_ttnn_matmul_56,
            self.cer["utils_constEvalFuncWrapper_109_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_57_0_0 = v57_ttnn_add_83
        v58_ttnn_add_84 = ttnn.add(
            v_215,
            CLIPMLP_57_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v58_ttnn_layer_norm_30 = ttnn.layer_norm(
            v58_ttnn_add_84,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.14.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.14.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_217, v_218 = v58_ttnn_layer_norm_30, v58_ttnn_add_84
        v59_ttnn_reshape_293 = ttnn.reshape(
            v_217,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_matmul_57 = ttnn.matmul(
            v59_ttnn_reshape_293,
            self.cer["utils_constEvalFuncWrapper_86_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v59_ttnn_add_85 = ttnn.add(
            v59_ttnn_matmul_57,
            self.cer["utils_constEvalFuncWrapper_101_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_slice_56 = ttnn.slice(
            v59_ttnn_add_85,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_slice_57 = ttnn.slice(
            v59_ttnn_add_85,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_slice_58 = ttnn.slice(
            v59_ttnn_add_85,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_reshape_294 = ttnn.reshape(
            v59_ttnn_slice_56,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_reshape_295 = ttnn.reshape(
            v59_ttnn_slice_57,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_reshape_296 = ttnn.reshape(
            v59_ttnn_slice_58,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_permute_62 = ttnn.permute(
            v59_ttnn_reshape_295,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v59_ttnn_permute_63 = ttnn.permute(
            v59_ttnn_reshape_296,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v59_ttnn_permute_64 = ttnn.permute(
            v59_ttnn_reshape_294,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v59_ttnn_pad_42 = ttnn.pad(
            v59_ttnn_permute_62,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_pad_43 = ttnn.pad(
            v59_ttnn_permute_63,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_pad_44 = ttnn.pad(
            v59_ttnn_permute_64,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_transformer_scaled_dot_product_attention_14 = (
            ttnn.transformer.scaled_dot_product_attention(
                v59_ttnn_pad_43,
                v59_ttnn_pad_42,
                v59_ttnn_pad_44,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v59_ttnn_slice_59 = ttnn.slice(
            v59_ttnn_transformer_scaled_dot_product_attention_14,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_permute_65 = ttnn.permute(
            v59_ttnn_slice_59,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v59_ttnn_reshape_297 = ttnn.reshape(
            v59_ttnn_permute_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v59_ttnn_matmul_58 = ttnn.matmul(
            v59_ttnn_reshape_297,
            self.weights[
                "image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v59_ttnn_add_86 = ttnn.add(
            v59_ttnn_matmul_58,
            self.cer["utils_constEvalFuncWrapper_11_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_59_0_0 = v59_ttnn_add_86
        v60_ttnn_add_87 = ttnn.add(
            v_218,
            CLIPAttention_59_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v60_ttnn_layer_norm_31 = ttnn.layer_norm(
            v60_ttnn_add_87,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.14.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.14.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_219, v_220 = v60_ttnn_add_87, v60_ttnn_layer_norm_31
        v61_ttnn_reshape_298 = ttnn.reshape(
            v_220,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v61_ttnn_matmul_59 = ttnn.matmul(
            v61_ttnn_reshape_298,
            self.weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v61_ttnn_add_88 = ttnn.add(
            v61_ttnn_matmul_59,
            self.cer["utils_constEvalFuncWrapper_141_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v61_ttnn_gelu_14 = ttnn.gelu(
            v61_ttnn_add_88,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v61_ttnn_reshape_299 = ttnn.reshape(
            v61_ttnn_gelu_14,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v61_ttnn_matmul_60 = ttnn.matmul(
            v61_ttnn_reshape_299,
            self.weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v61_ttnn_add_89 = ttnn.add(
            v61_ttnn_matmul_60,
            self.cer["utils_constEvalFuncWrapper_18_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_61_0_0 = v61_ttnn_add_89
        v62_ttnn_add_90 = ttnn.add(
            v_219,
            CLIPMLP_61_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v62_ttnn_layer_norm_32 = ttnn.layer_norm(
            v62_ttnn_add_90,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.15.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.15.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_221, v_222 = v62_ttnn_layer_norm_32, v62_ttnn_add_90
        v63_ttnn_reshape_300 = ttnn.reshape(
            v_221,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_matmul_61 = ttnn.matmul(
            v63_ttnn_reshape_300,
            self.cer["utils_constEvalFuncWrapper_23_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v63_ttnn_add_91 = ttnn.add(
            v63_ttnn_matmul_61,
            self.cer["utils_constEvalFuncWrapper_72_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_slice_60 = ttnn.slice(
            v63_ttnn_add_91,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_slice_61 = ttnn.slice(
            v63_ttnn_add_91,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_slice_62 = ttnn.slice(
            v63_ttnn_add_91,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_reshape_301 = ttnn.reshape(
            v63_ttnn_slice_60,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_reshape_302 = ttnn.reshape(
            v63_ttnn_slice_61,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_reshape_303 = ttnn.reshape(
            v63_ttnn_slice_62,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_permute_66 = ttnn.permute(
            v63_ttnn_reshape_302,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v63_ttnn_permute_67 = ttnn.permute(
            v63_ttnn_reshape_303,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v63_ttnn_permute_68 = ttnn.permute(
            v63_ttnn_reshape_301,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v63_ttnn_pad_45 = ttnn.pad(
            v63_ttnn_permute_66,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_pad_46 = ttnn.pad(
            v63_ttnn_permute_67,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_pad_47 = ttnn.pad(
            v63_ttnn_permute_68,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_transformer_scaled_dot_product_attention_15 = (
            ttnn.transformer.scaled_dot_product_attention(
                v63_ttnn_pad_46,
                v63_ttnn_pad_45,
                v63_ttnn_pad_47,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v63_ttnn_slice_63 = ttnn.slice(
            v63_ttnn_transformer_scaled_dot_product_attention_15,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_permute_69 = ttnn.permute(
            v63_ttnn_slice_63,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v63_ttnn_reshape_304 = ttnn.reshape(
            v63_ttnn_permute_69,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v63_ttnn_matmul_62 = ttnn.matmul(
            v63_ttnn_reshape_304,
            self.weights[
                "image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v63_ttnn_add_92 = ttnn.add(
            v63_ttnn_matmul_62,
            self.cer["utils_constEvalFuncWrapper_114_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_63_0_0 = v63_ttnn_add_92
        v64_ttnn_add_93 = ttnn.add(
            v_222,
            CLIPAttention_63_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v64_ttnn_layer_norm_33 = ttnn.layer_norm(
            v64_ttnn_add_93,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.15.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.15.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_223, v_224 = v64_ttnn_add_93, v64_ttnn_layer_norm_33
        v65_ttnn_reshape_305 = ttnn.reshape(
            v_224,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v65_ttnn_matmul_63 = ttnn.matmul(
            v65_ttnn_reshape_305,
            self.weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v65_ttnn_add_94 = ttnn.add(
            v65_ttnn_matmul_63,
            self.cer["utils_constEvalFuncWrapper_154_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v65_ttnn_gelu_15 = ttnn.gelu(
            v65_ttnn_add_94,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v65_ttnn_reshape_306 = ttnn.reshape(
            v65_ttnn_gelu_15,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v65_ttnn_matmul_64 = ttnn.matmul(
            v65_ttnn_reshape_306,
            self.weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v65_ttnn_add_95 = ttnn.add(
            v65_ttnn_matmul_64,
            self.cer["utils_constEvalFuncWrapper_83_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_65_0_0 = v65_ttnn_add_95
        v66_ttnn_add_96 = ttnn.add(
            v_223,
            CLIPMLP_65_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v66_ttnn_layer_norm_34 = ttnn.layer_norm(
            v66_ttnn_add_96,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.16.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.16.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_225, v_226 = v66_ttnn_add_96, v66_ttnn_layer_norm_34
        v67_ttnn_reshape_307 = ttnn.reshape(
            v_226,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_matmul_65 = ttnn.matmul(
            v67_ttnn_reshape_307,
            self.cer["utils_constEvalFuncWrapper_89_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v67_ttnn_add_97 = ttnn.add(
            v67_ttnn_matmul_65,
            self.cer["utils_constEvalFuncWrapper_118_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_slice_64 = ttnn.slice(
            v67_ttnn_add_97,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_slice_65 = ttnn.slice(
            v67_ttnn_add_97,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_slice_66 = ttnn.slice(
            v67_ttnn_add_97,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_reshape_308 = ttnn.reshape(
            v67_ttnn_slice_64,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_reshape_309 = ttnn.reshape(
            v67_ttnn_slice_65,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_reshape_310 = ttnn.reshape(
            v67_ttnn_slice_66,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_permute_70 = ttnn.permute(
            v67_ttnn_reshape_309,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v67_ttnn_permute_71 = ttnn.permute(
            v67_ttnn_reshape_310,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v67_ttnn_permute_72 = ttnn.permute(
            v67_ttnn_reshape_308,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v67_ttnn_pad_48 = ttnn.pad(
            v67_ttnn_permute_70,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_pad_49 = ttnn.pad(
            v67_ttnn_permute_71,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_pad_50 = ttnn.pad(
            v67_ttnn_permute_72,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_transformer_scaled_dot_product_attention_16 = (
            ttnn.transformer.scaled_dot_product_attention(
                v67_ttnn_pad_49,
                v67_ttnn_pad_48,
                v67_ttnn_pad_50,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v67_ttnn_slice_67 = ttnn.slice(
            v67_ttnn_transformer_scaled_dot_product_attention_16,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_permute_73 = ttnn.permute(
            v67_ttnn_slice_67,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v67_ttnn_reshape_311 = ttnn.reshape(
            v67_ttnn_permute_73,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v67_ttnn_matmul_66 = ttnn.matmul(
            v67_ttnn_reshape_311,
            self.weights[
                "image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v67_ttnn_add_98 = ttnn.add(
            v67_ttnn_matmul_66,
            self.cer["utils_constEvalFuncWrapper_63_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_67_0_0 = v67_ttnn_add_98
        v68_ttnn_add_99 = ttnn.add(
            v_225,
            CLIPAttention_67_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v68_ttnn_layer_norm_35 = ttnn.layer_norm(
            v68_ttnn_add_99,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.16.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.16.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_227, v_228 = v68_ttnn_add_99, v68_ttnn_layer_norm_35
        v69_ttnn_reshape_312 = ttnn.reshape(
            v_228,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v69_ttnn_matmul_67 = ttnn.matmul(
            v69_ttnn_reshape_312,
            self.weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v69_ttnn_add_100 = ttnn.add(
            v69_ttnn_matmul_67,
            self.cer["utils_constEvalFuncWrapper_130_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v69_ttnn_gelu_16 = ttnn.gelu(
            v69_ttnn_add_100,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v69_ttnn_reshape_313 = ttnn.reshape(
            v69_ttnn_gelu_16,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v69_ttnn_matmul_68 = ttnn.matmul(
            v69_ttnn_reshape_313,
            self.weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v69_ttnn_add_101 = ttnn.add(
            v69_ttnn_matmul_68,
            self.cer["utils_constEvalFuncWrapper_104_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_69_0_0 = v69_ttnn_add_101
        v70_ttnn_add_102 = ttnn.add(
            v_227,
            CLIPMLP_69_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v70_ttnn_layer_norm_36 = ttnn.layer_norm(
            v70_ttnn_add_102,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.17.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.17.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_229, v_230 = v70_ttnn_add_102, v70_ttnn_layer_norm_36
        v71_ttnn_reshape_314 = ttnn.reshape(
            v_230,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_matmul_69 = ttnn.matmul(
            v71_ttnn_reshape_314,
            self.cer["utils_constEvalFuncWrapper_34_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v71_ttnn_add_103 = ttnn.add(
            v71_ttnn_matmul_69,
            self.cer["utils_constEvalFuncWrapper_17_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_slice_68 = ttnn.slice(
            v71_ttnn_add_103,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_slice_69 = ttnn.slice(
            v71_ttnn_add_103,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_slice_70 = ttnn.slice(
            v71_ttnn_add_103,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_reshape_315 = ttnn.reshape(
            v71_ttnn_slice_68,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_reshape_316 = ttnn.reshape(
            v71_ttnn_slice_69,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_reshape_317 = ttnn.reshape(
            v71_ttnn_slice_70,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_permute_74 = ttnn.permute(
            v71_ttnn_reshape_316,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v71_ttnn_permute_75 = ttnn.permute(
            v71_ttnn_reshape_317,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v71_ttnn_permute_76 = ttnn.permute(
            v71_ttnn_reshape_315,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v71_ttnn_pad_51 = ttnn.pad(
            v71_ttnn_permute_74,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_pad_52 = ttnn.pad(
            v71_ttnn_permute_75,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_pad_53 = ttnn.pad(
            v71_ttnn_permute_76,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_transformer_scaled_dot_product_attention_17 = (
            ttnn.transformer.scaled_dot_product_attention(
                v71_ttnn_pad_52,
                v71_ttnn_pad_51,
                v71_ttnn_pad_53,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v71_ttnn_slice_71 = ttnn.slice(
            v71_ttnn_transformer_scaled_dot_product_attention_17,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_permute_77 = ttnn.permute(
            v71_ttnn_slice_71,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v71_ttnn_reshape_318 = ttnn.reshape(
            v71_ttnn_permute_77,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v71_ttnn_matmul_70 = ttnn.matmul(
            v71_ttnn_reshape_318,
            self.weights[
                "image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v71_ttnn_add_104 = ttnn.add(
            v71_ttnn_matmul_70,
            self.cer["utils_constEvalFuncWrapper_7_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_71_0_0 = v71_ttnn_add_104
        v72_ttnn_add_105 = ttnn.add(
            v_229,
            CLIPAttention_71_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v72_ttnn_layer_norm_37 = ttnn.layer_norm(
            v72_ttnn_add_105,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.17.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.17.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_231, v_232 = v72_ttnn_layer_norm_37, v72_ttnn_add_105
        v73_ttnn_reshape_319 = ttnn.reshape(
            v_231,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v73_ttnn_matmul_71 = ttnn.matmul(
            v73_ttnn_reshape_319,
            self.weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v73_ttnn_add_106 = ttnn.add(
            v73_ttnn_matmul_71,
            self.cer["utils_constEvalFuncWrapper_108_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v73_ttnn_gelu_17 = ttnn.gelu(
            v73_ttnn_add_106,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v73_ttnn_reshape_320 = ttnn.reshape(
            v73_ttnn_gelu_17,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v73_ttnn_matmul_72 = ttnn.matmul(
            v73_ttnn_reshape_320,
            self.weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v73_ttnn_add_107 = ttnn.add(
            v73_ttnn_matmul_72,
            self.cer["utils_constEvalFuncWrapper_19_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_73_0_0 = v73_ttnn_add_107
        v74_ttnn_add_108 = ttnn.add(
            v_232,
            CLIPMLP_73_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v74_ttnn_layer_norm_38 = ttnn.layer_norm(
            v74_ttnn_add_108,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.18.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.18.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_233, v_234 = v74_ttnn_layer_norm_38, v74_ttnn_add_108
        v75_ttnn_reshape_321 = ttnn.reshape(
            v_233,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_matmul_73 = ttnn.matmul(
            v75_ttnn_reshape_321,
            self.cer["utils_constEvalFuncWrapper_112_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v75_ttnn_add_109 = ttnn.add(
            v75_ttnn_matmul_73,
            self.cer["utils_constEvalFuncWrapper_134_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_slice_72 = ttnn.slice(
            v75_ttnn_add_109,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_slice_73 = ttnn.slice(
            v75_ttnn_add_109,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_slice_74 = ttnn.slice(
            v75_ttnn_add_109,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_reshape_322 = ttnn.reshape(
            v75_ttnn_slice_72,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_reshape_323 = ttnn.reshape(
            v75_ttnn_slice_73,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_reshape_324 = ttnn.reshape(
            v75_ttnn_slice_74,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_permute_78 = ttnn.permute(
            v75_ttnn_reshape_323,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v75_ttnn_permute_79 = ttnn.permute(
            v75_ttnn_reshape_324,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v75_ttnn_permute_80 = ttnn.permute(
            v75_ttnn_reshape_322,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v75_ttnn_pad_54 = ttnn.pad(
            v75_ttnn_permute_78,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_pad_55 = ttnn.pad(
            v75_ttnn_permute_79,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_pad_56 = ttnn.pad(
            v75_ttnn_permute_80,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_transformer_scaled_dot_product_attention_18 = (
            ttnn.transformer.scaled_dot_product_attention(
                v75_ttnn_pad_55,
                v75_ttnn_pad_54,
                v75_ttnn_pad_56,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v75_ttnn_slice_75 = ttnn.slice(
            v75_ttnn_transformer_scaled_dot_product_attention_18,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_permute_81 = ttnn.permute(
            v75_ttnn_slice_75,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v75_ttnn_reshape_325 = ttnn.reshape(
            v75_ttnn_permute_81,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v75_ttnn_matmul_74 = ttnn.matmul(
            v75_ttnn_reshape_325,
            self.weights[
                "image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v75_ttnn_add_110 = ttnn.add(
            v75_ttnn_matmul_74,
            self.cer["utils_constEvalFuncWrapper_100_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_75_0_0 = v75_ttnn_add_110
        v76_ttnn_add_111 = ttnn.add(
            v_234,
            CLIPAttention_75_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v76_ttnn_layer_norm_39 = ttnn.layer_norm(
            v76_ttnn_add_111,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.18.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.18.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_235, v_236 = v76_ttnn_layer_norm_39, v76_ttnn_add_111
        v77_ttnn_reshape_326 = ttnn.reshape(
            v_235,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v77_ttnn_matmul_75 = ttnn.matmul(
            v77_ttnn_reshape_326,
            self.weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v77_ttnn_add_112 = ttnn.add(
            v77_ttnn_matmul_75,
            self.cer["utils_constEvalFuncWrapper_94_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v77_ttnn_gelu_18 = ttnn.gelu(
            v77_ttnn_add_112,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v77_ttnn_reshape_327 = ttnn.reshape(
            v77_ttnn_gelu_18,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v77_ttnn_matmul_76 = ttnn.matmul(
            v77_ttnn_reshape_327,
            self.weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v77_ttnn_add_113 = ttnn.add(
            v77_ttnn_matmul_76,
            self.cer["utils_constEvalFuncWrapper_147_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_77_0_0 = v77_ttnn_add_113
        v78_ttnn_add_114 = ttnn.add(
            v_236,
            CLIPMLP_77_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v78_ttnn_layer_norm_40 = ttnn.layer_norm(
            v78_ttnn_add_114,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.19.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.19.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_237, v_238 = v78_ttnn_layer_norm_40, v78_ttnn_add_114
        v79_ttnn_reshape_328 = ttnn.reshape(
            v_237,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_matmul_77 = ttnn.matmul(
            v79_ttnn_reshape_328,
            self.cer["utils_constEvalFuncWrapper_12_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v79_ttnn_add_115 = ttnn.add(
            v79_ttnn_matmul_77,
            self.cer["utils_constEvalFuncWrapper_50_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_slice_76 = ttnn.slice(
            v79_ttnn_add_115,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_slice_77 = ttnn.slice(
            v79_ttnn_add_115,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_slice_78 = ttnn.slice(
            v79_ttnn_add_115,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_reshape_329 = ttnn.reshape(
            v79_ttnn_slice_76,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_reshape_330 = ttnn.reshape(
            v79_ttnn_slice_77,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_reshape_331 = ttnn.reshape(
            v79_ttnn_slice_78,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_permute_82 = ttnn.permute(
            v79_ttnn_reshape_330,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v79_ttnn_permute_83 = ttnn.permute(
            v79_ttnn_reshape_331,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v79_ttnn_permute_84 = ttnn.permute(
            v79_ttnn_reshape_329,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v79_ttnn_pad_57 = ttnn.pad(
            v79_ttnn_permute_82,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_pad_58 = ttnn.pad(
            v79_ttnn_permute_83,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_pad_59 = ttnn.pad(
            v79_ttnn_permute_84,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_transformer_scaled_dot_product_attention_19 = (
            ttnn.transformer.scaled_dot_product_attention(
                v79_ttnn_pad_58,
                v79_ttnn_pad_57,
                v79_ttnn_pad_59,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v79_ttnn_slice_79 = ttnn.slice(
            v79_ttnn_transformer_scaled_dot_product_attention_19,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_permute_85 = ttnn.permute(
            v79_ttnn_slice_79,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v79_ttnn_reshape_332 = ttnn.reshape(
            v79_ttnn_permute_85,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v79_ttnn_matmul_78 = ttnn.matmul(
            v79_ttnn_reshape_332,
            self.weights[
                "image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v79_ttnn_add_116 = ttnn.add(
            v79_ttnn_matmul_78,
            self.cer["utils_constEvalFuncWrapper_52_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_79_0_0 = v79_ttnn_add_116
        v80_ttnn_add_117 = ttnn.add(
            v_238,
            CLIPAttention_79_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v80_ttnn_layer_norm_41 = ttnn.layer_norm(
            v80_ttnn_add_117,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.19.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.19.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_239, v_240 = v80_ttnn_add_117, v80_ttnn_layer_norm_41
        v81_ttnn_reshape_333 = ttnn.reshape(
            v_240,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v81_ttnn_matmul_79 = ttnn.matmul(
            v81_ttnn_reshape_333,
            self.weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v81_ttnn_add_118 = ttnn.add(
            v81_ttnn_matmul_79,
            self.cer["utils_constEvalFuncWrapper_44_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v81_ttnn_gelu_19 = ttnn.gelu(
            v81_ttnn_add_118,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v81_ttnn_reshape_334 = ttnn.reshape(
            v81_ttnn_gelu_19,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v81_ttnn_matmul_80 = ttnn.matmul(
            v81_ttnn_reshape_334,
            self.weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v81_ttnn_add_119 = ttnn.add(
            v81_ttnn_matmul_80,
            self.cer["utils_constEvalFuncWrapper_28_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_81_0_0 = v81_ttnn_add_119
        v82_ttnn_add_120 = ttnn.add(
            v_239,
            CLIPMLP_81_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v82_ttnn_layer_norm_42 = ttnn.layer_norm(
            v82_ttnn_add_120,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.20.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.20.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_241, v_242 = v82_ttnn_add_120, v82_ttnn_layer_norm_42
        v83_ttnn_reshape_335 = ttnn.reshape(
            v_242,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_matmul_81 = ttnn.matmul(
            v83_ttnn_reshape_335,
            self.cer["utils_constEvalFuncWrapper_65_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v83_ttnn_add_121 = ttnn.add(
            v83_ttnn_matmul_81,
            self.cer["utils_constEvalFuncWrapper_60_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_slice_80 = ttnn.slice(
            v83_ttnn_add_121,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_slice_81 = ttnn.slice(
            v83_ttnn_add_121,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_slice_82 = ttnn.slice(
            v83_ttnn_add_121,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_reshape_336 = ttnn.reshape(
            v83_ttnn_slice_80,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_reshape_337 = ttnn.reshape(
            v83_ttnn_slice_81,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_reshape_338 = ttnn.reshape(
            v83_ttnn_slice_82,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_permute_86 = ttnn.permute(
            v83_ttnn_reshape_337,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v83_ttnn_permute_87 = ttnn.permute(
            v83_ttnn_reshape_338,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v83_ttnn_permute_88 = ttnn.permute(
            v83_ttnn_reshape_336,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v83_ttnn_pad_60 = ttnn.pad(
            v83_ttnn_permute_86,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_pad_61 = ttnn.pad(
            v83_ttnn_permute_87,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_pad_62 = ttnn.pad(
            v83_ttnn_permute_88,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_transformer_scaled_dot_product_attention_20 = (
            ttnn.transformer.scaled_dot_product_attention(
                v83_ttnn_pad_61,
                v83_ttnn_pad_60,
                v83_ttnn_pad_62,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v83_ttnn_slice_83 = ttnn.slice(
            v83_ttnn_transformer_scaled_dot_product_attention_20,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_permute_89 = ttnn.permute(
            v83_ttnn_slice_83,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v83_ttnn_reshape_339 = ttnn.reshape(
            v83_ttnn_permute_89,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v83_ttnn_matmul_82 = ttnn.matmul(
            v83_ttnn_reshape_339,
            self.weights[
                "image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v83_ttnn_add_122 = ttnn.add(
            v83_ttnn_matmul_82,
            self.cer["utils_constEvalFuncWrapper_78_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_83_0_0 = v83_ttnn_add_122
        v84_ttnn_add_123 = ttnn.add(
            v_241,
            CLIPAttention_83_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v84_ttnn_layer_norm_43 = ttnn.layer_norm(
            v84_ttnn_add_123,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.20.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.20.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_243, v_244 = v84_ttnn_add_123, v84_ttnn_layer_norm_43
        v85_ttnn_reshape_340 = ttnn.reshape(
            v_244,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v85_ttnn_matmul_83 = ttnn.matmul(
            v85_ttnn_reshape_340,
            self.weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v85_ttnn_add_124 = ttnn.add(
            v85_ttnn_matmul_83,
            self.cer["utils_constEvalFuncWrapper_107_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v85_ttnn_gelu_20 = ttnn.gelu(
            v85_ttnn_add_124,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v85_ttnn_reshape_341 = ttnn.reshape(
            v85_ttnn_gelu_20,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v85_ttnn_matmul_84 = ttnn.matmul(
            v85_ttnn_reshape_341,
            self.weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v85_ttnn_add_125 = ttnn.add(
            v85_ttnn_matmul_84,
            self.cer["utils_constEvalFuncWrapper_82_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_85_0_0 = v85_ttnn_add_125
        v86_ttnn_add_126 = ttnn.add(
            v_243,
            CLIPMLP_85_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v86_ttnn_layer_norm_44 = ttnn.layer_norm(
            v86_ttnn_add_126,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.21.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.21.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_245, v_246 = v86_ttnn_add_126, v86_ttnn_layer_norm_44
        v87_ttnn_reshape_342 = ttnn.reshape(
            v_246,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_matmul_85 = ttnn.matmul(
            v87_ttnn_reshape_342,
            self.cer["utils_constEvalFuncWrapper_37_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v87_ttnn_add_127 = ttnn.add(
            v87_ttnn_matmul_85,
            self.cer["utils_constEvalFuncWrapper_111_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_slice_84 = ttnn.slice(
            v87_ttnn_add_127,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_slice_85 = ttnn.slice(
            v87_ttnn_add_127,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_slice_86 = ttnn.slice(
            v87_ttnn_add_127,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_reshape_343 = ttnn.reshape(
            v87_ttnn_slice_84,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_reshape_344 = ttnn.reshape(
            v87_ttnn_slice_85,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_reshape_345 = ttnn.reshape(
            v87_ttnn_slice_86,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_permute_90 = ttnn.permute(
            v87_ttnn_reshape_344,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v87_ttnn_permute_91 = ttnn.permute(
            v87_ttnn_reshape_345,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v87_ttnn_permute_92 = ttnn.permute(
            v87_ttnn_reshape_343,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v87_ttnn_pad_63 = ttnn.pad(
            v87_ttnn_permute_90,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_pad_64 = ttnn.pad(
            v87_ttnn_permute_91,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_pad_65 = ttnn.pad(
            v87_ttnn_permute_92,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_transformer_scaled_dot_product_attention_21 = (
            ttnn.transformer.scaled_dot_product_attention(
                v87_ttnn_pad_64,
                v87_ttnn_pad_63,
                v87_ttnn_pad_65,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v87_ttnn_slice_87 = ttnn.slice(
            v87_ttnn_transformer_scaled_dot_product_attention_21,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_permute_93 = ttnn.permute(
            v87_ttnn_slice_87,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v87_ttnn_reshape_346 = ttnn.reshape(
            v87_ttnn_permute_93,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v87_ttnn_matmul_86 = ttnn.matmul(
            v87_ttnn_reshape_346,
            self.weights[
                "image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v87_ttnn_add_128 = ttnn.add(
            v87_ttnn_matmul_86,
            self.cer["utils_constEvalFuncWrapper_20_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_87_0_0 = v87_ttnn_add_128
        v88_ttnn_add_129 = ttnn.add(
            v_245,
            CLIPAttention_87_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v88_ttnn_layer_norm_45 = ttnn.layer_norm(
            v88_ttnn_add_129,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.21.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.21.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_247, v_248 = v88_ttnn_layer_norm_45, v88_ttnn_add_129
        v89_ttnn_reshape_347 = ttnn.reshape(
            v_247,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v89_ttnn_matmul_87 = ttnn.matmul(
            v89_ttnn_reshape_347,
            self.weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v89_ttnn_add_130 = ttnn.add(
            v89_ttnn_matmul_87,
            self.cer["utils_constEvalFuncWrapper_110_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v89_ttnn_gelu_21 = ttnn.gelu(
            v89_ttnn_add_130,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v89_ttnn_reshape_348 = ttnn.reshape(
            v89_ttnn_gelu_21,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v89_ttnn_matmul_88 = ttnn.matmul(
            v89_ttnn_reshape_348,
            self.weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v89_ttnn_add_131 = ttnn.add(
            v89_ttnn_matmul_88,
            self.cer["utils_constEvalFuncWrapper_160_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_89_0_0 = v89_ttnn_add_131
        v90_ttnn_add_132 = ttnn.add(
            v_248,
            CLIPMLP_89_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v90_ttnn_layer_norm_46 = ttnn.layer_norm(
            v90_ttnn_add_132,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.22.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.22.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_249, v_250 = v90_ttnn_layer_norm_46, v90_ttnn_add_132
        v91_ttnn_reshape_349 = ttnn.reshape(
            v_249,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_matmul_89 = ttnn.matmul(
            v91_ttnn_reshape_349,
            self.cer["utils_constEvalFuncWrapper_148_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v91_ttnn_add_133 = ttnn.add(
            v91_ttnn_matmul_89,
            self.cer["utils_constEvalFuncWrapper_57_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_slice_88 = ttnn.slice(
            v91_ttnn_add_133,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_slice_89 = ttnn.slice(
            v91_ttnn_add_133,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_slice_90 = ttnn.slice(
            v91_ttnn_add_133,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_reshape_350 = ttnn.reshape(
            v91_ttnn_slice_88,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_reshape_351 = ttnn.reshape(
            v91_ttnn_slice_89,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_reshape_352 = ttnn.reshape(
            v91_ttnn_slice_90,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_permute_94 = ttnn.permute(
            v91_ttnn_reshape_351,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v91_ttnn_permute_95 = ttnn.permute(
            v91_ttnn_reshape_352,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v91_ttnn_permute_96 = ttnn.permute(
            v91_ttnn_reshape_350,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v91_ttnn_pad_66 = ttnn.pad(
            v91_ttnn_permute_94,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_pad_67 = ttnn.pad(
            v91_ttnn_permute_95,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_pad_68 = ttnn.pad(
            v91_ttnn_permute_96,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_transformer_scaled_dot_product_attention_22 = (
            ttnn.transformer.scaled_dot_product_attention(
                v91_ttnn_pad_67,
                v91_ttnn_pad_66,
                v91_ttnn_pad_68,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v91_ttnn_slice_91 = ttnn.slice(
            v91_ttnn_transformer_scaled_dot_product_attention_22,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_permute_97 = ttnn.permute(
            v91_ttnn_slice_91,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v91_ttnn_reshape_353 = ttnn.reshape(
            v91_ttnn_permute_97,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v91_ttnn_matmul_90 = ttnn.matmul(
            v91_ttnn_reshape_353,
            self.weights[
                "image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v91_ttnn_add_134 = ttnn.add(
            v91_ttnn_matmul_90,
            self.cer["utils_constEvalFuncWrapper_33_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_91_0_0 = v91_ttnn_add_134
        v92_ttnn_add_135 = ttnn.add(
            v_250,
            CLIPAttention_91_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v92_ttnn_layer_norm_47 = ttnn.layer_norm(
            v92_ttnn_add_135,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.22.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.22.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_251, v_252 = v92_ttnn_add_135, v92_ttnn_layer_norm_47
        v93_ttnn_reshape_354 = ttnn.reshape(
            v_252,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v93_ttnn_matmul_91 = ttnn.matmul(
            v93_ttnn_reshape_354,
            self.weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v93_ttnn_add_136 = ttnn.add(
            v93_ttnn_matmul_91,
            self.cer["utils_constEvalFuncWrapper_4_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v93_ttnn_gelu_22 = ttnn.gelu(
            v93_ttnn_add_136,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v93_ttnn_reshape_355 = ttnn.reshape(
            v93_ttnn_gelu_22,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v93_ttnn_matmul_92 = ttnn.matmul(
            v93_ttnn_reshape_355,
            self.weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v93_ttnn_add_137 = ttnn.add(
            v93_ttnn_matmul_92,
            self.cer["utils_constEvalFuncWrapper_125_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_93_0_0 = v93_ttnn_add_137
        v94_ttnn_add_138 = ttnn.add(
            v_251,
            CLIPMLP_93_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v94_ttnn_layer_norm_48 = ttnn.layer_norm(
            v94_ttnn_add_138,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.23.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.23.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_253, v_254 = v94_ttnn_add_138, v94_ttnn_layer_norm_48
        v95_ttnn_reshape_356 = ttnn.reshape(
            v_254,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_matmul_93 = ttnn.matmul(
            v95_ttnn_reshape_356,
            self.cer["utils_constEvalFuncWrapper_36_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v95_ttnn_add_139 = ttnn.add(
            v95_ttnn_matmul_93,
            self.cer["utils_constEvalFuncWrapper_32_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_slice_92 = ttnn.slice(
            v95_ttnn_add_139,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_slice_93 = ttnn.slice(
            v95_ttnn_add_139,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_slice_94 = ttnn.slice(
            v95_ttnn_add_139,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_reshape_357 = ttnn.reshape(
            v95_ttnn_slice_92,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_reshape_358 = ttnn.reshape(
            v95_ttnn_slice_93,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_reshape_359 = ttnn.reshape(
            v95_ttnn_slice_94,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_permute_98 = ttnn.permute(
            v95_ttnn_reshape_358,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v95_ttnn_permute_99 = ttnn.permute(
            v95_ttnn_reshape_359,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v95_ttnn_permute_100 = ttnn.permute(
            v95_ttnn_reshape_357,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v95_ttnn_pad_69 = ttnn.pad(
            v95_ttnn_permute_98,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_pad_70 = ttnn.pad(
            v95_ttnn_permute_99,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_pad_71 = ttnn.pad(
            v95_ttnn_permute_100,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_transformer_scaled_dot_product_attention_23 = (
            ttnn.transformer.scaled_dot_product_attention(
                v95_ttnn_pad_70,
                v95_ttnn_pad_69,
                v95_ttnn_pad_71,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v95_ttnn_slice_95 = ttnn.slice(
            v95_ttnn_transformer_scaled_dot_product_attention_23,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_permute_101 = ttnn.permute(
            v95_ttnn_slice_95,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v95_ttnn_reshape_360 = ttnn.reshape(
            v95_ttnn_permute_101,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v95_ttnn_matmul_94 = ttnn.matmul(
            v95_ttnn_reshape_360,
            self.weights[
                "image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v95_ttnn_add_140 = ttnn.add(
            v95_ttnn_matmul_94,
            self.cer["utils_constEvalFuncWrapper_51_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_95_0_0 = v95_ttnn_add_140
        v96_ttnn_add_141 = ttnn.add(
            v_253,
            CLIPAttention_95_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v96_ttnn_layer_norm_49 = ttnn.layer_norm(
            v96_ttnn_add_141,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.23.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.23.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_255, v_256 = v96_ttnn_add_141, v96_ttnn_layer_norm_49
        v97_ttnn_reshape_361 = ttnn.reshape(
            v_256,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v97_ttnn_matmul_95 = ttnn.matmul(
            v97_ttnn_reshape_361,
            self.weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v97_ttnn_add_142 = ttnn.add(
            v97_ttnn_matmul_95,
            self.cer["utils_constEvalFuncWrapper_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v97_ttnn_gelu_23 = ttnn.gelu(
            v97_ttnn_add_142,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v97_ttnn_reshape_362 = ttnn.reshape(
            v97_ttnn_gelu_23,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v97_ttnn_matmul_96 = ttnn.matmul(
            v97_ttnn_reshape_362,
            self.weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v97_ttnn_add_143 = ttnn.add(
            v97_ttnn_matmul_96,
            self.cer["utils_constEvalFuncWrapper_22_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_97_0_0 = v97_ttnn_add_143
        v98_ttnn_add_144 = ttnn.add(
            v_255,
            CLIPMLP_97_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v98_ttnn_layer_norm_50 = ttnn.layer_norm(
            v98_ttnn_add_144,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.24.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.24.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_257, v_258 = v98_ttnn_add_144, v98_ttnn_layer_norm_50
        v99_ttnn_reshape_363 = ttnn.reshape(
            v_258,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_matmul_97 = ttnn.matmul(
            v99_ttnn_reshape_363,
            self.cer["utils_constEvalFuncWrapper_76_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v99_ttnn_add_145 = ttnn.add(
            v99_ttnn_matmul_97,
            self.cer["utils_constEvalFuncWrapper_59_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_slice_96 = ttnn.slice(
            v99_ttnn_add_145,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_slice_97 = ttnn.slice(
            v99_ttnn_add_145,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_slice_98 = ttnn.slice(
            v99_ttnn_add_145,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_reshape_364 = ttnn.reshape(
            v99_ttnn_slice_96,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_reshape_365 = ttnn.reshape(
            v99_ttnn_slice_97,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_reshape_366 = ttnn.reshape(
            v99_ttnn_slice_98,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_permute_102 = ttnn.permute(
            v99_ttnn_reshape_365,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v99_ttnn_permute_103 = ttnn.permute(
            v99_ttnn_reshape_366,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v99_ttnn_permute_104 = ttnn.permute(
            v99_ttnn_reshape_364,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v99_ttnn_pad_72 = ttnn.pad(
            v99_ttnn_permute_102,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_pad_73 = ttnn.pad(
            v99_ttnn_permute_103,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_pad_74 = ttnn.pad(
            v99_ttnn_permute_104,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_transformer_scaled_dot_product_attention_24 = (
            ttnn.transformer.scaled_dot_product_attention(
                v99_ttnn_pad_73,
                v99_ttnn_pad_72,
                v99_ttnn_pad_74,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v99_ttnn_slice_99 = ttnn.slice(
            v99_ttnn_transformer_scaled_dot_product_attention_24,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_permute_105 = ttnn.permute(
            v99_ttnn_slice_99,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v99_ttnn_reshape_367 = ttnn.reshape(
            v99_ttnn_permute_105,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v99_ttnn_matmul_98 = ttnn.matmul(
            v99_ttnn_reshape_367,
            self.weights[
                "image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v99_ttnn_add_146 = ttnn.add(
            v99_ttnn_matmul_98,
            self.cer["utils_constEvalFuncWrapper_143_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_99_0_0 = v99_ttnn_add_146
        v100_ttnn_add_147 = ttnn.add(
            v_257,
            CLIPAttention_99_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v100_ttnn_layer_norm_51 = ttnn.layer_norm(
            v100_ttnn_add_147,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.24.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.24.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_259, v_260 = v100_ttnn_add_147, v100_ttnn_layer_norm_51
        v101_ttnn_reshape_368 = ttnn.reshape(
            v_260,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v101_ttnn_matmul_99 = ttnn.matmul(
            v101_ttnn_reshape_368,
            self.weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v101_ttnn_add_148 = ttnn.add(
            v101_ttnn_matmul_99,
            self.cer["utils_constEvalFuncWrapper_139_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v101_ttnn_gelu_24 = ttnn.gelu(
            v101_ttnn_add_148,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v101_ttnn_reshape_369 = ttnn.reshape(
            v101_ttnn_gelu_24,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v101_ttnn_matmul_100 = ttnn.matmul(
            v101_ttnn_reshape_369,
            self.weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v101_ttnn_add_149 = ttnn.add(
            v101_ttnn_matmul_100,
            self.cer["utils_constEvalFuncWrapper_144_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_101_0_0 = v101_ttnn_add_149
        v102_ttnn_add_150 = ttnn.add(
            v_259,
            CLIPMLP_101_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v102_ttnn_layer_norm_52 = ttnn.layer_norm(
            v102_ttnn_add_150,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.25.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.25.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_261, v_262 = v102_ttnn_layer_norm_52, v102_ttnn_add_150
        v103_ttnn_reshape_370 = ttnn.reshape(
            v_261,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_matmul_101 = ttnn.matmul(
            v103_ttnn_reshape_370,
            self.cer["utils_constEvalFuncWrapper_61_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v103_ttnn_add_151 = ttnn.add(
            v103_ttnn_matmul_101,
            self.cer["utils_constEvalFuncWrapper_58_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_slice_100 = ttnn.slice(
            v103_ttnn_add_151,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_slice_101 = ttnn.slice(
            v103_ttnn_add_151,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_slice_102 = ttnn.slice(
            v103_ttnn_add_151,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_reshape_371 = ttnn.reshape(
            v103_ttnn_slice_100,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_reshape_372 = ttnn.reshape(
            v103_ttnn_slice_101,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_reshape_373 = ttnn.reshape(
            v103_ttnn_slice_102,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_permute_106 = ttnn.permute(
            v103_ttnn_reshape_372,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v103_ttnn_permute_107 = ttnn.permute(
            v103_ttnn_reshape_373,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v103_ttnn_permute_108 = ttnn.permute(
            v103_ttnn_reshape_371,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v103_ttnn_pad_75 = ttnn.pad(
            v103_ttnn_permute_106,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_pad_76 = ttnn.pad(
            v103_ttnn_permute_107,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_pad_77 = ttnn.pad(
            v103_ttnn_permute_108,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_transformer_scaled_dot_product_attention_25 = (
            ttnn.transformer.scaled_dot_product_attention(
                v103_ttnn_pad_76,
                v103_ttnn_pad_75,
                v103_ttnn_pad_77,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v103_ttnn_slice_103 = ttnn.slice(
            v103_ttnn_transformer_scaled_dot_product_attention_25,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_permute_109 = ttnn.permute(
            v103_ttnn_slice_103,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v103_ttnn_reshape_374 = ttnn.reshape(
            v103_ttnn_permute_109,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v103_ttnn_matmul_102 = ttnn.matmul(
            v103_ttnn_reshape_374,
            self.weights[
                "image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v103_ttnn_add_152 = ttnn.add(
            v103_ttnn_matmul_102,
            self.cer["utils_constEvalFuncWrapper_31_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_103_0_0 = v103_ttnn_add_152
        v104_ttnn_add_153 = ttnn.add(
            v_262,
            CLIPAttention_103_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v104_ttnn_layer_norm_53 = ttnn.layer_norm(
            v104_ttnn_add_153,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.25.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.25.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_263, v_264 = v104_ttnn_add_153, v104_ttnn_layer_norm_53
        v105_ttnn_reshape_375 = ttnn.reshape(
            v_264,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v105_ttnn_matmul_103 = ttnn.matmul(
            v105_ttnn_reshape_375,
            self.weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v105_ttnn_add_154 = ttnn.add(
            v105_ttnn_matmul_103,
            self.cer["utils_constEvalFuncWrapper_117_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v105_ttnn_gelu_25 = ttnn.gelu(
            v105_ttnn_add_154,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v105_ttnn_reshape_376 = ttnn.reshape(
            v105_ttnn_gelu_25,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v105_ttnn_matmul_104 = ttnn.matmul(
            v105_ttnn_reshape_376,
            self.weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v105_ttnn_add_155 = ttnn.add(
            v105_ttnn_matmul_104,
            self.cer["utils_constEvalFuncWrapper_39_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_105_0_0 = v105_ttnn_add_155
        v106_ttnn_add_156 = ttnn.add(
            v_263,
            CLIPMLP_105_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v106_ttnn_layer_norm_54 = ttnn.layer_norm(
            v106_ttnn_add_156,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.26.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.26.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_265, v_266 = v106_ttnn_layer_norm_54, v106_ttnn_add_156
        v107_ttnn_reshape_377 = ttnn.reshape(
            v_265,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_matmul_105 = ttnn.matmul(
            v107_ttnn_reshape_377,
            self.cer["utils_constEvalFuncWrapper_9_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v107_ttnn_add_157 = ttnn.add(
            v107_ttnn_matmul_105,
            self.cer["utils_constEvalFuncWrapper_77_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_slice_104 = ttnn.slice(
            v107_ttnn_add_157,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_slice_105 = ttnn.slice(
            v107_ttnn_add_157,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_slice_106 = ttnn.slice(
            v107_ttnn_add_157,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_reshape_378 = ttnn.reshape(
            v107_ttnn_slice_104,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_reshape_379 = ttnn.reshape(
            v107_ttnn_slice_105,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_reshape_380 = ttnn.reshape(
            v107_ttnn_slice_106,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_permute_110 = ttnn.permute(
            v107_ttnn_reshape_379,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v107_ttnn_permute_111 = ttnn.permute(
            v107_ttnn_reshape_380,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v107_ttnn_permute_112 = ttnn.permute(
            v107_ttnn_reshape_378,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v107_ttnn_pad_78 = ttnn.pad(
            v107_ttnn_permute_110,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_pad_79 = ttnn.pad(
            v107_ttnn_permute_111,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_pad_80 = ttnn.pad(
            v107_ttnn_permute_112,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_transformer_scaled_dot_product_attention_26 = (
            ttnn.transformer.scaled_dot_product_attention(
                v107_ttnn_pad_79,
                v107_ttnn_pad_78,
                v107_ttnn_pad_80,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v107_ttnn_slice_107 = ttnn.slice(
            v107_ttnn_transformer_scaled_dot_product_attention_26,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_permute_113 = ttnn.permute(
            v107_ttnn_slice_107,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v107_ttnn_reshape_381 = ttnn.reshape(
            v107_ttnn_permute_113,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v107_ttnn_matmul_106 = ttnn.matmul(
            v107_ttnn_reshape_381,
            self.weights[
                "image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v107_ttnn_add_158 = ttnn.add(
            v107_ttnn_matmul_106,
            self.cer["utils_constEvalFuncWrapper_105_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_107_0_0 = v107_ttnn_add_158
        v108_ttnn_add_159 = ttnn.add(
            v_266,
            CLIPAttention_107_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v108_ttnn_layer_norm_55 = ttnn.layer_norm(
            v108_ttnn_add_159,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.26.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.26.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_267, v_268 = v108_ttnn_layer_norm_55, v108_ttnn_add_159
        v109_ttnn_reshape_382 = ttnn.reshape(
            v_267,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v109_ttnn_matmul_107 = ttnn.matmul(
            v109_ttnn_reshape_382,
            self.weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v109_ttnn_add_160 = ttnn.add(
            v109_ttnn_matmul_107,
            self.cer["utils_constEvalFuncWrapper_98_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v109_ttnn_gelu_26 = ttnn.gelu(
            v109_ttnn_add_160,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v109_ttnn_reshape_383 = ttnn.reshape(
            v109_ttnn_gelu_26,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v109_ttnn_matmul_108 = ttnn.matmul(
            v109_ttnn_reshape_383,
            self.weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v109_ttnn_add_161 = ttnn.add(
            v109_ttnn_matmul_108,
            self.cer["utils_constEvalFuncWrapper_123_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_109_0_0 = v109_ttnn_add_161
        v110_ttnn_add_162 = ttnn.add(
            v_268,
            CLIPMLP_109_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v110_ttnn_layer_norm_56 = ttnn.layer_norm(
            v110_ttnn_add_162,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.27.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.27.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_269, v_270 = v110_ttnn_add_162, v110_ttnn_layer_norm_56
        v111_ttnn_reshape_384 = ttnn.reshape(
            v_270,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_matmul_109 = ttnn.matmul(
            v111_ttnn_reshape_384,
            self.cer["utils_constEvalFuncWrapper_159_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v111_ttnn_add_163 = ttnn.add(
            v111_ttnn_matmul_109,
            self.cer["utils_constEvalFuncWrapper_41_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_slice_108 = ttnn.slice(
            v111_ttnn_add_163,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_slice_109 = ttnn.slice(
            v111_ttnn_add_163,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_slice_110 = ttnn.slice(
            v111_ttnn_add_163,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_reshape_385 = ttnn.reshape(
            v111_ttnn_slice_108,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_reshape_386 = ttnn.reshape(
            v111_ttnn_slice_109,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_reshape_387 = ttnn.reshape(
            v111_ttnn_slice_110,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_permute_114 = ttnn.permute(
            v111_ttnn_reshape_386,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v111_ttnn_permute_115 = ttnn.permute(
            v111_ttnn_reshape_387,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v111_ttnn_permute_116 = ttnn.permute(
            v111_ttnn_reshape_385,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v111_ttnn_pad_81 = ttnn.pad(
            v111_ttnn_permute_114,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_pad_82 = ttnn.pad(
            v111_ttnn_permute_115,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_pad_83 = ttnn.pad(
            v111_ttnn_permute_116,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_transformer_scaled_dot_product_attention_27 = (
            ttnn.transformer.scaled_dot_product_attention(
                v111_ttnn_pad_82,
                v111_ttnn_pad_81,
                v111_ttnn_pad_83,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v111_ttnn_slice_111 = ttnn.slice(
            v111_ttnn_transformer_scaled_dot_product_attention_27,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_permute_117 = ttnn.permute(
            v111_ttnn_slice_111,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v111_ttnn_reshape_388 = ttnn.reshape(
            v111_ttnn_permute_117,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v111_ttnn_matmul_110 = ttnn.matmul(
            v111_ttnn_reshape_388,
            self.weights[
                "image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v111_ttnn_add_164 = ttnn.add(
            v111_ttnn_matmul_110,
            self.cer["utils_constEvalFuncWrapper_8_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_111_0_0 = v111_ttnn_add_164
        v112_ttnn_add_165 = ttnn.add(
            v_269,
            CLIPAttention_111_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v112_ttnn_layer_norm_57 = ttnn.layer_norm(
            v112_ttnn_add_165,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.27.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.27.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_271, v_272 = v112_ttnn_layer_norm_57, v112_ttnn_add_165
        v113_ttnn_reshape_389 = ttnn.reshape(
            v_271,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v113_ttnn_matmul_111 = ttnn.matmul(
            v113_ttnn_reshape_389,
            self.weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v113_ttnn_add_166 = ttnn.add(
            v113_ttnn_matmul_111,
            self.cer["utils_constEvalFuncWrapper_115_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v113_ttnn_gelu_27 = ttnn.gelu(
            v113_ttnn_add_166,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v113_ttnn_reshape_390 = ttnn.reshape(
            v113_ttnn_gelu_27,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v113_ttnn_matmul_112 = ttnn.matmul(
            v113_ttnn_reshape_390,
            self.weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v113_ttnn_add_167 = ttnn.add(
            v113_ttnn_matmul_112,
            self.cer["utils_constEvalFuncWrapper_129_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_113_0_0 = v113_ttnn_add_167
        v114_ttnn_add_168 = ttnn.add(
            v_272,
            CLIPMLP_113_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v114_ttnn_layer_norm_58 = ttnn.layer_norm(
            v114_ttnn_add_168,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.28.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.28.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_273, v_274 = v114_ttnn_add_168, v114_ttnn_layer_norm_58
        v115_ttnn_reshape_391 = ttnn.reshape(
            v_274,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_matmul_113 = ttnn.matmul(
            v115_ttnn_reshape_391,
            self.cer["utils_constEvalFuncWrapper_3_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v115_ttnn_add_169 = ttnn.add(
            v115_ttnn_matmul_113,
            self.cer["utils_constEvalFuncWrapper_16_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_slice_112 = ttnn.slice(
            v115_ttnn_add_169,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_slice_113 = ttnn.slice(
            v115_ttnn_add_169,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_slice_114 = ttnn.slice(
            v115_ttnn_add_169,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_reshape_392 = ttnn.reshape(
            v115_ttnn_slice_112,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_reshape_393 = ttnn.reshape(
            v115_ttnn_slice_113,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_reshape_394 = ttnn.reshape(
            v115_ttnn_slice_114,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_permute_118 = ttnn.permute(
            v115_ttnn_reshape_393,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v115_ttnn_permute_119 = ttnn.permute(
            v115_ttnn_reshape_394,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v115_ttnn_permute_120 = ttnn.permute(
            v115_ttnn_reshape_392,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v115_ttnn_pad_84 = ttnn.pad(
            v115_ttnn_permute_118,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_pad_85 = ttnn.pad(
            v115_ttnn_permute_119,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_pad_86 = ttnn.pad(
            v115_ttnn_permute_120,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_transformer_scaled_dot_product_attention_28 = (
            ttnn.transformer.scaled_dot_product_attention(
                v115_ttnn_pad_85,
                v115_ttnn_pad_84,
                v115_ttnn_pad_86,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v115_ttnn_slice_115 = ttnn.slice(
            v115_ttnn_transformer_scaled_dot_product_attention_28,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_permute_121 = ttnn.permute(
            v115_ttnn_slice_115,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v115_ttnn_reshape_395 = ttnn.reshape(
            v115_ttnn_permute_121,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v115_ttnn_matmul_114 = ttnn.matmul(
            v115_ttnn_reshape_395,
            self.weights[
                "image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v115_ttnn_add_170 = ttnn.add(
            v115_ttnn_matmul_114,
            self.cer["utils_constEvalFuncWrapper_121_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_115_0_0 = v115_ttnn_add_170
        v116_ttnn_add_171 = ttnn.add(
            v_273,
            CLIPAttention_115_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v116_ttnn_layer_norm_59 = ttnn.layer_norm(
            v116_ttnn_add_171,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.28.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.28.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_275, v_276 = v116_ttnn_layer_norm_59, v116_ttnn_add_171
        v117_ttnn_reshape_396 = ttnn.reshape(
            v_275,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v117_ttnn_matmul_115 = ttnn.matmul(
            v117_ttnn_reshape_396,
            self.weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v117_ttnn_add_172 = ttnn.add(
            v117_ttnn_matmul_115,
            self.cer["utils_constEvalFuncWrapper_14_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v117_ttnn_gelu_28 = ttnn.gelu(
            v117_ttnn_add_172,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v117_ttnn_reshape_397 = ttnn.reshape(
            v117_ttnn_gelu_28,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v117_ttnn_matmul_116 = ttnn.matmul(
            v117_ttnn_reshape_397,
            self.weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v117_ttnn_add_173 = ttnn.add(
            v117_ttnn_matmul_116,
            self.cer["utils_constEvalFuncWrapper_56_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_117_0_0 = v117_ttnn_add_173
        v118_ttnn_add_174 = ttnn.add(
            v_276,
            CLIPMLP_117_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v118_ttnn_layer_norm_60 = ttnn.layer_norm(
            v118_ttnn_add_174,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.29.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.29.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_277, v_278 = v118_ttnn_add_174, v118_ttnn_layer_norm_60
        v119_ttnn_reshape_398 = ttnn.reshape(
            v_278,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_matmul_117 = ttnn.matmul(
            v119_ttnn_reshape_398,
            self.cer["utils_constEvalFuncWrapper_75_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v119_ttnn_add_175 = ttnn.add(
            v119_ttnn_matmul_117,
            self.cer["utils_constEvalFuncWrapper_45_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_slice_116 = ttnn.slice(
            v119_ttnn_add_175,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_slice_117 = ttnn.slice(
            v119_ttnn_add_175,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_slice_118 = ttnn.slice(
            v119_ttnn_add_175,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_reshape_399 = ttnn.reshape(
            v119_ttnn_slice_116,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_reshape_400 = ttnn.reshape(
            v119_ttnn_slice_117,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_reshape_401 = ttnn.reshape(
            v119_ttnn_slice_118,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_permute_122 = ttnn.permute(
            v119_ttnn_reshape_400,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v119_ttnn_permute_123 = ttnn.permute(
            v119_ttnn_reshape_401,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v119_ttnn_permute_124 = ttnn.permute(
            v119_ttnn_reshape_399,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v119_ttnn_pad_87 = ttnn.pad(
            v119_ttnn_permute_122,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_pad_88 = ttnn.pad(
            v119_ttnn_permute_123,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_pad_89 = ttnn.pad(
            v119_ttnn_permute_124,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_transformer_scaled_dot_product_attention_29 = (
            ttnn.transformer.scaled_dot_product_attention(
                v119_ttnn_pad_88,
                v119_ttnn_pad_87,
                v119_ttnn_pad_89,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v119_ttnn_slice_119 = ttnn.slice(
            v119_ttnn_transformer_scaled_dot_product_attention_29,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_permute_125 = ttnn.permute(
            v119_ttnn_slice_119,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v119_ttnn_reshape_402 = ttnn.reshape(
            v119_ttnn_permute_125,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v119_ttnn_matmul_118 = ttnn.matmul(
            v119_ttnn_reshape_402,
            self.weights[
                "image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v119_ttnn_add_176 = ttnn.add(
            v119_ttnn_matmul_118,
            self.cer["utils_constEvalFuncWrapper_79_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_119_0_0 = v119_ttnn_add_176
        v120_ttnn_add_177 = ttnn.add(
            v_277,
            CLIPAttention_119_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v120_ttnn_layer_norm_61 = ttnn.layer_norm(
            v120_ttnn_add_177,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.29.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.29.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_279, v_280 = v120_ttnn_add_177, v120_ttnn_layer_norm_61
        v121_ttnn_reshape_403 = ttnn.reshape(
            v_280,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v121_ttnn_matmul_119 = ttnn.matmul(
            v121_ttnn_reshape_403,
            self.weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v121_ttnn_add_178 = ttnn.add(
            v121_ttnn_matmul_119,
            self.cer["utils_constEvalFuncWrapper_38_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v121_ttnn_gelu_29 = ttnn.gelu(
            v121_ttnn_add_178,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v121_ttnn_reshape_404 = ttnn.reshape(
            v121_ttnn_gelu_29,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v121_ttnn_matmul_120 = ttnn.matmul(
            v121_ttnn_reshape_404,
            self.weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v121_ttnn_add_179 = ttnn.add(
            v121_ttnn_matmul_120,
            self.cer["utils_constEvalFuncWrapper_35_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_121_0_0 = v121_ttnn_add_179
        v122_ttnn_add_180 = ttnn.add(
            v_279,
            CLIPMLP_121_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v122_ttnn_layer_norm_62 = ttnn.layer_norm(
            v122_ttnn_add_180,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.30.layer_norm1.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.30.layer_norm1.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_281, v_282 = v122_ttnn_layer_norm_62, v122_ttnn_add_180
        v123_ttnn_reshape_405 = ttnn.reshape(
            v_281,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_matmul_121 = ttnn.matmul(
            v123_ttnn_reshape_405,
            self.cer["utils_constEvalFuncWrapper_138_0"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v123_ttnn_add_181 = ttnn.add(
            v123_ttnn_matmul_121,
            self.cer["utils_constEvalFuncWrapper_131_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_slice_120 = ttnn.slice(
            v123_ttnn_add_181,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_slice_121 = ttnn.slice(
            v123_ttnn_add_181,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_slice_122 = ttnn.slice(
            v123_ttnn_add_181,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_reshape_406 = ttnn.reshape(
            v123_ttnn_slice_120,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_reshape_407 = ttnn.reshape(
            v123_ttnn_slice_121,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_reshape_408 = ttnn.reshape(
            v123_ttnn_slice_122,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_permute_126 = ttnn.permute(
            v123_ttnn_reshape_407,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v123_ttnn_permute_127 = ttnn.permute(
            v123_ttnn_reshape_408,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v123_ttnn_permute_128 = ttnn.permute(
            v123_ttnn_reshape_406,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v123_ttnn_pad_90 = ttnn.pad(
            v123_ttnn_permute_126,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_pad_91 = ttnn.pad(
            v123_ttnn_permute_127,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_pad_92 = ttnn.pad(
            v123_ttnn_permute_128,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_transformer_scaled_dot_product_attention_30 = (
            ttnn.transformer.scaled_dot_product_attention(
                v123_ttnn_pad_91,
                v123_ttnn_pad_90,
                v123_ttnn_pad_92,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v123_ttnn_slice_123 = ttnn.slice(
            v123_ttnn_transformer_scaled_dot_product_attention_30,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_permute_129 = ttnn.permute(
            v123_ttnn_slice_123,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v123_ttnn_reshape_409 = ttnn.reshape(
            v123_ttnn_permute_129,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v123_ttnn_matmul_122 = ttnn.matmul(
            v123_ttnn_reshape_409,
            self.weights[
                "image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v123_ttnn_add_182 = ttnn.add(
            v123_ttnn_matmul_122,
            self.cer["utils_constEvalFuncWrapper_48_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPAttention_123_0_0 = v123_ttnn_add_182
        v124_ttnn_add_183 = ttnn.add(
            v_282,
            CLIPAttention_123_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v124_ttnn_layer_norm_63 = ttnn.layer_norm(
            v124_ttnn_add_183,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[
                "image_encoder.vision_model.encoder.layers.30.layer_norm2.weight"
            ],
            bias=self.weights[
                "image_encoder.vision_model.encoder.layers.30.layer_norm2.bias"
            ],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_283, v_284 = v124_ttnn_add_183, v124_ttnn_layer_norm_63
        v125_ttnn_reshape_410 = ttnn.reshape(
            v_284,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v125_ttnn_matmul_123 = ttnn.matmul(
            v125_ttnn_reshape_410,
            self.weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v125_ttnn_add_184 = ttnn.add(
            v125_ttnn_matmul_123,
            self.cer["utils_constEvalFuncWrapper_135_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v125_ttnn_gelu_30 = ttnn.gelu(
            v125_ttnn_add_184,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v125_ttnn_reshape_411 = ttnn.reshape(
            v125_ttnn_gelu_30,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v125_ttnn_matmul_124 = ttnn.matmul(
            v125_ttnn_reshape_411,
            self.weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v125_ttnn_add_185 = ttnn.add(
            v125_ttnn_matmul_124,
            self.cer["utils_constEvalFuncWrapper_54_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPMLP_125_0_0 = v125_ttnn_add_185
        v126_ttnn_add_186 = ttnn.add(
            v_283,
            CLIPMLP_125_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v126_ttnn_reshape_412 = ttnn.reshape(
            v126_ttnn_add_186,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        CLIPEncoderLayer_126_0_0 = v126_ttnn_reshape_412
        v127_ttnn_matmul_125 = ttnn.matmul(
            CLIPEncoderLayer_126_0_0,
            self.weights["resampler.proj_in.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v127_ttnn_add_187 = ttnn.add(
            v127_ttnn_matmul_125,
            self.cer["utils_constEvalFuncWrapper_137_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Linear_127_0_0 = v127_ttnn_add_187
        v128_ttnn_layer_norm_64 = ttnn.layer_norm(
            Linear_127_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.1.ln0.weight"],
            bias=self.weights["resampler.layers.1.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v128_ttnn_layer_norm_65 = ttnn.layer_norm(
            Linear_127_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.2.ln0.weight"],
            bias=self.weights["resampler.layers.2.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v128_ttnn_layer_norm_66 = ttnn.layer_norm(
            Linear_127_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.3.ln0.weight"],
            bias=self.weights["resampler.layers.3.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v128_ttnn_layer_norm_67 = ttnn.layer_norm(
            Linear_127_0_0,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.0.ln0.weight"],
            bias=self.weights["resampler.layers.0.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v128_ttnn_reshape_413 = ttnn.reshape(
            v128_ttnn_layer_norm_64,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v128_ttnn_reshape_414 = ttnn.reshape(
            v128_ttnn_layer_norm_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v128_ttnn_reshape_415 = ttnn.reshape(
            v128_ttnn_layer_norm_66,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v128_ttnn_reshape_416 = ttnn.reshape(
            v128_ttnn_layer_norm_67,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v128_util_create_list_387 = [
            v128_ttnn_reshape_416,
            self.cer["utils_constEvalFuncWrapper_30_0"],
        ]
        v128_ttnn_concat_63 = ttnn.concat(
            v128_util_create_list_387,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v_285, v_286, v_287, v_288 = (
            v128_ttnn_reshape_414,
            v128_ttnn_concat_63,
            v128_ttnn_reshape_415,
            v128_ttnn_reshape_413,
        )
        v129_ttnn_matmul_126 = ttnn.matmul(
            v_286,
            self.weights["resampler.layers.0.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v129_ttnn_matmul_127 = ttnn.matmul(
            v_286,
            self.weights["resampler.layers.0.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v129_ttnn_typecast_3 = ttnn.typecast(
            v129_ttnn_matmul_126,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_typecast_4 = ttnn.typecast(
            v129_ttnn_matmul_127,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_reshape_417 = ttnn.reshape(
            v129_ttnn_typecast_3,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_reshape_418 = ttnn.reshape(
            v129_ttnn_typecast_4,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_permute_130 = ttnn.permute(
            v129_ttnn_reshape_417,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v129_ttnn_permute_131 = ttnn.permute(
            v129_ttnn_reshape_418,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v129_ttnn_permute_132 = ttnn.permute(
            v129_ttnn_permute_130,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v129_ttnn_typecast_5 = ttnn.typecast(
            v129_ttnn_permute_131,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_typecast_6 = ttnn.typecast(
            v129_ttnn_permute_132,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_transformer_scaled_dot_product_attention_31 = (
            ttnn.transformer.scaled_dot_product_attention(
                self.cer["utils_constEvalFuncWrapper_30_1"],
                v129_ttnn_typecast_6,
                v129_ttnn_typecast_5,
                attn_mask=None,
                is_causal=False,
                scale=0.35355338454246521,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v129_ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            v129_ttnn_transformer_scaled_dot_product_attention_31,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_reshape_419 = ttnn.reshape(
            v129_ttnn_transformer_concatenate_heads_0,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_matmul_128 = ttnn.matmul(
            v129_ttnn_reshape_419,
            self.weights["resampler.layers.0.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v129_ttnn_reshape_420 = ttnn.reshape(
            v129_ttnn_matmul_128,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v129_ttnn_divide_0 = ttnn.divide(
            v129_ttnn_reshape_420,
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Attention_129_0_0 = v129_ttnn_divide_0
        v130_ttnn_add_188 = ttnn.add(
            Attention_129_0_0,
            self.weights["resampler.latents"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v130_ttnn_layer_norm_68 = ttnn.layer_norm(
            v130_ttnn_add_188,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.0.ff.0.weight"],
            bias=self.weights["resampler.layers.0.ff.0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_289, v_290 = v130_ttnn_add_188, v130_ttnn_layer_norm_68
        v131_ttnn_reshape_421 = ttnn.reshape(
            v_290,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v131_ttnn_matmul_129 = ttnn.matmul(
            v131_ttnn_reshape_421,
            self.weights["resampler.layers.0.ff.1.net.0.proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        Linear_131_0_0 = v131_ttnn_matmul_129
        v132_ttnn_matmul_130 = ttnn.matmul(
            Linear_131_0_0,
            self.weights["resampler.layers.0.ff.1.net.2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v132_ttnn_reshape_422 = ttnn.reshape(
            v132_ttnn_matmul_130,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Linear_132_0_0 = v132_ttnn_reshape_422
        v133_ttnn_add_189 = ttnn.add(
            Linear_132_0_0,
            v_289,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v133_ttnn_layer_norm_69 = ttnn.layer_norm(
            v133_ttnn_add_189,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.1.ln1.weight"],
            bias=self.weights["resampler.layers.1.ln1.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v133_ttnn_reshape_423 = ttnn.reshape(
            v133_ttnn_layer_norm_69,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v133_util_create_list_388 = [v_288, v133_ttnn_reshape_423]
        v133_ttnn_concat_64 = ttnn.concat(
            v133_util_create_list_388,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v_291, v_292, v_293 = (
            v133_ttnn_add_189,
            v133_ttnn_concat_64,
            v133_ttnn_layer_norm_69,
        )
        v134_ttnn_reshape_424 = ttnn.reshape(
            v_293,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_matmul_131 = ttnn.matmul(
            v134_ttnn_reshape_424,
            self.weights["resampler.layers.1.attn.to_q.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v134_ttnn_matmul_132 = ttnn.matmul(
            v_292,
            self.weights["resampler.layers.1.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v134_ttnn_typecast_7 = ttnn.typecast(
            v134_ttnn_matmul_131,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_matmul_133 = ttnn.matmul(
            v_292,
            self.weights["resampler.layers.1.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v134_ttnn_typecast_8 = ttnn.typecast(
            v134_ttnn_matmul_132,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_reshape_425 = ttnn.reshape(
            v134_ttnn_typecast_7,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_typecast_9 = ttnn.typecast(
            v134_ttnn_matmul_133,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_reshape_426 = ttnn.reshape(
            v134_ttnn_typecast_8,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_reshape_427 = ttnn.reshape(
            v134_ttnn_typecast_9,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_permute_133 = ttnn.permute(
            v134_ttnn_reshape_425,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v134_ttnn_permute_134 = ttnn.permute(
            v134_ttnn_reshape_426,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v134_ttnn_permute_135 = ttnn.permute(
            v134_ttnn_reshape_427,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v134_ttnn_typecast_10 = ttnn.typecast(
            v134_ttnn_permute_133,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_typecast_11 = ttnn.typecast(
            v134_ttnn_permute_134,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_permute_136 = ttnn.permute(
            v134_ttnn_permute_135,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v134_ttnn_typecast_12 = ttnn.typecast(
            v134_ttnn_permute_136,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_transformer_scaled_dot_product_attention_32 = (
            ttnn.transformer.scaled_dot_product_attention(
                v134_ttnn_typecast_10,
                v134_ttnn_typecast_12,
                v134_ttnn_typecast_11,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v134_ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            v134_ttnn_transformer_scaled_dot_product_attention_32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_reshape_428 = ttnn.reshape(
            v134_ttnn_transformer_concatenate_heads_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_matmul_134 = ttnn.matmul(
            v134_ttnn_reshape_428,
            self.weights["resampler.layers.1.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v134_ttnn_reshape_429 = ttnn.reshape(
            v134_ttnn_matmul_134,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v134_ttnn_divide_1 = ttnn.divide(
            v134_ttnn_reshape_429,
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Attention_134_0_0 = v134_ttnn_divide_1
        v135_ttnn_add_190 = ttnn.add(
            Attention_134_0_0,
            v_291,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v135_ttnn_layer_norm_70 = ttnn.layer_norm(
            v135_ttnn_add_190,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.1.ff.0.weight"],
            bias=self.weights["resampler.layers.1.ff.0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_294, v_295 = v135_ttnn_add_190, v135_ttnn_layer_norm_70
        v136_ttnn_reshape_430 = ttnn.reshape(
            v_295,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v136_ttnn_matmul_135 = ttnn.matmul(
            v136_ttnn_reshape_430,
            self.weights["resampler.layers.1.ff.1.net.0.proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        Linear_136_0_0 = v136_ttnn_matmul_135
        v137_ttnn_matmul_136 = ttnn.matmul(
            Linear_136_0_0,
            self.weights["resampler.layers.1.ff.1.net.2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v137_ttnn_reshape_431 = ttnn.reshape(
            v137_ttnn_matmul_136,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Linear_137_0_0 = v137_ttnn_reshape_431
        v138_ttnn_add_191 = ttnn.add(
            Linear_137_0_0,
            v_294,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v138_ttnn_layer_norm_71 = ttnn.layer_norm(
            v138_ttnn_add_191,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.2.ln1.weight"],
            bias=self.weights["resampler.layers.2.ln1.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v138_ttnn_reshape_432 = ttnn.reshape(
            v138_ttnn_layer_norm_71,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v138_util_create_list_389 = [v_285, v138_ttnn_reshape_432]
        v138_ttnn_concat_65 = ttnn.concat(
            v138_util_create_list_389,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v_296, v_297, v_298 = (
            v138_ttnn_layer_norm_71,
            v138_ttnn_add_191,
            v138_ttnn_concat_65,
        )
        v139_ttnn_reshape_433 = ttnn.reshape(
            v_296,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_matmul_137 = ttnn.matmul(
            v139_ttnn_reshape_433,
            self.weights["resampler.layers.2.attn.to_q.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v139_ttnn_matmul_138 = ttnn.matmul(
            v_298,
            self.weights["resampler.layers.2.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v139_ttnn_typecast_13 = ttnn.typecast(
            v139_ttnn_matmul_137,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_matmul_139 = ttnn.matmul(
            v_298,
            self.weights["resampler.layers.2.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v139_ttnn_typecast_14 = ttnn.typecast(
            v139_ttnn_matmul_138,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_reshape_434 = ttnn.reshape(
            v139_ttnn_typecast_13,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_typecast_15 = ttnn.typecast(
            v139_ttnn_matmul_139,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_reshape_435 = ttnn.reshape(
            v139_ttnn_typecast_14,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_reshape_436 = ttnn.reshape(
            v139_ttnn_typecast_15,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_permute_137 = ttnn.permute(
            v139_ttnn_reshape_434,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v139_ttnn_permute_138 = ttnn.permute(
            v139_ttnn_reshape_435,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v139_ttnn_permute_139 = ttnn.permute(
            v139_ttnn_reshape_436,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v139_ttnn_typecast_16 = ttnn.typecast(
            v139_ttnn_permute_137,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_typecast_17 = ttnn.typecast(
            v139_ttnn_permute_138,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_permute_140 = ttnn.permute(
            v139_ttnn_permute_139,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v139_ttnn_typecast_18 = ttnn.typecast(
            v139_ttnn_permute_140,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_transformer_scaled_dot_product_attention_33 = (
            ttnn.transformer.scaled_dot_product_attention(
                v139_ttnn_typecast_16,
                v139_ttnn_typecast_18,
                v139_ttnn_typecast_17,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v139_ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
            v139_ttnn_transformer_scaled_dot_product_attention_33,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_reshape_437 = ttnn.reshape(
            v139_ttnn_transformer_concatenate_heads_2,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_matmul_140 = ttnn.matmul(
            v139_ttnn_reshape_437,
            self.weights["resampler.layers.2.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v139_ttnn_reshape_438 = ttnn.reshape(
            v139_ttnn_matmul_140,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v139_ttnn_divide_2 = ttnn.divide(
            v139_ttnn_reshape_438,
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Attention_139_0_0 = v139_ttnn_divide_2
        v140_ttnn_add_192 = ttnn.add(
            Attention_139_0_0,
            v_297,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v140_ttnn_layer_norm_72 = ttnn.layer_norm(
            v140_ttnn_add_192,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.2.ff.0.weight"],
            bias=self.weights["resampler.layers.2.ff.0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_299, v_300 = v140_ttnn_add_192, v140_ttnn_layer_norm_72
        v141_ttnn_reshape_439 = ttnn.reshape(
            v_300,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v141_ttnn_matmul_141 = ttnn.matmul(
            v141_ttnn_reshape_439,
            self.weights["resampler.layers.2.ff.1.net.0.proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        Linear_141_0_0 = v141_ttnn_matmul_141
        v142_ttnn_matmul_142 = ttnn.matmul(
            Linear_141_0_0,
            self.weights["resampler.layers.2.ff.1.net.2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v142_ttnn_reshape_440 = ttnn.reshape(
            v142_ttnn_matmul_142,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Linear_142_0_0 = v142_ttnn_reshape_440
        v143_ttnn_add_193 = ttnn.add(
            Linear_142_0_0,
            v_299,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v143_ttnn_layer_norm_73 = ttnn.layer_norm(
            v143_ttnn_add_193,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.3.ln1.weight"],
            bias=self.weights["resampler.layers.3.ln1.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v143_ttnn_reshape_441 = ttnn.reshape(
            v143_ttnn_layer_norm_73,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v143_util_create_list_390 = [v_287, v143_ttnn_reshape_441]
        v143_ttnn_concat_66 = ttnn.concat(
            v143_util_create_list_390,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v_301, v_302, v_303 = (
            v143_ttnn_layer_norm_73,
            v143_ttnn_concat_66,
            v143_ttnn_add_193,
        )
        v144_ttnn_reshape_442 = ttnn.reshape(
            v_301,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_matmul_143 = ttnn.matmul(
            v144_ttnn_reshape_442,
            self.weights["resampler.layers.3.attn.to_q.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v144_ttnn_matmul_144 = ttnn.matmul(
            v_302,
            self.weights["resampler.layers.3.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v144_ttnn_typecast_19 = ttnn.typecast(
            v144_ttnn_matmul_143,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_matmul_145 = ttnn.matmul(
            v_302,
            self.weights["resampler.layers.3.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v144_ttnn_typecast_20 = ttnn.typecast(
            v144_ttnn_matmul_144,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_reshape_443 = ttnn.reshape(
            v144_ttnn_typecast_19,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_typecast_21 = ttnn.typecast(
            v144_ttnn_matmul_145,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_reshape_444 = ttnn.reshape(
            v144_ttnn_typecast_20,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_reshape_445 = ttnn.reshape(
            v144_ttnn_typecast_21,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_permute_141 = ttnn.permute(
            v144_ttnn_reshape_443,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v144_ttnn_permute_142 = ttnn.permute(
            v144_ttnn_reshape_444,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v144_ttnn_permute_143 = ttnn.permute(
            v144_ttnn_reshape_445,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v144_ttnn_typecast_22 = ttnn.typecast(
            v144_ttnn_permute_141,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_typecast_23 = ttnn.typecast(
            v144_ttnn_permute_142,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_permute_144 = ttnn.permute(
            v144_ttnn_permute_143,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        v144_ttnn_typecast_24 = ttnn.typecast(
            v144_ttnn_permute_144,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_transformer_scaled_dot_product_attention_34 = (
            ttnn.transformer.scaled_dot_product_attention(
                v144_ttnn_typecast_22,
                v144_ttnn_typecast_24,
                v144_ttnn_typecast_23,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        v144_ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
            v144_ttnn_transformer_scaled_dot_product_attention_34,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_reshape_446 = ttnn.reshape(
            v144_ttnn_transformer_concatenate_heads_3,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_matmul_146 = ttnn.matmul(
            v144_ttnn_reshape_446,
            self.weights["resampler.layers.3.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v144_ttnn_reshape_447 = ttnn.reshape(
            v144_ttnn_matmul_146,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v144_ttnn_divide_3 = ttnn.divide(
            v144_ttnn_reshape_447,
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Attention_144_0_0 = v144_ttnn_divide_3
        v145_ttnn_add_194 = ttnn.add(
            Attention_144_0_0,
            v_303,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v145_ttnn_layer_norm_74 = ttnn.layer_norm(
            v145_ttnn_add_194,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.3.ff.0.weight"],
            bias=self.weights["resampler.layers.3.ff.0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        v_304, v_305 = v145_ttnn_layer_norm_74, v145_ttnn_add_194
        v146_ttnn_reshape_449 = ttnn.reshape(
            v_304,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v146_ttnn_matmul_147 = ttnn.matmul(
            v146_ttnn_reshape_449,
            self.weights["resampler.layers.3.ff.1.net.0.proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        v_306, v_307 = v146_ttnn_matmul_147, v146_ttnn_reshape_449
        v148_ttnn_reshape_448 = ttnn.reshape(
            v_305,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        Linear_146_0_0 = v148_ttnn_reshape_448
        v149_ttnn_matmul_148 = ttnn.matmul(
            v_306,
            self.weights["resampler.layers.3.ff.1.net.2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v149_ttnn_add_195 = ttnn.add(
            v149_ttnn_matmul_148,
            Linear_146_0_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        IPAdapterPlusImageProjectionBlock_148_0_0 = v149_ttnn_add_195
        v150_ttnn_matmul_149 = ttnn.matmul(
            IPAdapterPlusImageProjectionBlock_148_0_0,
            self.weights["resampler.proj_out.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        v150_ttnn_add_196 = ttnn.add(
            v150_ttnn_matmul_149,
            self.cer["utils_constEvalFuncWrapper_6_0"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        v150_ttnn_layer_norm_75 = ttnn.layer_norm(
            v150_ttnn_add_196,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.norm_out.weight"],
            bias=self.weights["resampler.norm_out.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        IPAdapterPlusImageProjection_149_0_0 = v150_ttnn_layer_norm_75
        util_create_list_385 = [IPAdapterPlusImageProjection_149_0_0]
        return util_create_list_385
