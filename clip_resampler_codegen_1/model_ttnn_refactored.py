# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model wrapper for CLIP Vision Encoder + IP-Adapter Resampler (Refactored)."""

import ttnn
import utils

# Import generated encoder layer classes
from clip_encoder_layers_generated import (
    CLIPEncoderLayerTTNN_0,
    CLIPEncoderLayerTTNN_1,
    CLIPEncoderLayerTTNN_2,
    CLIPEncoderLayerTTNN_3,
    CLIPEncoderLayerTTNN_4,
    CLIPEncoderLayerTTNN_5,
    CLIPEncoderLayerTTNN_6,
    CLIPEncoderLayerTTNN_7,
    CLIPEncoderLayerTTNN_8,
    CLIPEncoderLayerTTNN_9,
    CLIPEncoderLayerTTNN_10,
    CLIPEncoderLayerTTNN_11,
    CLIPEncoderLayerTTNN_12,
    CLIPEncoderLayerTTNN_13,
    CLIPEncoderLayerTTNN_14,
    CLIPEncoderLayerTTNN_15,
    CLIPEncoderLayerTTNN_16,
    CLIPEncoderLayerTTNN_17,
    CLIPEncoderLayerTTNN_18,
    CLIPEncoderLayerTTNN_19,
    CLIPEncoderLayerTTNN_20,
    CLIPEncoderLayerTTNN_21,
    CLIPEncoderLayerTTNN_22,
    CLIPEncoderLayerTTNN_23,
    CLIPEncoderLayerTTNN_24,
    CLIPEncoderLayerTTNN_25,
    CLIPEncoderLayerTTNN_26,
    CLIPEncoderLayerTTNN_27,
    CLIPEncoderLayerTTNN_28,
    CLIPEncoderLayerTTNN_29,
    CLIPEncoderLayerTTNN_30,
)
from consteval import run_const_evals
from models.common.lightweightmodule import LightweightModule

# List of layer classes for easy iteration
ENCODER_LAYER_CLASSES = [
    CLIPEncoderLayerTTNN_0,
    CLIPEncoderLayerTTNN_1,
    CLIPEncoderLayerTTNN_2,
    CLIPEncoderLayerTTNN_3,
    CLIPEncoderLayerTTNN_4,
    CLIPEncoderLayerTTNN_5,
    CLIPEncoderLayerTTNN_6,
    CLIPEncoderLayerTTNN_7,
    CLIPEncoderLayerTTNN_8,
    CLIPEncoderLayerTTNN_9,
    CLIPEncoderLayerTTNN_10,
    CLIPEncoderLayerTTNN_11,
    CLIPEncoderLayerTTNN_12,
    CLIPEncoderLayerTTNN_13,
    CLIPEncoderLayerTTNN_14,
    CLIPEncoderLayerTTNN_15,
    CLIPEncoderLayerTTNN_16,
    CLIPEncoderLayerTTNN_17,
    CLIPEncoderLayerTTNN_18,
    CLIPEncoderLayerTTNN_19,
    CLIPEncoderLayerTTNN_20,
    CLIPEncoderLayerTTNN_21,
    CLIPEncoderLayerTTNN_22,
    CLIPEncoderLayerTTNN_23,
    CLIPEncoderLayerTTNN_24,
    CLIPEncoderLayerTTNN_25,
    CLIPEncoderLayerTTNN_26,
    CLIPEncoderLayerTTNN_27,
    CLIPEncoderLayerTTNN_28,
    CLIPEncoderLayerTTNN_29,
    CLIPEncoderLayerTTNN_30,
]


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    def __init__(self, weights, cache, device):
        self.device = device
        self.weights = weights
        self.cer = run_const_evals(weights, cache)

        # Create encoder layers
        self.encoder_layers = [
            LayerClass(weights, self.cer) for LayerClass in ENCODER_LAYER_CLASSES
        ]

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

        # Vision Embeddings
        embeddings = self.CLIPVisionEmbeddings_0_0(
            pixel_values,
            self.cer["utils_constEvalFuncWrapper_142_0"],
            self.cer["utils_constEvalFuncWrapper_88_0"],
            self.cer["utils_constEvalFuncWrapper_66_0"],
        )

        # Pre-LayerNorm
        hidden_states = self.LayerNorm_1_0(
            self.weights["image_encoder.vision_model.pre_layrnorm.bias"],
            self.weights["image_encoder.vision_model.pre_layrnorm.weight"],
            embeddings,
        )

        # Encoder layers (31 layers, 0-30)
        residual = hidden_states
        for layer in self.encoder_layers:
            residual, hidden_states = layer(hidden_states, residual)

        # After last layer, hidden_states is None, residual contains the output
        encoder_output = residual

        # Continue with IP-Adapter resampler
        # Linear projection (proj_in equivalent)
        Linear_127_0_0 = self.Linear_127_0(
            self.cer["utils_constEvalFuncWrapper_137_0"],
            self.weights["resampler.proj_in.weight"],
            encoder_output,
        )

        # IP-Adapter blocks (keeping original method calls for now)
        v_285, v_286, v_287, v_288 = self.IPAdapterPlusImageProjectionBlock_128_0(
            self.cer["utils_constEvalFuncWrapper_30_0"],
            self.weights["resampler.layers.2.ln0.weight"],
            self.weights["resampler.layers.3.ln0.weight"],
            Linear_127_0_0,
            self.weights["resampler.layers.1.ln0.weight"],
            self.weights["resampler.layers.3.ln0.bias"],
            self.weights["resampler.layers.0.ln0.weight"],
            self.weights["resampler.layers.1.ln0.bias"],
            self.weights["resampler.layers.0.ln0.bias"],
            self.weights["resampler.layers.2.ln0.bias"],
        )
        Attention_129_0_0 = self.Attention_129_0(
            v_286,
            self.weights["resampler.layers.0.attn.to_out.0.weight"],
            self.weights["resampler.layers.0.attn.to_k.weight"],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights["resampler.layers.0.attn.to_v.weight"],
            self.cer["utils_constEvalFuncWrapper_30_1"],
        )
        v_289, v_290 = self.IPAdapterPlusImageProjectionBlock_130_0(
            self.weights["resampler.layers.0.ff.0.weight"],
            Attention_129_0_0,
            self.weights["resampler.latents"],
            self.weights["resampler.layers.0.ff.0.bias"],
        )
        Linear_131_0_0 = self.Linear_131_0(
            self.weights["resampler.layers.0.ff.1.net.0.proj.weight"], v_290
        )
        Linear_132_0_0 = self.Linear_132_0(
            self.weights["resampler.layers.0.ff.1.net.2.weight"], Linear_131_0_0
        )
        v_291, v_292, v_293 = self.IPAdapterPlusImageProjectionBlock_133_0(
            v_289,
            Linear_132_0_0,
            self.weights["resampler.layers.1.ln1.weight"],
            self.weights["resampler.layers.1.ln1.bias"],
            v_288,
        )
        Attention_134_0_0 = self.Attention_134_0(
            self.weights["resampler.layers.1.attn.to_q.weight"],
            self.weights["resampler.layers.1.attn.to_k.weight"],
            self.weights["resampler.layers.1.attn.to_out.0.weight"],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights["resampler.layers.1.attn.to_v.weight"],
            v_292,
            v_293,
        )
        v_294, v_295 = self.IPAdapterPlusImageProjectionBlock_135_0(
            v_291,
            self.weights["resampler.layers.1.ff.0.weight"],
            Attention_134_0_0,
            self.weights["resampler.layers.1.ff.0.bias"],
        )
        Linear_136_0_0 = self.Linear_136_0(
            self.weights["resampler.layers.1.ff.1.net.0.proj.weight"], v_295
        )
        Linear_137_0_0 = self.Linear_137_0(
            Linear_136_0_0, self.weights["resampler.layers.1.ff.1.net.2.weight"]
        )
        v_296, v_297, v_298 = self.IPAdapterPlusImageProjectionBlock_138_0(
            v_285,
            self.weights["resampler.layers.2.ln1.weight"],
            v_294,
            Linear_137_0_0,
            self.weights["resampler.layers.2.ln1.bias"],
        )
        Attention_139_0_0 = self.Attention_139_0(
            self.weights["resampler.layers.2.attn.to_k.weight"],
            self.weights["resampler.layers.2.attn.to_v.weight"],
            self.weights["resampler.layers.2.attn.to_q.weight"],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights["resampler.layers.2.attn.to_out.0.weight"],
            v_296,
            v_298,
        )
        v_299, v_300 = self.IPAdapterPlusImageProjectionBlock_140_0(
            self.weights["resampler.layers.2.ff.0.bias"],
            Attention_139_0_0,
            v_297,
            self.weights["resampler.layers.2.ff.0.weight"],
        )
        Linear_141_0_0 = self.Linear_141_0(
            self.weights["resampler.layers.2.ff.1.net.0.proj.weight"], v_300
        )
        Linear_142_0_0 = self.Linear_142_0(
            Linear_141_0_0, self.weights["resampler.layers.2.ff.1.net.2.weight"]
        )
        v_301, v_302, v_303 = self.IPAdapterPlusImageProjectionBlock_143_0(
            Linear_142_0_0,
            v_287,
            self.weights["resampler.layers.3.ln1.bias"],
            self.weights["resampler.layers.3.ln1.weight"],
            v_299,
        )
        Attention_144_0_0 = self.Attention_144_0(
            self.weights["resampler.layers.3.attn.to_k.weight"],
            self.weights["resampler.layers.3.attn.to_q.weight"],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights["resampler.layers.3.attn.to_v.weight"],
            self.weights["resampler.layers.3.attn.to_out.0.weight"],
            v_301,
            v_302,
        )
        v_304, v_305 = self.IPAdapterPlusImageProjectionBlock_145_0(
            self.weights["resampler.layers.3.ff.0.bias"],
            self.weights["resampler.layers.3.ff.0.weight"],
            Attention_144_0_0,
            v_303,
        )
        v_306, v_307 = self.Linear_147_0(
            v_304, self.weights["resampler.layers.3.ff.1.net.0.proj.weight"]
        )
        self.Linear_150_0(v_307)
        Linear_146_0_0 = self.Linear_146_0(v_305)
        IPAdapterPlusImageProjectionBlock_148_0_0 = (
            self.IPAdapterPlusImageProjectionBlock_148_0(
                Linear_146_0_0,
                v_306,
                self.weights["resampler.layers.3.ff.1.net.2.weight"],
            )
        )
        IPAdapterPlusImageProjection_149_0_0 = self.IPAdapterPlusImageProjection_149_0(
            v_305,
            self.weights["resampler.norm_out.weight"],
            self.cer["utils_constEvalFuncWrapper_6_0"],
            IPAdapterPlusImageProjectionBlock_148_0_0,
            self.weights["resampler.proj_out.weight"],
            self.weights["resampler.norm_out.bias"],
        )
        self.IPAdapterPlusImageProjectionBlock_151_0(v_306)
        util_create_list_385 = [IPAdapterPlusImageProjection_149_0_0]
        return util_create_list_385

    # Keep existing helper methods for embeddings, layernorm, and IP-adapter
    # (These will be copied from the original model_ttnn.py)
    def CLIPVisionEmbeddings_0_0(self, input_0, input_1, input_2, input_3):
        ttnn_to_layout_287 = ttnn.to_layout(
            input_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        utils_DeviceGetter_get_device_162 = utils.DeviceGetter.get_device((1, 1))
        ttnn_permute_3 = ttnn.permute(
            ttnn_to_layout_287,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_192 = ttnn.reshape(
            ttnn_permute_3,
            [1, 1, 50176, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_conv2d_0 = ttnn.conv2d(
            input_tensor=ttnn_reshape_192,
            weight_tensor=input_3,
            device=utils_DeviceGetter_get_device_162,
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
        ttnn_reshape_193 = ttnn.reshape(
            ttnn_conv2d_0,
            [1, 16, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_4 = ttnn.permute(
            ttnn_reshape_193,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_194 = ttnn.reshape(
            ttnn_permute_4,
            [1, 1280, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_386 = [input_1, ttnn_reshape_194]
        ttnn_concat_62 = ttnn.concat(
            util_create_list_386,
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_add_0 = ttnn.add(
            ttnn_concat_62,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_5 = ttnn.permute(
            ttnn_add_0,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        return ttnn_permute_5

    def LayerNorm_1_0(self, input_0, input_1, input_2):
        ttnn_layer_norm_1 = ttnn.layer_norm(
            input_2,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_1

    def Linear_127_0(self, input_0, input_1, input_2):
        ttnn_matmul_125 = ttnn.matmul(
            input_2,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_187 = ttnn.add(
            ttnn_matmul_125,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_187

    def IPAdapterPlusImageProjectionBlock_128_0(
        self,
        input_0,
        input_1,
        input_2,
        input_3,
        input_4,
        input_5,
        input_6,
        input_7,
        input_8,
        input_9,
    ):
        ttnn_layer_norm_64 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_4,
            bias=input_7,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_65 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_9,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_66 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_5,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_67 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_6,
            bias=input_8,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_413 = ttnn.reshape(
            ttnn_layer_norm_64,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_414 = ttnn.reshape(
            ttnn_layer_norm_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_415 = ttnn.reshape(
            ttnn_layer_norm_66,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_416 = ttnn.reshape(
            ttnn_layer_norm_67,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_387 = [ttnn_reshape_416, input_0]
        ttnn_concat_63 = ttnn.concat(
            util_create_list_387,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_414, ttnn_concat_63, ttnn_reshape_415, ttnn_reshape_413

    def Attention_129_0(self, input_0, input_1, input_2, input_3, input_4, input_5):
        ttnn_matmul_126 = ttnn.matmul(
            input_0,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_127 = ttnn.matmul(
            input_0,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_3 = ttnn.typecast(
            ttnn_matmul_126,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_4 = ttnn.typecast(
            ttnn_matmul_127,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_417 = ttnn.reshape(
            ttnn_typecast_3,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_418 = ttnn.reshape(
            ttnn_typecast_4,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_130 = ttnn.permute(
            ttnn_reshape_417,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_131 = ttnn.permute(
            ttnn_reshape_418,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_132 = ttnn.permute(
            ttnn_permute_130,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_5 = ttnn.typecast(
            ttnn_permute_131,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_permute_132,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_31 = (
            ttnn.transformer.scaled_dot_product_attention(
                input_5,
                ttnn_typecast_6,
                ttnn_typecast_5,
                attn_mask=None,
                is_causal=False,
                scale=0.35355338454246521,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_31,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_419 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_128 = ttnn.matmul(
            ttnn_reshape_419,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_420 = ttnn.reshape(
            ttnn_matmul_128,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_0 = ttnn.divide(
            ttnn_reshape_420,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_0

    def IPAdapterPlusImageProjectionBlock_130_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_188 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_68 = ttnn.layer_norm(
            ttnn_add_188,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_188, ttnn_layer_norm_68

    def Linear_131_0(self, input_0, input_1):
        ttnn_reshape_421 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_129 = ttnn.matmul(
            ttnn_reshape_421,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_129

    def Linear_132_0(self, input_0, input_1):
        ttnn_matmul_130 = ttnn.matmul(
            input_1,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_422 = ttnn.reshape(
            ttnn_matmul_130,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_422

    def IPAdapterPlusImageProjectionBlock_133_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_189 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_69 = ttnn.layer_norm(
            ttnn_add_189,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_423 = ttnn.reshape(
            ttnn_layer_norm_69,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_388 = [input_4, ttnn_reshape_423]
        ttnn_concat_64 = ttnn.concat(
            util_create_list_388,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_189, ttnn_concat_64, ttnn_layer_norm_69

    def Attention_134_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_424 = ttnn.reshape(
            input_6,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_131 = ttnn.matmul(
            ttnn_reshape_424,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_132 = ttnn.matmul(
            input_5,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_7 = ttnn.typecast(
            ttnn_matmul_131,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_133 = ttnn.matmul(
            input_5,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_8 = ttnn.typecast(
            ttnn_matmul_132,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_425 = ttnn.reshape(
            ttnn_typecast_7,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_9 = ttnn.typecast(
            ttnn_matmul_133,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_426 = ttnn.reshape(
            ttnn_typecast_8,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_427 = ttnn.reshape(
            ttnn_typecast_9,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_133 = ttnn.permute(
            ttnn_reshape_425,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_134 = ttnn.permute(
            ttnn_reshape_426,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_135 = ttnn.permute(
            ttnn_reshape_427,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_10 = ttnn.typecast(
            ttnn_permute_133,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_11 = ttnn.typecast(
            ttnn_permute_134,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_136 = ttnn.permute(
            ttnn_permute_135,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_12 = ttnn.typecast(
            ttnn_permute_136,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_32 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_10,
                ttnn_typecast_12,
                ttnn_typecast_11,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_428 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_134 = ttnn.matmul(
            ttnn_reshape_428,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_429 = ttnn.reshape(
            ttnn_matmul_134,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_1 = ttnn.divide(
            ttnn_reshape_429,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_1

    def IPAdapterPlusImageProjectionBlock_135_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_190 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_70 = ttnn.layer_norm(
            ttnn_add_190,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_190, ttnn_layer_norm_70

    def Linear_136_0(self, input_0, input_1):
        ttnn_reshape_430 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_135 = ttnn.matmul(
            ttnn_reshape_430,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_135

    def Linear_137_0(self, input_0, input_1):
        ttnn_matmul_136 = ttnn.matmul(
            input_0,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_431 = ttnn.reshape(
            ttnn_matmul_136,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_431

    def IPAdapterPlusImageProjectionBlock_138_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_191 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_71 = ttnn.layer_norm(
            ttnn_add_191,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_4,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_432 = ttnn.reshape(
            ttnn_layer_norm_71,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_389 = [input_0, ttnn_reshape_432]
        ttnn_concat_65 = ttnn.concat(
            util_create_list_389,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_layer_norm_71, ttnn_add_191, ttnn_concat_65

    def Attention_139_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_433 = ttnn.reshape(
            input_5,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_137 = ttnn.matmul(
            ttnn_reshape_433,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_138 = ttnn.matmul(
            input_6,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_13 = ttnn.typecast(
            ttnn_matmul_137,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_139 = ttnn.matmul(
            input_6,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_14 = ttnn.typecast(
            ttnn_matmul_138,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_434 = ttnn.reshape(
            ttnn_typecast_13,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_15 = ttnn.typecast(
            ttnn_matmul_139,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_435 = ttnn.reshape(
            ttnn_typecast_14,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_436 = ttnn.reshape(
            ttnn_typecast_15,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_137 = ttnn.permute(
            ttnn_reshape_434,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_138 = ttnn.permute(
            ttnn_reshape_435,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_139 = ttnn.permute(
            ttnn_reshape_436,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_16 = ttnn.typecast(
            ttnn_permute_137,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_17 = ttnn.typecast(
            ttnn_permute_138,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_140 = ttnn.permute(
            ttnn_permute_139,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_18 = ttnn.typecast(
            ttnn_permute_140,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_33 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_16,
                ttnn_typecast_18,
                ttnn_typecast_17,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_33,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_437 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_2,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_140 = ttnn.matmul(
            ttnn_reshape_437,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_438 = ttnn.reshape(
            ttnn_matmul_140,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_2 = ttnn.divide(
            ttnn_reshape_438,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_2

    def IPAdapterPlusImageProjectionBlock_140_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_192 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_72 = ttnn.layer_norm(
            ttnn_add_192,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_192, ttnn_layer_norm_72

    def Linear_141_0(self, input_0, input_1):
        ttnn_reshape_439 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_141 = ttnn.matmul(
            ttnn_reshape_439,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_141

    def Linear_142_0(self, input_0, input_1):
        ttnn_matmul_142 = ttnn.matmul(
            input_0,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_440 = ttnn.reshape(
            ttnn_matmul_142,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_440

    def IPAdapterPlusImageProjectionBlock_143_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_193 = ttnn.add(
            input_0,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_73 = ttnn.layer_norm(
            ttnn_add_193,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_441 = ttnn.reshape(
            ttnn_layer_norm_73,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_390 = [input_1, ttnn_reshape_441]
        ttnn_concat_66 = ttnn.concat(
            util_create_list_390,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_layer_norm_73, ttnn_concat_66, ttnn_add_193

    def Attention_144_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_442 = ttnn.reshape(
            input_5,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_143 = ttnn.matmul(
            ttnn_reshape_442,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_144 = ttnn.matmul(
            input_6,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_19 = ttnn.typecast(
            ttnn_matmul_143,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_145 = ttnn.matmul(
            input_6,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_20 = ttnn.typecast(
            ttnn_matmul_144,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_443 = ttnn.reshape(
            ttnn_typecast_19,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_21 = ttnn.typecast(
            ttnn_matmul_145,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_444 = ttnn.reshape(
            ttnn_typecast_20,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_445 = ttnn.reshape(
            ttnn_typecast_21,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_141 = ttnn.permute(
            ttnn_reshape_443,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_142 = ttnn.permute(
            ttnn_reshape_444,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_143 = ttnn.permute(
            ttnn_reshape_445,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_22 = ttnn.typecast(
            ttnn_permute_141,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_23 = ttnn.typecast(
            ttnn_permute_142,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_144 = ttnn.permute(
            ttnn_permute_143,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_24 = ttnn.typecast(
            ttnn_permute_144,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_34 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_22,
                ttnn_typecast_24,
                ttnn_typecast_23,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_34,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_446 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_3,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_146 = ttnn.matmul(
            ttnn_reshape_446,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_447 = ttnn.reshape(
            ttnn_matmul_146,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_3 = ttnn.divide(
            ttnn_reshape_447,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_3

    def IPAdapterPlusImageProjectionBlock_145_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_194 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_74 = ttnn.layer_norm(
            ttnn_add_194,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_74, ttnn_add_194

    def Linear_146_0(self, input):
        ttnn_reshape_448 = ttnn.reshape(
            input,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_448

    def Linear_147_0(self, input_0, input_1):
        ttnn_reshape_449 = ttnn.reshape(
            input_0,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_147 = ttnn.matmul(
            ttnn_reshape_449,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_147, ttnn_reshape_449

    def IPAdapterPlusImageProjectionBlock_148_0(self, input_0, input_1, input_2):
        ttnn_matmul_148 = ttnn.matmul(
            input_1,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_195 = ttnn.add(
            ttnn_matmul_148,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_195

    def IPAdapterPlusImageProjection_149_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5
    ):
        ttnn_matmul_149 = ttnn.matmul(
            input_3,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_196 = ttnn.add(
            ttnn_matmul_149,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_75 = ttnn.layer_norm(
            ttnn_add_196,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_5,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_75

    def Linear_150_0(self, input):
        return

    def IPAdapterPlusImageProjectionBlock_151_0(self, input):
        return
