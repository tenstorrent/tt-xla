# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for CLIP Resampler model."""

import ttnn
import utils


def _full_1_16_1280_ones():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 16, 1280]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn_full_0


def _single_weight_reshape_repeat_5120(input):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 5120],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([1, 257, 1]))
    return ttnn_repeat_0


def _three_weight_reshape_repeat_concat_dim2(input):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_2 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_3 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_1, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_2, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_3 = ttnn.repeat(ttnn_reshape_3, ttnn.Shape([1, 257, 1]))
    ttnn_concat_0 = ttnn.concat(
        [ttnn_repeat_1, ttnn_repeat_2, ttnn_repeat_3],
        2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn_concat_0


def _single_weight_reshape_repeat_1280(input):
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_4 = ttnn.to_device(
        input,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_4 = ttnn.repeat(ttnn_reshape_4, ttnn.Shape([1, 257, 1]))
    return ttnn_repeat_4


def _three_weight_concat_dim0(input):
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_5 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_6 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_7 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_concat_1 = ttnn.concat(
        [ttnn_to_layout_5, ttnn_to_layout_6, ttnn_to_layout_7],
        0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn_concat_1


def _single_weight_reshape_repeat_2048(input):
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_10 = ttnn.to_device(
        input,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_10,
        [1, 1, 2048],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_repeat_7 = ttnn.repeat(ttnn_reshape_7, ttnn.Shape([1, 16, 1]))
    return ttnn_repeat_7


def _resampler_attention_query(input):
    utils_DeviceGetter_get_device_31 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_50 = ttnn.to_device(
        input[3],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_50,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_51 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_52 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_52,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 64]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_layer_norm_0 = ttnn.layer_norm(
        input[0],
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_layout_51,
        bias=ttnn_to_layout_52,
        residual_input_tensor=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=None,
    )
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_29,
        ttnn_to_layout_50,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 16, 20, 64],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_30,
        [0, 2, 1, 3],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_permute_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_typecast_0,
        ttnn_full_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_multiply_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn_reshape_31, ttnn_typecast_1


def _prepare_conv_weights(input):
    utils_DeviceGetter_get_device_67 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=input,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=1280,
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[14, 14],
        stride=[14, 14],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=utils_DeviceGetter_get_device_67,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    return ttnn_prepare_conv_weights_0


def _position_embedding_lookup(input):
    utils_DeviceGetter_get_device_89 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_167 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_167 = ttnn.to_layout(
        ttnn_to_device_167,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_to_layout_167,
        ttnn.DataType.UINT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_168 = ttnn.to_layout(
        ttnn_typecast_2,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_device_168 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_168, ttnn_to_device_168, layout=ttnn.Layout.TILE
    )
    ttnn_permute_1 = ttnn.permute(
        ttnn_embedding_0,
        [0, 2, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    return ttnn_permute_1


def _reshape_permute_1280(input):
    utils_DeviceGetter_get_device_143 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_258 = ttnn.to_device(
        input,
        device=utils_DeviceGetter_get_device_143,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_258 = ttnn.to_layout(
        ttnn_to_device_258,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_reshape_175 = ttnn.reshape(
        ttnn_to_layout_258,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_175,
        [0, 2, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    return ttnn_permute_2


def run_const_evals(weights):
    # fmt: off
    weights["utils_constEvalFuncWrapperZeroArg_0_0"] = _full_1_16_1280_ones()
    weights["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"])
    weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"])
    weights["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"])
    weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"])
    weights["resampler.proj_in.bias"] = _single_weight_reshape_repeat_1280(weights["resampler.proj_in.bias"])
    weights["image_encoder.vision_model.embeddings.class_embedding"] = _reshape_permute_1280(weights["image_encoder.vision_model.embeddings.class_embedding"])
    weights["image_encoder.vision_model.embeddings.patch_embedding.weight"] = _prepare_conv_weights(weights["image_encoder.vision_model.embeddings.patch_embedding.weight"])
    weights["resampler.proj_out.bias"] = _single_weight_reshape_repeat_2048(weights["resampler.proj_out.bias"])
    weights["utils_constEvalFuncWrapper_47_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_70_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_157_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_62_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_25_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_80_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_26_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_90_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_127_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_43_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_158_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_96_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_128_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_99_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_49_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_84_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_29_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_40_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_119_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_133_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_152_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_71_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_116_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_67_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_136_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_87_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_102_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_1_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_101_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_86_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_23_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_72_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_118_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_89_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_17_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_34_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_112_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_134_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_12_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_50_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_60_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_65_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_111_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_37_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_148_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_57_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_32_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_36_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_59_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_76_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_58_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_61_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_77_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_9_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_159_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_41_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_16_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_3_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_45_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_75_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_131_0"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.bias"]])
    weights["utils_constEvalFuncWrapper_138_0"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.weight"]])
    weights["utils_constEvalFuncWrapper_88_0"] = _position_embedding_lookup([weights["__POSITION_IDS__"], weights["image_encoder.vision_model.embeddings.position_embedding.weight"]])
    _tmp_30 = _resampler_attention_query([weights["resampler.latents"], weights["resampler.layers.0.ln1.bias"], weights["resampler.layers.0.ln1.weight"], weights["resampler.layers.0.attn.to_q.weight"]])
    weights["utils_constEvalFuncWrapper_30_0"] = _tmp_30[0]
    weights["utils_constEvalFuncWrapper_30_1"] = _tmp_30[1]
    # fmt: on

    return weights
