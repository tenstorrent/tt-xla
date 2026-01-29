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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def _single_weight_reshape_repeat_5120(input):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([1, 257, 1]))
    util_create_list_1 = [ttnn_repeat_0]
    return util_create_list_1


def _three_weight_reshape_repeat_concat_dim2(input):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_2 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_3 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_1, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_2, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_3 = ttnn.repeat(ttnn_reshape_3, ttnn.Shape([1, 257, 1]))
    util_create_list_2 = [ttnn_repeat_1, ttnn_repeat_2, ttnn_repeat_3]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_2,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_3 = [ttnn_concat_0]
    return util_create_list_3


def _single_weight_reshape_repeat_1280(input):
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_4 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_4 = ttnn.repeat(ttnn_reshape_4, ttnn.Shape([1, 257, 1]))
    util_create_list_4 = [ttnn_repeat_4]
    return util_create_list_4


def _three_weight_concat_dim0(input):
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_5 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_6 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_7 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_5 = [ttnn_to_layout_5, ttnn_to_layout_6, ttnn_to_layout_7]
    ttnn_concat_1 = ttnn.concat(
        util_create_list_5,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_6 = [ttnn_concat_1]
    return util_create_list_6


def _single_weight_reshape_repeat_2048(input):
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_10 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_10,
        [1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_7 = ttnn.repeat(ttnn_reshape_7, ttnn.Shape([1, 16, 1]))
    util_create_list_9 = [ttnn_repeat_7]
    return util_create_list_9


def _resampler_attention_query(input):
    utils_DeviceGetter_get_device_31 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_50 = ttnn.to_device(
        input[3],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_50,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_51 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_52 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_52,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 64]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_0 = ttnn.layer_norm(
        input[0],
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_layout_51,
        bias=ttnn_to_layout_52,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_29,
        ttnn_to_layout_50,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 16, 20, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_30,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_permute_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_typecast_0,
        ttnn_full_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_multiply_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_41 = [ttnn_reshape_31, ttnn_typecast_1]
    return util_create_list_41


def _prepare_conv_weights(input):
    utils_DeviceGetter_get_device_67 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
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
    util_create_list_95 = [ttnn_prepare_conv_weights_0]
    return util_create_list_95


def _position_embedding_lookup(input):
    utils_DeviceGetter_get_device_89 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_167 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_167 = ttnn.to_layout(
        ttnn_to_device_167,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_to_layout_167,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_168 = ttnn.to_layout(
        ttnn_typecast_2,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_168 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_168, ttnn_to_device_168, layout=ttnn.Layout.TILE
    )
    ttnn_permute_1 = ttnn.permute(
        ttnn_embedding_0,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_128 = [ttnn_permute_1]
    return util_create_list_128


def _reshape_permute_1280(input):
    utils_DeviceGetter_get_device_143 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_258 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_143,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_258 = ttnn.to_layout(
        ttnn_to_device_258,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_175 = ttnn.reshape(
        ttnn_to_layout_258,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_175,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_200 = [ttnn_permute_2]
    return util_create_list_200


def run_const_evals(weights, cache):
    # fmt: off
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        _full_1_16_1280_ones, cache, "main_const_eval_0"
    )
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    utils_constEvalFuncWrapper_124_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"]], cache, "main_const_eval_125")[0]
    utils_constEvalFuncWrapper_42_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"]], cache, "main_const_eval_43")[0]
    utils_constEvalFuncWrapper_73_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"]], cache, "main_const_eval_74")[0]
    utils_constEvalFuncWrapper_13_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"]], cache, "main_const_eval_14")[0]
    utils_constEvalFuncWrapper_21_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"]], cache, "main_const_eval_22")[0]
    utils_constEvalFuncWrapper_55_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"]], cache, "main_const_eval_56")[0]
    utils_constEvalFuncWrapper_122_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"]], cache, "main_const_eval_123")[0]
    utils_constEvalFuncWrapper_146_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"]], cache, "main_const_eval_147")[0]
    utils_constEvalFuncWrapper_81_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"]], cache, "main_const_eval_82")[0]
    utils_constEvalFuncWrapper_10_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"]], cache, "main_const_eval_11")[0]
    utils_constEvalFuncWrapper_132_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"]], cache, "main_const_eval_133")[0]
    utils_constEvalFuncWrapper_145_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"]], cache, "main_const_eval_146")[0]
    utils_constEvalFuncWrapper_149_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"]], cache, "main_const_eval_150")[0]
    utils_constEvalFuncWrapper_150_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"]], cache, "main_const_eval_151")[0]
    utils_constEvalFuncWrapper_97_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"]], cache, "main_const_eval_98")[0]
    utils_constEvalFuncWrapper_106_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"]], cache, "main_const_eval_107")[0]
    utils_constEvalFuncWrapper_69_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"]], cache, "main_const_eval_70")[0]
    utils_constEvalFuncWrapper_91_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"]], cache, "main_const_eval_92")[0]
    utils_constEvalFuncWrapper_103_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"]], cache, "main_const_eval_104")[0]
    utils_constEvalFuncWrapper_46_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"]], cache, "main_const_eval_47")[0]
    utils_constEvalFuncWrapper_53_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"]], cache, "main_const_eval_54")[0]
    utils_constEvalFuncWrapper_120_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"]], cache, "main_const_eval_121")[0]
    utils_constEvalFuncWrapper_153_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"]], cache, "main_const_eval_154")[0]
    utils_constEvalFuncWrapper_27_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"]], cache, "main_const_eval_28")[0]
    utils_constEvalFuncWrapper_24_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"]], cache, "main_const_eval_25")[0]
    utils_constEvalFuncWrapper_74_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"]], cache, "main_const_eval_75")[0]
    utils_constEvalFuncWrapper_93_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"]], cache, "main_const_eval_94")[0]
    utils_constEvalFuncWrapper_113_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"]], cache, "main_const_eval_114")[0]
    utils_constEvalFuncWrapper_155_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"]], cache, "main_const_eval_156")[0]
    utils_constEvalFuncWrapper_2_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"]], cache, "main_const_eval_3")[0]
    utils_constEvalFuncWrapper_64_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"]], cache, "main_const_eval_65")[0]
    utils_constEvalFuncWrapper_85_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"]], cache, "main_const_eval_86")[0]
    utils_constEvalFuncWrapper_95_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"]], cache, "main_const_eval_96")[0]
    utils_constEvalFuncWrapper_140_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"]], cache, "main_const_eval_141")[0]
    utils_constEvalFuncWrapper_151_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"]], cache, "main_const_eval_152")[0]
    utils_constEvalFuncWrapper_156_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"]], cache, "main_const_eval_157")[0]
    utils_constEvalFuncWrapper_15_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"]], cache, "main_const_eval_16")[0]
    utils_constEvalFuncWrapper_5_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"]], cache, "main_const_eval_6")[0]
    utils_constEvalFuncWrapper_68_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"]], cache, "main_const_eval_69")[0]
    utils_constEvalFuncWrapper_109_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"]], cache, "main_const_eval_110")[0]
    utils_constEvalFuncWrapper_126_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"]], cache, "main_const_eval_127")[0]
    utils_constEvalFuncWrapper_92_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"]], cache, "main_const_eval_93")[0]
    utils_constEvalFuncWrapper_11_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"]], cache, "main_const_eval_12")[0]
    utils_constEvalFuncWrapper_141_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"]], cache, "main_const_eval_142")[0]
    utils_constEvalFuncWrapper_18_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"]], cache, "main_const_eval_19")[0]
    utils_constEvalFuncWrapper_114_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"]], cache, "main_const_eval_115")[0]
    utils_constEvalFuncWrapper_154_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"]], cache, "main_const_eval_155")[0]
    utils_constEvalFuncWrapper_83_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"]], cache, "main_const_eval_84")[0]
    utils_constEvalFuncWrapper_104_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"]], cache, "main_const_eval_105")[0]
    utils_constEvalFuncWrapper_130_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"]], cache, "main_const_eval_131")[0]
    utils_constEvalFuncWrapper_63_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"]], cache, "main_const_eval_64")[0]
    utils_constEvalFuncWrapper_108_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"]], cache, "main_const_eval_109")[0]
    utils_constEvalFuncWrapper_19_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"]], cache, "main_const_eval_20")[0]
    utils_constEvalFuncWrapper_7_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"]], cache, "main_const_eval_8")[0]
    utils_constEvalFuncWrapper_100_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"]], cache, "main_const_eval_101")[0]
    utils_constEvalFuncWrapper_147_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"]], cache, "main_const_eval_148")[0]
    utils_constEvalFuncWrapper_94_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"]], cache, "main_const_eval_95")[0]
    utils_constEvalFuncWrapper_28_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"]], cache, "main_const_eval_29")[0]
    utils_constEvalFuncWrapper_44_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"]], cache, "main_const_eval_45")[0]
    utils_constEvalFuncWrapper_52_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"]], cache, "main_const_eval_53")[0]
    utils_constEvalFuncWrapper_107_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"]], cache, "main_const_eval_108")[0]
    utils_constEvalFuncWrapper_78_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"]], cache, "main_const_eval_79")[0]
    utils_constEvalFuncWrapper_82_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"]], cache, "main_const_eval_83")[0]
    utils_constEvalFuncWrapper_110_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"]], cache, "main_const_eval_111")[0]
    utils_constEvalFuncWrapper_160_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"]], cache, "main_const_eval_161")[0]
    utils_constEvalFuncWrapper_20_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"]], cache, "main_const_eval_21")[0]
    utils_constEvalFuncWrapper_125_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"]], cache, "main_const_eval_126")[0]
    utils_constEvalFuncWrapper_33_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"]], cache, "main_const_eval_34")[0]
    utils_constEvalFuncWrapper_4_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"]], cache, "main_const_eval_5")[0]
    utils_constEvalFuncWrapper_0_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"]], cache, "main_const_eval_1")[0]
    utils_constEvalFuncWrapper_22_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"]], cache, "main_const_eval_23")[0]
    utils_constEvalFuncWrapper_51_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"]], cache, "main_const_eval_52")[0]
    utils_constEvalFuncWrapper_139_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"]], cache, "main_const_eval_140")[0]
    utils_constEvalFuncWrapper_143_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"]], cache, "main_const_eval_144")[0]
    utils_constEvalFuncWrapper_144_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"]], cache, "main_const_eval_145")[0]
    utils_constEvalFuncWrapper_117_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"]], cache, "main_const_eval_118")[0]
    utils_constEvalFuncWrapper_31_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"]], cache, "main_const_eval_32")[0]
    utils_constEvalFuncWrapper_39_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"]], cache, "main_const_eval_40")[0]
    utils_constEvalFuncWrapper_105_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"]], cache, "main_const_eval_106")[0]
    utils_constEvalFuncWrapper_123_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"]], cache, "main_const_eval_124")[0]
    utils_constEvalFuncWrapper_98_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"]], cache, "main_const_eval_99")[0]
    utils_constEvalFuncWrapper_115_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"]], cache, "main_const_eval_116")[0]
    utils_constEvalFuncWrapper_129_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"]], cache, "main_const_eval_130")[0]
    utils_constEvalFuncWrapper_8_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"]], cache, "main_const_eval_9")[0]
    utils_constEvalFuncWrapper_121_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"]], cache, "main_const_eval_122")[0]
    utils_constEvalFuncWrapper_14_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"]], cache, "main_const_eval_15")[0]
    utils_constEvalFuncWrapper_56_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"]], cache, "main_const_eval_57")[0]
    utils_constEvalFuncWrapper_35_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"]], cache, "main_const_eval_36")[0]
    utils_constEvalFuncWrapper_38_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"]], cache, "main_const_eval_39")[0]
    utils_constEvalFuncWrapper_79_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"]], cache, "main_const_eval_80")[0]
    utils_constEvalFuncWrapper_135_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"]], cache, "main_const_eval_136")[0]
    utils_constEvalFuncWrapper_48_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"]], cache, "main_const_eval_49")[0]
    utils_constEvalFuncWrapper_54_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"]], cache, "main_const_eval_55")[0]
    utils_constEvalFuncWrapper_137_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights["resampler.proj_in.bias"]], cache, "main_const_eval_138")[0]
    utils_constEvalFuncWrapper_142_0 = utils.constEvalFuncWrapper(_reshape_permute_1280, [weights["image_encoder.vision_model.embeddings.class_embedding"]], cache, "main_const_eval_143")[0]
    utils_constEvalFuncWrapper_66_0 = utils.constEvalFuncWrapper(_prepare_conv_weights, [weights["image_encoder.vision_model.embeddings.patch_embedding.weight"]], cache, "main_const_eval_67")[0]
    utils_constEvalFuncWrapper_6_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_2048, [weights["resampler.proj_out.bias"]], cache, "main_const_eval_7")[0]
    utils_constEvalFuncWrapper_47_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.bias"]], cache, "main_const_eval_48")[0]
    utils_constEvalFuncWrapper_70_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.weight"]], cache, "main_const_eval_71")[0]
    utils_constEvalFuncWrapper_157_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.weight"]], cache, "main_const_eval_158")[0]
    utils_constEvalFuncWrapper_62_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.bias"]], cache, "main_const_eval_63")[0]
    utils_constEvalFuncWrapper_25_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.weight"]], cache, "main_const_eval_26")[0]
    utils_constEvalFuncWrapper_80_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.bias"]], cache, "main_const_eval_81")[0]
    utils_constEvalFuncWrapper_26_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.weight"]], cache, "main_const_eval_27")[0]
    utils_constEvalFuncWrapper_90_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.bias"]], cache, "main_const_eval_91")[0]
    utils_constEvalFuncWrapper_127_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.weight"]], cache, "main_const_eval_128")[0]
    utils_constEvalFuncWrapper_43_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.bias"]], cache, "main_const_eval_44")[0]
    utils_constEvalFuncWrapper_158_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.bias"]], cache, "main_const_eval_159")[0]
    utils_constEvalFuncWrapper_96_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.weight"]], cache, "main_const_eval_97")[0]
    utils_constEvalFuncWrapper_128_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.weight"]], cache, "main_const_eval_129")[0]
    utils_constEvalFuncWrapper_99_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.bias"]], cache, "main_const_eval_100")[0]
    utils_constEvalFuncWrapper_49_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.weight"]], cache, "main_const_eval_50")[0]
    utils_constEvalFuncWrapper_84_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.bias"]], cache, "main_const_eval_85")[0]
    utils_constEvalFuncWrapper_29_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.weight"]], cache, "main_const_eval_30")[0]
    utils_constEvalFuncWrapper_40_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.bias"]], cache, "main_const_eval_41")[0]
    utils_constEvalFuncWrapper_119_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.weight"]], cache, "main_const_eval_120")[0]
    utils_constEvalFuncWrapper_133_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.bias"]], cache, "main_const_eval_134")[0]
    utils_constEvalFuncWrapper_152_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.weight"]], cache, "main_const_eval_153")[0]
    utils_constEvalFuncWrapper_71_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.bias"]], cache, "main_const_eval_72")[0]
    utils_constEvalFuncWrapper_116_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.bias"]], cache, "main_const_eval_117")[0]
    utils_constEvalFuncWrapper_67_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.weight"]], cache, "main_const_eval_68")[0]
    utils_constEvalFuncWrapper_136_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.bias"]], cache, "main_const_eval_137")[0]
    utils_constEvalFuncWrapper_87_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.weight"]], cache, "main_const_eval_88")[0]
    utils_constEvalFuncWrapper_102_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.weight"]], cache, "main_const_eval_103")[0]
    utils_constEvalFuncWrapper_1_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.bias"]], cache, "main_const_eval_2")[0]
    utils_constEvalFuncWrapper_101_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.bias"]], cache, "main_const_eval_102")[0]
    utils_constEvalFuncWrapper_86_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.weight"]], cache, "main_const_eval_87")[0]
    utils_constEvalFuncWrapper_23_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.weight"]], cache, "main_const_eval_24")[0]
    utils_constEvalFuncWrapper_72_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.bias"]], cache, "main_const_eval_73")[0]
    utils_constEvalFuncWrapper_118_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.bias"]], cache, "main_const_eval_119")[0]
    utils_constEvalFuncWrapper_89_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.weight"]], cache, "main_const_eval_90")[0]
    utils_constEvalFuncWrapper_17_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.bias"]], cache, "main_const_eval_18")[0]
    utils_constEvalFuncWrapper_34_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.weight"]], cache, "main_const_eval_35")[0]
    utils_constEvalFuncWrapper_112_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.weight"]], cache, "main_const_eval_113")[0]
    utils_constEvalFuncWrapper_134_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.bias"]], cache, "main_const_eval_135")[0]
    utils_constEvalFuncWrapper_12_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.weight"]], cache, "main_const_eval_13")[0]
    utils_constEvalFuncWrapper_50_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.bias"]], cache, "main_const_eval_51")[0]
    utils_constEvalFuncWrapper_60_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.bias"]], cache, "main_const_eval_61")[0]
    utils_constEvalFuncWrapper_65_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.weight"]], cache, "main_const_eval_66")[0]
    utils_constEvalFuncWrapper_111_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.bias"]], cache, "main_const_eval_112")[0]
    utils_constEvalFuncWrapper_37_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.weight"]], cache, "main_const_eval_38")[0]
    utils_constEvalFuncWrapper_148_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.weight"]], cache, "main_const_eval_149")[0]
    utils_constEvalFuncWrapper_57_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.bias"]], cache, "main_const_eval_58")[0]
    utils_constEvalFuncWrapper_32_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.bias"]], cache, "main_const_eval_33")[0]
    utils_constEvalFuncWrapper_36_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.weight"]], cache, "main_const_eval_37")[0]
    utils_constEvalFuncWrapper_59_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.bias"]], cache, "main_const_eval_60")[0]
    utils_constEvalFuncWrapper_76_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.weight"]], cache, "main_const_eval_77")[0]
    utils_constEvalFuncWrapper_58_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.bias"]], cache, "main_const_eval_59")[0]
    utils_constEvalFuncWrapper_61_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.weight"]], cache, "main_const_eval_62")[0]
    utils_constEvalFuncWrapper_77_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.bias"]], cache, "main_const_eval_78")[0]
    utils_constEvalFuncWrapper_9_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.weight"]], cache, "main_const_eval_10")[0]
    utils_constEvalFuncWrapper_159_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.weight"]], cache, "main_const_eval_160")[0]
    utils_constEvalFuncWrapper_41_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.bias"]], cache, "main_const_eval_42")[0]
    utils_constEvalFuncWrapper_16_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.bias"]], cache, "main_const_eval_17")[0]
    utils_constEvalFuncWrapper_3_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.weight"]], cache, "main_const_eval_4")[0]
    utils_constEvalFuncWrapper_45_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.bias"]], cache, "main_const_eval_46")[0]
    utils_constEvalFuncWrapper_75_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.weight"]], cache, "main_const_eval_76")[0]
    utils_constEvalFuncWrapper_131_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.bias"]], cache, "main_const_eval_132")[0]
    utils_constEvalFuncWrapper_138_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.weight"]], cache, "main_const_eval_139")[0]
    utils_constEvalFuncWrapper_88_0 = utils.constEvalFuncWrapper(_position_embedding_lookup, [weights["__POSITION_IDS__"], weights["image_encoder.vision_model.embeddings.position_embedding.weight"]], cache, "main_const_eval_89")[0]
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(_resampler_attention_query, [weights["resampler.latents"], weights["resampler.layers.0.ln1.bias"], weights["resampler.layers.0.ln1.weight"], weights["resampler.layers.0.attn.to_q.weight"]], cache, "main_const_eval_31")
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    utils_constEvalFuncWrapper_30_1 = utils_constEvalFuncWrapper_30[1]
    # fmt: on

    return {
        "utils_constEvalFuncWrapperZeroArg_0_0": utils_constEvalFuncWrapperZeroArg_0_0,
        "utils_constEvalFuncWrapper_0_0": utils_constEvalFuncWrapper_0_0,
        "utils_constEvalFuncWrapper_1_0": utils_constEvalFuncWrapper_1_0,
        "utils_constEvalFuncWrapper_2_0": utils_constEvalFuncWrapper_2_0,
        "utils_constEvalFuncWrapper_3_0": utils_constEvalFuncWrapper_3_0,
        "utils_constEvalFuncWrapper_4_0": utils_constEvalFuncWrapper_4_0,
        "utils_constEvalFuncWrapper_5_0": utils_constEvalFuncWrapper_5_0,
        "utils_constEvalFuncWrapper_6_0": utils_constEvalFuncWrapper_6_0,
        "utils_constEvalFuncWrapper_7_0": utils_constEvalFuncWrapper_7_0,
        "utils_constEvalFuncWrapper_8_0": utils_constEvalFuncWrapper_8_0,
        "utils_constEvalFuncWrapper_9_0": utils_constEvalFuncWrapper_9_0,
        "utils_constEvalFuncWrapper_10_0": utils_constEvalFuncWrapper_10_0,
        "utils_constEvalFuncWrapper_11_0": utils_constEvalFuncWrapper_11_0,
        "utils_constEvalFuncWrapper_12_0": utils_constEvalFuncWrapper_12_0,
        "utils_constEvalFuncWrapper_13_0": utils_constEvalFuncWrapper_13_0,
        "utils_constEvalFuncWrapper_14_0": utils_constEvalFuncWrapper_14_0,
        "utils_constEvalFuncWrapper_15_0": utils_constEvalFuncWrapper_15_0,
        "utils_constEvalFuncWrapper_16_0": utils_constEvalFuncWrapper_16_0,
        "utils_constEvalFuncWrapper_17_0": utils_constEvalFuncWrapper_17_0,
        "utils_constEvalFuncWrapper_18_0": utils_constEvalFuncWrapper_18_0,
        "utils_constEvalFuncWrapper_19_0": utils_constEvalFuncWrapper_19_0,
        "utils_constEvalFuncWrapper_20_0": utils_constEvalFuncWrapper_20_0,
        "utils_constEvalFuncWrapper_21_0": utils_constEvalFuncWrapper_21_0,
        "utils_constEvalFuncWrapper_22_0": utils_constEvalFuncWrapper_22_0,
        "utils_constEvalFuncWrapper_23_0": utils_constEvalFuncWrapper_23_0,
        "utils_constEvalFuncWrapper_24_0": utils_constEvalFuncWrapper_24_0,
        "utils_constEvalFuncWrapper_25_0": utils_constEvalFuncWrapper_25_0,
        "utils_constEvalFuncWrapper_26_0": utils_constEvalFuncWrapper_26_0,
        "utils_constEvalFuncWrapper_27_0": utils_constEvalFuncWrapper_27_0,
        "utils_constEvalFuncWrapper_28_0": utils_constEvalFuncWrapper_28_0,
        "utils_constEvalFuncWrapper_29_0": utils_constEvalFuncWrapper_29_0,
        "utils_constEvalFuncWrapper_30_0": utils_constEvalFuncWrapper_30_0,
        "utils_constEvalFuncWrapper_30_1": utils_constEvalFuncWrapper_30_1,
        "utils_constEvalFuncWrapper_31_0": utils_constEvalFuncWrapper_31_0,
        "utils_constEvalFuncWrapper_32_0": utils_constEvalFuncWrapper_32_0,
        "utils_constEvalFuncWrapper_33_0": utils_constEvalFuncWrapper_33_0,
        "utils_constEvalFuncWrapper_34_0": utils_constEvalFuncWrapper_34_0,
        "utils_constEvalFuncWrapper_35_0": utils_constEvalFuncWrapper_35_0,
        "utils_constEvalFuncWrapper_36_0": utils_constEvalFuncWrapper_36_0,
        "utils_constEvalFuncWrapper_37_0": utils_constEvalFuncWrapper_37_0,
        "utils_constEvalFuncWrapper_38_0": utils_constEvalFuncWrapper_38_0,
        "utils_constEvalFuncWrapper_39_0": utils_constEvalFuncWrapper_39_0,
        "utils_constEvalFuncWrapper_40_0": utils_constEvalFuncWrapper_40_0,
        "utils_constEvalFuncWrapper_41_0": utils_constEvalFuncWrapper_41_0,
        "utils_constEvalFuncWrapper_42_0": utils_constEvalFuncWrapper_42_0,
        "utils_constEvalFuncWrapper_43_0": utils_constEvalFuncWrapper_43_0,
        "utils_constEvalFuncWrapper_44_0": utils_constEvalFuncWrapper_44_0,
        "utils_constEvalFuncWrapper_45_0": utils_constEvalFuncWrapper_45_0,
        "utils_constEvalFuncWrapper_46_0": utils_constEvalFuncWrapper_46_0,
        "utils_constEvalFuncWrapper_47_0": utils_constEvalFuncWrapper_47_0,
        "utils_constEvalFuncWrapper_48_0": utils_constEvalFuncWrapper_48_0,
        "utils_constEvalFuncWrapper_49_0": utils_constEvalFuncWrapper_49_0,
        "utils_constEvalFuncWrapper_50_0": utils_constEvalFuncWrapper_50_0,
        "utils_constEvalFuncWrapper_51_0": utils_constEvalFuncWrapper_51_0,
        "utils_constEvalFuncWrapper_52_0": utils_constEvalFuncWrapper_52_0,
        "utils_constEvalFuncWrapper_53_0": utils_constEvalFuncWrapper_53_0,
        "utils_constEvalFuncWrapper_54_0": utils_constEvalFuncWrapper_54_0,
        "utils_constEvalFuncWrapper_55_0": utils_constEvalFuncWrapper_55_0,
        "utils_constEvalFuncWrapper_56_0": utils_constEvalFuncWrapper_56_0,
        "utils_constEvalFuncWrapper_57_0": utils_constEvalFuncWrapper_57_0,
        "utils_constEvalFuncWrapper_58_0": utils_constEvalFuncWrapper_58_0,
        "utils_constEvalFuncWrapper_59_0": utils_constEvalFuncWrapper_59_0,
        "utils_constEvalFuncWrapper_60_0": utils_constEvalFuncWrapper_60_0,
        "utils_constEvalFuncWrapper_61_0": utils_constEvalFuncWrapper_61_0,
        "utils_constEvalFuncWrapper_62_0": utils_constEvalFuncWrapper_62_0,
        "utils_constEvalFuncWrapper_63_0": utils_constEvalFuncWrapper_63_0,
        "utils_constEvalFuncWrapper_64_0": utils_constEvalFuncWrapper_64_0,
        "utils_constEvalFuncWrapper_65_0": utils_constEvalFuncWrapper_65_0,
        "utils_constEvalFuncWrapper_66_0": utils_constEvalFuncWrapper_66_0,
        "utils_constEvalFuncWrapper_67_0": utils_constEvalFuncWrapper_67_0,
        "utils_constEvalFuncWrapper_68_0": utils_constEvalFuncWrapper_68_0,
        "utils_constEvalFuncWrapper_69_0": utils_constEvalFuncWrapper_69_0,
        "utils_constEvalFuncWrapper_70_0": utils_constEvalFuncWrapper_70_0,
        "utils_constEvalFuncWrapper_71_0": utils_constEvalFuncWrapper_71_0,
        "utils_constEvalFuncWrapper_72_0": utils_constEvalFuncWrapper_72_0,
        "utils_constEvalFuncWrapper_73_0": utils_constEvalFuncWrapper_73_0,
        "utils_constEvalFuncWrapper_74_0": utils_constEvalFuncWrapper_74_0,
        "utils_constEvalFuncWrapper_75_0": utils_constEvalFuncWrapper_75_0,
        "utils_constEvalFuncWrapper_76_0": utils_constEvalFuncWrapper_76_0,
        "utils_constEvalFuncWrapper_77_0": utils_constEvalFuncWrapper_77_0,
        "utils_constEvalFuncWrapper_78_0": utils_constEvalFuncWrapper_78_0,
        "utils_constEvalFuncWrapper_79_0": utils_constEvalFuncWrapper_79_0,
        "utils_constEvalFuncWrapper_80_0": utils_constEvalFuncWrapper_80_0,
        "utils_constEvalFuncWrapper_81_0": utils_constEvalFuncWrapper_81_0,
        "utils_constEvalFuncWrapper_82_0": utils_constEvalFuncWrapper_82_0,
        "utils_constEvalFuncWrapper_83_0": utils_constEvalFuncWrapper_83_0,
        "utils_constEvalFuncWrapper_84_0": utils_constEvalFuncWrapper_84_0,
        "utils_constEvalFuncWrapper_85_0": utils_constEvalFuncWrapper_85_0,
        "utils_constEvalFuncWrapper_86_0": utils_constEvalFuncWrapper_86_0,
        "utils_constEvalFuncWrapper_87_0": utils_constEvalFuncWrapper_87_0,
        "utils_constEvalFuncWrapper_88_0": utils_constEvalFuncWrapper_88_0,
        "utils_constEvalFuncWrapper_89_0": utils_constEvalFuncWrapper_89_0,
        "utils_constEvalFuncWrapper_90_0": utils_constEvalFuncWrapper_90_0,
        "utils_constEvalFuncWrapper_91_0": utils_constEvalFuncWrapper_91_0,
        "utils_constEvalFuncWrapper_92_0": utils_constEvalFuncWrapper_92_0,
        "utils_constEvalFuncWrapper_93_0": utils_constEvalFuncWrapper_93_0,
        "utils_constEvalFuncWrapper_94_0": utils_constEvalFuncWrapper_94_0,
        "utils_constEvalFuncWrapper_95_0": utils_constEvalFuncWrapper_95_0,
        "utils_constEvalFuncWrapper_96_0": utils_constEvalFuncWrapper_96_0,
        "utils_constEvalFuncWrapper_97_0": utils_constEvalFuncWrapper_97_0,
        "utils_constEvalFuncWrapper_98_0": utils_constEvalFuncWrapper_98_0,
        "utils_constEvalFuncWrapper_99_0": utils_constEvalFuncWrapper_99_0,
        "utils_constEvalFuncWrapper_100_0": utils_constEvalFuncWrapper_100_0,
        "utils_constEvalFuncWrapper_101_0": utils_constEvalFuncWrapper_101_0,
        "utils_constEvalFuncWrapper_102_0": utils_constEvalFuncWrapper_102_0,
        "utils_constEvalFuncWrapper_103_0": utils_constEvalFuncWrapper_103_0,
        "utils_constEvalFuncWrapper_104_0": utils_constEvalFuncWrapper_104_0,
        "utils_constEvalFuncWrapper_105_0": utils_constEvalFuncWrapper_105_0,
        "utils_constEvalFuncWrapper_106_0": utils_constEvalFuncWrapper_106_0,
        "utils_constEvalFuncWrapper_107_0": utils_constEvalFuncWrapper_107_0,
        "utils_constEvalFuncWrapper_108_0": utils_constEvalFuncWrapper_108_0,
        "utils_constEvalFuncWrapper_109_0": utils_constEvalFuncWrapper_109_0,
        "utils_constEvalFuncWrapper_110_0": utils_constEvalFuncWrapper_110_0,
        "utils_constEvalFuncWrapper_111_0": utils_constEvalFuncWrapper_111_0,
        "utils_constEvalFuncWrapper_112_0": utils_constEvalFuncWrapper_112_0,
        "utils_constEvalFuncWrapper_113_0": utils_constEvalFuncWrapper_113_0,
        "utils_constEvalFuncWrapper_114_0": utils_constEvalFuncWrapper_114_0,
        "utils_constEvalFuncWrapper_115_0": utils_constEvalFuncWrapper_115_0,
        "utils_constEvalFuncWrapper_116_0": utils_constEvalFuncWrapper_116_0,
        "utils_constEvalFuncWrapper_117_0": utils_constEvalFuncWrapper_117_0,
        "utils_constEvalFuncWrapper_118_0": utils_constEvalFuncWrapper_118_0,
        "utils_constEvalFuncWrapper_119_0": utils_constEvalFuncWrapper_119_0,
        "utils_constEvalFuncWrapper_120_0": utils_constEvalFuncWrapper_120_0,
        "utils_constEvalFuncWrapper_121_0": utils_constEvalFuncWrapper_121_0,
        "utils_constEvalFuncWrapper_122_0": utils_constEvalFuncWrapper_122_0,
        "utils_constEvalFuncWrapper_123_0": utils_constEvalFuncWrapper_123_0,
        "utils_constEvalFuncWrapper_124_0": utils_constEvalFuncWrapper_124_0,
        "utils_constEvalFuncWrapper_125_0": utils_constEvalFuncWrapper_125_0,
        "utils_constEvalFuncWrapper_126_0": utils_constEvalFuncWrapper_126_0,
        "utils_constEvalFuncWrapper_127_0": utils_constEvalFuncWrapper_127_0,
        "utils_constEvalFuncWrapper_128_0": utils_constEvalFuncWrapper_128_0,
        "utils_constEvalFuncWrapper_129_0": utils_constEvalFuncWrapper_129_0,
        "utils_constEvalFuncWrapper_130_0": utils_constEvalFuncWrapper_130_0,
        "utils_constEvalFuncWrapper_131_0": utils_constEvalFuncWrapper_131_0,
        "utils_constEvalFuncWrapper_132_0": utils_constEvalFuncWrapper_132_0,
        "utils_constEvalFuncWrapper_133_0": utils_constEvalFuncWrapper_133_0,
        "utils_constEvalFuncWrapper_134_0": utils_constEvalFuncWrapper_134_0,
        "utils_constEvalFuncWrapper_135_0": utils_constEvalFuncWrapper_135_0,
        "utils_constEvalFuncWrapper_136_0": utils_constEvalFuncWrapper_136_0,
        "utils_constEvalFuncWrapper_137_0": utils_constEvalFuncWrapper_137_0,
        "utils_constEvalFuncWrapper_138_0": utils_constEvalFuncWrapper_138_0,
        "utils_constEvalFuncWrapper_139_0": utils_constEvalFuncWrapper_139_0,
        "utils_constEvalFuncWrapper_140_0": utils_constEvalFuncWrapper_140_0,
        "utils_constEvalFuncWrapper_141_0": utils_constEvalFuncWrapper_141_0,
        "utils_constEvalFuncWrapper_142_0": utils_constEvalFuncWrapper_142_0,
        "utils_constEvalFuncWrapper_143_0": utils_constEvalFuncWrapper_143_0,
        "utils_constEvalFuncWrapper_144_0": utils_constEvalFuncWrapper_144_0,
        "utils_constEvalFuncWrapper_145_0": utils_constEvalFuncWrapper_145_0,
        "utils_constEvalFuncWrapper_146_0": utils_constEvalFuncWrapper_146_0,
        "utils_constEvalFuncWrapper_147_0": utils_constEvalFuncWrapper_147_0,
        "utils_constEvalFuncWrapper_148_0": utils_constEvalFuncWrapper_148_0,
        "utils_constEvalFuncWrapper_149_0": utils_constEvalFuncWrapper_149_0,
        "utils_constEvalFuncWrapper_150_0": utils_constEvalFuncWrapper_150_0,
        "utils_constEvalFuncWrapper_151_0": utils_constEvalFuncWrapper_151_0,
        "utils_constEvalFuncWrapper_152_0": utils_constEvalFuncWrapper_152_0,
        "utils_constEvalFuncWrapper_153_0": utils_constEvalFuncWrapper_153_0,
        "utils_constEvalFuncWrapper_154_0": utils_constEvalFuncWrapper_154_0,
        "utils_constEvalFuncWrapper_155_0": utils_constEvalFuncWrapper_155_0,
        "utils_constEvalFuncWrapper_156_0": utils_constEvalFuncWrapper_156_0,
        "utils_constEvalFuncWrapper_157_0": utils_constEvalFuncWrapper_157_0,
        "utils_constEvalFuncWrapper_158_0": utils_constEvalFuncWrapper_158_0,
        "utils_constEvalFuncWrapper_159_0": utils_constEvalFuncWrapper_159_0,
        "utils_constEvalFuncWrapper_160_0": utils_constEvalFuncWrapper_160_0,
    }
