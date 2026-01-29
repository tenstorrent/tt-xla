# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for CLIP Resampler model."""

import ttnn
import utils


def main_const_eval_0():
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


def main_const_eval_1(input):
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


def main_const_eval_2(input):
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


def main_const_eval_3(input):
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


def main_const_eval_4(input):
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


def main_const_eval_7(input):
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


def main_const_eval_31(input):
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


def main_const_eval_67(input):
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


def main_const_eval_89(input):
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


def main_const_eval_143(input):
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


def run_const_evals(input, cache):
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, cache, const_1
    )
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_2 = main_const_eval_1
    util_create_list_224 = [
        input["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"]
    ]
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_2, util_create_list_224, cache, const_3
    )
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_4 = main_const_eval_2
    util_create_list_225 = [
        input["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.bias"],
    ]
    const_5 = "main_const_eval_2"
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_4, util_create_list_225, cache, const_5
    )
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_6 = main_const_eval_3
    util_create_list_226 = [
        input["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"]
    ]
    const_7 = "main_const_eval_3"
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_6, util_create_list_226, cache, const_7
    )
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_8 = main_const_eval_4
    util_create_list_227 = [
        input["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.weight"],
    ]
    const_9 = "main_const_eval_4"
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_8, util_create_list_227, cache, const_9
    )
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_10 = main_const_eval_1
    util_create_list_228 = [
        input["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"]
    ]
    const_11 = "main_const_eval_5"
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_10, util_create_list_228, cache, const_11
    )
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_12 = main_const_eval_1
    util_create_list_229 = [
        input["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"]
    ]
    const_13 = "main_const_eval_6"
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(
        const_12, util_create_list_229, cache, const_13
    )
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_14 = main_const_eval_7
    util_create_list_230 = [input["resampler.proj_out.bias"]]
    const_15 = "main_const_eval_7"
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(
        const_14, util_create_list_230, cache, const_15
    )
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_16 = main_const_eval_3
    util_create_list_231 = [
        input["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"]
    ]
    const_17 = "main_const_eval_8"
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(
        const_16, util_create_list_231, cache, const_17
    )
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_18 = main_const_eval_3
    util_create_list_232 = [
        input["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"]
    ]
    const_19 = "main_const_eval_9"
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(
        const_18, util_create_list_232, cache, const_19
    )
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_20 = main_const_eval_4
    util_create_list_233 = [
        input["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.weight"],
    ]
    const_21 = "main_const_eval_10"
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(
        const_20, util_create_list_233, cache, const_21
    )
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_22 = main_const_eval_3
    util_create_list_234 = [
        input["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"]
    ]
    const_23 = "main_const_eval_11"
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_22, util_create_list_234, cache, const_23
    )
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_24 = main_const_eval_3
    util_create_list_235 = [
        input["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"]
    ]
    const_25 = "main_const_eval_12"
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_24, util_create_list_235, cache, const_25
    )
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_26 = main_const_eval_4
    util_create_list_236 = [
        input["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.weight"],
    ]
    const_27 = "main_const_eval_13"
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_26, util_create_list_236, cache, const_27
    )
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_28 = main_const_eval_1
    util_create_list_237 = [
        input["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"]
    ]
    const_29 = "main_const_eval_14"
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_28, util_create_list_237, cache, const_29
    )
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_30 = main_const_eval_1
    util_create_list_238 = [
        input["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"]
    ]
    const_31 = "main_const_eval_15"
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_30, util_create_list_238, cache, const_31
    )
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_32 = main_const_eval_3
    util_create_list_239 = [
        input["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"]
    ]
    const_33 = "main_const_eval_16"
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_32, util_create_list_239, cache, const_33
    )
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_34 = main_const_eval_2
    util_create_list_240 = [
        input["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.bias"],
    ]
    const_35 = "main_const_eval_17"
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_34, util_create_list_240, cache, const_35
    )
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_36 = main_const_eval_2
    util_create_list_241 = [
        input["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.bias"],
    ]
    const_37 = "main_const_eval_18"
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_36, util_create_list_241, cache, const_37
    )
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_38 = main_const_eval_3
    util_create_list_242 = [
        input["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"]
    ]
    const_39 = "main_const_eval_19"
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_38, util_create_list_242, cache, const_39
    )
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_40 = main_const_eval_3
    util_create_list_243 = [
        input["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"]
    ]
    const_41 = "main_const_eval_20"
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_40, util_create_list_243, cache, const_41
    )
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_42 = main_const_eval_3
    util_create_list_244 = [
        input["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"]
    ]
    const_43 = "main_const_eval_21"
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_42, util_create_list_244, cache, const_43
    )
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_44 = main_const_eval_3
    util_create_list_245 = [
        input["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"]
    ]
    const_45 = "main_const_eval_22"
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_44, util_create_list_245, cache, const_45
    )
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_46 = main_const_eval_3
    util_create_list_246 = [
        input["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"]
    ]
    const_47 = "main_const_eval_23"
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_46, util_create_list_246, cache, const_47
    )
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_48 = main_const_eval_4
    util_create_list_247 = [
        input["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.weight"],
    ]
    const_49 = "main_const_eval_24"
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_48, util_create_list_247, cache, const_49
    )
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_50 = main_const_eval_1
    util_create_list_248 = [
        input["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"]
    ]
    const_51 = "main_const_eval_25"
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_50, util_create_list_248, cache, const_51
    )
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_52 = main_const_eval_4
    util_create_list_249 = [
        input["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.weight"],
    ]
    const_53 = "main_const_eval_26"
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_52, util_create_list_249, cache, const_53
    )
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_54 = main_const_eval_4
    util_create_list_250 = [
        input["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.weight"],
    ]
    const_55 = "main_const_eval_27"
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_54, util_create_list_250, cache, const_55
    )
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_56 = main_const_eval_1
    util_create_list_251 = [
        input["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"]
    ]
    const_57 = "main_const_eval_28"
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_56, util_create_list_251, cache, const_57
    )
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_58 = main_const_eval_3
    util_create_list_252 = [
        input["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"]
    ]
    const_59 = "main_const_eval_29"
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_58, util_create_list_252, cache, const_59
    )
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_60 = main_const_eval_4
    util_create_list_253 = [
        input["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.weight"],
    ]
    const_61 = "main_const_eval_30"
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_60, util_create_list_253, cache, const_61
    )
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_62 = main_const_eval_31
    util_create_list_254 = [
        input["resampler.latents"],
        input["resampler.layers.0.ln1.bias"],
        input["resampler.layers.0.ln1.weight"],
        input["resampler.layers.0.attn.to_q.weight"],
    ]
    const_63 = "main_const_eval_31"
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_62, util_create_list_254, cache, const_63
    )
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    utils_constEvalFuncWrapper_30_1 = utils_constEvalFuncWrapper_30[1]
    const_64 = main_const_eval_3
    util_create_list_255 = [
        input["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"]
    ]
    const_65 = "main_const_eval_32"
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_64, util_create_list_255, cache, const_65
    )
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_66 = main_const_eval_2
    util_create_list_256 = [
        input["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.bias"],
    ]
    const_67 = "main_const_eval_33"
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_66, util_create_list_256, cache, const_67
    )
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_68 = main_const_eval_3
    util_create_list_257 = [
        input["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"]
    ]
    const_69 = "main_const_eval_34"
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_68, util_create_list_257, cache, const_69
    )
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_70 = main_const_eval_4
    util_create_list_258 = [
        input["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.weight"],
    ]
    const_71 = "main_const_eval_35"
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_70, util_create_list_258, cache, const_71
    )
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_72 = main_const_eval_3
    util_create_list_259 = [
        input["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"]
    ]
    const_73 = "main_const_eval_36"
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_72, util_create_list_259, cache, const_73
    )
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_74 = main_const_eval_4
    util_create_list_260 = [
        input["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.weight"],
    ]
    const_75 = "main_const_eval_37"
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_74, util_create_list_260, cache, const_75
    )
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_76 = main_const_eval_4
    util_create_list_261 = [
        input["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.weight"],
    ]
    const_77 = "main_const_eval_38"
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_76, util_create_list_261, cache, const_77
    )
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_78 = main_const_eval_1
    util_create_list_262 = [
        input["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"]
    ]
    const_79 = "main_const_eval_39"
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_78, util_create_list_262, cache, const_79
    )
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_80 = main_const_eval_3
    util_create_list_263 = [
        input["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"]
    ]
    const_81 = "main_const_eval_40"
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_80, util_create_list_263, cache, const_81
    )
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_82 = main_const_eval_2
    util_create_list_264 = [
        input["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.bias"],
    ]
    const_83 = "main_const_eval_41"
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_82, util_create_list_264, cache, const_83
    )
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_84 = main_const_eval_2
    util_create_list_265 = [
        input["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.bias"],
    ]
    const_85 = "main_const_eval_42"
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_84, util_create_list_265, cache, const_85
    )
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_86 = main_const_eval_1
    util_create_list_266 = [
        input["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"]
    ]
    const_87 = "main_const_eval_43"
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_86, util_create_list_266, cache, const_87
    )
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_88 = main_const_eval_2
    util_create_list_267 = [
        input["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.bias"],
    ]
    const_89 = "main_const_eval_44"
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_88, util_create_list_267, cache, const_89
    )
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_90 = main_const_eval_1
    util_create_list_268 = [
        input["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"]
    ]
    const_91 = "main_const_eval_45"
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_90, util_create_list_268, cache, const_91
    )
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_92 = main_const_eval_2
    util_create_list_269 = [
        input["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.bias"],
    ]
    const_93 = "main_const_eval_46"
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_92, util_create_list_269, cache, const_93
    )
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_94 = main_const_eval_3
    util_create_list_270 = [
        input["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"]
    ]
    const_95 = "main_const_eval_47"
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_94, util_create_list_270, cache, const_95
    )
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_96 = main_const_eval_2
    util_create_list_271 = [
        input["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.bias"],
    ]
    const_97 = "main_const_eval_48"
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_96, util_create_list_271, cache, const_97
    )
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_98 = main_const_eval_3
    util_create_list_272 = [
        input["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"]
    ]
    const_99 = "main_const_eval_49"
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_98, util_create_list_272, cache, const_99
    )
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_100 = main_const_eval_4
    util_create_list_273 = [
        input["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.weight"],
    ]
    const_101 = "main_const_eval_50"
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_100, util_create_list_273, cache, const_101
    )
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_102 = main_const_eval_2
    util_create_list_274 = [
        input["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.bias"],
    ]
    const_103 = "main_const_eval_51"
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_102, util_create_list_274, cache, const_103
    )
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_104 = main_const_eval_3
    util_create_list_275 = [
        input["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"]
    ]
    const_105 = "main_const_eval_52"
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_104, util_create_list_275, cache, const_105
    )
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_106 = main_const_eval_3
    util_create_list_276 = [
        input["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"]
    ]
    const_107 = "main_const_eval_53"
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_106, util_create_list_276, cache, const_107
    )
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_108 = main_const_eval_1
    util_create_list_277 = [
        input["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"]
    ]
    const_109 = "main_const_eval_54"
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_108, util_create_list_277, cache, const_109
    )
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_110 = main_const_eval_3
    util_create_list_278 = [
        input["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"]
    ]
    const_111 = "main_const_eval_55"
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_110, util_create_list_278, cache, const_111
    )
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_112 = main_const_eval_3
    util_create_list_279 = [
        input["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"]
    ]
    const_113 = "main_const_eval_56"
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_112, util_create_list_279, cache, const_113
    )
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_114 = main_const_eval_3
    util_create_list_280 = [
        input["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"]
    ]
    const_115 = "main_const_eval_57"
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_114, util_create_list_280, cache, const_115
    )
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_116 = main_const_eval_2
    util_create_list_281 = [
        input["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.bias"],
    ]
    const_117 = "main_const_eval_58"
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_116, util_create_list_281, cache, const_117
    )
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_118 = main_const_eval_2
    util_create_list_282 = [
        input["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.bias"],
    ]
    const_119 = "main_const_eval_59"
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_118, util_create_list_282, cache, const_119
    )
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_120 = main_const_eval_2
    util_create_list_283 = [
        input["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.bias"],
    ]
    const_121 = "main_const_eval_60"
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_120, util_create_list_283, cache, const_121
    )
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_122 = main_const_eval_2
    util_create_list_284 = [
        input["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.bias"],
    ]
    const_123 = "main_const_eval_61"
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_122, util_create_list_284, cache, const_123
    )
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_124 = main_const_eval_4
    util_create_list_285 = [
        input["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.weight"],
    ]
    const_125 = "main_const_eval_62"
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_124, util_create_list_285, cache, const_125
    )
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_126 = main_const_eval_2
    util_create_list_286 = [
        input["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.bias"],
    ]
    const_127 = "main_const_eval_63"
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_126, util_create_list_286, cache, const_127
    )
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_128 = main_const_eval_3
    util_create_list_287 = [
        input["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"]
    ]
    const_129 = "main_const_eval_64"
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_128, util_create_list_287, cache, const_129
    )
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_130 = main_const_eval_3
    util_create_list_288 = [
        input["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"]
    ]
    const_131 = "main_const_eval_65"
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_130, util_create_list_288, cache, const_131
    )
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_132 = main_const_eval_4
    util_create_list_289 = [
        input["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.weight"],
    ]
    const_133 = "main_const_eval_66"
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_132, util_create_list_289, cache, const_133
    )
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_134 = main_const_eval_67
    util_create_list_290 = [
        input["image_encoder.vision_model.embeddings.patch_embedding.weight"]
    ]
    const_135 = "main_const_eval_67"
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_134, util_create_list_290, cache, const_135
    )
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_136 = main_const_eval_4
    util_create_list_291 = [
        input["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.weight"],
    ]
    const_137 = "main_const_eval_68"
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_136, util_create_list_291, cache, const_137
    )
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_138 = main_const_eval_3
    util_create_list_292 = [
        input["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"]
    ]
    const_139 = "main_const_eval_69"
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_138, util_create_list_292, cache, const_139
    )
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_140 = main_const_eval_3
    util_create_list_293 = [
        input["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"]
    ]
    const_141 = "main_const_eval_70"
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_140, util_create_list_293, cache, const_141
    )
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_142 = main_const_eval_4
    util_create_list_294 = [
        input["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.weight"],
    ]
    const_143 = "main_const_eval_71"
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_142, util_create_list_294, cache, const_143
    )
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_144 = main_const_eval_2
    util_create_list_295 = [
        input["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.bias"],
    ]
    const_145 = "main_const_eval_72"
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_144, util_create_list_295, cache, const_145
    )
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_146 = main_const_eval_2
    util_create_list_296 = [
        input["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.bias"],
    ]
    const_147 = "main_const_eval_73"
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_146, util_create_list_296, cache, const_147
    )
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_148 = main_const_eval_3
    util_create_list_297 = [
        input["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"]
    ]
    const_149 = "main_const_eval_74"
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_148, util_create_list_297, cache, const_149
    )
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_150 = main_const_eval_3
    util_create_list_298 = [
        input["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"]
    ]
    const_151 = "main_const_eval_75"
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_150, util_create_list_298, cache, const_151
    )
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_152 = main_const_eval_4
    util_create_list_299 = [
        input["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.weight"],
    ]
    const_153 = "main_const_eval_76"
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_152, util_create_list_299, cache, const_153
    )
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_154 = main_const_eval_4
    util_create_list_300 = [
        input["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.weight"],
    ]
    const_155 = "main_const_eval_77"
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_154, util_create_list_300, cache, const_155
    )
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_156 = main_const_eval_2
    util_create_list_301 = [
        input["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.bias"],
    ]
    const_157 = "main_const_eval_78"
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_156, util_create_list_301, cache, const_157
    )
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_158 = main_const_eval_3
    util_create_list_302 = [
        input["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"]
    ]
    const_159 = "main_const_eval_79"
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_158, util_create_list_302, cache, const_159
    )
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_160 = main_const_eval_3
    util_create_list_303 = [
        input["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"]
    ]
    const_161 = "main_const_eval_80"
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_160, util_create_list_303, cache, const_161
    )
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_162 = main_const_eval_2
    util_create_list_304 = [
        input["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.bias"],
    ]
    const_163 = "main_const_eval_81"
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_162, util_create_list_304, cache, const_163
    )
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_164 = main_const_eval_1
    util_create_list_305 = [
        input["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"]
    ]
    const_165 = "main_const_eval_82"
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_164, util_create_list_305, cache, const_165
    )
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_166 = main_const_eval_3
    util_create_list_306 = [
        input["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"]
    ]
    const_167 = "main_const_eval_83"
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_166, util_create_list_306, cache, const_167
    )
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_168 = main_const_eval_3
    util_create_list_307 = [
        input["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"]
    ]
    const_169 = "main_const_eval_84"
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_168, util_create_list_307, cache, const_169
    )
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_170 = main_const_eval_2
    util_create_list_308 = [
        input["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.bias"],
    ]
    const_171 = "main_const_eval_85"
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_170, util_create_list_308, cache, const_171
    )
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_172 = main_const_eval_3
    util_create_list_309 = [
        input["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"]
    ]
    const_173 = "main_const_eval_86"
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_172, util_create_list_309, cache, const_173
    )
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_174 = main_const_eval_4
    util_create_list_310 = [
        input["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.weight"],
    ]
    const_175 = "main_const_eval_87"
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_174, util_create_list_310, cache, const_175
    )
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_176 = main_const_eval_4
    util_create_list_311 = [
        input["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.weight"],
    ]
    const_177 = "main_const_eval_88"
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_176, util_create_list_311, cache, const_177
    )
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_178 = main_const_eval_89
    util_create_list_312 = [
        input["__POSITION_IDS__"],
        input["image_encoder.vision_model.embeddings.position_embedding.weight"],
    ]
    const_179 = "main_const_eval_89"
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_178, util_create_list_312, cache, const_179
    )
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_180 = main_const_eval_4
    util_create_list_313 = [
        input["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.weight"],
    ]
    const_181 = "main_const_eval_90"
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_180, util_create_list_313, cache, const_181
    )
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_182 = main_const_eval_2
    util_create_list_314 = [
        input["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.bias"],
    ]
    const_183 = "main_const_eval_91"
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_182, util_create_list_314, cache, const_183
    )
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_184 = main_const_eval_1
    util_create_list_315 = [
        input["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"]
    ]
    const_185 = "main_const_eval_92"
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_184, util_create_list_315, cache, const_185
    )
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_186 = main_const_eval_1
    util_create_list_316 = [
        input["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"]
    ]
    const_187 = "main_const_eval_93"
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_186, util_create_list_316, cache, const_187
    )
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_188 = main_const_eval_3
    util_create_list_317 = [
        input["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"]
    ]
    const_189 = "main_const_eval_94"
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_188, util_create_list_317, cache, const_189
    )
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_190 = main_const_eval_1
    util_create_list_318 = [
        input["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"]
    ]
    const_191 = "main_const_eval_95"
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_190, util_create_list_318, cache, const_191
    )
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_192 = main_const_eval_1
    util_create_list_319 = [
        input["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"]
    ]
    const_193 = "main_const_eval_96"
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_192, util_create_list_319, cache, const_193
    )
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_194 = main_const_eval_4
    util_create_list_320 = [
        input["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.weight"],
    ]
    const_195 = "main_const_eval_97"
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_194, util_create_list_320, cache, const_195
    )
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_196 = main_const_eval_3
    util_create_list_321 = [
        input["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"]
    ]
    const_197 = "main_const_eval_98"
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_196, util_create_list_321, cache, const_197
    )
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_198 = main_const_eval_1
    util_create_list_322 = [
        input["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"]
    ]
    const_199 = "main_const_eval_99"
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_198, util_create_list_322, cache, const_199
    )
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_200 = main_const_eval_2
    util_create_list_323 = [
        input["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.bias"],
    ]
    const_201 = "main_const_eval_100"
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_200, util_create_list_323, cache, const_201
    )
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_202 = main_const_eval_3
    util_create_list_324 = [
        input["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"]
    ]
    const_203 = "main_const_eval_101"
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_202, util_create_list_324, cache, const_203
    )
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_204 = main_const_eval_2
    util_create_list_325 = [
        input["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.bias"],
    ]
    const_205 = "main_const_eval_102"
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_204, util_create_list_325, cache, const_205
    )
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_206 = main_const_eval_4
    util_create_list_326 = [
        input["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.weight"],
    ]
    const_207 = "main_const_eval_103"
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_206, util_create_list_326, cache, const_207
    )
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_208 = main_const_eval_3
    util_create_list_327 = [
        input["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"]
    ]
    const_209 = "main_const_eval_104"
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_208, util_create_list_327, cache, const_209
    )
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_210 = main_const_eval_3
    util_create_list_328 = [
        input["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"]
    ]
    const_211 = "main_const_eval_105"
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_210, util_create_list_328, cache, const_211
    )
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_212 = main_const_eval_3
    util_create_list_329 = [
        input["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"]
    ]
    const_213 = "main_const_eval_106"
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_212, util_create_list_329, cache, const_213
    )
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_214 = main_const_eval_3
    util_create_list_330 = [
        input["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"]
    ]
    const_215 = "main_const_eval_107"
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_214, util_create_list_330, cache, const_215
    )
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_216 = main_const_eval_1
    util_create_list_331 = [
        input["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"]
    ]
    const_217 = "main_const_eval_108"
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_216, util_create_list_331, cache, const_217
    )
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_218 = main_const_eval_1
    util_create_list_332 = [
        input["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"]
    ]
    const_219 = "main_const_eval_109"
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_218, util_create_list_332, cache, const_219
    )
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_220 = main_const_eval_3
    util_create_list_333 = [
        input["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"]
    ]
    const_221 = "main_const_eval_110"
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_220, util_create_list_333, cache, const_221
    )
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_222 = main_const_eval_1
    util_create_list_334 = [
        input["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"]
    ]
    const_223 = "main_const_eval_111"
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_222, util_create_list_334, cache, const_223
    )
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_224 = main_const_eval_2
    util_create_list_335 = [
        input["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.bias"],
    ]
    const_225 = "main_const_eval_112"
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_224, util_create_list_335, cache, const_225
    )
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_226 = main_const_eval_4
    util_create_list_336 = [
        input["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.weight"],
    ]
    const_227 = "main_const_eval_113"
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_226, util_create_list_336, cache, const_227
    )
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_228 = main_const_eval_3
    util_create_list_337 = [
        input["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"]
    ]
    const_229 = "main_const_eval_114"
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_228, util_create_list_337, cache, const_229
    )
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_230 = main_const_eval_3
    util_create_list_338 = [
        input["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"]
    ]
    const_231 = "main_const_eval_115"
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_230, util_create_list_338, cache, const_231
    )
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_232 = main_const_eval_1
    util_create_list_339 = [
        input["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"]
    ]
    const_233 = "main_const_eval_116"
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_232, util_create_list_339, cache, const_233
    )
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_234 = main_const_eval_2
    util_create_list_340 = [
        input["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.bias"],
    ]
    const_235 = "main_const_eval_117"
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_234, util_create_list_340, cache, const_235
    )
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_236 = main_const_eval_1
    util_create_list_341 = [
        input["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"]
    ]
    const_237 = "main_const_eval_118"
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_236, util_create_list_341, cache, const_237
    )
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_238 = main_const_eval_2
    util_create_list_342 = [
        input["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.bias"],
    ]
    const_239 = "main_const_eval_119"
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_238, util_create_list_342, cache, const_239
    )
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_240 = main_const_eval_4
    util_create_list_343 = [
        input["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.weight"],
    ]
    const_241 = "main_const_eval_120"
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_240, util_create_list_343, cache, const_241
    )
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_242 = main_const_eval_3
    util_create_list_344 = [
        input["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"]
    ]
    const_243 = "main_const_eval_121"
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_242, util_create_list_344, cache, const_243
    )
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_244 = main_const_eval_3
    util_create_list_345 = [
        input["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"]
    ]
    const_245 = "main_const_eval_122"
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_244, util_create_list_345, cache, const_245
    )
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_246 = main_const_eval_3
    util_create_list_346 = [
        input["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"]
    ]
    const_247 = "main_const_eval_123"
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_246, util_create_list_346, cache, const_247
    )
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_248 = main_const_eval_3
    util_create_list_347 = [
        input["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"]
    ]
    const_249 = "main_const_eval_124"
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_248, util_create_list_347, cache, const_249
    )
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_250 = main_const_eval_3
    util_create_list_348 = [
        input["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"]
    ]
    const_251 = "main_const_eval_125"
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_250, util_create_list_348, cache, const_251
    )
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_252 = main_const_eval_3
    util_create_list_349 = [
        input["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"]
    ]
    const_253 = "main_const_eval_126"
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_252, util_create_list_349, cache, const_253
    )
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    const_254 = main_const_eval_3
    util_create_list_350 = [
        input["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"]
    ]
    const_255 = "main_const_eval_127"
    utils_constEvalFuncWrapper_126 = utils.constEvalFuncWrapper(
        const_254, util_create_list_350, cache, const_255
    )
    utils_constEvalFuncWrapper_126_0 = utils_constEvalFuncWrapper_126[0]
    const_256 = main_const_eval_4
    util_create_list_351 = [
        input["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.weight"],
    ]
    const_257 = "main_const_eval_128"
    utils_constEvalFuncWrapper_127 = utils.constEvalFuncWrapper(
        const_256, util_create_list_351, cache, const_257
    )
    utils_constEvalFuncWrapper_127_0 = utils_constEvalFuncWrapper_127[0]
    const_258 = main_const_eval_4
    util_create_list_352 = [
        input["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.weight"],
    ]
    const_259 = "main_const_eval_129"
    utils_constEvalFuncWrapper_128 = utils.constEvalFuncWrapper(
        const_258, util_create_list_352, cache, const_259
    )
    utils_constEvalFuncWrapper_128_0 = utils_constEvalFuncWrapper_128[0]
    const_260 = main_const_eval_3
    util_create_list_353 = [
        input["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"]
    ]
    const_261 = "main_const_eval_130"
    utils_constEvalFuncWrapper_129 = utils.constEvalFuncWrapper(
        const_260, util_create_list_353, cache, const_261
    )
    utils_constEvalFuncWrapper_129_0 = utils_constEvalFuncWrapper_129[0]
    const_262 = main_const_eval_1
    util_create_list_354 = [
        input["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"]
    ]
    const_263 = "main_const_eval_131"
    utils_constEvalFuncWrapper_130 = utils.constEvalFuncWrapper(
        const_262, util_create_list_354, cache, const_263
    )
    utils_constEvalFuncWrapper_130_0 = utils_constEvalFuncWrapper_130[0]
    const_264 = main_const_eval_2
    util_create_list_355 = [
        input["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.bias"],
    ]
    const_265 = "main_const_eval_132"
    utils_constEvalFuncWrapper_131 = utils.constEvalFuncWrapper(
        const_264, util_create_list_355, cache, const_265
    )
    utils_constEvalFuncWrapper_131_0 = utils_constEvalFuncWrapper_131[0]
    const_266 = main_const_eval_3
    util_create_list_356 = [
        input["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"]
    ]
    const_267 = "main_const_eval_133"
    utils_constEvalFuncWrapper_132 = utils.constEvalFuncWrapper(
        const_266, util_create_list_356, cache, const_267
    )
    utils_constEvalFuncWrapper_132_0 = utils_constEvalFuncWrapper_132[0]
    const_268 = main_const_eval_2
    util_create_list_357 = [
        input["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.bias"],
    ]
    const_269 = "main_const_eval_134"
    utils_constEvalFuncWrapper_133 = utils.constEvalFuncWrapper(
        const_268, util_create_list_357, cache, const_269
    )
    utils_constEvalFuncWrapper_133_0 = utils_constEvalFuncWrapper_133[0]
    const_270 = main_const_eval_2
    util_create_list_358 = [
        input["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.bias"],
    ]
    const_271 = "main_const_eval_135"
    utils_constEvalFuncWrapper_134 = utils.constEvalFuncWrapper(
        const_270, util_create_list_358, cache, const_271
    )
    utils_constEvalFuncWrapper_134_0 = utils_constEvalFuncWrapper_134[0]
    const_272 = main_const_eval_1
    util_create_list_359 = [
        input["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"]
    ]
    const_273 = "main_const_eval_136"
    utils_constEvalFuncWrapper_135 = utils.constEvalFuncWrapper(
        const_272, util_create_list_359, cache, const_273
    )
    utils_constEvalFuncWrapper_135_0 = utils_constEvalFuncWrapper_135[0]
    const_274 = main_const_eval_2
    util_create_list_360 = [
        input["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.bias"],
    ]
    const_275 = "main_const_eval_137"
    utils_constEvalFuncWrapper_136 = utils.constEvalFuncWrapper(
        const_274, util_create_list_360, cache, const_275
    )
    utils_constEvalFuncWrapper_136_0 = utils_constEvalFuncWrapper_136[0]
    const_276 = main_const_eval_3
    util_create_list_361 = [input["resampler.proj_in.bias"]]
    const_277 = "main_const_eval_138"
    utils_constEvalFuncWrapper_137 = utils.constEvalFuncWrapper(
        const_276, util_create_list_361, cache, const_277
    )
    utils_constEvalFuncWrapper_137_0 = utils_constEvalFuncWrapper_137[0]
    const_278 = main_const_eval_4
    util_create_list_362 = [
        input["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.weight"],
    ]
    const_279 = "main_const_eval_139"
    utils_constEvalFuncWrapper_138 = utils.constEvalFuncWrapper(
        const_278, util_create_list_362, cache, const_279
    )
    utils_constEvalFuncWrapper_138_0 = utils_constEvalFuncWrapper_138[0]
    const_280 = main_const_eval_1
    util_create_list_363 = [
        input["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"]
    ]
    const_281 = "main_const_eval_140"
    utils_constEvalFuncWrapper_139 = utils.constEvalFuncWrapper(
        const_280, util_create_list_363, cache, const_281
    )
    utils_constEvalFuncWrapper_139_0 = utils_constEvalFuncWrapper_139[0]
    const_282 = main_const_eval_3
    util_create_list_364 = [
        input["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"]
    ]
    const_283 = "main_const_eval_141"
    utils_constEvalFuncWrapper_140 = utils.constEvalFuncWrapper(
        const_282, util_create_list_364, cache, const_283
    )
    utils_constEvalFuncWrapper_140_0 = utils_constEvalFuncWrapper_140[0]
    const_284 = main_const_eval_1
    util_create_list_365 = [
        input["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"]
    ]
    const_285 = "main_const_eval_142"
    utils_constEvalFuncWrapper_141 = utils.constEvalFuncWrapper(
        const_284, util_create_list_365, cache, const_285
    )
    utils_constEvalFuncWrapper_141_0 = utils_constEvalFuncWrapper_141[0]
    const_286 = main_const_eval_143
    util_create_list_366 = [
        input["image_encoder.vision_model.embeddings.class_embedding"]
    ]
    const_287 = "main_const_eval_143"
    utils_constEvalFuncWrapper_142 = utils.constEvalFuncWrapper(
        const_286, util_create_list_366, cache, const_287
    )
    utils_constEvalFuncWrapper_142_0 = utils_constEvalFuncWrapper_142[0]
    const_288 = main_const_eval_3
    util_create_list_367 = [
        input["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"]
    ]
    const_289 = "main_const_eval_144"
    utils_constEvalFuncWrapper_143 = utils.constEvalFuncWrapper(
        const_288, util_create_list_367, cache, const_289
    )
    utils_constEvalFuncWrapper_143_0 = utils_constEvalFuncWrapper_143[0]
    const_290 = main_const_eval_3
    util_create_list_368 = [
        input["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"]
    ]
    const_291 = "main_const_eval_145"
    utils_constEvalFuncWrapper_144 = utils.constEvalFuncWrapper(
        const_290, util_create_list_368, cache, const_291
    )
    utils_constEvalFuncWrapper_144_0 = utils_constEvalFuncWrapper_144[0]
    const_292 = main_const_eval_1
    util_create_list_369 = [
        input["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"]
    ]
    const_293 = "main_const_eval_146"
    utils_constEvalFuncWrapper_145 = utils.constEvalFuncWrapper(
        const_292, util_create_list_369, cache, const_293
    )
    utils_constEvalFuncWrapper_145_0 = utils_constEvalFuncWrapper_145[0]
    const_294 = main_const_eval_3
    util_create_list_370 = [
        input["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"]
    ]
    const_295 = "main_const_eval_147"
    utils_constEvalFuncWrapper_146 = utils.constEvalFuncWrapper(
        const_294, util_create_list_370, cache, const_295
    )
    utils_constEvalFuncWrapper_146_0 = utils_constEvalFuncWrapper_146[0]
    const_296 = main_const_eval_3
    util_create_list_371 = [
        input["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"]
    ]
    const_297 = "main_const_eval_148"
    utils_constEvalFuncWrapper_147 = utils.constEvalFuncWrapper(
        const_296, util_create_list_371, cache, const_297
    )
    utils_constEvalFuncWrapper_147_0 = utils_constEvalFuncWrapper_147[0]
    const_298 = main_const_eval_4
    util_create_list_372 = [
        input["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.weight"],
    ]
    const_299 = "main_const_eval_149"
    utils_constEvalFuncWrapper_148 = utils.constEvalFuncWrapper(
        const_298, util_create_list_372, cache, const_299
    )
    utils_constEvalFuncWrapper_148_0 = utils_constEvalFuncWrapper_148[0]
    const_300 = main_const_eval_3
    util_create_list_373 = [
        input["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"]
    ]
    const_301 = "main_const_eval_150"
    utils_constEvalFuncWrapper_149 = utils.constEvalFuncWrapper(
        const_300, util_create_list_373, cache, const_301
    )
    utils_constEvalFuncWrapper_149_0 = utils_constEvalFuncWrapper_149[0]
    const_302 = main_const_eval_1
    util_create_list_374 = [
        input["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"]
    ]
    const_303 = "main_const_eval_151"
    utils_constEvalFuncWrapper_150 = utils.constEvalFuncWrapper(
        const_302, util_create_list_374, cache, const_303
    )
    utils_constEvalFuncWrapper_150_0 = utils_constEvalFuncWrapper_150[0]
    const_304 = main_const_eval_3
    util_create_list_375 = [
        input["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"]
    ]
    const_305 = "main_const_eval_152"
    utils_constEvalFuncWrapper_151 = utils.constEvalFuncWrapper(
        const_304, util_create_list_375, cache, const_305
    )
    utils_constEvalFuncWrapper_151_0 = utils_constEvalFuncWrapper_151[0]
    const_306 = main_const_eval_4
    util_create_list_376 = [
        input["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.weight"],
    ]
    const_307 = "main_const_eval_153"
    utils_constEvalFuncWrapper_152 = utils.constEvalFuncWrapper(
        const_306, util_create_list_376, cache, const_307
    )
    utils_constEvalFuncWrapper_152_0 = utils_constEvalFuncWrapper_152[0]
    const_308 = main_const_eval_3
    util_create_list_377 = [
        input["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"]
    ]
    const_309 = "main_const_eval_154"
    utils_constEvalFuncWrapper_153 = utils.constEvalFuncWrapper(
        const_308, util_create_list_377, cache, const_309
    )
    utils_constEvalFuncWrapper_153_0 = utils_constEvalFuncWrapper_153[0]
    const_310 = main_const_eval_1
    util_create_list_378 = [
        input["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"]
    ]
    const_311 = "main_const_eval_155"
    utils_constEvalFuncWrapper_154 = utils.constEvalFuncWrapper(
        const_310, util_create_list_378, cache, const_311
    )
    utils_constEvalFuncWrapper_154_0 = utils_constEvalFuncWrapper_154[0]
    const_312 = main_const_eval_1
    util_create_list_379 = [
        input["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"]
    ]
    const_313 = "main_const_eval_156"
    utils_constEvalFuncWrapper_155 = utils.constEvalFuncWrapper(
        const_312, util_create_list_379, cache, const_313
    )
    utils_constEvalFuncWrapper_155_0 = utils_constEvalFuncWrapper_155[0]
    const_314 = main_const_eval_1
    util_create_list_380 = [
        input["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"]
    ]
    const_315 = "main_const_eval_157"
    utils_constEvalFuncWrapper_156 = utils.constEvalFuncWrapper(
        const_314, util_create_list_380, cache, const_315
    )
    utils_constEvalFuncWrapper_156_0 = utils_constEvalFuncWrapper_156[0]
    const_316 = main_const_eval_4
    util_create_list_381 = [
        input["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.weight"],
    ]
    const_317 = "main_const_eval_158"
    utils_constEvalFuncWrapper_157 = utils.constEvalFuncWrapper(
        const_316, util_create_list_381, cache, const_317
    )
    utils_constEvalFuncWrapper_157_0 = utils_constEvalFuncWrapper_157[0]
    const_318 = main_const_eval_2
    util_create_list_382 = [
        input["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.bias"],
        input["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.bias"],
    ]
    const_319 = "main_const_eval_159"
    utils_constEvalFuncWrapper_158 = utils.constEvalFuncWrapper(
        const_318, util_create_list_382, cache, const_319
    )
    utils_constEvalFuncWrapper_158_0 = utils_constEvalFuncWrapper_158[0]
    const_320 = main_const_eval_4
    util_create_list_383 = [
        input["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.weight"],
        input["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.weight"],
    ]
    const_321 = "main_const_eval_160"
    utils_constEvalFuncWrapper_159 = utils.constEvalFuncWrapper(
        const_320, util_create_list_383, cache, const_321
    )
    utils_constEvalFuncWrapper_159_0 = utils_constEvalFuncWrapper_159[0]
    const_322 = main_const_eval_3
    util_create_list_384 = [
        input["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"]
    ]
    const_323 = "main_const_eval_161"
    utils_constEvalFuncWrapper_160 = utils.constEvalFuncWrapper(
        const_322, util_create_list_384, cache, const_323
    )
    utils_constEvalFuncWrapper_160_0 = utils_constEvalFuncWrapper_160[0]

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
