# This file contains the const-eval functions for the CLIPVision model

import ttnn
import utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([2, 1]),
        fill_value=1.0013580322265625e-05,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    ttnn_reshape_0 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_repeat_0 = ttnn.repeat(ttnn_permute_0, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_repeat_0,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_1,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    util_create_list_1 = [ttnn_reshape_1]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    ttnn_reshape_2 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_2,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_permute_2, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_1, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_0,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
    util_create_list_2 = [ttnn_reshape_3]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    ttnn_reshape_4 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_3 = [ttnn_reshape_4]
    return util_create_list_3


def main_const_eval_4(input):
    input_0 = input[0]
    ttnn_reshape_5 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_4 = [ttnn_reshape_5]
    return util_create_list_4


def main_const_eval_5(input):
    input_0 = input[0]
    ttnn_reshape_6 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_5 = [ttnn_reshape_6]
    return util_create_list_5


def main_const_eval_6(input):
    input_0 = input[0]
    ttnn_reshape_7 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_6 = [ttnn_reshape_7]
    return util_create_list_6


def main_const_eval_7(input):
    input_0 = input[0]
    ttnn_reshape_8 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_7 = [ttnn_reshape_8]
    return util_create_list_7


def main_const_eval_8(input):
    input_0 = input[0]
    ttnn_reshape_9 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_8 = [ttnn_reshape_9]
    return util_create_list_8


def main_const_eval_9(input):
    input_0 = input[0]
    ttnn_reshape_10 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_10, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_repeat_2,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_2, False)
    util_create_list_9 = [ttnn_reshape_11]
    return util_create_list_9


def main_const_eval_10(input):
    input_0 = input[0]
    ttnn_reshape_12 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_3 = ttnn.repeat(ttnn_reshape_12, ttnn.Shape([2, 1, 1]))
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_repeat_3,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_3, False)
    util_create_list_10 = [ttnn_permute_3]
    return util_create_list_10


def main_const_eval_11(input):
    input_0 = input[0]
    ttnn_reshape_13 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_4 = ttnn.permute(
        ttnn_reshape_13,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_repeat_4 = ttnn.repeat(ttnn_permute_4, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_4, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_repeat_4,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_4, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_permute_5,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    util_create_list_11 = [ttnn_reshape_14]
    return util_create_list_11


def main_const_eval_12(input):
    input_0 = input[0]
    ttnn_reshape_15 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_6 = ttnn.permute(
        ttnn_reshape_15,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_repeat_5 = ttnn.repeat(ttnn_permute_6, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_6, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_repeat_5,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_5, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_permute_7,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_7, False)
    util_create_list_12 = [ttnn_reshape_16]
    return util_create_list_12


def main_const_eval_13(input):
    input_0 = input[0]
    ttnn_reshape_17 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_17,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_17, False)
    ttnn_repeat_6 = ttnn.repeat(ttnn_permute_8, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_8, False)
    ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_6, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_1,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
    util_create_list_13 = [ttnn_reshape_18]
    return util_create_list_13


def main_const_eval_14(input):
    input_0 = input[0]
    ttnn_reshape_19 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_7 = ttnn.repeat(ttnn_reshape_19, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_repeat_7,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_7, False)
    util_create_list_14 = [ttnn_reshape_20]
    return util_create_list_14


def main_const_eval_15(input):
    input_0 = input[0]
    ttnn_reshape_21 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_15 = [ttnn_reshape_21]
    return util_create_list_15


def main_const_eval_16(input):
    input_0 = input[0]
    ttnn_reshape_22 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_8 = ttnn.repeat(ttnn_reshape_22, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_repeat_8,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_8, False)
    util_create_list_16 = [ttnn_reshape_23]
    return util_create_list_16


def main_const_eval_17(input):
    input_0 = input[0]
    ttnn_reshape_24 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_9 = ttnn.permute(
        ttnn_reshape_24,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_repeat_9 = ttnn.repeat(ttnn_permute_9, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_9, False)
    ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_9,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_9, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_2,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_2, False)
    util_create_list_17 = [ttnn_reshape_25]
    return util_create_list_17


def main_const_eval_18(input):
    input_0 = input[0]
    ttnn_reshape_26 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_10 = ttnn.permute(
        ttnn_reshape_26,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn_repeat_10 = ttnn.repeat(ttnn_permute_10, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_10, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_repeat_10,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_10, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_permute_11,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_11, False)
    util_create_list_18 = [ttnn_reshape_27]
    return util_create_list_18


def main_const_eval_19(input):
    input_0 = input[0]
    ttnn_reshape_28 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_11 = ttnn.repeat(ttnn_reshape_28, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_28, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_repeat_11,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_11, False)
    util_create_list_19 = [ttnn_reshape_29]
    return util_create_list_19


def main_const_eval_20(input):
    input_0 = input[0]
    ttnn_reshape_30 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_12 = ttnn.permute(
        ttnn_reshape_30,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_repeat_12 = ttnn.repeat(ttnn_permute_12, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_12, False)
    ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_12,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_12, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_3,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_3, False)
    util_create_list_20 = [ttnn_reshape_31]
    return util_create_list_20


def main_const_eval_21(input):
    input_0 = input[0]
    ttnn_reshape_32 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_13 = ttnn.repeat(ttnn_reshape_32, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_32, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_repeat_13,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_13, False)
    util_create_list_21 = [ttnn_reshape_33]
    return util_create_list_21


def main_const_eval_22(input):
    input_0 = input[0]
    ttnn_reshape_34 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_14 = ttnn.repeat(ttnn_reshape_34, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_34, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_repeat_14,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_14, False)
    util_create_list_22 = [ttnn_reshape_35]
    return util_create_list_22


def main_const_eval_23(input):
    input_0 = input[0]
    ttnn_reshape_36 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_23 = [ttnn_reshape_36]
    return util_create_list_23


def main_const_eval_24(input):
    input_0 = input[0]
    ttnn_reshape_37 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_24 = [ttnn_reshape_37]
    return util_create_list_24


def main_const_eval_25(input):
    input_0 = input[0]
    ttnn_reshape_38 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_25 = [ttnn_reshape_38]
    return util_create_list_25


def main_const_eval_26(input):
    input_0 = input[0]
    ttnn_reshape_39 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_26 = [ttnn_reshape_39]
    return util_create_list_26


def main_const_eval_27(input):
    input_0 = input[0]
    ttnn_reshape_40 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_15 = ttnn.repeat(ttnn_reshape_40, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_40, False)
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_repeat_15,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_15, False)
    util_create_list_27 = [ttnn_reshape_41]
    return util_create_list_27


def main_const_eval_28(input):
    input_0 = input[0]
    ttnn_reshape_42 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_28 = [ttnn_reshape_42]
    return util_create_list_28


def main_const_eval_29(input):
    input_0 = input[0]
    ttnn_reshape_43 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_29 = [ttnn_reshape_43]
    return util_create_list_29


def main_const_eval_30(input):
    input_0 = input[0]
    ttnn_reshape_44 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_13 = ttnn.permute(
        ttnn_reshape_44,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_44, False)
    ttnn_repeat_16 = ttnn.repeat(ttnn_permute_13, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_13, False)
    ttnn_transformer_concatenate_heads_4 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_16, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_4,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_4, False)
    util_create_list_30 = [ttnn_reshape_45]
    return util_create_list_30


def main_const_eval_31(input):
    input_0 = input[0]
    ttnn_reshape_46 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_31 = [ttnn_reshape_46]
    return util_create_list_31


def main_const_eval_32(input):
    input_0 = input[0]
    ttnn_reshape_47 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_32 = [ttnn_reshape_47]
    return util_create_list_32


def main_const_eval_33(input):
    input_0 = input[0]
    ttnn_reshape_48 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_17 = ttnn.repeat(ttnn_reshape_48, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_48, False)
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_repeat_17,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_17, False)
    util_create_list_33 = [ttnn_reshape_49]
    return util_create_list_33


def main_const_eval_34(input):
    input_0 = input[0]
    ttnn_reshape_50 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_34 = [ttnn_reshape_50]
    return util_create_list_34


def main_const_eval_35(input):
    input_0 = input[0]
    ttnn_reshape_51 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_18 = ttnn.repeat(ttnn_reshape_51, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_51, False)
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_repeat_18,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_18, False)
    util_create_list_35 = [ttnn_reshape_52]
    return util_create_list_35


def main_const_eval_36(input):
    input_0 = input[0]
    ttnn_reshape_53 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_14 = ttnn.permute(
        ttnn_reshape_53,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_53, False)
    ttnn_repeat_19 = ttnn.repeat(ttnn_permute_14, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_14, False)
    ttnn_transformer_concatenate_heads_5 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_19,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_19, False)
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_5,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_5, False)
    util_create_list_36 = [ttnn_reshape_54]
    return util_create_list_36


def main_const_eval_37(input):
    input_0 = input[0]
    ttnn_reshape_55 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_15 = ttnn.permute(
        ttnn_reshape_55,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_55, False)
    ttnn_repeat_20 = ttnn.repeat(ttnn_permute_15, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_15, False)
    ttnn_transformer_concatenate_heads_6 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_20,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_20, False)
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_6,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_6, False)
    util_create_list_37 = [ttnn_reshape_56]
    return util_create_list_37


def main_const_eval_38(input):
    input_0 = input[0]
    ttnn_reshape_57 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_21 = ttnn.repeat(ttnn_reshape_57, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_57, False)
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_repeat_21,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_21, False)
    util_create_list_38 = [ttnn_reshape_58]
    return util_create_list_38


def main_const_eval_39(input):
    input_0 = input[0]
    ttnn_reshape_59 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_16 = ttnn.permute(
        ttnn_reshape_59,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_59, False)
    ttnn_repeat_22 = ttnn.repeat(ttnn_permute_16, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_16, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_repeat_22,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_22, False)
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_permute_17,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_17, False)
    util_create_list_39 = [ttnn_reshape_60]
    return util_create_list_39


def main_const_eval_40(input):
    input_0 = input[0]
    ttnn_reshape_61 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_40 = [ttnn_reshape_61]
    return util_create_list_40


def main_const_eval_41():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([2, 50]),
        fill_value=0.00130462646484375,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_41 = [ttnn_full_1]
    return util_create_list_41


def main_const_eval_42(input):
    input_0 = input[0]
    ttnn_reshape_62 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_42 = [ttnn_reshape_62]
    return util_create_list_42


def main_const_eval_43(input):
    input_0 = input[0]
    ttnn_reshape_63 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_18 = ttnn.permute(
        ttnn_reshape_63,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_63, False)
    ttnn_repeat_23 = ttnn.repeat(ttnn_permute_18, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_18, False)
    ttnn_transformer_concatenate_heads_7 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_23,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_23, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_7,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_7, False)
    util_create_list_43 = [ttnn_reshape_64]
    return util_create_list_43


def main_const_eval_44(input):
    input_0 = input[0]
    ttnn_reshape_65 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_44 = [ttnn_reshape_65]
    return util_create_list_44


def main_const_eval_45(input):
    input_0 = input[0]
    ttnn_reshape_66 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_45 = [ttnn_reshape_66]
    return util_create_list_45


def main_const_eval_46(input):
    input_0 = input[0]
    ttnn_reshape_67 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_46 = [ttnn_reshape_67]
    return util_create_list_46


def main_const_eval_47(input):
    input_0 = input[0]
    ttnn_reshape_68 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_24 = ttnn.repeat(ttnn_reshape_68, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_68, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_repeat_24,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_24, False)
    util_create_list_47 = [ttnn_reshape_69]
    return util_create_list_47


def main_const_eval_48(input):
    input_0 = input[0]
    ttnn_reshape_70 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_48 = [ttnn_reshape_70]
    return util_create_list_48


def main_const_eval_49(input):
    input_0 = input[0]
    ttnn_reshape_71 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_49 = [ttnn_reshape_71]
    return util_create_list_49


def main_const_eval_50(input):
    input_0 = input[0]
    ttnn_reshape_72 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_50 = [ttnn_reshape_72]
    return util_create_list_50


def main_const_eval_51():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([2]),
        fill_value=0.00130462646484375,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_51 = [ttnn_full_2]
    return util_create_list_51


def main_const_eval_52(input):
    input_0 = input[0]
    ttnn_reshape_73 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_19 = ttnn.permute(
        ttnn_reshape_73,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_73, False)
    ttnn_repeat_25 = ttnn.repeat(ttnn_permute_19, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_19, False)
    ttnn_transformer_concatenate_heads_8 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_25,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_25, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_8,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_8, False)
    util_create_list_52 = [ttnn_reshape_74]
    return util_create_list_52


def main_const_eval_53(input):
    input_0 = input[0]
    ttnn_reshape_75 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_53 = [ttnn_reshape_75]
    return util_create_list_53


def main_const_eval_54(input):
    input_0 = input[0]
    ttnn_reshape_76 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_54 = [ttnn_reshape_76]
    return util_create_list_54


def main_const_eval_55(input):
    input_0 = input[0]
    ttnn_reshape_77 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_55 = [ttnn_reshape_77]
    return util_create_list_55


def main_const_eval_56(input):
    input_0 = input[0]
    ttnn_reshape_78 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_20 = ttnn.permute(
        ttnn_reshape_78,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_78, False)
    ttnn_repeat_26 = ttnn.repeat(ttnn_permute_20, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_20, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_repeat_26,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_26, False)
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_permute_21,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_21, False)
    util_create_list_56 = [ttnn_reshape_79]
    return util_create_list_56


def main_const_eval_57():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([2, 50, 1]),
        fill_value=1.0013580322265625e-05,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_83 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_85 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_87 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_88 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_90 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_92 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_102 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_103 = ttnn.reshape(
        ttnn_full_3,
        [2, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_57 = [
        ttnn_full_3,
        ttnn_reshape_80,
        ttnn_reshape_81,
        ttnn_reshape_82,
        ttnn_reshape_83,
        ttnn_reshape_84,
        ttnn_reshape_85,
        ttnn_reshape_86,
        ttnn_reshape_87,
        ttnn_reshape_88,
        ttnn_reshape_89,
        ttnn_reshape_90,
        ttnn_reshape_91,
        ttnn_reshape_92,
        ttnn_reshape_93,
        ttnn_reshape_94,
        ttnn_reshape_95,
        ttnn_reshape_96,
        ttnn_reshape_97,
        ttnn_reshape_98,
        ttnn_reshape_99,
        ttnn_reshape_100,
        ttnn_reshape_101,
        ttnn_reshape_102,
        ttnn_reshape_103,
    ]
    return util_create_list_57


def main_const_eval_58(input):
    input_0 = input[0]
    ttnn_reshape_104 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_58 = [ttnn_reshape_104]
    return util_create_list_58


def main_const_eval_59(input):
    input_0 = input[0]
    ttnn_reshape_105 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_22 = ttnn.permute(
        ttnn_reshape_105,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_105, False)
    ttnn_repeat_27 = ttnn.repeat(ttnn_permute_22, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_22, False)
    ttnn_transformer_concatenate_heads_9 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_27,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_27, False)
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_9,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_9, False)
    util_create_list_59 = [ttnn_reshape_106]
    return util_create_list_59


def main_const_eval_60(input):
    input_0 = input[0]
    ttnn_reshape_107 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_23 = ttnn.permute(
        ttnn_reshape_107,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_107, False)
    ttnn_repeat_28 = ttnn.repeat(ttnn_permute_23, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_23, False)
    ttnn_transformer_concatenate_heads_10 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_28,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_28, False)
    ttnn_reshape_108 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_10,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_10, False)
    util_create_list_60 = [ttnn_reshape_108]
    return util_create_list_60


def main_const_eval_61(input):
    input_0 = input[0]
    ttnn_reshape_109 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_61 = [ttnn_reshape_109]
    return util_create_list_61


def main_const_eval_62(input):
    input_0 = input[0]
    ttnn_reshape_110 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_24 = ttnn.permute(
        ttnn_reshape_110,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_110, False)
    ttnn_repeat_29 = ttnn.repeat(ttnn_permute_24, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_24, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_repeat_29,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_29, False)
    ttnn_reshape_111 = ttnn.reshape(
        ttnn_permute_25,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_25, False)
    util_create_list_62 = [ttnn_reshape_111]
    return util_create_list_62


def main_const_eval_63(input):
    input_0 = input[0]
    ttnn_reshape_112 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_63 = [ttnn_reshape_112]
    return util_create_list_63


def main_const_eval_64(input):
    input_0 = input[0]
    ttnn_reshape_113 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_26 = ttnn.permute(
        ttnn_reshape_113,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_113, False)
    ttnn_repeat_30 = ttnn.repeat(ttnn_permute_26, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_26, False)
    ttnn_transformer_concatenate_heads_11 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_30,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_30, False)
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_11,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_11, False)
    util_create_list_64 = [ttnn_reshape_114]
    return util_create_list_64


def main_const_eval_65(input):
    input_0 = input[0]
    ttnn_reshape_115 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_27 = ttnn.permute(
        ttnn_reshape_115,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_115, False)
    ttnn_repeat_31 = ttnn.repeat(ttnn_permute_27, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_27, False)
    ttnn_transformer_concatenate_heads_12 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_31,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_31, False)
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_12,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_12, False)
    util_create_list_65 = [ttnn_reshape_116]
    return util_create_list_65


def main_const_eval_66(input):
    input_0 = input[0]
    ttnn_reshape_117 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_66 = [ttnn_reshape_117]
    return util_create_list_66


def main_const_eval_67(input):
    input_0 = input[0]
    ttnn_reshape_118 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_67 = [ttnn_reshape_118]
    return util_create_list_67


def main_const_eval_68(input):
    input_0 = input[0]
    ttnn_reshape_119 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_68 = [ttnn_reshape_119]
    return util_create_list_68


def main_const_eval_69(input):
    input_0 = input[0]
    ttnn_reshape_120 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_69 = [ttnn_reshape_120]
    return util_create_list_69


def main_const_eval_70(input):
    input_0 = input[0]
    ttnn_reshape_121 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_70 = [ttnn_reshape_121]
    return util_create_list_70


def main_const_eval_71(input):
    input_0 = input[0]
    ttnn_reshape_122 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_32 = ttnn.repeat(ttnn_reshape_122, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_122, False)
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_repeat_32,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_32, False)
    util_create_list_71 = [ttnn_reshape_123]
    return util_create_list_71


def main_const_eval_72(input):
    input_0 = input[0]
    ttnn_reshape_124 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_72 = [ttnn_reshape_124]
    return util_create_list_72


def main_const_eval_73(input):
    input_0 = input[0]
    ttnn_reshape_125 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_73 = [ttnn_reshape_125]
    return util_create_list_73


def main_const_eval_74(input):
    input_0 = input[0]
    ttnn_reshape_126 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_74 = [ttnn_reshape_126]
    return util_create_list_74


def main_const_eval_75(input):
    input_0 = input[0]
    ttnn_reshape_127 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_75 = [ttnn_reshape_127]
    return util_create_list_75


def main_const_eval_76(input):
    input_0 = input[0]
    ttnn_reshape_128 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_76 = [ttnn_reshape_128]
    return util_create_list_76


def main_const_eval_77(input):
    input_0 = input[0]
    ttnn_reshape_129 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_28 = ttnn.permute(
        ttnn_reshape_129,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_129, False)
    ttnn_repeat_33 = ttnn.repeat(ttnn_permute_28, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_28, False)
    ttnn_transformer_concatenate_heads_13 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_33,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_33, False)
    ttnn_reshape_130 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_13,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_13, False)
    util_create_list_77 = [ttnn_reshape_130]
    return util_create_list_77


def main_const_eval_78(input):
    input_0 = input[0]
    ttnn_reshape_131 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_34 = ttnn.repeat(ttnn_reshape_131, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_131, False)
    ttnn_reshape_132 = ttnn.reshape(
        ttnn_repeat_34,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_34, False)
    util_create_list_78 = [ttnn_reshape_132]
    return util_create_list_78


def main_const_eval_79(input):
    input_0 = input[0]
    ttnn_reshape_133 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_79 = [ttnn_reshape_133]
    return util_create_list_79


def main_const_eval_80(input):
    input_0 = input[0]
    ttnn_reshape_134 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_29 = ttnn.permute(
        ttnn_reshape_134,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_134, False)
    ttnn_repeat_35 = ttnn.repeat(ttnn_permute_29, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_transformer_concatenate_heads_14 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_35,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_35, False)
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_14,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_14, False)
    util_create_list_80 = [ttnn_reshape_135]
    return util_create_list_80


def main_const_eval_81(input):
    input_0 = input[0]
    ttnn_reshape_136 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_81 = [ttnn_reshape_136]
    return util_create_list_81


def main_const_eval_82(input):
    input_0 = input[0]
    input_1 = input[1]
    ttnn_typecast_0 = ttnn.typecast(
        input_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_embedding_0 = ttnn.embedding(ttnn_typecast_0, input_1)
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_repeat_36 = ttnn.repeat(ttnn_embedding_0, ttnn.Shape([2, 1, 1]))
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_repeat_36,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_36, False)
    util_create_list_82 = [ttnn_permute_30]
    return util_create_list_82


def main_const_eval_83(input):
    input_0 = input[0]
    ttnn_reshape_137 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_83 = [ttnn_reshape_137]
    return util_create_list_83


def main_const_eval_84(input):
    input_0 = input[0]
    ttnn_reshape_138 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_31 = ttnn.permute(
        ttnn_reshape_138,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_138, False)
    ttnn_repeat_37 = ttnn.repeat(ttnn_permute_31, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_31, False)
    ttnn_permute_32 = ttnn.permute(
        ttnn_repeat_37,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_37, False)
    ttnn_reshape_139 = ttnn.reshape(
        ttnn_permute_32,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_32, False)
    util_create_list_84 = [ttnn_reshape_139]
    return util_create_list_84


def main_const_eval_85(input):
    input_0 = input[0]
    ttnn_reshape_140 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_85 = [ttnn_reshape_140]
    return util_create_list_85


def main_const_eval_86(input):
    input_0 = input[0]
    ttnn_reshape_141 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_86 = [ttnn_reshape_141]
    return util_create_list_86


def main_const_eval_87(input):
    input_0 = input[0]
    ttnn_reshape_142 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_87 = [ttnn_reshape_142]
    return util_create_list_87


def main_const_eval_88(input):
    input_0 = input[0]
    ttnn_reshape_143 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_33 = ttnn.permute(
        ttnn_reshape_143,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn_repeat_38 = ttnn.repeat(ttnn_permute_33, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_33, False)
    ttnn_transformer_concatenate_heads_15 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_38,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_38, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_15,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_15, False)
    util_create_list_88 = [ttnn_reshape_144]
    return util_create_list_88


def main_const_eval_89(input):
    input_0 = input[0]
    ttnn_reshape_145 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_39 = ttnn.repeat(ttnn_reshape_145, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_145, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_repeat_39,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_39, False)
    util_create_list_89 = [ttnn_reshape_146]
    return util_create_list_89


def main_const_eval_90(input):
    input_0 = input[0]
    ttnn_reshape_147 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_40 = ttnn.repeat(ttnn_reshape_147, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_reshape_148 = ttnn.reshape(
        ttnn_repeat_40,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_40, False)
    util_create_list_90 = [ttnn_reshape_148]
    return util_create_list_90


def main_const_eval_91(input):
    input_0 = input[0]
    ttnn_reshape_149 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_41 = ttnn.repeat(ttnn_reshape_149, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_149, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_repeat_41,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_41, False)
    util_create_list_91 = [ttnn_reshape_150]
    return util_create_list_91


def main_const_eval_92(input):
    input_0 = input[0]
    ttnn_reshape_151 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_42 = ttnn.repeat(ttnn_reshape_151, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_repeat_42,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_42, False)
    util_create_list_92 = [ttnn_reshape_152]
    return util_create_list_92


def main_const_eval_93(input):
    input_0 = input[0]
    ttnn_reshape_153 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_93 = [ttnn_reshape_153]
    return util_create_list_93


def main_const_eval_94(input):
    input_0 = input[0]
    ttnn_reshape_154 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_94 = [ttnn_reshape_154]
    return util_create_list_94


def main_const_eval_95(input):
    input_0 = input[0]
    ttnn_reshape_155 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_95 = [ttnn_reshape_155]
    return util_create_list_95


def main_const_eval_96(input):
    input_0 = input[0]
    ttnn_reshape_156 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_34 = ttnn.permute(
        ttnn_reshape_156,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_156, False)
    ttnn_repeat_43 = ttnn.repeat(ttnn_permute_34, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_34, False)
    ttnn_transformer_concatenate_heads_16 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_43,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_43, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_16,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_16, False)
    util_create_list_96 = [ttnn_reshape_157]
    return util_create_list_96


def main_const_eval_97(input):
    input_0 = input[0]
    ttnn_reshape_158 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_97 = [ttnn_reshape_158]
    return util_create_list_97


def main_const_eval_98(input):
    input_0 = input[0]
    ttnn_reshape_159 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_44 = ttnn.repeat(ttnn_reshape_159, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_159, False)
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_repeat_44,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_44, False)
    util_create_list_98 = [ttnn_reshape_160]
    return util_create_list_98


def main_const_eval_99(input):
    input_0 = input[0]
    ttnn_reshape_161 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_35 = ttnn.permute(
        ttnn_reshape_161,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn_repeat_45 = ttnn.repeat(ttnn_permute_35, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_35, False)
    ttnn_transformer_concatenate_heads_17 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_45,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_45, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_17,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_17, False)
    util_create_list_99 = [ttnn_reshape_162]
    return util_create_list_99


def main_const_eval_100(input):
    input_0 = input[0]
    ttnn_reshape_163 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_36 = ttnn.permute(
        ttnn_reshape_163,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_163, False)
    ttnn_repeat_46 = ttnn.repeat(ttnn_permute_36, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_36, False)
    ttnn_transformer_concatenate_heads_18 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_46,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_46, False)
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_18,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_18, False)
    util_create_list_100 = [ttnn_reshape_164]
    return util_create_list_100


def main_const_eval_101(input):
    input_0 = input[0]
    ttnn_reshape_165 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_37 = ttnn.permute(
        ttnn_reshape_165,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_165, False)
    ttnn_repeat_47 = ttnn.repeat(ttnn_permute_37, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_37, False)
    ttnn_permute_38 = ttnn.permute(
        ttnn_repeat_47,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_47, False)
    ttnn_reshape_166 = ttnn.reshape(
        ttnn_permute_38,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_38, False)
    util_create_list_101 = [ttnn_reshape_166]
    return util_create_list_101


def main_const_eval_102(input):
    input_0 = input[0]
    ttnn_reshape_167 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_48 = ttnn.repeat(ttnn_reshape_167, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_167, False)
    ttnn_reshape_168 = ttnn.reshape(
        ttnn_repeat_48,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_48, False)
    util_create_list_102 = [ttnn_reshape_168]
    return util_create_list_102


def main_const_eval_103(input):
    input_0 = input[0]
    ttnn_reshape_169 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_39 = ttnn.permute(
        ttnn_reshape_169,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_169, False)
    ttnn_repeat_49 = ttnn.repeat(ttnn_permute_39, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_39, False)
    ttnn_transformer_concatenate_heads_19 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_49,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_49, False)
    ttnn_reshape_170 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_19,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_19, False)
    util_create_list_103 = [ttnn_reshape_170]
    return util_create_list_103


def main_const_eval_104(input):
    input_0 = input[0]
    ttnn_reshape_171 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_104 = [ttnn_reshape_171]
    return util_create_list_104


def main_const_eval_105(input):
    input_0 = input[0]
    ttnn_reshape_172 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_105 = [ttnn_reshape_172]
    return util_create_list_105


def main_const_eval_106(input):
    input_0 = input[0]
    ttnn_reshape_173 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_106 = [ttnn_reshape_173]
    return util_create_list_106


def main_const_eval_107(input):
    input_0 = input[0]
    ttnn_reshape_174 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_107 = [ttnn_reshape_174]
    return util_create_list_107


def main_const_eval_108(input):
    input_0 = input[0]
    ttnn_reshape_175 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_40 = ttnn.permute(
        ttnn_reshape_175,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_175, False)
    ttnn_repeat_50 = ttnn.repeat(ttnn_permute_40, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_40, False)
    ttnn_permute_41 = ttnn.permute(
        ttnn_repeat_50,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_50, False)
    ttnn_reshape_176 = ttnn.reshape(
        ttnn_permute_41,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_41, False)
    util_create_list_108 = [ttnn_reshape_176]
    return util_create_list_108


def main_const_eval_109(input):
    input_0 = input[0]
    ttnn_reshape_177 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_109 = [ttnn_reshape_177]
    return util_create_list_109


def main_const_eval_110(input):
    input_0 = input[0]
    ttnn_reshape_178 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_51 = ttnn.repeat(ttnn_reshape_178, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_178, False)
    ttnn_reshape_179 = ttnn.reshape(
        ttnn_repeat_51,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_51, False)
    util_create_list_110 = [ttnn_reshape_179]
    return util_create_list_110


def main_const_eval_111(input):
    input_0 = input[0]
    ttnn_reshape_180 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_111 = [ttnn_reshape_180]
    return util_create_list_111


def main_const_eval_112(input):
    input_0 = input[0]
    ttnn_reshape_181 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_42 = ttnn.permute(
        ttnn_reshape_181,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_181, False)
    ttnn_repeat_52 = ttnn.repeat(ttnn_permute_42, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_42, False)
    ttnn_transformer_concatenate_heads_20 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_52,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_52, False)
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_20,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_20, False)
    util_create_list_112 = [ttnn_reshape_182]
    return util_create_list_112


def main_const_eval_113(input):
    input_0 = input[0]
    ttnn_reshape_183 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_53 = ttnn.repeat(ttnn_reshape_183, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_183, False)
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_repeat_53,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_53, False)
    util_create_list_113 = [ttnn_reshape_184]
    return util_create_list_113


def main_const_eval_114(input):
    input_0 = input[0]
    ttnn_reshape_185 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_114 = [ttnn_reshape_185]
    return util_create_list_114


def main_const_eval_115(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=input_0,
        input_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=3,
        out_channels=768,
        batch_size=2,
        input_height=224,
        input_width=224,
        kernel_size=[32, 32],
        stride=[32, 32],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=utils_DeviceGetter_get_device_4,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=32,
            act_block_w_div=1,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            transpose_shards=False,
            output_layout=ttnn.Layout.TILE,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            in_place=False,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    util_create_list_115 = [ttnn_prepare_conv_weights_0]
    return util_create_list_115


def main_const_eval_116(input):
    input_0 = input[0]
    ttnn_reshape_186 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_116 = [ttnn_reshape_186]
    return util_create_list_116


def main_const_eval_117(input):
    input_0 = input[0]
    ttnn_reshape_187 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_43 = ttnn.permute(
        ttnn_reshape_187,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_187, False)
    ttnn_repeat_54 = ttnn.repeat(ttnn_permute_43, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_43, False)
    ttnn_transformer_concatenate_heads_21 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_54,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_54, False)
    ttnn_reshape_188 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_21,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_21, False)
    util_create_list_117 = [ttnn_reshape_188]
    return util_create_list_117


def main_const_eval_118(input):
    input_0 = input[0]
    ttnn_reshape_189 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_118 = [ttnn_reshape_189]
    return util_create_list_118


def main_const_eval_119():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_4 = ttnn.full(
        shape=ttnn.Shape([2, 50, 3072]),
        fill_value=1.703125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_192 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_193 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_194 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_195 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_196 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_197 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_198 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_199 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_200 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_201 = ttnn.reshape(
        ttnn_full_4,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_full_4, False)
    util_create_list_119 = [
        ttnn_reshape_190,
        ttnn_reshape_191,
        ttnn_reshape_192,
        ttnn_reshape_193,
        ttnn_reshape_194,
        ttnn_reshape_195,
        ttnn_reshape_196,
        ttnn_reshape_197,
        ttnn_reshape_198,
        ttnn_reshape_199,
        ttnn_reshape_200,
        ttnn_reshape_201,
    ]
    return util_create_list_119


def main_const_eval_120(input):
    input_0 = input[0]
    ttnn_reshape_202 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_120 = [ttnn_reshape_202]
    return util_create_list_120


def main_const_eval_121(input):
    input_0 = input[0]
    ttnn_reshape_203 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_121 = [ttnn_reshape_203]
    return util_create_list_121


def main_const_eval_122(input):
    input_0 = input[0]
    ttnn_reshape_204 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_44 = ttnn.permute(
        ttnn_reshape_204,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_204, False)
    ttnn_repeat_55 = ttnn.repeat(ttnn_permute_44, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_44, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_repeat_55,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_55, False)
    ttnn_reshape_205 = ttnn.reshape(
        ttnn_permute_45,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    util_create_list_122 = [ttnn_reshape_205]
    return util_create_list_122


def main_const_eval_123(input):
    input_0 = input[0]
    ttnn_reshape_206 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_123 = [ttnn_reshape_206]
    return util_create_list_123


def main_const_eval_124(input):
    input_0 = input[0]
    ttnn_reshape_207 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_124 = [ttnn_reshape_207]
    return util_create_list_124


def main_const_eval_125(input):
    input_0 = input[0]
    ttnn_reshape_208 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_46 = ttnn.permute(
        ttnn_reshape_208,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_208, False)
    ttnn_repeat_56 = ttnn.repeat(ttnn_permute_46, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_46, False)
    ttnn_transformer_concatenate_heads_22 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_56,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_56, False)
    ttnn_reshape_209 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_22,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_22, False)
    util_create_list_125 = [ttnn_reshape_209]
    return util_create_list_125


def main_const_eval_126(input):
    input_0 = input[0]
    ttnn_reshape_210 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_126 = [ttnn_reshape_210]
    return util_create_list_126


def main_const_eval_127(input):
    input_0 = input[0]
    ttnn_reshape_211 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_57 = ttnn.repeat(ttnn_reshape_211, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_211, False)
    ttnn_reshape_212 = ttnn.reshape(
        ttnn_repeat_57,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_57, False)
    util_create_list_127 = [ttnn_reshape_212]
    return util_create_list_127


def main_const_eval_128():
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_5 = ttnn.full(
        shape=ttnn.Shape([2, 12, 50, 50]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_128 = [ttnn_full_5]
    return util_create_list_128


def main_const_eval_129(input):
    input_0 = input[0]
    ttnn_reshape_213 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_47 = ttnn.permute(
        ttnn_reshape_213,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_213, False)
    ttnn_repeat_58 = ttnn.repeat(ttnn_permute_47, ttnn.Shape([2, 1, 50, 1]))
    ttnn.deallocate(ttnn_permute_47, False)
    ttnn_transformer_concatenate_heads_23 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_58,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_58, False)
    ttnn_reshape_214 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_23,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_23, False)
    util_create_list_129 = [ttnn_reshape_214]
    return util_create_list_129


def main_const_eval_130(input):
    input_0 = input[0]
    ttnn_reshape_215 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_48 = ttnn.permute(
        ttnn_reshape_215,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_215, False)
    ttnn_repeat_59 = ttnn.repeat(ttnn_permute_48, ttnn.Shape([2, 1, 1, 50]))
    ttnn.deallocate(ttnn_permute_48, False)
    ttnn_permute_49 = ttnn.permute(
        ttnn_repeat_59,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_repeat_59, False)
    ttnn_reshape_216 = ttnn.reshape(
        ttnn_permute_49,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_49, False)
    util_create_list_130 = [ttnn_reshape_216]
    return util_create_list_130


def main_const_eval_131(input):
    input_0 = input[0]
    ttnn_reshape_217 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_60 = ttnn.repeat(ttnn_reshape_217, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_217, False)
    ttnn_reshape_218 = ttnn.reshape(
        ttnn_repeat_60,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_60, False)
    util_create_list_131 = [ttnn_reshape_218]
    return util_create_list_131


def main_const_eval_132(input):
    input_0 = input[0]
    ttnn_reshape_219 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_61 = ttnn.repeat(ttnn_reshape_219, ttnn.Shape([2, 50, 1]))
    ttnn.deallocate(ttnn_reshape_219, False)
    ttnn_reshape_220 = ttnn.reshape(
        ttnn_repeat_61,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_61, False)
    util_create_list_132 = [ttnn_reshape_220]
    return util_create_list_132


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None
CACHED_main_const_eval_4 = None
CACHED_main_const_eval_5 = None
CACHED_main_const_eval_6 = None
CACHED_main_const_eval_7 = None
CACHED_main_const_eval_8 = None
CACHED_main_const_eval_9 = None
CACHED_main_const_eval_10 = None
CACHED_main_const_eval_11 = None
CACHED_main_const_eval_12 = None
CACHED_main_const_eval_13 = None
CACHED_main_const_eval_14 = None
CACHED_main_const_eval_15 = None
CACHED_main_const_eval_16 = None
CACHED_main_const_eval_17 = None
CACHED_main_const_eval_18 = None
CACHED_main_const_eval_19 = None
CACHED_main_const_eval_20 = None
CACHED_main_const_eval_21 = None
CACHED_main_const_eval_22 = None
CACHED_main_const_eval_23 = None
CACHED_main_const_eval_24 = None
CACHED_main_const_eval_25 = None
CACHED_main_const_eval_26 = None
CACHED_main_const_eval_27 = None
CACHED_main_const_eval_28 = None
CACHED_main_const_eval_29 = None
CACHED_main_const_eval_30 = None
CACHED_main_const_eval_31 = None
CACHED_main_const_eval_32 = None
CACHED_main_const_eval_33 = None
CACHED_main_const_eval_34 = None
CACHED_main_const_eval_35 = None
CACHED_main_const_eval_36 = None
CACHED_main_const_eval_37 = None
CACHED_main_const_eval_38 = None
CACHED_main_const_eval_39 = None
CACHED_main_const_eval_40 = None
CACHED_main_const_eval_41 = None
CACHED_main_const_eval_42 = None
CACHED_main_const_eval_43 = None
CACHED_main_const_eval_44 = None
CACHED_main_const_eval_45 = None
CACHED_main_const_eval_46 = None
CACHED_main_const_eval_47 = None
CACHED_main_const_eval_48 = None
CACHED_main_const_eval_49 = None
CACHED_main_const_eval_50 = None
CACHED_main_const_eval_51 = None
CACHED_main_const_eval_52 = None
CACHED_main_const_eval_53 = None
CACHED_main_const_eval_54 = None
CACHED_main_const_eval_55 = None
CACHED_main_const_eval_56 = None
CACHED_main_const_eval_57 = None
CACHED_main_const_eval_58 = None
CACHED_main_const_eval_59 = None
CACHED_main_const_eval_60 = None
CACHED_main_const_eval_61 = None
CACHED_main_const_eval_62 = None
CACHED_main_const_eval_63 = None
CACHED_main_const_eval_64 = None
CACHED_main_const_eval_65 = None
CACHED_main_const_eval_66 = None
CACHED_main_const_eval_67 = None
CACHED_main_const_eval_68 = None
CACHED_main_const_eval_69 = None
CACHED_main_const_eval_70 = None
CACHED_main_const_eval_71 = None
CACHED_main_const_eval_72 = None
CACHED_main_const_eval_73 = None
CACHED_main_const_eval_74 = None
CACHED_main_const_eval_75 = None
CACHED_main_const_eval_76 = None
CACHED_main_const_eval_77 = None
CACHED_main_const_eval_78 = None
CACHED_main_const_eval_79 = None
CACHED_main_const_eval_80 = None
CACHED_main_const_eval_81 = None
CACHED_main_const_eval_82 = None
CACHED_main_const_eval_83 = None
CACHED_main_const_eval_84 = None
CACHED_main_const_eval_85 = None
CACHED_main_const_eval_86 = None
CACHED_main_const_eval_87 = None
CACHED_main_const_eval_88 = None
CACHED_main_const_eval_89 = None
CACHED_main_const_eval_90 = None
CACHED_main_const_eval_91 = None
CACHED_main_const_eval_92 = None
CACHED_main_const_eval_93 = None
CACHED_main_const_eval_94 = None
CACHED_main_const_eval_95 = None
CACHED_main_const_eval_96 = None
CACHED_main_const_eval_97 = None
CACHED_main_const_eval_98 = None
CACHED_main_const_eval_99 = None
CACHED_main_const_eval_100 = None
CACHED_main_const_eval_101 = None
CACHED_main_const_eval_102 = None
CACHED_main_const_eval_103 = None
CACHED_main_const_eval_104 = None
CACHED_main_const_eval_105 = None
CACHED_main_const_eval_106 = None
CACHED_main_const_eval_107 = None
CACHED_main_const_eval_108 = None
CACHED_main_const_eval_109 = None
CACHED_main_const_eval_110 = None
CACHED_main_const_eval_111 = None
CACHED_main_const_eval_112 = None
CACHED_main_const_eval_113 = None
CACHED_main_const_eval_114 = None
CACHED_main_const_eval_115 = None
CACHED_main_const_eval_116 = None
CACHED_main_const_eval_117 = None
CACHED_main_const_eval_118 = None
CACHED_main_const_eval_119 = None
CACHED_main_const_eval_120 = None
CACHED_main_const_eval_121 = None
CACHED_main_const_eval_122 = None
CACHED_main_const_eval_123 = None
CACHED_main_const_eval_124 = None
CACHED_main_const_eval_125 = None
CACHED_main_const_eval_126 = None
CACHED_main_const_eval_127 = None
CACHED_main_const_eval_128 = None
CACHED_main_const_eval_129 = None
CACHED_main_const_eval_130 = None
CACHED_main_const_eval_131 = None
CACHED_main_const_eval_132 = None


# Cache for const_eval results
_CONST_EVAL_CACHE = {i: None for i in range(133)}


def execute_all_const_evals(inputs):
    """Execute all const_eval functions and cache results.

    Args:
        inputs: List of input tensors (indexed 0-201)

    Returns:
        Dictionary mapping const_eval index to full result tuple/list
    """
    global _CONST_EVAL_CACHE
    results = {}

    # const_eval_0: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_0, _CONST_EVAL_CACHE[0]
    )
    _CONST_EVAL_CACHE[0] = result
    results[0] = result

    # const_eval_1: inputs [178]
    result = utils.constEvalFuncWrapper(
        main_const_eval_1, [inputs[178]], _CONST_EVAL_CACHE[1]
    )
    _CONST_EVAL_CACHE[1] = result
    results[1] = result

    # const_eval_2: inputs [59]
    result = utils.constEvalFuncWrapper(
        main_const_eval_2, [inputs[59]], _CONST_EVAL_CACHE[2]
    )
    _CONST_EVAL_CACHE[2] = result
    results[2] = result

    # const_eval_3: inputs [145]
    result = utils.constEvalFuncWrapper(
        main_const_eval_3, [inputs[145]], _CONST_EVAL_CACHE[3]
    )
    _CONST_EVAL_CACHE[3] = result
    results[3] = result

    # const_eval_4: inputs [137]
    result = utils.constEvalFuncWrapper(
        main_const_eval_4, [inputs[137]], _CONST_EVAL_CACHE[4]
    )
    _CONST_EVAL_CACHE[4] = result
    results[4] = result

    # const_eval_5: inputs [20]
    result = utils.constEvalFuncWrapper(
        main_const_eval_5, [inputs[20]], _CONST_EVAL_CACHE[5]
    )
    _CONST_EVAL_CACHE[5] = result
    results[5] = result

    # const_eval_6: inputs [127]
    result = utils.constEvalFuncWrapper(
        main_const_eval_6, [inputs[127]], _CONST_EVAL_CACHE[6]
    )
    _CONST_EVAL_CACHE[6] = result
    results[6] = result

    # const_eval_7: inputs [31]
    result = utils.constEvalFuncWrapper(
        main_const_eval_7, [inputs[31]], _CONST_EVAL_CACHE[7]
    )
    _CONST_EVAL_CACHE[7] = result
    results[7] = result

    # const_eval_8: inputs [116]
    result = utils.constEvalFuncWrapper(
        main_const_eval_8, [inputs[116]], _CONST_EVAL_CACHE[8]
    )
    _CONST_EVAL_CACHE[8] = result
    results[8] = result

    # const_eval_9: inputs [123]
    result = utils.constEvalFuncWrapper(
        main_const_eval_9, [inputs[123]], _CONST_EVAL_CACHE[9]
    )
    _CONST_EVAL_CACHE[9] = result
    results[9] = result

    # const_eval_10: inputs [153]
    result = utils.constEvalFuncWrapper(
        main_const_eval_10, [inputs[153]], _CONST_EVAL_CACHE[10]
    )
    _CONST_EVAL_CACHE[10] = result
    results[10] = result

    # const_eval_11: inputs [186]
    result = utils.constEvalFuncWrapper(
        main_const_eval_11, [inputs[186]], _CONST_EVAL_CACHE[11]
    )
    _CONST_EVAL_CACHE[11] = result
    results[11] = result

    # const_eval_12: inputs [190]
    result = utils.constEvalFuncWrapper(
        main_const_eval_12, [inputs[190]], _CONST_EVAL_CACHE[12]
    )
    _CONST_EVAL_CACHE[12] = result
    results[12] = result

    # const_eval_13: inputs [188]
    result = utils.constEvalFuncWrapper(
        main_const_eval_13, [inputs[188]], _CONST_EVAL_CACHE[13]
    )
    _CONST_EVAL_CACHE[13] = result
    results[13] = result

    # const_eval_14: inputs [3]
    result = utils.constEvalFuncWrapper(
        main_const_eval_14, [inputs[3]], _CONST_EVAL_CACHE[14]
    )
    _CONST_EVAL_CACHE[14] = result
    results[14] = result

    # const_eval_15: inputs [125]
    result = utils.constEvalFuncWrapper(
        main_const_eval_15, [inputs[125]], _CONST_EVAL_CACHE[15]
    )
    _CONST_EVAL_CACHE[15] = result
    results[15] = result

    # const_eval_16: inputs [21]
    result = utils.constEvalFuncWrapper(
        main_const_eval_16, [inputs[21]], _CONST_EVAL_CACHE[16]
    )
    _CONST_EVAL_CACHE[16] = result
    results[16] = result

    # const_eval_17: inputs [156]
    result = utils.constEvalFuncWrapper(
        main_const_eval_17, [inputs[156]], _CONST_EVAL_CACHE[17]
    )
    _CONST_EVAL_CACHE[17] = result
    results[17] = result

    # const_eval_18: inputs [194]
    result = utils.constEvalFuncWrapper(
        main_const_eval_18, [inputs[194]], _CONST_EVAL_CACHE[18]
    )
    _CONST_EVAL_CACHE[18] = result
    results[18] = result

    # const_eval_19: inputs [15]
    result = utils.constEvalFuncWrapper(
        main_const_eval_19, [inputs[15]], _CONST_EVAL_CACHE[19]
    )
    _CONST_EVAL_CACHE[19] = result
    results[19] = result

    # const_eval_20: inputs [131]
    result = utils.constEvalFuncWrapper(
        main_const_eval_20, [inputs[131]], _CONST_EVAL_CACHE[20]
    )
    _CONST_EVAL_CACHE[20] = result
    results[20] = result

    # const_eval_21: inputs [111]
    result = utils.constEvalFuncWrapper(
        main_const_eval_21, [inputs[111]], _CONST_EVAL_CACHE[21]
    )
    _CONST_EVAL_CACHE[21] = result
    results[21] = result

    # const_eval_22: inputs [27]
    result = utils.constEvalFuncWrapper(
        main_const_eval_22, [inputs[27]], _CONST_EVAL_CACHE[22]
    )
    _CONST_EVAL_CACHE[22] = result
    results[22] = result

    # const_eval_23: inputs [97]
    result = utils.constEvalFuncWrapper(
        main_const_eval_23, [inputs[97]], _CONST_EVAL_CACHE[23]
    )
    _CONST_EVAL_CACHE[23] = result
    results[23] = result

    # const_eval_24: inputs [77]
    result = utils.constEvalFuncWrapper(
        main_const_eval_24, [inputs[77]], _CONST_EVAL_CACHE[24]
    )
    _CONST_EVAL_CACHE[24] = result
    results[24] = result

    # const_eval_25: inputs [62]
    result = utils.constEvalFuncWrapper(
        main_const_eval_25, [inputs[62]], _CONST_EVAL_CACHE[25]
    )
    _CONST_EVAL_CACHE[25] = result
    results[25] = result

    # const_eval_26: inputs [110]
    result = utils.constEvalFuncWrapper(
        main_const_eval_26, [inputs[110]], _CONST_EVAL_CACHE[26]
    )
    _CONST_EVAL_CACHE[26] = result
    results[26] = result

    # const_eval_27: inputs [141]
    result = utils.constEvalFuncWrapper(
        main_const_eval_27, [inputs[141]], _CONST_EVAL_CACHE[27]
    )
    _CONST_EVAL_CACHE[27] = result
    results[27] = result

    # const_eval_28: inputs [65]
    result = utils.constEvalFuncWrapper(
        main_const_eval_28, [inputs[65]], _CONST_EVAL_CACHE[28]
    )
    _CONST_EVAL_CACHE[28] = result
    results[28] = result

    # const_eval_29: inputs [79]
    result = utils.constEvalFuncWrapper(
        main_const_eval_29, [inputs[79]], _CONST_EVAL_CACHE[29]
    )
    _CONST_EVAL_CACHE[29] = result
    results[29] = result

    # const_eval_30: inputs [192]
    result = utils.constEvalFuncWrapper(
        main_const_eval_30, [inputs[192]], _CONST_EVAL_CACHE[30]
    )
    _CONST_EVAL_CACHE[30] = result
    results[30] = result

    # const_eval_31: inputs [128]
    result = utils.constEvalFuncWrapper(
        main_const_eval_31, [inputs[128]], _CONST_EVAL_CACHE[31]
    )
    _CONST_EVAL_CACHE[31] = result
    results[31] = result

    # const_eval_32: inputs [80]
    result = utils.constEvalFuncWrapper(
        main_const_eval_32, [inputs[80]], _CONST_EVAL_CACHE[32]
    )
    _CONST_EVAL_CACHE[32] = result
    results[32] = result

    # const_eval_33: inputs [117]
    result = utils.constEvalFuncWrapper(
        main_const_eval_33, [inputs[117]], _CONST_EVAL_CACHE[33]
    )
    _CONST_EVAL_CACHE[33] = result
    results[33] = result

    # const_eval_34: inputs [115]
    result = utils.constEvalFuncWrapper(
        main_const_eval_34, [inputs[115]], _CONST_EVAL_CACHE[34]
    )
    _CONST_EVAL_CACHE[34] = result
    results[34] = result

    # const_eval_35: inputs [135]
    result = utils.constEvalFuncWrapper(
        main_const_eval_35, [inputs[135]], _CONST_EVAL_CACHE[35]
    )
    _CONST_EVAL_CACHE[35] = result
    results[35] = result

    # const_eval_36: inputs [168]
    result = utils.constEvalFuncWrapper(
        main_const_eval_36, [inputs[168]], _CONST_EVAL_CACHE[36]
    )
    _CONST_EVAL_CACHE[36] = result
    results[36] = result

    # const_eval_37: inputs [172]
    result = utils.constEvalFuncWrapper(
        main_const_eval_37, [inputs[172]], _CONST_EVAL_CACHE[37]
    )
    _CONST_EVAL_CACHE[37] = result
    results[37] = result

    # const_eval_38: inputs [75]
    result = utils.constEvalFuncWrapper(
        main_const_eval_38, [inputs[75]], _CONST_EVAL_CACHE[38]
    )
    _CONST_EVAL_CACHE[38] = result
    results[38] = result

    # const_eval_39: inputs [174]
    result = utils.constEvalFuncWrapper(
        main_const_eval_39, [inputs[174]], _CONST_EVAL_CACHE[39]
    )
    _CONST_EVAL_CACHE[39] = result
    results[39] = result

    # const_eval_40: inputs [13]
    result = utils.constEvalFuncWrapper(
        main_const_eval_40, [inputs[13]], _CONST_EVAL_CACHE[40]
    )
    _CONST_EVAL_CACHE[40] = result
    results[40] = result

    # const_eval_41: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_41, _CONST_EVAL_CACHE[41]
    )
    _CONST_EVAL_CACHE[41] = result
    results[41] = result

    # const_eval_42: inputs [50]
    result = utils.constEvalFuncWrapper(
        main_const_eval_42, [inputs[50]], _CONST_EVAL_CACHE[42]
    )
    _CONST_EVAL_CACHE[42] = result
    results[42] = result

    # const_eval_43: inputs [23]
    result = utils.constEvalFuncWrapper(
        main_const_eval_43, [inputs[23]], _CONST_EVAL_CACHE[43]
    )
    _CONST_EVAL_CACHE[43] = result
    results[43] = result

    # const_eval_44: inputs [85]
    result = utils.constEvalFuncWrapper(
        main_const_eval_44, [inputs[85]], _CONST_EVAL_CACHE[44]
    )
    _CONST_EVAL_CACHE[44] = result
    results[44] = result

    # const_eval_45: inputs [98]
    result = utils.constEvalFuncWrapper(
        main_const_eval_45, [inputs[98]], _CONST_EVAL_CACHE[45]
    )
    _CONST_EVAL_CACHE[45] = result
    results[45] = result

    # const_eval_46: inputs [2]
    result = utils.constEvalFuncWrapper(
        main_const_eval_46, [inputs[2]], _CONST_EVAL_CACHE[46]
    )
    _CONST_EVAL_CACHE[46] = result
    results[46] = result

    # const_eval_47: inputs [45]
    result = utils.constEvalFuncWrapper(
        main_const_eval_47, [inputs[45]], _CONST_EVAL_CACHE[47]
    )
    _CONST_EVAL_CACHE[47] = result
    results[47] = result

    # const_eval_48: inputs [43]
    result = utils.constEvalFuncWrapper(
        main_const_eval_48, [inputs[43]], _CONST_EVAL_CACHE[48]
    )
    _CONST_EVAL_CACHE[48] = result
    results[48] = result

    # const_eval_49: inputs [8]
    result = utils.constEvalFuncWrapper(
        main_const_eval_49, [inputs[8]], _CONST_EVAL_CACHE[49]
    )
    _CONST_EVAL_CACHE[49] = result
    results[49] = result

    # const_eval_50: inputs [19]
    result = utils.constEvalFuncWrapper(
        main_const_eval_50, [inputs[19]], _CONST_EVAL_CACHE[50]
    )
    _CONST_EVAL_CACHE[50] = result
    results[50] = result

    # const_eval_51: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_51, _CONST_EVAL_CACHE[51]
    )
    _CONST_EVAL_CACHE[51] = result
    results[51] = result

    # const_eval_52: inputs [83]
    result = utils.constEvalFuncWrapper(
        main_const_eval_52, [inputs[83]], _CONST_EVAL_CACHE[52]
    )
    _CONST_EVAL_CACHE[52] = result
    results[52] = result

    # const_eval_53: inputs [1]
    result = utils.constEvalFuncWrapper(
        main_const_eval_53, [inputs[1]], _CONST_EVAL_CACHE[53]
    )
    _CONST_EVAL_CACHE[53] = result
    results[53] = result

    # const_eval_54: inputs [17]
    result = utils.constEvalFuncWrapper(
        main_const_eval_54, [inputs[17]], _CONST_EVAL_CACHE[54]
    )
    _CONST_EVAL_CACHE[54] = result
    results[54] = result

    # const_eval_55: inputs [14]
    result = utils.constEvalFuncWrapper(
        main_const_eval_55, [inputs[14]], _CONST_EVAL_CACHE[55]
    )
    _CONST_EVAL_CACHE[55] = result
    results[55] = result

    # const_eval_56: inputs [198]
    result = utils.constEvalFuncWrapper(
        main_const_eval_56, [inputs[198]], _CONST_EVAL_CACHE[56]
    )
    _CONST_EVAL_CACHE[56] = result
    results[56] = result

    # const_eval_57: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_57, _CONST_EVAL_CACHE[57]
    )
    _CONST_EVAL_CACHE[57] = result
    results[57] = result

    # const_eval_58: inputs [122]
    result = utils.constEvalFuncWrapper(
        main_const_eval_58, [inputs[122]], _CONST_EVAL_CACHE[58]
    )
    _CONST_EVAL_CACHE[58] = result
    results[58] = result

    # const_eval_59: inputs [184]
    result = utils.constEvalFuncWrapper(
        main_const_eval_59, [inputs[184]], _CONST_EVAL_CACHE[59]
    )
    _CONST_EVAL_CACHE[59] = result
    results[59] = result

    # const_eval_60: inputs [35]
    result = utils.constEvalFuncWrapper(
        main_const_eval_60, [inputs[35]], _CONST_EVAL_CACHE[60]
    )
    _CONST_EVAL_CACHE[60] = result
    results[60] = result

    # const_eval_61: inputs [148]
    result = utils.constEvalFuncWrapper(
        main_const_eval_61, [inputs[148]], _CONST_EVAL_CACHE[61]
    )
    _CONST_EVAL_CACHE[61] = result
    results[61] = result

    # const_eval_62: inputs [166]
    result = utils.constEvalFuncWrapper(
        main_const_eval_62, [inputs[166]], _CONST_EVAL_CACHE[62]
    )
    _CONST_EVAL_CACHE[62] = result
    results[62] = result

    # const_eval_63: inputs [61]
    result = utils.constEvalFuncWrapper(
        main_const_eval_63, [inputs[61]], _CONST_EVAL_CACHE[63]
    )
    _CONST_EVAL_CACHE[63] = result
    results[63] = result

    # const_eval_64: inputs [160]
    result = utils.constEvalFuncWrapper(
        main_const_eval_64, [inputs[160]], _CONST_EVAL_CACHE[64]
    )
    _CONST_EVAL_CACHE[64] = result
    results[64] = result

    # const_eval_65: inputs [71]
    result = utils.constEvalFuncWrapper(
        main_const_eval_65, [inputs[71]], _CONST_EVAL_CACHE[65]
    )
    _CONST_EVAL_CACHE[65] = result
    results[65] = result

    # const_eval_66: inputs [92]
    result = utils.constEvalFuncWrapper(
        main_const_eval_66, [inputs[92]], _CONST_EVAL_CACHE[66]
    )
    _CONST_EVAL_CACHE[66] = result
    results[66] = result

    # const_eval_67: inputs [53]
    result = utils.constEvalFuncWrapper(
        main_const_eval_67, [inputs[53]], _CONST_EVAL_CACHE[67]
    )
    _CONST_EVAL_CACHE[67] = result
    results[67] = result

    # const_eval_68: inputs [55]
    result = utils.constEvalFuncWrapper(
        main_const_eval_68, [inputs[55]], _CONST_EVAL_CACHE[68]
    )
    _CONST_EVAL_CACHE[68] = result
    results[68] = result

    # const_eval_69: inputs [140]
    result = utils.constEvalFuncWrapper(
        main_const_eval_69, [inputs[140]], _CONST_EVAL_CACHE[69]
    )
    _CONST_EVAL_CACHE[69] = result
    results[69] = result

    # const_eval_70: inputs [68]
    result = utils.constEvalFuncWrapper(
        main_const_eval_70, [inputs[68]], _CONST_EVAL_CACHE[70]
    )
    _CONST_EVAL_CACHE[70] = result
    results[70] = result

    # const_eval_71: inputs [9]
    result = utils.constEvalFuncWrapper(
        main_const_eval_71, [inputs[9]], _CONST_EVAL_CACHE[71]
    )
    _CONST_EVAL_CACHE[71] = result
    results[71] = result

    # const_eval_72: inputs [104]
    result = utils.constEvalFuncWrapper(
        main_const_eval_72, [inputs[104]], _CONST_EVAL_CACHE[72]
    )
    _CONST_EVAL_CACHE[72] = result
    results[72] = result

    # const_eval_73: inputs [25]
    result = utils.constEvalFuncWrapper(
        main_const_eval_73, [inputs[25]], _CONST_EVAL_CACHE[73]
    )
    _CONST_EVAL_CACHE[73] = result
    results[73] = result

    # const_eval_74: inputs [133]
    result = utils.constEvalFuncWrapper(
        main_const_eval_74, [inputs[133]], _CONST_EVAL_CACHE[74]
    )
    _CONST_EVAL_CACHE[74] = result
    results[74] = result

    # const_eval_75: inputs [86]
    result = utils.constEvalFuncWrapper(
        main_const_eval_75, [inputs[86]], _CONST_EVAL_CACHE[75]
    )
    _CONST_EVAL_CACHE[75] = result
    results[75] = result

    # const_eval_76: inputs [103]
    result = utils.constEvalFuncWrapper(
        main_const_eval_76, [inputs[103]], _CONST_EVAL_CACHE[76]
    )
    _CONST_EVAL_CACHE[76] = result
    results[76] = result

    # const_eval_77: inputs [176]
    result = utils.constEvalFuncWrapper(
        main_const_eval_77, [inputs[176]], _CONST_EVAL_CACHE[77]
    )
    _CONST_EVAL_CACHE[77] = result
    results[77] = result

    # const_eval_78: inputs [57]
    result = utils.constEvalFuncWrapper(
        main_const_eval_78, [inputs[57]], _CONST_EVAL_CACHE[78]
    )
    _CONST_EVAL_CACHE[78] = result
    results[78] = result

    # const_eval_79: inputs [49]
    result = utils.constEvalFuncWrapper(
        main_const_eval_79, [inputs[49]], _CONST_EVAL_CACHE[79]
    )
    _CONST_EVAL_CACHE[79] = result
    results[79] = result

    # const_eval_80: inputs [119]
    result = utils.constEvalFuncWrapper(
        main_const_eval_80, [inputs[119]], _CONST_EVAL_CACHE[80]
    )
    _CONST_EVAL_CACHE[80] = result
    results[80] = result

    # const_eval_81: inputs [91]
    result = utils.constEvalFuncWrapper(
        main_const_eval_81, [inputs[91]], _CONST_EVAL_CACHE[81]
    )
    _CONST_EVAL_CACHE[81] = result
    results[81] = result

    # const_eval_82: inputs [149, 150]
    result = utils.constEvalFuncWrapper(
        main_const_eval_82, [inputs[149], inputs[150]], _CONST_EVAL_CACHE[82]
    )
    _CONST_EVAL_CACHE[82] = result
    results[82] = result

    # const_eval_83: inputs [74]
    result = utils.constEvalFuncWrapper(
        main_const_eval_83, [inputs[74]], _CONST_EVAL_CACHE[83]
    )
    _CONST_EVAL_CACHE[83] = result
    results[83] = result

    # const_eval_84: inputs [170]
    result = utils.constEvalFuncWrapper(
        main_const_eval_84, [inputs[170]], _CONST_EVAL_CACHE[84]
    )
    _CONST_EVAL_CACHE[84] = result
    results[84] = result

    # const_eval_85: inputs [73]
    result = utils.constEvalFuncWrapper(
        main_const_eval_85, [inputs[73]], _CONST_EVAL_CACHE[85]
    )
    _CONST_EVAL_CACHE[85] = result
    results[85] = result

    # const_eval_86: inputs [5]
    result = utils.constEvalFuncWrapper(
        main_const_eval_86, [inputs[5]], _CONST_EVAL_CACHE[86]
    )
    _CONST_EVAL_CACHE[86] = result
    results[86] = result

    # const_eval_87: inputs [67]
    result = utils.constEvalFuncWrapper(
        main_const_eval_87, [inputs[67]], _CONST_EVAL_CACHE[87]
    )
    _CONST_EVAL_CACHE[87] = result
    results[87] = result

    # const_eval_88: inputs [180]
    result = utils.constEvalFuncWrapper(
        main_const_eval_88, [inputs[180]], _CONST_EVAL_CACHE[88]
    )
    _CONST_EVAL_CACHE[88] = result
    results[88] = result

    # const_eval_89: inputs [105]
    result = utils.constEvalFuncWrapper(
        main_const_eval_89, [inputs[105]], _CONST_EVAL_CACHE[89]
    )
    _CONST_EVAL_CACHE[89] = result
    results[89] = result

    # const_eval_90: inputs [93]
    result = utils.constEvalFuncWrapper(
        main_const_eval_90, [inputs[93]], _CONST_EVAL_CACHE[90]
    )
    _CONST_EVAL_CACHE[90] = result
    results[90] = result

    # const_eval_91: inputs [99]
    result = utils.constEvalFuncWrapper(
        main_const_eval_91, [inputs[99]], _CONST_EVAL_CACHE[91]
    )
    _CONST_EVAL_CACHE[91] = result
    results[91] = result

    # const_eval_92: inputs [63]
    result = utils.constEvalFuncWrapper(
        main_const_eval_92, [inputs[63]], _CONST_EVAL_CACHE[92]
    )
    _CONST_EVAL_CACHE[92] = result
    results[92] = result

    # const_eval_93: inputs [109]
    result = utils.constEvalFuncWrapper(
        main_const_eval_93, [inputs[109]], _CONST_EVAL_CACHE[93]
    )
    _CONST_EVAL_CACHE[93] = result
    results[93] = result

    # const_eval_94: inputs [26]
    result = utils.constEvalFuncWrapper(
        main_const_eval_94, [inputs[26]], _CONST_EVAL_CACHE[94]
    )
    _CONST_EVAL_CACHE[94] = result
    results[94] = result

    # const_eval_95: inputs [146]
    result = utils.constEvalFuncWrapper(
        main_const_eval_95, [inputs[146]], _CONST_EVAL_CACHE[95]
    )
    _CONST_EVAL_CACHE[95] = result
    results[95] = result

    # const_eval_96: inputs [200]
    result = utils.constEvalFuncWrapper(
        main_const_eval_96, [inputs[200]], _CONST_EVAL_CACHE[96]
    )
    _CONST_EVAL_CACHE[96] = result
    results[96] = result

    # const_eval_97: inputs [113]
    result = utils.constEvalFuncWrapper(
        main_const_eval_97, [inputs[113]], _CONST_EVAL_CACHE[97]
    )
    _CONST_EVAL_CACHE[97] = result
    results[97] = result

    # const_eval_98: inputs [129]
    result = utils.constEvalFuncWrapper(
        main_const_eval_98, [inputs[129]], _CONST_EVAL_CACHE[98]
    )
    _CONST_EVAL_CACHE[98] = result
    results[98] = result

    # const_eval_99: inputs [47]
    result = utils.constEvalFuncWrapper(
        main_const_eval_99, [inputs[47]], _CONST_EVAL_CACHE[99]
    )
    _CONST_EVAL_CACHE[99] = result
    results[99] = result

    # const_eval_100: inputs [164]
    result = utils.constEvalFuncWrapper(
        main_const_eval_100, [inputs[164]], _CONST_EVAL_CACHE[100]
    )
    _CONST_EVAL_CACHE[100] = result
    results[100] = result

    # const_eval_101: inputs [158]
    result = utils.constEvalFuncWrapper(
        main_const_eval_101, [inputs[158]], _CONST_EVAL_CACHE[101]
    )
    _CONST_EVAL_CACHE[101] = result
    results[101] = result

    # const_eval_102: inputs [33]
    result = utils.constEvalFuncWrapper(
        main_const_eval_102, [inputs[33]], _CONST_EVAL_CACHE[102]
    )
    _CONST_EVAL_CACHE[102] = result
    results[102] = result

    # const_eval_103: inputs [196]
    result = utils.constEvalFuncWrapper(
        main_const_eval_103, [inputs[196]], _CONST_EVAL_CACHE[103]
    )
    _CONST_EVAL_CACHE[103] = result
    results[103] = result

    # const_eval_104: inputs [147]
    result = utils.constEvalFuncWrapper(
        main_const_eval_104, [inputs[147]], _CONST_EVAL_CACHE[104]
    )
    _CONST_EVAL_CACHE[104] = result
    results[104] = result

    # const_eval_105: inputs [134]
    result = utils.constEvalFuncWrapper(
        main_const_eval_105, [inputs[134]], _CONST_EVAL_CACHE[105]
    )
    _CONST_EVAL_CACHE[105] = result
    results[105] = result

    # const_eval_106: inputs [44]
    result = utils.constEvalFuncWrapper(
        main_const_eval_106, [inputs[44]], _CONST_EVAL_CACHE[106]
    )
    _CONST_EVAL_CACHE[106] = result
    results[106] = result

    # const_eval_107: inputs [37]
    result = utils.constEvalFuncWrapper(
        main_const_eval_107, [inputs[37]], _CONST_EVAL_CACHE[107]
    )
    _CONST_EVAL_CACHE[107] = result
    results[107] = result

    # const_eval_108: inputs [162]
    result = utils.constEvalFuncWrapper(
        main_const_eval_108, [inputs[162]], _CONST_EVAL_CACHE[108]
    )
    _CONST_EVAL_CACHE[108] = result
    results[108] = result

    # const_eval_109: inputs [7]
    result = utils.constEvalFuncWrapper(
        main_const_eval_109, [inputs[7]], _CONST_EVAL_CACHE[109]
    )
    _CONST_EVAL_CACHE[109] = result
    results[109] = result

    # const_eval_110: inputs [39]
    result = utils.constEvalFuncWrapper(
        main_const_eval_110, [inputs[39]], _CONST_EVAL_CACHE[110]
    )
    _CONST_EVAL_CACHE[110] = result
    results[110] = result

    # const_eval_111: inputs [41]
    result = utils.constEvalFuncWrapper(
        main_const_eval_111, [inputs[41]], _CONST_EVAL_CACHE[111]
    )
    _CONST_EVAL_CACHE[111] = result
    results[111] = result

    # const_eval_112: inputs [143]
    result = utils.constEvalFuncWrapper(
        main_const_eval_112, [inputs[143]], _CONST_EVAL_CACHE[112]
    )
    _CONST_EVAL_CACHE[112] = result
    results[112] = result

    # const_eval_113: inputs [81]
    result = utils.constEvalFuncWrapper(
        main_const_eval_113, [inputs[81]], _CONST_EVAL_CACHE[113]
    )
    _CONST_EVAL_CACHE[113] = result
    results[113] = result

    # const_eval_114: inputs [121]
    result = utils.constEvalFuncWrapper(
        main_const_eval_114, [inputs[121]], _CONST_EVAL_CACHE[114]
    )
    _CONST_EVAL_CACHE[114] = result
    results[114] = result

    # const_eval_115: inputs [151]
    result = utils.constEvalFuncWrapper(
        main_const_eval_115, [inputs[151]], _CONST_EVAL_CACHE[115]
    )
    _CONST_EVAL_CACHE[115] = result
    results[115] = result

    # const_eval_116: inputs [139]
    result = utils.constEvalFuncWrapper(
        main_const_eval_116, [inputs[139]], _CONST_EVAL_CACHE[116]
    )
    _CONST_EVAL_CACHE[116] = result
    results[116] = result

    # const_eval_117: inputs [11]
    result = utils.constEvalFuncWrapper(
        main_const_eval_117, [inputs[11]], _CONST_EVAL_CACHE[117]
    )
    _CONST_EVAL_CACHE[117] = result
    results[117] = result

    # const_eval_118: inputs [89]
    result = utils.constEvalFuncWrapper(
        main_const_eval_118, [inputs[89]], _CONST_EVAL_CACHE[118]
    )
    _CONST_EVAL_CACHE[118] = result
    results[118] = result

    # const_eval_119: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_119, _CONST_EVAL_CACHE[119]
    )
    _CONST_EVAL_CACHE[119] = result
    results[119] = result

    # const_eval_120: inputs [101]
    result = utils.constEvalFuncWrapper(
        main_const_eval_120, [inputs[101]], _CONST_EVAL_CACHE[120]
    )
    _CONST_EVAL_CACHE[120] = result
    results[120] = result

    # const_eval_121: inputs [56]
    result = utils.constEvalFuncWrapper(
        main_const_eval_121, [inputs[56]], _CONST_EVAL_CACHE[121]
    )
    _CONST_EVAL_CACHE[121] = result
    results[121] = result

    # const_eval_122: inputs [154]
    result = utils.constEvalFuncWrapper(
        main_const_eval_122, [inputs[154]], _CONST_EVAL_CACHE[122]
    )
    _CONST_EVAL_CACHE[122] = result
    results[122] = result

    # const_eval_123: inputs [32]
    result = utils.constEvalFuncWrapper(
        main_const_eval_123, [inputs[32]], _CONST_EVAL_CACHE[123]
    )
    _CONST_EVAL_CACHE[123] = result
    results[123] = result

    # const_eval_124: inputs [38]
    result = utils.constEvalFuncWrapper(
        main_const_eval_124, [inputs[38]], _CONST_EVAL_CACHE[124]
    )
    _CONST_EVAL_CACHE[124] = result
    results[124] = result

    # const_eval_125: inputs [107]
    result = utils.constEvalFuncWrapper(
        main_const_eval_125, [inputs[107]], _CONST_EVAL_CACHE[125]
    )
    _CONST_EVAL_CACHE[125] = result
    results[125] = result

    # const_eval_126: inputs [29]
    result = utils.constEvalFuncWrapper(
        main_const_eval_126, [inputs[29]], _CONST_EVAL_CACHE[126]
    )
    _CONST_EVAL_CACHE[126] = result
    results[126] = result

    # const_eval_127: inputs [87]
    result = utils.constEvalFuncWrapper(
        main_const_eval_127, [inputs[87]], _CONST_EVAL_CACHE[127]
    )
    _CONST_EVAL_CACHE[127] = result
    results[127] = result

    # const_eval_128: no inputs
    result = utils.constEvalFuncWrapperZeroArg(
        main_const_eval_128, _CONST_EVAL_CACHE[128]
    )
    _CONST_EVAL_CACHE[128] = result
    results[128] = result

    # const_eval_129: inputs [95]
    result = utils.constEvalFuncWrapper(
        main_const_eval_129, [inputs[95]], _CONST_EVAL_CACHE[129]
    )
    _CONST_EVAL_CACHE[129] = result
    results[129] = result

    # const_eval_130: inputs [182]
    result = utils.constEvalFuncWrapper(
        main_const_eval_130, [inputs[182]], _CONST_EVAL_CACHE[130]
    )
    _CONST_EVAL_CACHE[130] = result
    results[130] = result

    # const_eval_131: inputs [69]
    result = utils.constEvalFuncWrapper(
        main_const_eval_131, [inputs[69]], _CONST_EVAL_CACHE[131]
    )
    _CONST_EVAL_CACHE[131] = result
    results[131] = result

    # const_eval_132: inputs [51]
    result = utils.constEvalFuncWrapper(
        main_const_eval_132, [inputs[51]], _CONST_EVAL_CACHE[132]
    )
    _CONST_EVAL_CACHE[132] = result
    results[132] = result

    return results
