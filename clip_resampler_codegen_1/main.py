# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import model_pt
import ttnn
import utils

_CONST_EVAL_CACHE = {}


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


def main_const_eval_5(input):
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_8 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_to_device_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_to_layout_8,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_5 = ttnn.repeat(ttnn_reshape_5, ttnn.Shape([1, 257, 1]))
    util_create_list_7 = [ttnn_repeat_5]
    return util_create_list_7


def main_const_eval_6(input):
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_9 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_9,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_to_layout_9,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_6 = ttnn.repeat(ttnn_reshape_6, ttnn.Shape([1, 257, 1]))
    util_create_list_8 = [ttnn_repeat_6]
    return util_create_list_8


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


def main_const_eval_8(input):
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_11 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_11,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_to_layout_11,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_8 = ttnn.repeat(ttnn_reshape_8, ttnn.Shape([1, 257, 1]))
    util_create_list_10 = [ttnn_repeat_8]
    return util_create_list_10


def main_const_eval_9(input):
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_12 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_to_layout_12,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_9 = ttnn.repeat(ttnn_reshape_9, ttnn.Shape([1, 257, 1]))
    util_create_list_11 = [ttnn_repeat_9]
    return util_create_list_11


def main_const_eval_10(input):
    utils_DeviceGetter_get_device_10 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_13 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_to_device_13,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_14 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_to_device_14,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_15 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_to_device_15,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_12 = [ttnn_to_layout_13, ttnn_to_layout_14, ttnn_to_layout_15]
    ttnn_concat_2 = ttnn.concat(
        util_create_list_12,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_13 = [ttnn_concat_2]
    return util_create_list_13


def main_const_eval_11(input):
    utils_DeviceGetter_get_device_11 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_16 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_to_layout_16,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_10 = ttnn.repeat(ttnn_reshape_10, ttnn.Shape([1, 257, 1]))
    util_create_list_14 = [ttnn_repeat_10]
    return util_create_list_14


def main_const_eval_12(input):
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_17 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_to_device_17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_to_layout_17,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_11 = ttnn.repeat(ttnn_reshape_11, ttnn.Shape([1, 257, 1]))
    util_create_list_15 = [ttnn_repeat_11]
    return util_create_list_15


def main_const_eval_13(input):
    utils_DeviceGetter_get_device_13 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_18 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_18,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_19 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_to_device_19,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_20 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_20,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_16 = [ttnn_to_layout_18, ttnn_to_layout_19, ttnn_to_layout_20]
    ttnn_concat_3 = ttnn.concat(
        util_create_list_16,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_17 = [ttnn_concat_3]
    return util_create_list_17


def main_const_eval_14(input):
    utils_DeviceGetter_get_device_14 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_21 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_21,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_to_layout_21,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_12 = ttnn.repeat(ttnn_reshape_12, ttnn.Shape([1, 257, 1]))
    util_create_list_18 = [ttnn_repeat_12]
    return util_create_list_18


def main_const_eval_15(input):
    utils_DeviceGetter_get_device_15 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_22 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_to_device_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_to_layout_22,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_13 = ttnn.repeat(ttnn_reshape_13, ttnn.Shape([1, 257, 1]))
    util_create_list_19 = [ttnn_repeat_13]
    return util_create_list_19


def main_const_eval_16(input):
    utils_DeviceGetter_get_device_16 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_23 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_to_device_23,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_to_layout_23,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_14 = ttnn.repeat(ttnn_reshape_14, ttnn.Shape([1, 257, 1]))
    util_create_list_20 = [ttnn_repeat_14]
    return util_create_list_20


def main_const_eval_17(input):
    utils_DeviceGetter_get_device_17 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_24 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_to_device_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_25 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_to_device_25,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_26 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_26,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_to_layout_24,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_15 = ttnn.repeat(ttnn_reshape_15, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_to_layout_25,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_16 = ttnn.repeat(ttnn_reshape_16, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_to_layout_26,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_17 = ttnn.repeat(ttnn_reshape_17, ttnn.Shape([1, 257, 1]))
    util_create_list_21 = [ttnn_repeat_15, ttnn_repeat_16, ttnn_repeat_17]
    ttnn_concat_4 = ttnn.concat(
        util_create_list_21,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_22 = [ttnn_concat_4]
    return util_create_list_22


def main_const_eval_18(input):
    utils_DeviceGetter_get_device_18 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_27 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_to_device_27,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_28 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_28,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_29 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_29,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_to_layout_27,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_18 = ttnn.repeat(ttnn_reshape_18, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_to_layout_28,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_19 = ttnn.repeat(ttnn_reshape_19, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_to_layout_29,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_20 = ttnn.repeat(ttnn_reshape_20, ttnn.Shape([1, 257, 1]))
    util_create_list_23 = [ttnn_repeat_18, ttnn_repeat_19, ttnn_repeat_20]
    ttnn_concat_5 = ttnn.concat(
        util_create_list_23,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_24 = [ttnn_concat_5]
    return util_create_list_24


def main_const_eval_19(input):
    utils_DeviceGetter_get_device_19 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_30 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_19,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_30,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_to_layout_30,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_21 = ttnn.repeat(ttnn_reshape_21, ttnn.Shape([1, 257, 1]))
    util_create_list_25 = [ttnn_repeat_21]
    return util_create_list_25


def main_const_eval_20(input):
    utils_DeviceGetter_get_device_20 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_31 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_31,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_to_layout_31,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_22 = ttnn.repeat(ttnn_reshape_22, ttnn.Shape([1, 257, 1]))
    util_create_list_26 = [ttnn_repeat_22]
    return util_create_list_26


def main_const_eval_21(input):
    utils_DeviceGetter_get_device_21 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_32 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_to_device_32,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_to_layout_32,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_23 = ttnn.repeat(ttnn_reshape_23, ttnn.Shape([1, 257, 1]))
    util_create_list_27 = [ttnn_repeat_23]
    return util_create_list_27


def main_const_eval_22(input):
    utils_DeviceGetter_get_device_22 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_33 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_to_device_33,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_to_layout_33,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_24 = ttnn.repeat(ttnn_reshape_24, ttnn.Shape([1, 257, 1]))
    util_create_list_28 = [ttnn_repeat_24]
    return util_create_list_28


def main_const_eval_23(input):
    utils_DeviceGetter_get_device_23 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_34 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_23,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_34,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_to_layout_34,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_25 = ttnn.repeat(ttnn_reshape_25, ttnn.Shape([1, 257, 1]))
    util_create_list_29 = [ttnn_repeat_25]
    return util_create_list_29


def main_const_eval_24(input):
    utils_DeviceGetter_get_device_24 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_35 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_to_device_35,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_36 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_36,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_37 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_to_device_37,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_30 = [ttnn_to_layout_35, ttnn_to_layout_36, ttnn_to_layout_37]
    ttnn_concat_6 = ttnn.concat(
        util_create_list_30,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_31 = [ttnn_concat_6]
    return util_create_list_31


def main_const_eval_25(input):
    utils_DeviceGetter_get_device_25 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_38 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_38,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_to_layout_38,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_26 = ttnn.repeat(ttnn_reshape_26, ttnn.Shape([1, 257, 1]))
    util_create_list_32 = [ttnn_repeat_26]
    return util_create_list_32


def main_const_eval_26(input):
    utils_DeviceGetter_get_device_26 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_39 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_40 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_to_device_40,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_41 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_41,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_33 = [ttnn_to_layout_39, ttnn_to_layout_40, ttnn_to_layout_41]
    ttnn_concat_7 = ttnn.concat(
        util_create_list_33,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_34 = [ttnn_concat_7]
    return util_create_list_34


def main_const_eval_27(input):
    utils_DeviceGetter_get_device_27 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_42 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_42,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_43 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_to_device_43,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_44 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_44,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_35 = [ttnn_to_layout_42, ttnn_to_layout_43, ttnn_to_layout_44]
    ttnn_concat_8 = ttnn.concat(
        util_create_list_35,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_36 = [ttnn_concat_8]
    return util_create_list_36


def main_const_eval_28(input):
    utils_DeviceGetter_get_device_28 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_45 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_28,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_to_device_45,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_to_layout_45,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_27 = ttnn.repeat(ttnn_reshape_27, ttnn.Shape([1, 257, 1]))
    util_create_list_37 = [ttnn_repeat_27]
    return util_create_list_37


def main_const_eval_29(input):
    utils_DeviceGetter_get_device_29 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_46 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_to_device_46,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_to_layout_46,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_28 = ttnn.repeat(ttnn_reshape_28, ttnn.Shape([1, 257, 1]))
    util_create_list_38 = [ttnn_repeat_28]
    return util_create_list_38


def main_const_eval_30(input):
    utils_DeviceGetter_get_device_30 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_47 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_47,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_48 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_to_device_48,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_49 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_to_device_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_39 = [ttnn_to_layout_47, ttnn_to_layout_48, ttnn_to_layout_49]
    ttnn_concat_9 = ttnn.concat(
        util_create_list_39,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_40 = [ttnn_concat_9]
    return util_create_list_40


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


def main_const_eval_32(input):
    utils_DeviceGetter_get_device_32 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_53 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_to_device_53,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_to_layout_53,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_29 = ttnn.repeat(ttnn_reshape_32, ttnn.Shape([1, 257, 1]))
    util_create_list_42 = [ttnn_repeat_29]
    return util_create_list_42


def main_const_eval_33(input):
    utils_DeviceGetter_get_device_33 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_54 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_33,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_to_device_54,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_55 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_33,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_to_device_55,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_56 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_33,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_to_device_56,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_to_layout_54,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_30 = ttnn.repeat(ttnn_reshape_33, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_to_layout_55,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_31 = ttnn.repeat(ttnn_reshape_34, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_to_layout_56,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_32 = ttnn.repeat(ttnn_reshape_35, ttnn.Shape([1, 257, 1]))
    util_create_list_43 = [ttnn_repeat_30, ttnn_repeat_31, ttnn_repeat_32]
    ttnn_concat_10 = ttnn.concat(
        util_create_list_43,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_44 = [ttnn_concat_10]
    return util_create_list_44


def main_const_eval_34(input):
    utils_DeviceGetter_get_device_34 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_57 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_34,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_to_device_57,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_to_layout_57,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_33 = ttnn.repeat(ttnn_reshape_36, ttnn.Shape([1, 257, 1]))
    util_create_list_45 = [ttnn_repeat_33]
    return util_create_list_45


def main_const_eval_35(input):
    utils_DeviceGetter_get_device_35 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_58 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_35,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_to_device_58,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_59 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_35,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_to_device_59,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_60 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_35,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_to_device_60,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_46 = [ttnn_to_layout_58, ttnn_to_layout_59, ttnn_to_layout_60]
    ttnn_concat_11 = ttnn.concat(
        util_create_list_46,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_47 = [ttnn_concat_11]
    return util_create_list_47


def main_const_eval_36(input):
    utils_DeviceGetter_get_device_36 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_61 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_36,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_to_device_61,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_to_layout_61,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_34 = ttnn.repeat(ttnn_reshape_37, ttnn.Shape([1, 257, 1]))
    util_create_list_48 = [ttnn_repeat_34]
    return util_create_list_48


def main_const_eval_37(input):
    utils_DeviceGetter_get_device_37 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_62 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_37,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_to_device_62,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_63 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_37,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_to_device_63,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_64 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_37,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_to_device_64,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_49 = [ttnn_to_layout_62, ttnn_to_layout_63, ttnn_to_layout_64]
    ttnn_concat_12 = ttnn.concat(
        util_create_list_49,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_50 = [ttnn_concat_12]
    return util_create_list_50


def main_const_eval_38(input):
    utils_DeviceGetter_get_device_38 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_65 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_38,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_to_device_65,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_66 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_38,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_to_device_66,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_67 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_38,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_to_device_67,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_51 = [ttnn_to_layout_65, ttnn_to_layout_66, ttnn_to_layout_67]
    ttnn_concat_13 = ttnn.concat(
        util_create_list_51,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_52 = [ttnn_concat_13]
    return util_create_list_52


def main_const_eval_39(input):
    utils_DeviceGetter_get_device_39 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_68 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_39,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_to_device_68,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_to_layout_68,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_35 = ttnn.repeat(ttnn_reshape_38, ttnn.Shape([1, 257, 1]))
    util_create_list_53 = [ttnn_repeat_35]
    return util_create_list_53


def main_const_eval_40(input):
    utils_DeviceGetter_get_device_40 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_69 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_40,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_to_device_69,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_39 = ttnn.reshape(
        ttnn_to_layout_69,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_36 = ttnn.repeat(ttnn_reshape_39, ttnn.Shape([1, 257, 1]))
    util_create_list_54 = [ttnn_repeat_36]
    return util_create_list_54


def main_const_eval_41(input):
    utils_DeviceGetter_get_device_41 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_70 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_41,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_to_device_70,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_71 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_41,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_to_device_71,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_72 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_41,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_to_device_72,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_40 = ttnn.reshape(
        ttnn_to_layout_70,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_37 = ttnn.repeat(ttnn_reshape_40, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_to_layout_71,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_38 = ttnn.repeat(ttnn_reshape_41, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_to_layout_72,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_39 = ttnn.repeat(ttnn_reshape_42, ttnn.Shape([1, 257, 1]))
    util_create_list_55 = [ttnn_repeat_37, ttnn_repeat_38, ttnn_repeat_39]
    ttnn_concat_14 = ttnn.concat(
        util_create_list_55,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_56 = [ttnn_concat_14]
    return util_create_list_56


def main_const_eval_42(input):
    utils_DeviceGetter_get_device_42 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_73 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_to_device_73,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_74 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_74 = ttnn.to_layout(
        ttnn_to_device_74,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_75 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_75 = ttnn.to_layout(
        ttnn_to_device_75,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_to_layout_73,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_40 = ttnn.repeat(ttnn_reshape_43, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_to_layout_74,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_41 = ttnn.repeat(ttnn_reshape_44, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_to_layout_75,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_42 = ttnn.repeat(ttnn_reshape_45, ttnn.Shape([1, 257, 1]))
    util_create_list_57 = [ttnn_repeat_40, ttnn_repeat_41, ttnn_repeat_42]
    ttnn_concat_15 = ttnn.concat(
        util_create_list_57,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_58 = [ttnn_concat_15]
    return util_create_list_58


def main_const_eval_43(input):
    utils_DeviceGetter_get_device_43 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_76 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_43,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_to_device_76,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_to_layout_76,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_43 = ttnn.repeat(ttnn_reshape_46, ttnn.Shape([1, 257, 1]))
    util_create_list_59 = [ttnn_repeat_43]
    return util_create_list_59


def main_const_eval_44(input):
    utils_DeviceGetter_get_device_44 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_77 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_44,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_to_device_77,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_78 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_44,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_78 = ttnn.to_layout(
        ttnn_to_device_78,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_79 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_44,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_to_device_79,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_to_layout_77,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_44 = ttnn.repeat(ttnn_reshape_47, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_48 = ttnn.reshape(
        ttnn_to_layout_78,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_45 = ttnn.repeat(ttnn_reshape_48, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_to_layout_79,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_46 = ttnn.repeat(ttnn_reshape_49, ttnn.Shape([1, 257, 1]))
    util_create_list_60 = [ttnn_repeat_44, ttnn_repeat_45, ttnn_repeat_46]
    ttnn_concat_16 = ttnn.concat(
        util_create_list_60,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_61 = [ttnn_concat_16]
    return util_create_list_61


def main_const_eval_45(input):
    utils_DeviceGetter_get_device_45 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_80 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_45,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_to_device_80,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_to_layout_80,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_47 = ttnn.repeat(ttnn_reshape_50, ttnn.Shape([1, 257, 1]))
    util_create_list_62 = [ttnn_repeat_47]
    return util_create_list_62


def main_const_eval_46(input):
    utils_DeviceGetter_get_device_46 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_81 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_46,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_to_device_81,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_82 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_46,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_82 = ttnn.to_layout(
        ttnn_to_device_82,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_83 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_46,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_83 = ttnn.to_layout(
        ttnn_to_device_83,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_51 = ttnn.reshape(
        ttnn_to_layout_81,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_48 = ttnn.repeat(ttnn_reshape_51, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_to_layout_82,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_49 = ttnn.repeat(ttnn_reshape_52, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_53 = ttnn.reshape(
        ttnn_to_layout_83,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_50 = ttnn.repeat(ttnn_reshape_53, ttnn.Shape([1, 257, 1]))
    util_create_list_63 = [ttnn_repeat_48, ttnn_repeat_49, ttnn_repeat_50]
    ttnn_concat_17 = ttnn.concat(
        util_create_list_63,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_64 = [ttnn_concat_17]
    return util_create_list_64


def main_const_eval_47(input):
    utils_DeviceGetter_get_device_47 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_84 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_47,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_84 = ttnn.to_layout(
        ttnn_to_device_84,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_to_layout_84,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_51 = ttnn.repeat(ttnn_reshape_54, ttnn.Shape([1, 257, 1]))
    util_create_list_65 = [ttnn_repeat_51]
    return util_create_list_65


def main_const_eval_48(input):
    utils_DeviceGetter_get_device_48 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_85 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_48,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_85 = ttnn.to_layout(
        ttnn_to_device_85,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_86 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_48,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_86 = ttnn.to_layout(
        ttnn_to_device_86,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_87 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_48,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_87 = ttnn.to_layout(
        ttnn_to_device_87,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_55 = ttnn.reshape(
        ttnn_to_layout_85,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_52 = ttnn.repeat(ttnn_reshape_55, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_to_layout_86,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_53 = ttnn.repeat(ttnn_reshape_56, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_57 = ttnn.reshape(
        ttnn_to_layout_87,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_54 = ttnn.repeat(ttnn_reshape_57, ttnn.Shape([1, 257, 1]))
    util_create_list_66 = [ttnn_repeat_52, ttnn_repeat_53, ttnn_repeat_54]
    ttnn_concat_18 = ttnn.concat(
        util_create_list_66,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_67 = [ttnn_concat_18]
    return util_create_list_67


def main_const_eval_49(input):
    utils_DeviceGetter_get_device_49 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_88 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_49,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_88 = ttnn.to_layout(
        ttnn_to_device_88,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_to_layout_88,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_55 = ttnn.repeat(ttnn_reshape_58, ttnn.Shape([1, 257, 1]))
    util_create_list_68 = [ttnn_repeat_55]
    return util_create_list_68


def main_const_eval_50(input):
    utils_DeviceGetter_get_device_50 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_89 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_50,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_89 = ttnn.to_layout(
        ttnn_to_device_89,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_90 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_50,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_90 = ttnn.to_layout(
        ttnn_to_device_90,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_91 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_50,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_91 = ttnn.to_layout(
        ttnn_to_device_91,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_69 = [ttnn_to_layout_89, ttnn_to_layout_90, ttnn_to_layout_91]
    ttnn_concat_19 = ttnn.concat(
        util_create_list_69,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_70 = [ttnn_concat_19]
    return util_create_list_70


def main_const_eval_51(input):
    utils_DeviceGetter_get_device_51 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_92 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_92 = ttnn.to_layout(
        ttnn_to_device_92,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_93 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_93 = ttnn.to_layout(
        ttnn_to_device_93,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_94 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_94 = ttnn.to_layout(
        ttnn_to_device_94,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_to_layout_92,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_56 = ttnn.repeat(ttnn_reshape_59, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_to_layout_93,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_57 = ttnn.repeat(ttnn_reshape_60, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_to_layout_94,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_58 = ttnn.repeat(ttnn_reshape_61, ttnn.Shape([1, 257, 1]))
    util_create_list_71 = [ttnn_repeat_56, ttnn_repeat_57, ttnn_repeat_58]
    ttnn_concat_20 = ttnn.concat(
        util_create_list_71,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_72 = [ttnn_concat_20]
    return util_create_list_72


def main_const_eval_52(input):
    utils_DeviceGetter_get_device_52 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_95 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_52,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_95 = ttnn.to_layout(
        ttnn_to_device_95,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_59 = ttnn.repeat(ttnn_reshape_62, ttnn.Shape([1, 257, 1]))
    util_create_list_73 = [ttnn_repeat_59]
    return util_create_list_73


def main_const_eval_53(input):
    utils_DeviceGetter_get_device_53 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_96 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_53,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_96 = ttnn.to_layout(
        ttnn_to_device_96,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_to_layout_96,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_60 = ttnn.repeat(ttnn_reshape_63, ttnn.Shape([1, 257, 1]))
    util_create_list_74 = [ttnn_repeat_60]
    return util_create_list_74


def main_const_eval_54(input):
    utils_DeviceGetter_get_device_54 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_97 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_54,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_97 = ttnn.to_layout(
        ttnn_to_device_97,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_to_layout_97,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_61 = ttnn.repeat(ttnn_reshape_64, ttnn.Shape([1, 257, 1]))
    util_create_list_75 = [ttnn_repeat_61]
    return util_create_list_75


def main_const_eval_55(input):
    utils_DeviceGetter_get_device_55 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_98 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_55,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_98 = ttnn.to_layout(
        ttnn_to_device_98,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_to_layout_98,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_62 = ttnn.repeat(ttnn_reshape_65, ttnn.Shape([1, 257, 1]))
    util_create_list_76 = [ttnn_repeat_62]
    return util_create_list_76


def main_const_eval_56(input):
    utils_DeviceGetter_get_device_56 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_99 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_56,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_99 = ttnn.to_layout(
        ttnn_to_device_99,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_to_layout_99,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_63 = ttnn.repeat(ttnn_reshape_66, ttnn.Shape([1, 257, 1]))
    util_create_list_77 = [ttnn_repeat_63]
    return util_create_list_77


def main_const_eval_57(input):
    utils_DeviceGetter_get_device_57 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_100 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_57,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_100 = ttnn.to_layout(
        ttnn_to_device_100,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_to_layout_100,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_64 = ttnn.repeat(ttnn_reshape_67, ttnn.Shape([1, 257, 1]))
    util_create_list_78 = [ttnn_repeat_64]
    return util_create_list_78


def main_const_eval_58(input):
    utils_DeviceGetter_get_device_58 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_101 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_58,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_101 = ttnn.to_layout(
        ttnn_to_device_101,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_102 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_58,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_102 = ttnn.to_layout(
        ttnn_to_device_102,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_103 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_58,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_103 = ttnn.to_layout(
        ttnn_to_device_103,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_to_layout_101,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_65 = ttnn.repeat(ttnn_reshape_68, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_to_layout_102,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_66 = ttnn.repeat(ttnn_reshape_69, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_to_layout_103,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_67 = ttnn.repeat(ttnn_reshape_70, ttnn.Shape([1, 257, 1]))
    util_create_list_79 = [ttnn_repeat_65, ttnn_repeat_66, ttnn_repeat_67]
    ttnn_concat_21 = ttnn.concat(
        util_create_list_79,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_80 = [ttnn_concat_21]
    return util_create_list_80


def main_const_eval_59(input):
    utils_DeviceGetter_get_device_59 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_104 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_59,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_104 = ttnn.to_layout(
        ttnn_to_device_104,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_105 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_59,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_105 = ttnn.to_layout(
        ttnn_to_device_105,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_106 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_59,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_106 = ttnn.to_layout(
        ttnn_to_device_106,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_to_layout_104,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_68 = ttnn.repeat(ttnn_reshape_71, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_to_layout_105,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_69 = ttnn.repeat(ttnn_reshape_72, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_to_layout_106,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_70 = ttnn.repeat(ttnn_reshape_73, ttnn.Shape([1, 257, 1]))
    util_create_list_81 = [ttnn_repeat_68, ttnn_repeat_69, ttnn_repeat_70]
    ttnn_concat_22 = ttnn.concat(
        util_create_list_81,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_82 = [ttnn_concat_22]
    return util_create_list_82


def main_const_eval_60(input):
    utils_DeviceGetter_get_device_60 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_107 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_60,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_107 = ttnn.to_layout(
        ttnn_to_device_107,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_108 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_60,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_108 = ttnn.to_layout(
        ttnn_to_device_108,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_109 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_60,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_109 = ttnn.to_layout(
        ttnn_to_device_109,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_to_layout_107,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_71 = ttnn.repeat(ttnn_reshape_74, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_to_layout_108,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_72 = ttnn.repeat(ttnn_reshape_75, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_to_layout_109,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_73 = ttnn.repeat(ttnn_reshape_76, ttnn.Shape([1, 257, 1]))
    util_create_list_83 = [ttnn_repeat_71, ttnn_repeat_72, ttnn_repeat_73]
    ttnn_concat_23 = ttnn.concat(
        util_create_list_83,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_84 = [ttnn_concat_23]
    return util_create_list_84


def main_const_eval_61(input):
    utils_DeviceGetter_get_device_61 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_110 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_61,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_110 = ttnn.to_layout(
        ttnn_to_device_110,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_111 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_61,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_111 = ttnn.to_layout(
        ttnn_to_device_111,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_112 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_61,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_112 = ttnn.to_layout(
        ttnn_to_device_112,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_77 = ttnn.reshape(
        ttnn_to_layout_110,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_74 = ttnn.repeat(ttnn_reshape_77, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_to_layout_111,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_75 = ttnn.repeat(ttnn_reshape_78, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_to_layout_112,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_76 = ttnn.repeat(ttnn_reshape_79, ttnn.Shape([1, 257, 1]))
    util_create_list_85 = [ttnn_repeat_74, ttnn_repeat_75, ttnn_repeat_76]
    ttnn_concat_24 = ttnn.concat(
        util_create_list_85,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_86 = [ttnn_concat_24]
    return util_create_list_86


def main_const_eval_62(input):
    utils_DeviceGetter_get_device_62 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_113 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_113 = ttnn.to_layout(
        ttnn_to_device_113,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_114 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_114 = ttnn.to_layout(
        ttnn_to_device_114,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_115 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_115 = ttnn.to_layout(
        ttnn_to_device_115,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_87 = [ttnn_to_layout_113, ttnn_to_layout_114, ttnn_to_layout_115]
    ttnn_concat_25 = ttnn.concat(
        util_create_list_87,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_88 = [ttnn_concat_25]
    return util_create_list_88


def main_const_eval_63(input):
    utils_DeviceGetter_get_device_63 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_116 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_63,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_116 = ttnn.to_layout(
        ttnn_to_device_116,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_117 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_63,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_117 = ttnn.to_layout(
        ttnn_to_device_117,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_118 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_63,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_118 = ttnn.to_layout(
        ttnn_to_device_118,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_to_layout_116,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_77 = ttnn.repeat(ttnn_reshape_80, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_to_layout_117,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_78 = ttnn.repeat(ttnn_reshape_81, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_to_layout_118,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_79 = ttnn.repeat(ttnn_reshape_82, ttnn.Shape([1, 257, 1]))
    util_create_list_89 = [ttnn_repeat_77, ttnn_repeat_78, ttnn_repeat_79]
    ttnn_concat_26 = ttnn.concat(
        util_create_list_89,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_90 = [ttnn_concat_26]
    return util_create_list_90


def main_const_eval_64(input):
    utils_DeviceGetter_get_device_64 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_119 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_64,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_119 = ttnn.to_layout(
        ttnn_to_device_119,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_83 = ttnn.reshape(
        ttnn_to_layout_119,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_80 = ttnn.repeat(ttnn_reshape_83, ttnn.Shape([1, 257, 1]))
    util_create_list_91 = [ttnn_repeat_80]
    return util_create_list_91


def main_const_eval_65(input):
    utils_DeviceGetter_get_device_65 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_120 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_65,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_120 = ttnn.to_layout(
        ttnn_to_device_120,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_to_layout_120,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_81 = ttnn.repeat(ttnn_reshape_84, ttnn.Shape([1, 257, 1]))
    util_create_list_92 = [ttnn_repeat_81]
    return util_create_list_92


def main_const_eval_66(input):
    utils_DeviceGetter_get_device_66 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_121 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_66,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_121 = ttnn.to_layout(
        ttnn_to_device_121,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_122 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_66,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_122 = ttnn.to_layout(
        ttnn_to_device_122,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_123 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_66,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_123 = ttnn.to_layout(
        ttnn_to_device_123,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_93 = [ttnn_to_layout_121, ttnn_to_layout_122, ttnn_to_layout_123]
    ttnn_concat_27 = ttnn.concat(
        util_create_list_93,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_94 = [ttnn_concat_27]
    return util_create_list_94


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


def main_const_eval_68(input):
    utils_DeviceGetter_get_device_68 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_124 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_68,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_124 = ttnn.to_layout(
        ttnn_to_device_124,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_125 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_68,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_125 = ttnn.to_layout(
        ttnn_to_device_125,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_126 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_68,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_126 = ttnn.to_layout(
        ttnn_to_device_126,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_96 = [ttnn_to_layout_124, ttnn_to_layout_125, ttnn_to_layout_126]
    ttnn_concat_28 = ttnn.concat(
        util_create_list_96,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_97 = [ttnn_concat_28]
    return util_create_list_97


def main_const_eval_69(input):
    utils_DeviceGetter_get_device_69 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_127 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_69,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_127 = ttnn.to_layout(
        ttnn_to_device_127,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_85 = ttnn.reshape(
        ttnn_to_layout_127,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_82 = ttnn.repeat(ttnn_reshape_85, ttnn.Shape([1, 257, 1]))
    util_create_list_98 = [ttnn_repeat_82]
    return util_create_list_98


def main_const_eval_70(input):
    utils_DeviceGetter_get_device_70 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_128 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_70,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_128 = ttnn.to_layout(
        ttnn_to_device_128,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_to_layout_128,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_83 = ttnn.repeat(ttnn_reshape_86, ttnn.Shape([1, 257, 1]))
    util_create_list_99 = [ttnn_repeat_83]
    return util_create_list_99


def main_const_eval_71(input):
    utils_DeviceGetter_get_device_71 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_129 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_71,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_129 = ttnn.to_layout(
        ttnn_to_device_129,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_130 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_71,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_130 = ttnn.to_layout(
        ttnn_to_device_130,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_131 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_71,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_131 = ttnn.to_layout(
        ttnn_to_device_131,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_100 = [ttnn_to_layout_129, ttnn_to_layout_130, ttnn_to_layout_131]
    ttnn_concat_29 = ttnn.concat(
        util_create_list_100,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_101 = [ttnn_concat_29]
    return util_create_list_101


def main_const_eval_72(input):
    utils_DeviceGetter_get_device_72 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_132 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_72,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_132 = ttnn.to_layout(
        ttnn_to_device_132,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_133 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_72,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_133 = ttnn.to_layout(
        ttnn_to_device_133,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_134 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_72,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_134 = ttnn.to_layout(
        ttnn_to_device_134,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_87 = ttnn.reshape(
        ttnn_to_layout_132,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_84 = ttnn.repeat(ttnn_reshape_87, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_88 = ttnn.reshape(
        ttnn_to_layout_133,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_85 = ttnn.repeat(ttnn_reshape_88, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_to_layout_134,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_86 = ttnn.repeat(ttnn_reshape_89, ttnn.Shape([1, 257, 1]))
    util_create_list_102 = [ttnn_repeat_84, ttnn_repeat_85, ttnn_repeat_86]
    ttnn_concat_30 = ttnn.concat(
        util_create_list_102,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_103 = [ttnn_concat_30]
    return util_create_list_103


def main_const_eval_73(input):
    utils_DeviceGetter_get_device_73 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_135 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_73,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_135 = ttnn.to_layout(
        ttnn_to_device_135,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_136 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_73,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_136 = ttnn.to_layout(
        ttnn_to_device_136,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_137 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_73,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_137 = ttnn.to_layout(
        ttnn_to_device_137,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_90 = ttnn.reshape(
        ttnn_to_layout_135,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_87 = ttnn.repeat(ttnn_reshape_90, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_to_layout_136,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_88 = ttnn.repeat(ttnn_reshape_91, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_92 = ttnn.reshape(
        ttnn_to_layout_137,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_89 = ttnn.repeat(ttnn_reshape_92, ttnn.Shape([1, 257, 1]))
    util_create_list_104 = [ttnn_repeat_87, ttnn_repeat_88, ttnn_repeat_89]
    ttnn_concat_31 = ttnn.concat(
        util_create_list_104,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_105 = [ttnn_concat_31]
    return util_create_list_105


def main_const_eval_74(input):
    utils_DeviceGetter_get_device_74 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_138 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_74,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_138 = ttnn.to_layout(
        ttnn_to_device_138,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_to_layout_138,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_90 = ttnn.repeat(ttnn_reshape_93, ttnn.Shape([1, 257, 1]))
    util_create_list_106 = [ttnn_repeat_90]
    return util_create_list_106


def main_const_eval_75(input):
    utils_DeviceGetter_get_device_75 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_139 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_75,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_139 = ttnn.to_layout(
        ttnn_to_device_139,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_to_layout_139,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_91 = ttnn.repeat(ttnn_reshape_94, ttnn.Shape([1, 257, 1]))
    util_create_list_107 = [ttnn_repeat_91]
    return util_create_list_107


def main_const_eval_76(input):
    utils_DeviceGetter_get_device_76 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_140 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_76,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_140 = ttnn.to_layout(
        ttnn_to_device_140,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_141 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_76,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_141 = ttnn.to_layout(
        ttnn_to_device_141,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_142 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_76,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_142 = ttnn.to_layout(
        ttnn_to_device_142,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_108 = [ttnn_to_layout_140, ttnn_to_layout_141, ttnn_to_layout_142]
    ttnn_concat_32 = ttnn.concat(
        util_create_list_108,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_109 = [ttnn_concat_32]
    return util_create_list_109


def main_const_eval_77(input):
    utils_DeviceGetter_get_device_77 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_143 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_77,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_143 = ttnn.to_layout(
        ttnn_to_device_143,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_144 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_77,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_144 = ttnn.to_layout(
        ttnn_to_device_144,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_145 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_77,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_145 = ttnn.to_layout(
        ttnn_to_device_145,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_110 = [ttnn_to_layout_143, ttnn_to_layout_144, ttnn_to_layout_145]
    ttnn_concat_33 = ttnn.concat(
        util_create_list_110,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_111 = [ttnn_concat_33]
    return util_create_list_111


def main_const_eval_78(input):
    utils_DeviceGetter_get_device_78 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_146 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_78,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_146 = ttnn.to_layout(
        ttnn_to_device_146,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_147 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_78,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_147 = ttnn.to_layout(
        ttnn_to_device_147,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_148 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_78,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_148 = ttnn.to_layout(
        ttnn_to_device_148,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_to_layout_146,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_92 = ttnn.repeat(ttnn_reshape_95, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_to_layout_147,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_93 = ttnn.repeat(ttnn_reshape_96, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_to_layout_148,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_94 = ttnn.repeat(ttnn_reshape_97, ttnn.Shape([1, 257, 1]))
    util_create_list_112 = [ttnn_repeat_92, ttnn_repeat_93, ttnn_repeat_94]
    ttnn_concat_34 = ttnn.concat(
        util_create_list_112,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_113 = [ttnn_concat_34]
    return util_create_list_113


def main_const_eval_79(input):
    utils_DeviceGetter_get_device_79 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_149 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_79,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_149 = ttnn.to_layout(
        ttnn_to_device_149,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_to_layout_149,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_95 = ttnn.repeat(ttnn_reshape_98, ttnn.Shape([1, 257, 1]))
    util_create_list_114 = [ttnn_repeat_95]
    return util_create_list_114


def main_const_eval_80(input):
    utils_DeviceGetter_get_device_80 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_150 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_80,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_150 = ttnn.to_layout(
        ttnn_to_device_150,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_to_layout_150,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_96 = ttnn.repeat(ttnn_reshape_99, ttnn.Shape([1, 257, 1]))
    util_create_list_115 = [ttnn_repeat_96]
    return util_create_list_115


def main_const_eval_81(input):
    utils_DeviceGetter_get_device_81 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_151 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_81,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_151 = ttnn.to_layout(
        ttnn_to_device_151,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_152 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_81,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_152 = ttnn.to_layout(
        ttnn_to_device_152,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_153 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_81,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_153 = ttnn.to_layout(
        ttnn_to_device_153,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_to_layout_151,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_97 = ttnn.repeat(ttnn_reshape_100, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_to_layout_152,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_98 = ttnn.repeat(ttnn_reshape_101, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_102 = ttnn.reshape(
        ttnn_to_layout_153,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_99 = ttnn.repeat(ttnn_reshape_102, ttnn.Shape([1, 257, 1]))
    util_create_list_116 = [ttnn_repeat_97, ttnn_repeat_98, ttnn_repeat_99]
    ttnn_concat_35 = ttnn.concat(
        util_create_list_116,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_117 = [ttnn_concat_35]
    return util_create_list_117


def main_const_eval_82(input):
    utils_DeviceGetter_get_device_82 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_154 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_82,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_154 = ttnn.to_layout(
        ttnn_to_device_154,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_103 = ttnn.reshape(
        ttnn_to_layout_154,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_100 = ttnn.repeat(ttnn_reshape_103, ttnn.Shape([1, 257, 1]))
    util_create_list_118 = [ttnn_repeat_100]
    return util_create_list_118


def main_const_eval_83(input):
    utils_DeviceGetter_get_device_83 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_155 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_83,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_155 = ttnn.to_layout(
        ttnn_to_device_155,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_104 = ttnn.reshape(
        ttnn_to_layout_155,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_101 = ttnn.repeat(ttnn_reshape_104, ttnn.Shape([1, 257, 1]))
    util_create_list_119 = [ttnn_repeat_101]
    return util_create_list_119


def main_const_eval_84(input):
    utils_DeviceGetter_get_device_84 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_156 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_84,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_156 = ttnn.to_layout(
        ttnn_to_device_156,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_105 = ttnn.reshape(
        ttnn_to_layout_156,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_102 = ttnn.repeat(ttnn_reshape_105, ttnn.Shape([1, 257, 1]))
    util_create_list_120 = [ttnn_repeat_102]
    return util_create_list_120


def main_const_eval_85(input):
    utils_DeviceGetter_get_device_85 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_157 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_85,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_157 = ttnn.to_layout(
        ttnn_to_device_157,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_158 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_85,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_158 = ttnn.to_layout(
        ttnn_to_device_158,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_159 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_85,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_159 = ttnn.to_layout(
        ttnn_to_device_159,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_to_layout_157,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_103 = ttnn.repeat(ttnn_reshape_106, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_107 = ttnn.reshape(
        ttnn_to_layout_158,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_104 = ttnn.repeat(ttnn_reshape_107, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_108 = ttnn.reshape(
        ttnn_to_layout_159,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_105 = ttnn.repeat(ttnn_reshape_108, ttnn.Shape([1, 257, 1]))
    util_create_list_121 = [ttnn_repeat_103, ttnn_repeat_104, ttnn_repeat_105]
    ttnn_concat_36 = ttnn.concat(
        util_create_list_121,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_122 = [ttnn_concat_36]
    return util_create_list_122


def main_const_eval_86(input):
    utils_DeviceGetter_get_device_86 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_160 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_86,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_160 = ttnn.to_layout(
        ttnn_to_device_160,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_109 = ttnn.reshape(
        ttnn_to_layout_160,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_106 = ttnn.repeat(ttnn_reshape_109, ttnn.Shape([1, 257, 1]))
    util_create_list_123 = [ttnn_repeat_106]
    return util_create_list_123


def main_const_eval_87(input):
    utils_DeviceGetter_get_device_87 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_161 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_87,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_161 = ttnn.to_layout(
        ttnn_to_device_161,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_162 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_87,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_162 = ttnn.to_layout(
        ttnn_to_device_162,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_163 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_87,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_163 = ttnn.to_layout(
        ttnn_to_device_163,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_124 = [ttnn_to_layout_161, ttnn_to_layout_162, ttnn_to_layout_163]
    ttnn_concat_37 = ttnn.concat(
        util_create_list_124,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_125 = [ttnn_concat_37]
    return util_create_list_125


def main_const_eval_88(input):
    utils_DeviceGetter_get_device_88 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_164 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_88,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_164 = ttnn.to_layout(
        ttnn_to_device_164,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_165 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_88,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_165 = ttnn.to_layout(
        ttnn_to_device_165,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_166 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_88,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_166 = ttnn.to_layout(
        ttnn_to_device_166,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_126 = [ttnn_to_layout_164, ttnn_to_layout_165, ttnn_to_layout_166]
    ttnn_concat_38 = ttnn.concat(
        util_create_list_126,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_127 = [ttnn_concat_38]
    return util_create_list_127


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


def main_const_eval_90(input):
    utils_DeviceGetter_get_device_90 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_169 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_90,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_169 = ttnn.to_layout(
        ttnn_to_device_169,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_170 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_90,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_170 = ttnn.to_layout(
        ttnn_to_device_170,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_171 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_90,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_171 = ttnn.to_layout(
        ttnn_to_device_171,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_129 = [ttnn_to_layout_169, ttnn_to_layout_170, ttnn_to_layout_171]
    ttnn_concat_39 = ttnn.concat(
        util_create_list_129,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_130 = [ttnn_concat_39]
    return util_create_list_130


def main_const_eval_91(input):
    utils_DeviceGetter_get_device_91 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_172 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_91,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_172 = ttnn.to_layout(
        ttnn_to_device_172,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_173 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_91,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_173 = ttnn.to_layout(
        ttnn_to_device_173,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_174 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_91,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_174 = ttnn.to_layout(
        ttnn_to_device_174,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_110 = ttnn.reshape(
        ttnn_to_layout_172,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_107 = ttnn.repeat(ttnn_reshape_110, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_111 = ttnn.reshape(
        ttnn_to_layout_173,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_108 = ttnn.repeat(ttnn_reshape_111, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_112 = ttnn.reshape(
        ttnn_to_layout_174,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_109 = ttnn.repeat(ttnn_reshape_112, ttnn.Shape([1, 257, 1]))
    util_create_list_131 = [ttnn_repeat_107, ttnn_repeat_108, ttnn_repeat_109]
    ttnn_concat_40 = ttnn.concat(
        util_create_list_131,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_132 = [ttnn_concat_40]
    return util_create_list_132


def main_const_eval_92(input):
    utils_DeviceGetter_get_device_92 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_175 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_92,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_175 = ttnn.to_layout(
        ttnn_to_device_175,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_113 = ttnn.reshape(
        ttnn_to_layout_175,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_110 = ttnn.repeat(ttnn_reshape_113, ttnn.Shape([1, 257, 1]))
    util_create_list_133 = [ttnn_repeat_110]
    return util_create_list_133


def main_const_eval_93(input):
    utils_DeviceGetter_get_device_93 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_176 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_93,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_176 = ttnn.to_layout(
        ttnn_to_device_176,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_to_layout_176,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_111 = ttnn.repeat(ttnn_reshape_114, ttnn.Shape([1, 257, 1]))
    util_create_list_134 = [ttnn_repeat_111]
    return util_create_list_134


def main_const_eval_94(input):
    utils_DeviceGetter_get_device_94 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_177 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_94,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_177 = ttnn.to_layout(
        ttnn_to_device_177,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_115 = ttnn.reshape(
        ttnn_to_layout_177,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_112 = ttnn.repeat(ttnn_reshape_115, ttnn.Shape([1, 257, 1]))
    util_create_list_135 = [ttnn_repeat_112]
    return util_create_list_135


def main_const_eval_95(input):
    utils_DeviceGetter_get_device_95 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_178 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_95,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_178 = ttnn.to_layout(
        ttnn_to_device_178,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_to_layout_178,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_113 = ttnn.repeat(ttnn_reshape_116, ttnn.Shape([1, 257, 1]))
    util_create_list_136 = [ttnn_repeat_113]
    return util_create_list_136


def main_const_eval_96(input):
    utils_DeviceGetter_get_device_96 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_179 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_96,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_179 = ttnn.to_layout(
        ttnn_to_device_179,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_to_layout_179,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_114 = ttnn.repeat(ttnn_reshape_117, ttnn.Shape([1, 257, 1]))
    util_create_list_137 = [ttnn_repeat_114]
    return util_create_list_137


def main_const_eval_97(input):
    utils_DeviceGetter_get_device_97 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_180 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_97,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_180 = ttnn.to_layout(
        ttnn_to_device_180,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_181 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_97,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_181 = ttnn.to_layout(
        ttnn_to_device_181,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_182 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_97,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_182 = ttnn.to_layout(
        ttnn_to_device_182,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_138 = [ttnn_to_layout_180, ttnn_to_layout_181, ttnn_to_layout_182]
    ttnn_concat_41 = ttnn.concat(
        util_create_list_138,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_139 = [ttnn_concat_41]
    return util_create_list_139


def main_const_eval_98(input):
    utils_DeviceGetter_get_device_98 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_183 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_98,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_183 = ttnn.to_layout(
        ttnn_to_device_183,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_to_layout_183,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_115 = ttnn.repeat(ttnn_reshape_118, ttnn.Shape([1, 257, 1]))
    util_create_list_140 = [ttnn_repeat_115]
    return util_create_list_140


def main_const_eval_99(input):
    utils_DeviceGetter_get_device_99 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_184 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_99,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_184 = ttnn.to_layout(
        ttnn_to_device_184,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_to_layout_184,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_116 = ttnn.repeat(ttnn_reshape_119, ttnn.Shape([1, 257, 1]))
    util_create_list_141 = [ttnn_repeat_116]
    return util_create_list_141


def main_const_eval_100(input):
    utils_DeviceGetter_get_device_100 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_185 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_100,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_185 = ttnn.to_layout(
        ttnn_to_device_185,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_186 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_100,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_186 = ttnn.to_layout(
        ttnn_to_device_186,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_187 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_100,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_187 = ttnn.to_layout(
        ttnn_to_device_187,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_to_layout_185,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_117 = ttnn.repeat(ttnn_reshape_120, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_to_layout_186,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_118 = ttnn.repeat(ttnn_reshape_121, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_to_layout_187,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_119 = ttnn.repeat(ttnn_reshape_122, ttnn.Shape([1, 257, 1]))
    util_create_list_142 = [ttnn_repeat_117, ttnn_repeat_118, ttnn_repeat_119]
    ttnn_concat_42 = ttnn.concat(
        util_create_list_142,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_143 = [ttnn_concat_42]
    return util_create_list_143


def main_const_eval_101(input):
    utils_DeviceGetter_get_device_101 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_188 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_101,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_188 = ttnn.to_layout(
        ttnn_to_device_188,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_to_layout_188,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_120 = ttnn.repeat(ttnn_reshape_123, ttnn.Shape([1, 257, 1]))
    util_create_list_144 = [ttnn_repeat_120]
    return util_create_list_144


def main_const_eval_102(input):
    utils_DeviceGetter_get_device_102 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_189 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_102,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_189 = ttnn.to_layout(
        ttnn_to_device_189,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_190 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_102,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_190 = ttnn.to_layout(
        ttnn_to_device_190,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_191 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_102,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_191 = ttnn.to_layout(
        ttnn_to_device_191,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_124 = ttnn.reshape(
        ttnn_to_layout_189,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_121 = ttnn.repeat(ttnn_reshape_124, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_125 = ttnn.reshape(
        ttnn_to_layout_190,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_122 = ttnn.repeat(ttnn_reshape_125, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_126 = ttnn.reshape(
        ttnn_to_layout_191,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_123 = ttnn.repeat(ttnn_reshape_126, ttnn.Shape([1, 257, 1]))
    util_create_list_145 = [ttnn_repeat_121, ttnn_repeat_122, ttnn_repeat_123]
    ttnn_concat_43 = ttnn.concat(
        util_create_list_145,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_146 = [ttnn_concat_43]
    return util_create_list_146


def main_const_eval_103(input):
    utils_DeviceGetter_get_device_103 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_192 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_103,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_192 = ttnn.to_layout(
        ttnn_to_device_192,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_193 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_103,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_193 = ttnn.to_layout(
        ttnn_to_device_193,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_194 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_103,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_194 = ttnn.to_layout(
        ttnn_to_device_194,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_147 = [ttnn_to_layout_192, ttnn_to_layout_193, ttnn_to_layout_194]
    ttnn_concat_44 = ttnn.concat(
        util_create_list_147,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_148 = [ttnn_concat_44]
    return util_create_list_148


def main_const_eval_104(input):
    utils_DeviceGetter_get_device_104 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_195 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_104,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_195 = ttnn.to_layout(
        ttnn_to_device_195,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_127 = ttnn.reshape(
        ttnn_to_layout_195,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_124 = ttnn.repeat(ttnn_reshape_127, ttnn.Shape([1, 257, 1]))
    util_create_list_149 = [ttnn_repeat_124]
    return util_create_list_149


def main_const_eval_105(input):
    utils_DeviceGetter_get_device_105 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_196 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_105,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_196 = ttnn.to_layout(
        ttnn_to_device_196,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_128 = ttnn.reshape(
        ttnn_to_layout_196,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_125 = ttnn.repeat(ttnn_reshape_128, ttnn.Shape([1, 257, 1]))
    util_create_list_150 = [ttnn_repeat_125]
    return util_create_list_150


def main_const_eval_106(input):
    utils_DeviceGetter_get_device_106 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_197 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_106,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_197 = ttnn.to_layout(
        ttnn_to_device_197,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_129 = ttnn.reshape(
        ttnn_to_layout_197,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_126 = ttnn.repeat(ttnn_reshape_129, ttnn.Shape([1, 257, 1]))
    util_create_list_151 = [ttnn_repeat_126]
    return util_create_list_151


def main_const_eval_107(input):
    utils_DeviceGetter_get_device_107 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_198 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_107,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_198 = ttnn.to_layout(
        ttnn_to_device_198,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_130 = ttnn.reshape(
        ttnn_to_layout_198,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_127 = ttnn.repeat(ttnn_reshape_130, ttnn.Shape([1, 257, 1]))
    util_create_list_152 = [ttnn_repeat_127]
    return util_create_list_152


def main_const_eval_108(input):
    utils_DeviceGetter_get_device_108 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_199 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_108,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_199 = ttnn.to_layout(
        ttnn_to_device_199,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_131 = ttnn.reshape(
        ttnn_to_layout_199,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_128 = ttnn.repeat(ttnn_reshape_131, ttnn.Shape([1, 257, 1]))
    util_create_list_153 = [ttnn_repeat_128]
    return util_create_list_153


def main_const_eval_109(input):
    utils_DeviceGetter_get_device_109 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_200 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_109,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_200 = ttnn.to_layout(
        ttnn_to_device_200,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_132 = ttnn.reshape(
        ttnn_to_layout_200,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_129 = ttnn.repeat(ttnn_reshape_132, ttnn.Shape([1, 257, 1]))
    util_create_list_154 = [ttnn_repeat_129]
    return util_create_list_154


def main_const_eval_110(input):
    utils_DeviceGetter_get_device_110 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_201 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_110,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_201 = ttnn.to_layout(
        ttnn_to_device_201,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_133 = ttnn.reshape(
        ttnn_to_layout_201,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_130 = ttnn.repeat(ttnn_reshape_133, ttnn.Shape([1, 257, 1]))
    util_create_list_155 = [ttnn_repeat_130]
    return util_create_list_155


def main_const_eval_111(input):
    utils_DeviceGetter_get_device_111 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_202 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_111,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_202 = ttnn.to_layout(
        ttnn_to_device_202,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_134 = ttnn.reshape(
        ttnn_to_layout_202,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_131 = ttnn.repeat(ttnn_reshape_134, ttnn.Shape([1, 257, 1]))
    util_create_list_156 = [ttnn_repeat_131]
    return util_create_list_156


def main_const_eval_112(input):
    utils_DeviceGetter_get_device_112 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_203 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_112,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_203 = ttnn.to_layout(
        ttnn_to_device_203,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_204 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_112,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_204 = ttnn.to_layout(
        ttnn_to_device_204,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_205 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_112,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_205 = ttnn.to_layout(
        ttnn_to_device_205,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_to_layout_203,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_132 = ttnn.repeat(ttnn_reshape_135, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_136 = ttnn.reshape(
        ttnn_to_layout_204,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_133 = ttnn.repeat(ttnn_reshape_136, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_137 = ttnn.reshape(
        ttnn_to_layout_205,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_134 = ttnn.repeat(ttnn_reshape_137, ttnn.Shape([1, 257, 1]))
    util_create_list_157 = [ttnn_repeat_132, ttnn_repeat_133, ttnn_repeat_134]
    ttnn_concat_45 = ttnn.concat(
        util_create_list_157,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_158 = [ttnn_concat_45]
    return util_create_list_158


def main_const_eval_113(input):
    utils_DeviceGetter_get_device_113 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_206 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_113,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_206 = ttnn.to_layout(
        ttnn_to_device_206,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_207 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_113,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_207 = ttnn.to_layout(
        ttnn_to_device_207,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_208 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_113,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_208 = ttnn.to_layout(
        ttnn_to_device_208,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_159 = [ttnn_to_layout_206, ttnn_to_layout_207, ttnn_to_layout_208]
    ttnn_concat_46 = ttnn.concat(
        util_create_list_159,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_160 = [ttnn_concat_46]
    return util_create_list_160


def main_const_eval_114(input):
    utils_DeviceGetter_get_device_114 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_209 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_114,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_209 = ttnn.to_layout(
        ttnn_to_device_209,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_138 = ttnn.reshape(
        ttnn_to_layout_209,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_135 = ttnn.repeat(ttnn_reshape_138, ttnn.Shape([1, 257, 1]))
    util_create_list_161 = [ttnn_repeat_135]
    return util_create_list_161


def main_const_eval_115(input):
    utils_DeviceGetter_get_device_115 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_210 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_115,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_210 = ttnn.to_layout(
        ttnn_to_device_210,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_139 = ttnn.reshape(
        ttnn_to_layout_210,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_136 = ttnn.repeat(ttnn_reshape_139, ttnn.Shape([1, 257, 1]))
    util_create_list_162 = [ttnn_repeat_136]
    return util_create_list_162


def main_const_eval_116(input):
    utils_DeviceGetter_get_device_116 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_211 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_116,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_211 = ttnn.to_layout(
        ttnn_to_device_211,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_to_layout_211,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_137 = ttnn.repeat(ttnn_reshape_140, ttnn.Shape([1, 257, 1]))
    util_create_list_163 = [ttnn_repeat_137]
    return util_create_list_163


def main_const_eval_117(input):
    utils_DeviceGetter_get_device_117 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_212 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_117,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_212 = ttnn.to_layout(
        ttnn_to_device_212,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_213 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_117,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_213 = ttnn.to_layout(
        ttnn_to_device_213,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_214 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_117,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_214 = ttnn.to_layout(
        ttnn_to_device_214,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_to_layout_212,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_138 = ttnn.repeat(ttnn_reshape_141, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_to_layout_213,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_139 = ttnn.repeat(ttnn_reshape_142, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_to_layout_214,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_140 = ttnn.repeat(ttnn_reshape_143, ttnn.Shape([1, 257, 1]))
    util_create_list_164 = [ttnn_repeat_138, ttnn_repeat_139, ttnn_repeat_140]
    ttnn_concat_47 = ttnn.concat(
        util_create_list_164,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_165 = [ttnn_concat_47]
    return util_create_list_165


def main_const_eval_118(input):
    utils_DeviceGetter_get_device_118 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_215 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_118,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_215 = ttnn.to_layout(
        ttnn_to_device_215,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_to_layout_215,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_141 = ttnn.repeat(ttnn_reshape_144, ttnn.Shape([1, 257, 1]))
    util_create_list_166 = [ttnn_repeat_141]
    return util_create_list_166


def main_const_eval_119(input):
    utils_DeviceGetter_get_device_119 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_216 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_119,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_216 = ttnn.to_layout(
        ttnn_to_device_216,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_217 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_119,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_217 = ttnn.to_layout(
        ttnn_to_device_217,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_218 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_119,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_218 = ttnn.to_layout(
        ttnn_to_device_218,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_to_layout_216,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_142 = ttnn.repeat(ttnn_reshape_145, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_to_layout_217,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_143 = ttnn.repeat(ttnn_reshape_146, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_to_layout_218,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_144 = ttnn.repeat(ttnn_reshape_147, ttnn.Shape([1, 257, 1]))
    util_create_list_167 = [ttnn_repeat_142, ttnn_repeat_143, ttnn_repeat_144]
    ttnn_concat_48 = ttnn.concat(
        util_create_list_167,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_168 = [ttnn_concat_48]
    return util_create_list_168


def main_const_eval_120(input):
    utils_DeviceGetter_get_device_120 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_219 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_120,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_219 = ttnn.to_layout(
        ttnn_to_device_219,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_220 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_120,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_220 = ttnn.to_layout(
        ttnn_to_device_220,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_221 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_120,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_221 = ttnn.to_layout(
        ttnn_to_device_221,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_169 = [ttnn_to_layout_219, ttnn_to_layout_220, ttnn_to_layout_221]
    ttnn_concat_49 = ttnn.concat(
        util_create_list_169,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_170 = [ttnn_concat_49]
    return util_create_list_170


def main_const_eval_121(input):
    utils_DeviceGetter_get_device_121 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_222 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_121,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_222 = ttnn.to_layout(
        ttnn_to_device_222,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_148 = ttnn.reshape(
        ttnn_to_layout_222,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_145 = ttnn.repeat(ttnn_reshape_148, ttnn.Shape([1, 257, 1]))
    util_create_list_171 = [ttnn_repeat_145]
    return util_create_list_171


def main_const_eval_122(input):
    utils_DeviceGetter_get_device_122 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_223 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_122,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_223 = ttnn.to_layout(
        ttnn_to_device_223,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_149 = ttnn.reshape(
        ttnn_to_layout_223,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_146 = ttnn.repeat(ttnn_reshape_149, ttnn.Shape([1, 257, 1]))
    util_create_list_172 = [ttnn_repeat_146]
    return util_create_list_172


def main_const_eval_123(input):
    utils_DeviceGetter_get_device_123 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_224 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_123,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_224 = ttnn.to_layout(
        ttnn_to_device_224,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_to_layout_224,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_147 = ttnn.repeat(ttnn_reshape_150, ttnn.Shape([1, 257, 1]))
    util_create_list_173 = [ttnn_repeat_147]
    return util_create_list_173


def main_const_eval_124(input):
    utils_DeviceGetter_get_device_124 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_225 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_124,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_225 = ttnn.to_layout(
        ttnn_to_device_225,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_to_layout_225,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_148 = ttnn.repeat(ttnn_reshape_151, ttnn.Shape([1, 257, 1]))
    util_create_list_174 = [ttnn_repeat_148]
    return util_create_list_174


def main_const_eval_125(input):
    utils_DeviceGetter_get_device_125 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_226 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_125,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_226 = ttnn.to_layout(
        ttnn_to_device_226,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_to_layout_226,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_149 = ttnn.repeat(ttnn_reshape_152, ttnn.Shape([1, 257, 1]))
    util_create_list_175 = [ttnn_repeat_149]
    return util_create_list_175


def main_const_eval_126(input):
    utils_DeviceGetter_get_device_126 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_227 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_126,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_227 = ttnn.to_layout(
        ttnn_to_device_227,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_to_layout_227,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_150 = ttnn.repeat(ttnn_reshape_153, ttnn.Shape([1, 257, 1]))
    util_create_list_176 = [ttnn_repeat_150]
    return util_create_list_176


def main_const_eval_127(input):
    utils_DeviceGetter_get_device_127 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_228 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_127,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_228 = ttnn.to_layout(
        ttnn_to_device_228,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_to_layout_228,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_151 = ttnn.repeat(ttnn_reshape_154, ttnn.Shape([1, 257, 1]))
    util_create_list_177 = [ttnn_repeat_151]
    return util_create_list_177


def main_const_eval_128(input):
    utils_DeviceGetter_get_device_128 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_229 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_128,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_229 = ttnn.to_layout(
        ttnn_to_device_229,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_230 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_128,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_230 = ttnn.to_layout(
        ttnn_to_device_230,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_231 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_128,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_231 = ttnn.to_layout(
        ttnn_to_device_231,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_178 = [ttnn_to_layout_229, ttnn_to_layout_230, ttnn_to_layout_231]
    ttnn_concat_50 = ttnn.concat(
        util_create_list_178,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_179 = [ttnn_concat_50]
    return util_create_list_179


def main_const_eval_129(input):
    utils_DeviceGetter_get_device_129 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_232 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_129,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_232 = ttnn.to_layout(
        ttnn_to_device_232,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_233 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_129,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_233 = ttnn.to_layout(
        ttnn_to_device_233,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_234 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_129,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_234 = ttnn.to_layout(
        ttnn_to_device_234,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_180 = [ttnn_to_layout_232, ttnn_to_layout_233, ttnn_to_layout_234]
    ttnn_concat_51 = ttnn.concat(
        util_create_list_180,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_181 = [ttnn_concat_51]
    return util_create_list_181


def main_const_eval_130(input):
    utils_DeviceGetter_get_device_130 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_235 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_130,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_235 = ttnn.to_layout(
        ttnn_to_device_235,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_to_layout_235,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_152 = ttnn.repeat(ttnn_reshape_155, ttnn.Shape([1, 257, 1]))
    util_create_list_182 = [ttnn_repeat_152]
    return util_create_list_182


def main_const_eval_131(input):
    utils_DeviceGetter_get_device_131 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_236 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_131,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_236 = ttnn.to_layout(
        ttnn_to_device_236,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_to_layout_236,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_153 = ttnn.repeat(ttnn_reshape_156, ttnn.Shape([1, 257, 1]))
    util_create_list_183 = [ttnn_repeat_153]
    return util_create_list_183


def main_const_eval_132(input):
    utils_DeviceGetter_get_device_132 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_237 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_132,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_237 = ttnn.to_layout(
        ttnn_to_device_237,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_238 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_132,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_238 = ttnn.to_layout(
        ttnn_to_device_238,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_239 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_132,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_239 = ttnn.to_layout(
        ttnn_to_device_239,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_to_layout_237,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_154 = ttnn.repeat(ttnn_reshape_157, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_to_layout_238,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_155 = ttnn.repeat(ttnn_reshape_158, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_159 = ttnn.reshape(
        ttnn_to_layout_239,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_156 = ttnn.repeat(ttnn_reshape_159, ttnn.Shape([1, 257, 1]))
    util_create_list_184 = [ttnn_repeat_154, ttnn_repeat_155, ttnn_repeat_156]
    ttnn_concat_52 = ttnn.concat(
        util_create_list_184,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_185 = [ttnn_concat_52]
    return util_create_list_185


def main_const_eval_133(input):
    utils_DeviceGetter_get_device_133 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_240 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_133,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_240 = ttnn.to_layout(
        ttnn_to_device_240,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_to_layout_240,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_157 = ttnn.repeat(ttnn_reshape_160, ttnn.Shape([1, 257, 1]))
    util_create_list_186 = [ttnn_repeat_157]
    return util_create_list_186


def main_const_eval_134(input):
    utils_DeviceGetter_get_device_134 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_241 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_134,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_241 = ttnn.to_layout(
        ttnn_to_device_241,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_242 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_134,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_242 = ttnn.to_layout(
        ttnn_to_device_242,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_243 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_134,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_243 = ttnn.to_layout(
        ttnn_to_device_243,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_to_layout_241,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_158 = ttnn.repeat(ttnn_reshape_161, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_to_layout_242,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_159 = ttnn.repeat(ttnn_reshape_162, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_163 = ttnn.reshape(
        ttnn_to_layout_243,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_160 = ttnn.repeat(ttnn_reshape_163, ttnn.Shape([1, 257, 1]))
    util_create_list_187 = [ttnn_repeat_158, ttnn_repeat_159, ttnn_repeat_160]
    ttnn_concat_53 = ttnn.concat(
        util_create_list_187,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_188 = [ttnn_concat_53]
    return util_create_list_188


def main_const_eval_135(input):
    utils_DeviceGetter_get_device_135 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_244 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_135,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_244 = ttnn.to_layout(
        ttnn_to_device_244,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_245 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_135,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_245 = ttnn.to_layout(
        ttnn_to_device_245,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_246 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_135,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_246 = ttnn.to_layout(
        ttnn_to_device_246,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_to_layout_244,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_161 = ttnn.repeat(ttnn_reshape_164, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_165 = ttnn.reshape(
        ttnn_to_layout_245,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_162 = ttnn.repeat(ttnn_reshape_165, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_166 = ttnn.reshape(
        ttnn_to_layout_246,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_163 = ttnn.repeat(ttnn_reshape_166, ttnn.Shape([1, 257, 1]))
    util_create_list_189 = [ttnn_repeat_161, ttnn_repeat_162, ttnn_repeat_163]
    ttnn_concat_54 = ttnn.concat(
        util_create_list_189,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_190 = [ttnn_concat_54]
    return util_create_list_190


def main_const_eval_136(input):
    utils_DeviceGetter_get_device_136 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_247 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_136,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_247 = ttnn.to_layout(
        ttnn_to_device_247,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_167 = ttnn.reshape(
        ttnn_to_layout_247,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_164 = ttnn.repeat(ttnn_reshape_167, ttnn.Shape([1, 257, 1]))
    util_create_list_191 = [ttnn_repeat_164]
    return util_create_list_191


def main_const_eval_137(input):
    utils_DeviceGetter_get_device_137 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_248 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_137,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_248 = ttnn.to_layout(
        ttnn_to_device_248,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_249 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_137,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_249 = ttnn.to_layout(
        ttnn_to_device_249,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_250 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_137,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_250 = ttnn.to_layout(
        ttnn_to_device_250,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_168 = ttnn.reshape(
        ttnn_to_layout_248,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_165 = ttnn.repeat(ttnn_reshape_168, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_169 = ttnn.reshape(
        ttnn_to_layout_249,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_166 = ttnn.repeat(ttnn_reshape_169, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_170 = ttnn.reshape(
        ttnn_to_layout_250,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_167 = ttnn.repeat(ttnn_reshape_170, ttnn.Shape([1, 257, 1]))
    util_create_list_192 = [ttnn_repeat_165, ttnn_repeat_166, ttnn_repeat_167]
    ttnn_concat_55 = ttnn.concat(
        util_create_list_192,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_193 = [ttnn_concat_55]
    return util_create_list_193


def main_const_eval_138(input):
    utils_DeviceGetter_get_device_138 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_251 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_138,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_251 = ttnn.to_layout(
        ttnn_to_device_251,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_171 = ttnn.reshape(
        ttnn_to_layout_251,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_168 = ttnn.repeat(ttnn_reshape_171, ttnn.Shape([1, 257, 1]))
    util_create_list_194 = [ttnn_repeat_168]
    return util_create_list_194


def main_const_eval_139(input):
    utils_DeviceGetter_get_device_139 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_252 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_139,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_252 = ttnn.to_layout(
        ttnn_to_device_252,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_253 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_139,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_253 = ttnn.to_layout(
        ttnn_to_device_253,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_254 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_139,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_254 = ttnn.to_layout(
        ttnn_to_device_254,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_195 = [ttnn_to_layout_252, ttnn_to_layout_253, ttnn_to_layout_254]
    ttnn_concat_56 = ttnn.concat(
        util_create_list_195,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_196 = [ttnn_concat_56]
    return util_create_list_196


def main_const_eval_140(input):
    utils_DeviceGetter_get_device_140 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_255 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_140,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_255 = ttnn.to_layout(
        ttnn_to_device_255,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_172 = ttnn.reshape(
        ttnn_to_layout_255,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_169 = ttnn.repeat(ttnn_reshape_172, ttnn.Shape([1, 257, 1]))
    util_create_list_197 = [ttnn_repeat_169]
    return util_create_list_197


def main_const_eval_141(input):
    utils_DeviceGetter_get_device_141 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_256 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_141,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_256 = ttnn.to_layout(
        ttnn_to_device_256,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_173 = ttnn.reshape(
        ttnn_to_layout_256,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_170 = ttnn.repeat(ttnn_reshape_173, ttnn.Shape([1, 257, 1]))
    util_create_list_198 = [ttnn_repeat_170]
    return util_create_list_198


def main_const_eval_142(input):
    utils_DeviceGetter_get_device_142 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_257 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_142,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_257 = ttnn.to_layout(
        ttnn_to_device_257,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_174 = ttnn.reshape(
        ttnn_to_layout_257,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_171 = ttnn.repeat(ttnn_reshape_174, ttnn.Shape([1, 257, 1]))
    util_create_list_199 = [ttnn_repeat_171]
    return util_create_list_199


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


def main_const_eval_144(input):
    utils_DeviceGetter_get_device_144 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_259 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_144,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_259 = ttnn.to_layout(
        ttnn_to_device_259,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_176 = ttnn.reshape(
        ttnn_to_layout_259,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_172 = ttnn.repeat(ttnn_reshape_176, ttnn.Shape([1, 257, 1]))
    util_create_list_201 = [ttnn_repeat_172]
    return util_create_list_201


def main_const_eval_145(input):
    utils_DeviceGetter_get_device_145 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_260 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_145,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_260 = ttnn.to_layout(
        ttnn_to_device_260,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_177 = ttnn.reshape(
        ttnn_to_layout_260,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_173 = ttnn.repeat(ttnn_reshape_177, ttnn.Shape([1, 257, 1]))
    util_create_list_202 = [ttnn_repeat_173]
    return util_create_list_202


def main_const_eval_146(input):
    utils_DeviceGetter_get_device_146 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_261 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_146,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_261 = ttnn.to_layout(
        ttnn_to_device_261,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_178 = ttnn.reshape(
        ttnn_to_layout_261,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_174 = ttnn.repeat(ttnn_reshape_178, ttnn.Shape([1, 257, 1]))
    util_create_list_203 = [ttnn_repeat_174]
    return util_create_list_203


def main_const_eval_147(input):
    utils_DeviceGetter_get_device_147 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_262 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_147,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_262 = ttnn.to_layout(
        ttnn_to_device_262,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_179 = ttnn.reshape(
        ttnn_to_layout_262,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_175 = ttnn.repeat(ttnn_reshape_179, ttnn.Shape([1, 257, 1]))
    util_create_list_204 = [ttnn_repeat_175]
    return util_create_list_204


def main_const_eval_148(input):
    utils_DeviceGetter_get_device_148 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_263 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_148,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_263 = ttnn.to_layout(
        ttnn_to_device_263,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_180 = ttnn.reshape(
        ttnn_to_layout_263,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_176 = ttnn.repeat(ttnn_reshape_180, ttnn.Shape([1, 257, 1]))
    util_create_list_205 = [ttnn_repeat_176]
    return util_create_list_205


def main_const_eval_149(input):
    utils_DeviceGetter_get_device_149 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_264 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_149,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_264 = ttnn.to_layout(
        ttnn_to_device_264,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_265 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_149,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_265 = ttnn.to_layout(
        ttnn_to_device_265,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_266 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_149,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_266 = ttnn.to_layout(
        ttnn_to_device_266,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_206 = [ttnn_to_layout_264, ttnn_to_layout_265, ttnn_to_layout_266]
    ttnn_concat_57 = ttnn.concat(
        util_create_list_206,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_207 = [ttnn_concat_57]
    return util_create_list_207


def main_const_eval_150(input):
    utils_DeviceGetter_get_device_150 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_267 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_150,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_267 = ttnn.to_layout(
        ttnn_to_device_267,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_181 = ttnn.reshape(
        ttnn_to_layout_267,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_177 = ttnn.repeat(ttnn_reshape_181, ttnn.Shape([1, 257, 1]))
    util_create_list_208 = [ttnn_repeat_177]
    return util_create_list_208


def main_const_eval_151(input):
    utils_DeviceGetter_get_device_151 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_268 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_151,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_268 = ttnn.to_layout(
        ttnn_to_device_268,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_to_layout_268,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_178 = ttnn.repeat(ttnn_reshape_182, ttnn.Shape([1, 257, 1]))
    util_create_list_209 = [ttnn_repeat_178]
    return util_create_list_209


def main_const_eval_152(input):
    utils_DeviceGetter_get_device_152 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_269 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_152,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_269 = ttnn.to_layout(
        ttnn_to_device_269,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_183 = ttnn.reshape(
        ttnn_to_layout_269,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_179 = ttnn.repeat(ttnn_reshape_183, ttnn.Shape([1, 257, 1]))
    util_create_list_210 = [ttnn_repeat_179]
    return util_create_list_210


def main_const_eval_153(input):
    utils_DeviceGetter_get_device_153 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_270 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_153,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_270 = ttnn.to_layout(
        ttnn_to_device_270,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_271 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_153,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_271 = ttnn.to_layout(
        ttnn_to_device_271,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_272 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_153,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_272 = ttnn.to_layout(
        ttnn_to_device_272,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_211 = [ttnn_to_layout_270, ttnn_to_layout_271, ttnn_to_layout_272]
    ttnn_concat_58 = ttnn.concat(
        util_create_list_211,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_212 = [ttnn_concat_58]
    return util_create_list_212


def main_const_eval_154(input):
    utils_DeviceGetter_get_device_154 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_273 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_154,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_273 = ttnn.to_layout(
        ttnn_to_device_273,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_to_layout_273,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_180 = ttnn.repeat(ttnn_reshape_184, ttnn.Shape([1, 257, 1]))
    util_create_list_213 = [ttnn_repeat_180]
    return util_create_list_213


def main_const_eval_155(input):
    utils_DeviceGetter_get_device_155 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_274 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_155,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_274 = ttnn.to_layout(
        ttnn_to_device_274,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_185 = ttnn.reshape(
        ttnn_to_layout_274,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_181 = ttnn.repeat(ttnn_reshape_185, ttnn.Shape([1, 257, 1]))
    util_create_list_214 = [ttnn_repeat_181]
    return util_create_list_214


def main_const_eval_156(input):
    utils_DeviceGetter_get_device_156 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_275 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_156,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_275 = ttnn.to_layout(
        ttnn_to_device_275,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_186 = ttnn.reshape(
        ttnn_to_layout_275,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_182 = ttnn.repeat(ttnn_reshape_186, ttnn.Shape([1, 257, 1]))
    util_create_list_215 = [ttnn_repeat_182]
    return util_create_list_215


def main_const_eval_157(input):
    utils_DeviceGetter_get_device_157 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_276 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_157,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_276 = ttnn.to_layout(
        ttnn_to_device_276,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_187 = ttnn.reshape(
        ttnn_to_layout_276,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_183 = ttnn.repeat(ttnn_reshape_187, ttnn.Shape([1, 257, 1]))
    util_create_list_216 = [ttnn_repeat_183]
    return util_create_list_216


def main_const_eval_158(input):
    utils_DeviceGetter_get_device_158 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_277 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_158,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_277 = ttnn.to_layout(
        ttnn_to_device_277,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_278 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_158,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_278 = ttnn.to_layout(
        ttnn_to_device_278,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_279 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_158,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_279 = ttnn.to_layout(
        ttnn_to_device_279,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_217 = [ttnn_to_layout_277, ttnn_to_layout_278, ttnn_to_layout_279]
    ttnn_concat_59 = ttnn.concat(
        util_create_list_217,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_218 = [ttnn_concat_59]
    return util_create_list_218


def main_const_eval_159(input):
    utils_DeviceGetter_get_device_159 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_280 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_159,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_280 = ttnn.to_layout(
        ttnn_to_device_280,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_281 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_159,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_281 = ttnn.to_layout(
        ttnn_to_device_281,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_282 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_159,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_282 = ttnn.to_layout(
        ttnn_to_device_282,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_188 = ttnn.reshape(
        ttnn_to_layout_280,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_184 = ttnn.repeat(ttnn_reshape_188, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_189 = ttnn.reshape(
        ttnn_to_layout_281,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_185 = ttnn.repeat(ttnn_reshape_189, ttnn.Shape([1, 257, 1]))
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_to_layout_282,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_186 = ttnn.repeat(ttnn_reshape_190, ttnn.Shape([1, 257, 1]))
    util_create_list_219 = [ttnn_repeat_184, ttnn_repeat_185, ttnn_repeat_186]
    ttnn_concat_60 = ttnn.concat(
        util_create_list_219,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_220 = [ttnn_concat_60]
    return util_create_list_220


def main_const_eval_160(input):
    utils_DeviceGetter_get_device_160 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_283 = ttnn.to_device(
        input[2],
        device=utils_DeviceGetter_get_device_160,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_283 = ttnn.to_layout(
        ttnn_to_device_283,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_284 = ttnn.to_device(
        input[1],
        device=utils_DeviceGetter_get_device_160,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_284 = ttnn.to_layout(
        ttnn_to_device_284,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_285 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_160,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_285 = ttnn.to_layout(
        ttnn_to_device_285,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_221 = [ttnn_to_layout_283, ttnn_to_layout_284, ttnn_to_layout_285]
    ttnn_concat_61 = ttnn.concat(
        util_create_list_221,
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_222 = [ttnn_concat_61]
    return util_create_list_222


def main_const_eval_161(input):
    utils_DeviceGetter_get_device_161 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_286 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_161,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_286 = ttnn.to_layout(
        ttnn_to_device_286,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_to_layout_286,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_187 = ttnn.repeat(ttnn_reshape_191, ttnn.Shape([1, 257, 1]))
    util_create_list_223 = [ttnn_repeat_187]
    return util_create_list_223


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, _CONST_EVAL_CACHE, const_1
    )
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_2 = main_const_eval_1
    util_create_list_224 = [input[99]]
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_2, util_create_list_224, _CONST_EVAL_CACHE, const_3
    )
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_4 = main_const_eval_2
    util_create_list_225 = [input[225], input[444], input[446]]
    const_5 = "main_const_eval_2"
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_4, util_create_list_225, _CONST_EVAL_CACHE, const_5
    )
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_6 = main_const_eval_3
    util_create_list_226 = [input[265]]
    const_7 = "main_const_eval_3"
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_6, util_create_list_226, _CONST_EVAL_CACHE, const_7
    )
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_8 = main_const_eval_4
    util_create_list_227 = [input[46], input[505], input[507]]
    const_9 = "main_const_eval_4"
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_8, util_create_list_227, _CONST_EVAL_CACHE, const_9
    )
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_10 = main_const_eval_5
    util_create_list_228 = [input[111]]
    const_11 = "main_const_eval_5"
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_10, util_create_list_228, _CONST_EVAL_CACHE, const_11
    )
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_12 = main_const_eval_6
    util_create_list_229 = [input[231]]
    const_13 = "main_const_eval_6"
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(
        const_12, util_create_list_229, _CONST_EVAL_CACHE, const_13
    )
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_14 = main_const_eval_7
    util_create_list_230 = [input[2]]
    const_15 = "main_const_eval_7"
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(
        const_14, util_create_list_230, _CONST_EVAL_CACHE, const_15
    )
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_16 = main_const_eval_8
    util_create_list_231 = [input[175]]
    const_17 = "main_const_eval_8"
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(
        const_16, util_create_list_231, _CONST_EVAL_CACHE, const_17
    )
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_18 = main_const_eval_9
    util_create_list_232 = [input[55]]
    const_19 = "main_const_eval_9"
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(
        const_18, util_create_list_232, _CONST_EVAL_CACHE, const_19
    )
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_20 = main_const_eval_10
    util_create_list_233 = [input[70], input[497], input[499]]
    const_21 = "main_const_eval_10"
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(
        const_20, util_create_list_233, _CONST_EVAL_CACHE, const_21
    )
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_22 = main_const_eval_11
    util_create_list_234 = [input[337]]
    const_23 = "main_const_eval_11"
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_22, util_create_list_234, _CONST_EVAL_CACHE, const_23
    )
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_24 = main_const_eval_12
    util_create_list_235 = [input[211]]
    const_25 = "main_const_eval_12"
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_24, util_create_list_235, _CONST_EVAL_CACHE, const_25
    )
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_26 = main_const_eval_13
    util_create_list_236 = [input[154], input[469], input[471]]
    const_27 = "main_const_eval_13"
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_26, util_create_list_236, _CONST_EVAL_CACHE, const_27
    )
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_28 = main_const_eval_14
    util_create_list_237 = [input[363]]
    const_29 = "main_const_eval_14"
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_28, util_create_list_237, _CONST_EVAL_CACHE, const_29
    )
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_30 = main_const_eval_15
    util_create_list_238 = [input[39]]
    const_31 = "main_const_eval_15"
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_30, util_create_list_238, _CONST_EVAL_CACHE, const_31
    )
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_32 = main_const_eval_16
    util_create_list_239 = [input[229]]
    const_33 = "main_const_eval_16"
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_32, util_create_list_239, _CONST_EVAL_CACHE, const_33
    )
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_34 = main_const_eval_17
    util_create_list_240 = [input[45], input[504], input[506]]
    const_35 = "main_const_eval_17"
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_34, util_create_list_240, _CONST_EVAL_CACHE, const_35
    )
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_36 = main_const_eval_18
    util_create_list_241 = [input[177], input[460], input[462]]
    const_37 = "main_const_eval_18"
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_36, util_create_list_241, _CONST_EVAL_CACHE, const_37
    )
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_38 = main_const_eval_19
    util_create_list_242 = [input[205]]
    const_39 = "main_const_eval_19"
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_38, util_create_list_242, _CONST_EVAL_CACHE, const_39
    )
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_40 = main_const_eval_20
    util_create_list_243 = [input[169]]
    const_41 = "main_const_eval_20"
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_40, util_create_list_243, _CONST_EVAL_CACHE, const_41
    )
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_42 = main_const_eval_21
    util_create_list_244 = [input[127]]
    const_43 = "main_const_eval_21"
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_42, util_create_list_244, _CONST_EVAL_CACHE, const_43
    )
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_44 = main_const_eval_22
    util_create_list_245 = [input[361]]
    const_45 = "main_const_eval_22"
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_44, util_create_list_245, _CONST_EVAL_CACHE, const_45
    )
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_46 = main_const_eval_23
    util_create_list_246 = [input[97]]
    const_47 = "main_const_eval_23"
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_46, util_create_list_246, _CONST_EVAL_CACHE, const_47
    )
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_48 = main_const_eval_24
    util_create_list_247 = [input[202], input[453], input[455]]
    const_49 = "main_const_eval_24"
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_48, util_create_list_247, _CONST_EVAL_CACHE, const_49
    )
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_50 = main_const_eval_25
    util_create_list_248 = [input[279]]
    const_51 = "main_const_eval_25"
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_50, util_create_list_248, _CONST_EVAL_CACHE, const_51
    )
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_52 = main_const_eval_26
    util_create_list_249 = [input[358], input[401], input[403]]
    const_53 = "main_const_eval_26"
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_52, util_create_list_249, _CONST_EVAL_CACHE, const_53
    )
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_54 = main_const_eval_27
    util_create_list_250 = [input[346], input[405], input[407]]
    const_55 = "main_const_eval_27"
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_54, util_create_list_250, _CONST_EVAL_CACHE, const_55
    )
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_56 = main_const_eval_28
    util_create_list_251 = [input[291]]
    const_57 = "main_const_eval_28"
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_56, util_create_list_251, _CONST_EVAL_CACHE, const_57
    )
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_58 = main_const_eval_29
    util_create_list_252 = [input[145]]
    const_59 = "main_const_eval_29"
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_58, util_create_list_252, _CONST_EVAL_CACHE, const_59
    )
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_60 = main_const_eval_30
    util_create_list_253 = [input[286], input[425], input[427]]
    const_61 = "main_const_eval_30"
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_60, util_create_list_253, _CONST_EVAL_CACHE, const_61
    )
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_62 = main_const_eval_31
    util_create_list_254 = [input[4], input[7], input[8], input[517]]
    const_63 = "main_const_eval_31"
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_62, util_create_list_254, _CONST_EVAL_CACHE, const_63
    )
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    utils_constEvalFuncWrapper_30_1 = utils_constEvalFuncWrapper_30[1]
    const_64 = main_const_eval_32
    util_create_list_255 = [input[79]]
    const_65 = "main_const_eval_32"
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_64, util_create_list_255, _CONST_EVAL_CACHE, const_65
    )
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_66 = main_const_eval_33
    util_create_list_256 = [input[105], input[484], input[486]]
    const_67 = "main_const_eval_33"
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_66, util_create_list_256, _CONST_EVAL_CACHE, const_67
    )
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_68 = main_const_eval_34
    util_create_list_257 = [input[115]]
    const_69 = "main_const_eval_34"
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_68, util_create_list_257, _CONST_EVAL_CACHE, const_69
    )
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_70 = main_const_eval_35
    util_create_list_258 = [input[178], input[461], input[463]]
    const_71 = "main_const_eval_35"
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_70, util_create_list_258, _CONST_EVAL_CACHE, const_71
    )
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_72 = main_const_eval_36
    util_create_list_259 = [input[25]]
    const_73 = "main_const_eval_36"
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_72, util_create_list_259, _CONST_EVAL_CACHE, const_73
    )
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_74 = main_const_eval_37
    util_create_list_260 = [input[106], input[485], input[487]]
    const_75 = "main_const_eval_37"
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_74, util_create_list_260, _CONST_EVAL_CACHE, const_75
    )
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_76 = main_const_eval_38
    util_create_list_261 = [input[130], input[477], input[479]]
    const_77 = "main_const_eval_38"
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_76, util_create_list_261, _CONST_EVAL_CACHE, const_77
    )
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_78 = main_const_eval_39
    util_create_list_262 = [input[27]]
    const_79 = "main_const_eval_39"
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_78, util_create_list_262, _CONST_EVAL_CACHE, const_79
    )
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_80 = main_const_eval_40
    util_create_list_263 = [input[73]]
    const_81 = "main_const_eval_40"
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_80, util_create_list_263, _CONST_EVAL_CACHE, const_81
    )
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_82 = main_const_eval_41
    util_create_list_264 = [input[285], input[424], input[426]]
    const_83 = "main_const_eval_41"
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_82, util_create_list_264, _CONST_EVAL_CACHE, const_83
    )
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_84 = main_const_eval_42
    util_create_list_265 = [input[57], input[500], input[502]]
    const_85 = "main_const_eval_42"
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_84, util_create_list_265, _CONST_EVAL_CACHE, const_85
    )
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_86 = main_const_eval_43
    util_create_list_266 = [input[375]]
    const_87 = "main_const_eval_43"
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_86, util_create_list_266, _CONST_EVAL_CACHE, const_87
    )
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_88 = main_const_eval_44
    util_create_list_267 = [input[333], input[408], input[410]]
    const_89 = "main_const_eval_44"
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_88, util_create_list_267, _CONST_EVAL_CACHE, const_89
    )
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_90 = main_const_eval_45
    util_create_list_268 = [input[147]]
    const_91 = "main_const_eval_45"
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_90, util_create_list_268, _CONST_EVAL_CACHE, const_91
    )
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_92 = main_const_eval_46
    util_create_list_269 = [input[33], input[508], input[510]]
    const_93 = "main_const_eval_46"
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_92, util_create_list_269, _CONST_EVAL_CACHE, const_93
    )
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_94 = main_const_eval_47
    util_create_list_270 = [input[307]]
    const_95 = "main_const_eval_47"
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_94, util_create_list_270, _CONST_EVAL_CACHE, const_95
    )
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_96 = main_const_eval_48
    util_create_list_271 = [input[381], input[392], input[394]]
    const_97 = "main_const_eval_48"
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_96, util_create_list_271, _CONST_EVAL_CACHE, const_97
    )
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_98 = main_const_eval_49
    util_create_list_272 = [input[19]]
    const_99 = "main_const_eval_49"
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_98, util_create_list_272, _CONST_EVAL_CACHE, const_99
    )
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_100 = main_const_eval_50
    util_create_list_273 = [input[298], input[421], input[423]]
    const_101 = "main_const_eval_50"
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_100, util_create_list_273, _CONST_EVAL_CACHE, const_101
    )
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_102 = main_const_eval_51
    util_create_list_274 = [input[153], input[468], input[470]]
    const_103 = "main_const_eval_51"
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_102, util_create_list_274, _CONST_EVAL_CACHE, const_103
    )
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_104 = main_const_eval_52
    util_create_list_275 = [input[103]]
    const_105 = "main_const_eval_52"
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_104, util_create_list_275, _CONST_EVAL_CACHE, const_105
    )
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_106 = main_const_eval_53
    util_create_list_276 = [input[151]]
    const_107 = "main_const_eval_53"
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_106, util_create_list_276, _CONST_EVAL_CACHE, const_107
    )
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_108 = main_const_eval_54
    util_create_list_277 = [input[303]]
    const_109 = "main_const_eval_54"
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_108, util_create_list_277, _CONST_EVAL_CACHE, const_109
    )
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_110 = main_const_eval_55
    util_create_list_278 = [input[13]]
    const_111 = "main_const_eval_55"
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_110, util_create_list_278, _CONST_EVAL_CACHE, const_111
    )
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_112 = main_const_eval_56
    util_create_list_279 = [input[367]]
    const_113 = "main_const_eval_56"
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_112, util_create_list_279, _CONST_EVAL_CACHE, const_113
    )
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_114 = main_const_eval_57
    util_create_list_280 = [input[37]]
    const_115 = "main_const_eval_57"
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_114, util_create_list_280, _CONST_EVAL_CACHE, const_115
    )
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_116 = main_const_eval_58
    util_create_list_281 = [input[117], input[480], input[482]]
    const_117 = "main_const_eval_58"
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_116, util_create_list_281, _CONST_EVAL_CACHE, const_117
    )
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_118 = main_const_eval_59
    util_create_list_282 = [input[81], input[492], input[494]]
    const_119 = "main_const_eval_59"
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_118, util_create_list_282, _CONST_EVAL_CACHE, const_119
    )
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_120 = main_const_eval_60
    util_create_list_283 = [input[93], input[488], input[490]]
    const_121 = "main_const_eval_60"
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_120, util_create_list_283, _CONST_EVAL_CACHE, const_121
    )
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_122 = main_const_eval_61
    util_create_list_284 = [input[141], input[472], input[474]]
    const_123 = "main_const_eval_61"
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_122, util_create_list_284, _CONST_EVAL_CACHE, const_123
    )
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_124 = main_const_eval_62
    util_create_list_285 = [input[82], input[493], input[495]]
    const_125 = "main_const_eval_62"
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_124, util_create_list_285, _CONST_EVAL_CACHE, const_125
    )
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_126 = main_const_eval_63
    util_create_list_286 = [input[369], input[396], input[398]]
    const_127 = "main_const_eval_63"
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_126, util_create_list_286, _CONST_EVAL_CACHE, const_127
    )
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_128 = main_const_eval_64
    util_create_list_287 = [input[187]]
    const_129 = "main_const_eval_64"
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_128, util_create_list_287, _CONST_EVAL_CACHE, const_129
    )
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_130 = main_const_eval_65
    util_create_list_288 = [input[259]]
    const_131 = "main_const_eval_65"
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_130, util_create_list_288, _CONST_EVAL_CACHE, const_131
    )
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_132 = main_const_eval_66
    util_create_list_289 = [input[142], input[473], input[475]]
    const_133 = "main_const_eval_66"
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_132, util_create_list_289, _CONST_EVAL_CACHE, const_133
    )
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_134 = main_const_eval_67
    util_create_list_290 = [input[389]]
    const_135 = "main_const_eval_67"
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_134, util_create_list_290, _CONST_EVAL_CACHE, const_135
    )
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_136 = main_const_eval_68
    util_create_list_291 = [input[250], input[437], input[439]]
    const_137 = "main_const_eval_68"
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_136, util_create_list_291, _CONST_EVAL_CACHE, const_137
    )
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_138 = main_const_eval_69
    util_create_list_292 = [input[235]]
    const_139 = "main_const_eval_69"
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_138, util_create_list_292, _CONST_EVAL_CACHE, const_139
    )
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_140 = main_const_eval_70
    util_create_list_293 = [input[319]]
    const_141 = "main_const_eval_70"
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_140, util_create_list_293, _CONST_EVAL_CACHE, const_141
    )
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_142 = main_const_eval_71
    util_create_list_294 = [input[382], input[393], input[395]]
    const_143 = "main_const_eval_71"
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_142, util_create_list_294, _CONST_EVAL_CACHE, const_143
    )
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_144 = main_const_eval_72
    util_create_list_295 = [input[261], input[432], input[434]]
    const_145 = "main_const_eval_72"
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_144, util_create_list_295, _CONST_EVAL_CACHE, const_145
    )
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_146 = main_const_eval_73
    util_create_list_296 = [input[201], input[452], input[454]]
    const_147 = "main_const_eval_73"
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_146, util_create_list_296, _CONST_EVAL_CACHE, const_147
    )
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_148 = main_const_eval_74
    util_create_list_297 = [input[373]]
    const_149 = "main_const_eval_74"
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_148, util_create_list_297, _CONST_EVAL_CACHE, const_149
    )
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_150 = main_const_eval_75
    util_create_list_298 = [input[283]]
    const_151 = "main_const_eval_75"
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_150, util_create_list_298, _CONST_EVAL_CACHE, const_151
    )
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_152 = main_const_eval_76
    util_create_list_299 = [input[34], input[509], input[511]]
    const_153 = "main_const_eval_76"
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_152, util_create_list_299, _CONST_EVAL_CACHE, const_153
    )
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_154 = main_const_eval_77
    util_create_list_300 = [input[94], input[489], input[491]]
    const_155 = "main_const_eval_77"
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_154, util_create_list_300, _CONST_EVAL_CACHE, const_155
    )
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_156 = main_const_eval_78
    util_create_list_301 = [input[69], input[496], input[498]]
    const_157 = "main_const_eval_78"
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_156, util_create_list_301, _CONST_EVAL_CACHE, const_157
    )
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_158 = main_const_eval_79
    util_create_list_302 = [input[139]]
    const_159 = "main_const_eval_79"
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_158, util_create_list_302, _CONST_EVAL_CACHE, const_159
    )
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_160 = main_const_eval_80
    util_create_list_303 = [input[31]]
    const_161 = "main_const_eval_80"
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_160, util_create_list_303, _CONST_EVAL_CACHE, const_161
    )
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_162 = main_const_eval_81
    util_create_list_304 = [input[357], input[400], input[402]]
    const_163 = "main_const_eval_81"
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_162, util_create_list_304, _CONST_EVAL_CACHE, const_163
    )
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_164 = main_const_eval_82
    util_create_list_305 = [input[351]]
    const_165 = "main_const_eval_82"
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_164, util_create_list_305, _CONST_EVAL_CACHE, const_165
    )
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_166 = main_const_eval_83
    util_create_list_306 = [input[133]]
    const_167 = "main_const_eval_83"
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_166, util_create_list_306, _CONST_EVAL_CACHE, const_167
    )
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_168 = main_const_eval_84
    util_create_list_307 = [input[193]]
    const_169 = "main_const_eval_84"
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_168, util_create_list_307, _CONST_EVAL_CACHE, const_169
    )
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_170 = main_const_eval_85
    util_create_list_308 = [input[297], input[420], input[422]]
    const_171 = "main_const_eval_85"
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_170, util_create_list_308, _CONST_EVAL_CACHE, const_171
    )
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_172 = main_const_eval_86
    util_create_list_309 = [input[253]]
    const_173 = "main_const_eval_86"
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_172, util_create_list_309, _CONST_EVAL_CACHE, const_173
    )
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_174 = main_const_eval_87
    util_create_list_310 = [input[214], input[449], input[451]]
    const_175 = "main_const_eval_87"
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_174, util_create_list_310, _CONST_EVAL_CACHE, const_175
    )
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_176 = main_const_eval_88
    util_create_list_311 = [input[238], input[441], input[443]]
    const_177 = "main_const_eval_88"
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_176, util_create_list_311, _CONST_EVAL_CACHE, const_177
    )
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_178 = main_const_eval_89
    util_create_list_312 = [input[387], input[388]]
    const_179 = "main_const_eval_89"
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_178, util_create_list_312, _CONST_EVAL_CACHE, const_179
    )
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_180 = main_const_eval_90
    util_create_list_313 = [input[190], input[457], input[459]]
    const_181 = "main_const_eval_90"
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_180, util_create_list_313, _CONST_EVAL_CACHE, const_181
    )
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_182 = main_const_eval_91
    util_create_list_314 = [input[345], input[404], input[406]]
    const_183 = "main_const_eval_91"
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_182, util_create_list_314, _CONST_EVAL_CACHE, const_183
    )
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_184 = main_const_eval_92
    util_create_list_315 = [input[315]]
    const_185 = "main_const_eval_92"
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_184, util_create_list_315, _CONST_EVAL_CACHE, const_185
    )
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_186 = main_const_eval_93
    util_create_list_316 = [input[219]]
    const_187 = "main_const_eval_93"
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_186, util_create_list_316, _CONST_EVAL_CACHE, const_187
    )
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_188 = main_const_eval_94
    util_create_list_317 = [input[277]]
    const_189 = "main_const_eval_94"
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_188, util_create_list_317, _CONST_EVAL_CACHE, const_189
    )
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_190 = main_const_eval_95
    util_create_list_318 = [input[159]]
    const_191 = "main_const_eval_95"
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_190, util_create_list_318, _CONST_EVAL_CACHE, const_191
    )
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_192 = main_const_eval_96
    util_create_list_319 = [input[255]]
    const_193 = "main_const_eval_96"
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_192, util_create_list_319, _CONST_EVAL_CACHE, const_193
    )
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_194 = main_const_eval_97
    util_create_list_320 = [input[322], input[413], input[415]]
    const_195 = "main_const_eval_97"
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_194, util_create_list_320, _CONST_EVAL_CACHE, const_195
    )
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_196 = main_const_eval_98
    util_create_list_321 = [input[331]]
    const_197 = "main_const_eval_98"
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_196, util_create_list_321, _CONST_EVAL_CACHE, const_197
    )
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_198 = main_const_eval_99
    util_create_list_322 = [input[63]]
    const_199 = "main_const_eval_99"
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_198, util_create_list_322, _CONST_EVAL_CACHE, const_199
    )
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_200 = main_const_eval_100
    util_create_list_323 = [input[309], input[416], input[418]]
    const_201 = "main_const_eval_100"
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_200, util_create_list_323, _CONST_EVAL_CACHE, const_201
    )
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_202 = main_const_eval_101
    util_create_list_324 = [input[163]]
    const_203 = "main_const_eval_101"
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_202, util_create_list_324, _CONST_EVAL_CACHE, const_203
    )
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_204 = main_const_eval_102
    util_create_list_325 = [input[213], input[448], input[450]]
    const_205 = "main_const_eval_102"
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_204, util_create_list_325, _CONST_EVAL_CACHE, const_205
    )
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_206 = main_const_eval_103
    util_create_list_326 = [input[226], input[445], input[447]]
    const_207 = "main_const_eval_103"
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_206, util_create_list_326, _CONST_EVAL_CACHE, const_207
    )
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_208 = main_const_eval_104
    util_create_list_327 = [input[301]]
    const_209 = "main_const_eval_104"
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_208, util_create_list_327, _CONST_EVAL_CACHE, const_209
    )
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_210 = main_const_eval_105
    util_create_list_328 = [input[181]]
    const_211 = "main_const_eval_105"
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_210, util_create_list_328, _CONST_EVAL_CACHE, const_211
    )
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_212 = main_const_eval_106
    util_create_list_329 = [input[67]]
    const_213 = "main_const_eval_106"
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_212, util_create_list_329, _CONST_EVAL_CACHE, const_213
    )
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_214 = main_const_eval_107
    util_create_list_330 = [input[313]]
    const_215 = "main_const_eval_107"
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_214, util_create_list_330, _CONST_EVAL_CACHE, const_215
    )
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_216 = main_const_eval_108
    util_create_list_331 = [input[135]]
    const_217 = "main_const_eval_108"
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_216, util_create_list_331, _CONST_EVAL_CACHE, const_217
    )
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_218 = main_const_eval_109
    util_create_list_332 = [input[171]]
    const_219 = "main_const_eval_109"
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_218, util_create_list_332, _CONST_EVAL_CACHE, const_219
    )
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_220 = main_const_eval_110
    util_create_list_333 = [input[217]]
    const_221 = "main_const_eval_110"
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_220, util_create_list_333, _CONST_EVAL_CACHE, const_221
    )
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_222 = main_const_eval_111
    util_create_list_334 = [input[123]]
    const_223 = "main_const_eval_111"
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_222, util_create_list_334, _CONST_EVAL_CACHE, const_223
    )
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_224 = main_const_eval_112
    util_create_list_335 = [input[129], input[476], input[478]]
    const_225 = "main_const_eval_112"
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_224, util_create_list_335, _CONST_EVAL_CACHE, const_225
    )
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_226 = main_const_eval_113
    util_create_list_336 = [input[166], input[465], input[467]]
    const_227 = "main_const_eval_113"
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_226, util_create_list_336, _CONST_EVAL_CACHE, const_227
    )
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_228 = main_const_eval_114
    util_create_list_337 = [input[271]]
    const_229 = "main_const_eval_114"
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_228, util_create_list_337, _CONST_EVAL_CACHE, const_229
    )
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_230 = main_const_eval_115
    util_create_list_338 = [input[199]]
    const_231 = "main_const_eval_115"
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_230, util_create_list_338, _CONST_EVAL_CACHE, const_231
    )
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_232 = main_const_eval_116
    util_create_list_339 = [input[51]]
    const_233 = "main_const_eval_116"
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_232, util_create_list_339, _CONST_EVAL_CACHE, const_233
    )
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_234 = main_const_eval_117
    util_create_list_340 = [input[249], input[436], input[438]]
    const_235 = "main_const_eval_117"
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_234, util_create_list_340, _CONST_EVAL_CACHE, const_235
    )
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_236 = main_const_eval_118
    util_create_list_341 = [input[75]]
    const_237 = "main_const_eval_118"
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_236, util_create_list_341, _CONST_EVAL_CACHE, const_237
    )
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_238 = main_const_eval_119
    util_create_list_342 = [input[189], input[456], input[458]]
    const_239 = "main_const_eval_119"
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_238, util_create_list_342, _CONST_EVAL_CACHE, const_239
    )
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_240 = main_const_eval_120
    util_create_list_343 = [input[274], input[429], input[431]]
    const_241 = "main_const_eval_120"
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_240, util_create_list_343, _CONST_EVAL_CACHE, const_241
    )
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_242 = main_const_eval_121
    util_create_list_344 = [input[295]]
    const_243 = "main_const_eval_121"
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_242, util_create_list_344, _CONST_EVAL_CACHE, const_243
    )
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_244 = main_const_eval_122
    util_create_list_345 = [input[43]]
    const_245 = "main_const_eval_122"
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_244, util_create_list_345, _CONST_EVAL_CACHE, const_245
    )
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_246 = main_const_eval_123
    util_create_list_346 = [input[355]]
    const_247 = "main_const_eval_123"
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_246, util_create_list_346, _CONST_EVAL_CACHE, const_247
    )
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_248 = main_const_eval_124
    util_create_list_347 = [input[61]]
    const_249 = "main_const_eval_124"
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_248, util_create_list_347, _CONST_EVAL_CACHE, const_249
    )
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_250 = main_const_eval_125
    util_create_list_348 = [input[379]]
    const_251 = "main_const_eval_125"
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_250, util_create_list_348, _CONST_EVAL_CACHE, const_251
    )
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_252 = main_const_eval_126
    util_create_list_349 = [input[109]]
    const_253 = "main_const_eval_126"
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_252, util_create_list_349, _CONST_EVAL_CACHE, const_253
    )
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    const_254 = main_const_eval_127
    util_create_list_350 = [input[223]]
    const_255 = "main_const_eval_127"
    utils_constEvalFuncWrapper_126 = utils.constEvalFuncWrapper(
        const_254, util_create_list_350, _CONST_EVAL_CACHE, const_255
    )
    utils_constEvalFuncWrapper_126_0 = utils_constEvalFuncWrapper_126[0]
    const_256 = main_const_eval_128
    util_create_list_351 = [input[334], input[409], input[411]]
    const_257 = "main_const_eval_128"
    utils_constEvalFuncWrapper_127 = utils.constEvalFuncWrapper(
        const_256, util_create_list_351, _CONST_EVAL_CACHE, const_257
    )
    utils_constEvalFuncWrapper_127_0 = utils_constEvalFuncWrapper_127[0]
    const_258 = main_const_eval_129
    util_create_list_352 = [input[310], input[417], input[419]]
    const_259 = "main_const_eval_129"
    utils_constEvalFuncWrapper_128 = utils.constEvalFuncWrapper(
        const_258, util_create_list_352, _CONST_EVAL_CACHE, const_259
    )
    utils_constEvalFuncWrapper_128_0 = utils_constEvalFuncWrapper_128[0]
    const_260 = main_const_eval_130
    util_create_list_353 = [input[49]]
    const_261 = "main_const_eval_130"
    utils_constEvalFuncWrapper_129 = utils.constEvalFuncWrapper(
        const_260, util_create_list_353, _CONST_EVAL_CACHE, const_261
    )
    utils_constEvalFuncWrapper_129_0 = utils_constEvalFuncWrapper_129[0]
    const_262 = main_const_eval_131
    util_create_list_354 = [input[183]]
    const_263 = "main_const_eval_131"
    utils_constEvalFuncWrapper_130 = utils.constEvalFuncWrapper(
        const_262, util_create_list_354, _CONST_EVAL_CACHE, const_263
    )
    utils_constEvalFuncWrapper_130_0 = utils_constEvalFuncWrapper_130[0]
    const_264 = main_const_eval_132
    util_create_list_355 = [input[21], input[512], input[514]]
    const_265 = "main_const_eval_132"
    utils_constEvalFuncWrapper_131 = utils.constEvalFuncWrapper(
        const_264, util_create_list_355, _CONST_EVAL_CACHE, const_265
    )
    utils_constEvalFuncWrapper_131_0 = utils_constEvalFuncWrapper_131[0]
    const_266 = main_const_eval_133
    util_create_list_356 = [input[343]]
    const_267 = "main_const_eval_133"
    utils_constEvalFuncWrapper_132 = utils.constEvalFuncWrapper(
        const_266, util_create_list_356, _CONST_EVAL_CACHE, const_267
    )
    utils_constEvalFuncWrapper_132_0 = utils_constEvalFuncWrapper_132[0]
    const_268 = main_const_eval_134
    util_create_list_357 = [input[273], input[428], input[430]]
    const_269 = "main_const_eval_134"
    utils_constEvalFuncWrapper_133 = utils.constEvalFuncWrapper(
        const_268, util_create_list_357, _CONST_EVAL_CACHE, const_269
    )
    utils_constEvalFuncWrapper_133_0 = utils_constEvalFuncWrapper_133[0]
    const_270 = main_const_eval_135
    util_create_list_358 = [input[165], input[464], input[466]]
    const_271 = "main_const_eval_135"
    utils_constEvalFuncWrapper_134 = utils.constEvalFuncWrapper(
        const_270, util_create_list_358, _CONST_EVAL_CACHE, const_271
    )
    utils_constEvalFuncWrapper_134_0 = utils_constEvalFuncWrapper_134[0]
    const_272 = main_const_eval_136
    util_create_list_359 = [input[15]]
    const_273 = "main_const_eval_136"
    utils_constEvalFuncWrapper_135 = utils.constEvalFuncWrapper(
        const_272, util_create_list_359, _CONST_EVAL_CACHE, const_273
    )
    utils_constEvalFuncWrapper_135_0 = utils_constEvalFuncWrapper_135[0]
    const_274 = main_const_eval_137
    util_create_list_360 = [input[237], input[440], input[442]]
    const_275 = "main_const_eval_137"
    utils_constEvalFuncWrapper_136 = utils.constEvalFuncWrapper(
        const_274, util_create_list_360, _CONST_EVAL_CACHE, const_275
    )
    utils_constEvalFuncWrapper_136_0 = utils_constEvalFuncWrapper_136[0]
    const_276 = main_const_eval_138
    util_create_list_361 = [input[11]]
    const_277 = "main_const_eval_138"
    utils_constEvalFuncWrapper_137 = utils.constEvalFuncWrapper(
        const_276, util_create_list_361, _CONST_EVAL_CACHE, const_277
    )
    utils_constEvalFuncWrapper_137_0 = utils_constEvalFuncWrapper_137[0]
    const_278 = main_const_eval_139
    util_create_list_362 = [input[22], input[513], input[515]]
    const_279 = "main_const_eval_139"
    utils_constEvalFuncWrapper_138 = utils.constEvalFuncWrapper(
        const_278, util_create_list_362, _CONST_EVAL_CACHE, const_279
    )
    utils_constEvalFuncWrapper_138_0 = utils_constEvalFuncWrapper_138[0]
    const_280 = main_const_eval_140
    util_create_list_363 = [input[87]]
    const_281 = "main_const_eval_140"
    utils_constEvalFuncWrapper_139 = utils.constEvalFuncWrapper(
        const_280, util_create_list_363, _CONST_EVAL_CACHE, const_281
    )
    utils_constEvalFuncWrapper_139_0 = utils_constEvalFuncWrapper_139[0]
    const_282 = main_const_eval_141
    util_create_list_364 = [input[247]]
    const_283 = "main_const_eval_141"
    utils_constEvalFuncWrapper_140 = utils.constEvalFuncWrapper(
        const_282, util_create_list_364, _CONST_EVAL_CACHE, const_283
    )
    utils_constEvalFuncWrapper_140_0 = utils_constEvalFuncWrapper_140[0]
    const_284 = main_const_eval_142
    util_create_list_365 = [input[207]]
    const_285 = "main_const_eval_142"
    utils_constEvalFuncWrapper_141 = utils.constEvalFuncWrapper(
        const_284, util_create_list_365, _CONST_EVAL_CACHE, const_285
    )
    utils_constEvalFuncWrapper_141_0 = utils_constEvalFuncWrapper_141[0]
    const_286 = main_const_eval_143
    util_create_list_366 = [input[391]]
    const_287 = "main_const_eval_143"
    utils_constEvalFuncWrapper_142 = utils.constEvalFuncWrapper(
        const_286, util_create_list_366, _CONST_EVAL_CACHE, const_287
    )
    utils_constEvalFuncWrapper_142_0 = utils_constEvalFuncWrapper_142[0]
    const_288 = main_const_eval_144
    util_create_list_367 = [input[91]]
    const_289 = "main_const_eval_144"
    utils_constEvalFuncWrapper_143 = utils.constEvalFuncWrapper(
        const_288, util_create_list_367, _CONST_EVAL_CACHE, const_289
    )
    utils_constEvalFuncWrapper_143_0 = utils_constEvalFuncWrapper_143[0]
    const_290 = main_const_eval_145
    util_create_list_368 = [input[85]]
    const_291 = "main_const_eval_145"
    utils_constEvalFuncWrapper_144 = utils.constEvalFuncWrapper(
        const_290, util_create_list_368, _CONST_EVAL_CACHE, const_291
    )
    utils_constEvalFuncWrapper_144_0 = utils_constEvalFuncWrapper_144[0]
    const_292 = main_const_eval_146
    util_create_list_369 = [input[339]]
    const_293 = "main_const_eval_146"
    utils_constEvalFuncWrapper_145 = utils.constEvalFuncWrapper(
        const_292, util_create_list_369, _CONST_EVAL_CACHE, const_293
    )
    utils_constEvalFuncWrapper_145_0 = utils_constEvalFuncWrapper_145[0]
    const_294 = main_const_eval_147
    util_create_list_370 = [input[349]]
    const_295 = "main_const_eval_147"
    utils_constEvalFuncWrapper_146 = utils.constEvalFuncWrapper(
        const_294, util_create_list_370, _CONST_EVAL_CACHE, const_295
    )
    utils_constEvalFuncWrapper_146_0 = utils_constEvalFuncWrapper_146[0]
    const_296 = main_const_eval_148
    util_create_list_371 = [input[157]]
    const_297 = "main_const_eval_148"
    utils_constEvalFuncWrapper_147 = utils.constEvalFuncWrapper(
        const_296, util_create_list_371, _CONST_EVAL_CACHE, const_297
    )
    utils_constEvalFuncWrapper_147_0 = utils_constEvalFuncWrapper_147[0]
    const_298 = main_const_eval_149
    util_create_list_372 = [input[118], input[481], input[483]]
    const_299 = "main_const_eval_149"
    utils_constEvalFuncWrapper_148 = utils.constEvalFuncWrapper(
        const_298, util_create_list_372, _CONST_EVAL_CACHE, const_299
    )
    utils_constEvalFuncWrapper_148_0 = utils_constEvalFuncWrapper_148[0]
    const_300 = main_const_eval_150
    util_create_list_373 = [input[325]]
    const_301 = "main_const_eval_150"
    utils_constEvalFuncWrapper_149 = utils.constEvalFuncWrapper(
        const_300, util_create_list_373, _CONST_EVAL_CACHE, const_301
    )
    utils_constEvalFuncWrapper_149_0 = utils_constEvalFuncWrapper_149[0]
    const_302 = main_const_eval_151
    util_create_list_374 = [input[327]]
    const_303 = "main_const_eval_151"
    utils_constEvalFuncWrapper_150 = utils.constEvalFuncWrapper(
        const_302, util_create_list_374, _CONST_EVAL_CACHE, const_303
    )
    utils_constEvalFuncWrapper_150_0 = utils_constEvalFuncWrapper_150[0]
    const_304 = main_const_eval_152
    util_create_list_375 = [input[241]]
    const_305 = "main_const_eval_152"
    utils_constEvalFuncWrapper_151 = utils.constEvalFuncWrapper(
        const_304, util_create_list_375, _CONST_EVAL_CACHE, const_305
    )
    utils_constEvalFuncWrapper_151_0 = utils_constEvalFuncWrapper_151[0]
    const_306 = main_const_eval_153
    util_create_list_376 = [input[262], input[433], input[435]]
    const_307 = "main_const_eval_153"
    utils_constEvalFuncWrapper_152 = utils.constEvalFuncWrapper(
        const_306, util_create_list_376, _CONST_EVAL_CACHE, const_307
    )
    utils_constEvalFuncWrapper_152_0 = utils_constEvalFuncWrapper_152[0]
    const_308 = main_const_eval_154
    util_create_list_377 = [input[289]]
    const_309 = "main_const_eval_154"
    utils_constEvalFuncWrapper_153 = utils.constEvalFuncWrapper(
        const_308, util_create_list_377, _CONST_EVAL_CACHE, const_309
    )
    utils_constEvalFuncWrapper_153_0 = utils_constEvalFuncWrapper_153[0]
    const_310 = main_const_eval_155
    util_create_list_378 = [input[195]]
    const_311 = "main_const_eval_155"
    utils_constEvalFuncWrapper_154 = utils.constEvalFuncWrapper(
        const_310, util_create_list_378, _CONST_EVAL_CACHE, const_311
    )
    utils_constEvalFuncWrapper_154_0 = utils_constEvalFuncWrapper_154[0]
    const_312 = main_const_eval_156
    util_create_list_379 = [input[267]]
    const_313 = "main_const_eval_156"
    utils_constEvalFuncWrapper_155 = utils.constEvalFuncWrapper(
        const_312, util_create_list_379, _CONST_EVAL_CACHE, const_313
    )
    utils_constEvalFuncWrapper_155_0 = utils_constEvalFuncWrapper_155[0]
    const_314 = main_const_eval_157
    util_create_list_380 = [input[243]]
    const_315 = "main_const_eval_157"
    utils_constEvalFuncWrapper_156 = utils.constEvalFuncWrapper(
        const_314, util_create_list_380, _CONST_EVAL_CACHE, const_315
    )
    utils_constEvalFuncWrapper_156_0 = utils_constEvalFuncWrapper_156[0]
    const_316 = main_const_eval_158
    util_create_list_381 = [input[370], input[397], input[399]]
    const_317 = "main_const_eval_158"
    utils_constEvalFuncWrapper_157 = utils.constEvalFuncWrapper(
        const_316, util_create_list_381, _CONST_EVAL_CACHE, const_317
    )
    utils_constEvalFuncWrapper_157_0 = utils_constEvalFuncWrapper_157[0]
    const_318 = main_const_eval_159
    util_create_list_382 = [input[321], input[412], input[414]]
    const_319 = "main_const_eval_159"
    utils_constEvalFuncWrapper_158 = utils.constEvalFuncWrapper(
        const_318, util_create_list_382, _CONST_EVAL_CACHE, const_319
    )
    utils_constEvalFuncWrapper_158_0 = utils_constEvalFuncWrapper_158[0]
    const_320 = main_const_eval_160
    util_create_list_383 = [input[58], input[501], input[503]]
    const_321 = "main_const_eval_160"
    utils_constEvalFuncWrapper_159 = utils.constEvalFuncWrapper(
        const_320, util_create_list_383, _CONST_EVAL_CACHE, const_321
    )
    utils_constEvalFuncWrapper_159_0 = utils_constEvalFuncWrapper_159[0]
    const_322 = main_const_eval_161
    util_create_list_384 = [input[121]]
    const_323 = "main_const_eval_161"
    utils_constEvalFuncWrapper_160 = utils.constEvalFuncWrapper(
        const_322, util_create_list_384, _CONST_EVAL_CACHE, const_323
    )
    utils_constEvalFuncWrapper_160_0 = utils_constEvalFuncWrapper_160[0]
    CLIPVisionEmbeddings_0_0_0 = CLIPVisionEmbeddings_0_0(
        input[390],
        utils_constEvalFuncWrapper_142_0,
        utils_constEvalFuncWrapper_88_0,
        utils_constEvalFuncWrapper_66_0,
    )
    LayerNorm_1_0_0 = LayerNorm_1_0(input[385], input[386], CLIPVisionEmbeddings_0_0_0)
    CLIPEncoderLayer_2_0_0 = CLIPEncoderLayer_2_0(
        input[383], LayerNorm_1_0_0, input[384]
    )
    CLIPAttention_3_0_0 = CLIPAttention_3_0(
        input[380],
        utils_constEvalFuncWrapper_47_0,
        utils_constEvalFuncWrapper_124_0,
        utils_constEvalFuncWrapper_70_0,
        CLIPEncoderLayer_2_0_0,
    )
    v_163, v_164 = CLIPEncoderLayer_4_0(
        input[377], input[378], LayerNorm_1_0_0, CLIPAttention_3_0_0
    )
    CLIPMLP_5_0_0 = CLIPMLP_5_0(
        v_163,
        utils_constEvalFuncWrapper_73_0,
        utils_constEvalFuncWrapper_42_0,
        input[374],
        input[376],
    )
    v_165, v_166 = CLIPEncoderLayer_6_0(CLIPMLP_5_0_0, input[372], v_164, input[371])
    CLIPAttention_7_0_0 = CLIPAttention_7_0(
        utils_constEvalFuncWrapper_62_0,
        utils_constEvalFuncWrapper_55_0,
        v_166,
        input[368],
        utils_constEvalFuncWrapper_157_0,
    )
    v_167, v_168 = CLIPEncoderLayer_8_0(
        input[366], v_165, input[365], CLIPAttention_7_0_0
    )
    CLIPMLP_9_0_0 = CLIPMLP_9_0(
        utils_constEvalFuncWrapper_13_0,
        input[362],
        utils_constEvalFuncWrapper_21_0,
        input[364],
        v_168,
    )
    v_169, v_170 = CLIPEncoderLayer_10_0(v_167, input[360], input[359], CLIPMLP_9_0_0)
    CLIPAttention_11_0_0 = CLIPAttention_11_0(
        utils_constEvalFuncWrapper_80_0,
        utils_constEvalFuncWrapper_25_0,
        v_170,
        utils_constEvalFuncWrapper_122_0,
        input[356],
    )
    v_171, v_172 = CLIPEncoderLayer_12_0(
        v_169, input[353], CLIPAttention_11_0_0, input[354]
    )
    CLIPMLP_13_0_0 = CLIPMLP_13_0(
        utils_constEvalFuncWrapper_81_0,
        input[352],
        input[350],
        v_172,
        utils_constEvalFuncWrapper_146_0,
    )
    v_173, v_174 = CLIPEncoderLayer_14_0(input[347], CLIPMLP_13_0_0, v_171, input[348])
    CLIPAttention_15_0_0 = CLIPAttention_15_0(
        utils_constEvalFuncWrapper_90_0,
        input[344],
        utils_constEvalFuncWrapper_132_0,
        v_174,
        utils_constEvalFuncWrapper_26_0,
    )
    v_175, v_176 = CLIPEncoderLayer_16_0(
        CLIPAttention_15_0_0, v_173, input[342], input[341]
    )
    CLIPMLP_17_0_0 = CLIPMLP_17_0(
        v_175,
        utils_constEvalFuncWrapper_145_0,
        input[338],
        input[340],
        utils_constEvalFuncWrapper_10_0,
    )
    v_177, v_178 = CLIPEncoderLayer_18_0(CLIPMLP_17_0_0, input[335], v_176, input[336])
    CLIPAttention_19_0_0 = CLIPAttention_19_0(
        utils_constEvalFuncWrapper_97_0,
        v_177,
        utils_constEvalFuncWrapper_43_0,
        utils_constEvalFuncWrapper_127_0,
        input[332],
    )
    v_179, v_180 = CLIPEncoderLayer_20_0(
        input[329], v_178, CLIPAttention_19_0_0, input[330]
    )
    CLIPMLP_21_0_0 = CLIPMLP_21_0(
        utils_constEvalFuncWrapper_150_0,
        input[326],
        input[328],
        v_179,
        utils_constEvalFuncWrapper_149_0,
    )
    v_181, v_182 = CLIPEncoderLayer_22_0(CLIPMLP_21_0_0, input[324], v_180, input[323])
    CLIPAttention_23_0_0 = CLIPAttention_23_0(
        input[320],
        utils_constEvalFuncWrapper_158_0,
        utils_constEvalFuncWrapper_69_0,
        v_182,
        utils_constEvalFuncWrapper_96_0,
    )
    v_183, v_184 = CLIPEncoderLayer_24_0(
        input[318], CLIPAttention_23_0_0, v_181, input[317]
    )
    CLIPMLP_25_0_0 = CLIPMLP_25_0(
        utils_constEvalFuncWrapper_91_0,
        v_183,
        utils_constEvalFuncWrapper_106_0,
        input[316],
        input[314],
    )
    v_185, v_186 = CLIPEncoderLayer_26_0(input[312], input[311], CLIPMLP_25_0_0, v_184)
    CLIPAttention_27_0_0 = CLIPAttention_27_0(
        utils_constEvalFuncWrapper_99_0,
        v_185,
        utils_constEvalFuncWrapper_46_0,
        utils_constEvalFuncWrapper_128_0,
        input[308],
    )
    v_187, v_188 = CLIPEncoderLayer_28_0(
        CLIPAttention_27_0_0, input[305], input[306], v_186
    )
    CLIPMLP_29_0_0 = CLIPMLP_29_0(
        utils_constEvalFuncWrapper_53_0,
        input[304],
        input[302],
        v_188,
        utils_constEvalFuncWrapper_103_0,
    )
    v_189, v_190 = CLIPEncoderLayer_30_0(CLIPMLP_29_0_0, v_187, input[299], input[300])
    CLIPAttention_31_0_0 = CLIPAttention_31_0(
        utils_constEvalFuncWrapper_120_0,
        utils_constEvalFuncWrapper_84_0,
        v_189,
        input[296],
        utils_constEvalFuncWrapper_49_0,
    )
    v_191, v_192 = CLIPEncoderLayer_32_0(
        input[293], input[294], CLIPAttention_31_0_0, v_190
    )
    CLIPMLP_33_0_0 = CLIPMLP_33_0(
        utils_constEvalFuncWrapper_153_0,
        utils_constEvalFuncWrapper_27_0,
        v_192,
        input[290],
        input[292],
    )
    v_193, v_194 = CLIPEncoderLayer_34_0(input[288], input[287], v_191, CLIPMLP_33_0_0)
    CLIPAttention_35_0_0 = CLIPAttention_35_0(
        v_193,
        utils_constEvalFuncWrapper_40_0,
        input[284],
        utils_constEvalFuncWrapper_74_0,
        utils_constEvalFuncWrapper_29_0,
    )
    v_195, v_196 = CLIPEncoderLayer_36_0(
        CLIPAttention_35_0_0, input[281], input[282], v_194
    )
    CLIPMLP_37_0_0 = CLIPMLP_37_0(
        utils_constEvalFuncWrapper_24_0,
        input[278],
        input[280],
        utils_constEvalFuncWrapper_93_0,
        v_195,
    )
    v_197, v_198 = CLIPEncoderLayer_38_0(input[276], input[275], CLIPMLP_37_0_0, v_196)
    CLIPAttention_39_0_0 = CLIPAttention_39_0(
        v_197,
        utils_constEvalFuncWrapper_119_0,
        utils_constEvalFuncWrapper_133_0,
        utils_constEvalFuncWrapper_113_0,
        input[272],
    )
    v_199, v_200 = CLIPEncoderLayer_40_0(
        input[269], input[270], v_198, CLIPAttention_39_0_0
    )
    CLIPMLP_41_0_0 = CLIPMLP_41_0(
        utils_constEvalFuncWrapper_2_0,
        v_200,
        input[266],
        input[268],
        utils_constEvalFuncWrapper_155_0,
    )
    v_201, v_202 = CLIPEncoderLayer_42_0(input[264], CLIPMLP_41_0_0, v_199, input[263])
    CLIPAttention_43_0_0 = CLIPAttention_43_0(
        input[260],
        utils_constEvalFuncWrapper_152_0,
        utils_constEvalFuncWrapper_64_0,
        v_202,
        utils_constEvalFuncWrapper_71_0,
    )
    v_203, v_204 = CLIPEncoderLayer_44_0(
        CLIPAttention_43_0_0, input[258], v_201, input[257]
    )
    CLIPMLP_45_0_0 = CLIPMLP_45_0(
        input[256],
        utils_constEvalFuncWrapper_95_0,
        utils_constEvalFuncWrapper_85_0,
        v_204,
        input[254],
    )
    v_205, v_206 = CLIPEncoderLayer_46_0(v_203, input[252], input[251], CLIPMLP_45_0_0)
    CLIPAttention_47_0_0 = CLIPAttention_47_0(
        utils_constEvalFuncWrapper_67_0,
        utils_constEvalFuncWrapper_140_0,
        v_205,
        utils_constEvalFuncWrapper_116_0,
        input[248],
    )
    v_207, v_208 = CLIPEncoderLayer_48_0(
        input[245], input[246], CLIPAttention_47_0_0, v_206
    )
    CLIPMLP_49_0_0 = CLIPMLP_49_0(
        v_207,
        utils_constEvalFuncWrapper_156_0,
        utils_constEvalFuncWrapper_151_0,
        input[244],
        input[242],
    )
    v_209, v_210 = CLIPEncoderLayer_50_0(input[240], v_208, CLIPMLP_49_0_0, input[239])
    CLIPAttention_51_0_0 = CLIPAttention_51_0(
        input[236],
        utils_constEvalFuncWrapper_87_0,
        v_209,
        utils_constEvalFuncWrapper_136_0,
        utils_constEvalFuncWrapper_68_0,
    )
    v_211, v_212 = CLIPEncoderLayer_52_0(
        CLIPAttention_51_0_0, input[233], v_210, input[234]
    )
    CLIPMLP_53_0_0 = CLIPMLP_53_0(
        input[232],
        utils_constEvalFuncWrapper_5_0,
        utils_constEvalFuncWrapper_15_0,
        v_211,
        input[230],
    )
    v_213, v_214 = CLIPEncoderLayer_54_0(input[227], CLIPMLP_53_0_0, input[228], v_212)
    CLIPAttention_55_0_0 = CLIPAttention_55_0(
        input[224],
        utils_constEvalFuncWrapper_1_0,
        utils_constEvalFuncWrapper_126_0,
        utils_constEvalFuncWrapper_102_0,
        v_214,
    )
    v_215, v_216 = CLIPEncoderLayer_56_0(
        CLIPAttention_55_0_0, input[221], v_213, input[222]
    )
    CLIPMLP_57_0_0 = CLIPMLP_57_0(
        utils_constEvalFuncWrapper_92_0,
        utils_constEvalFuncWrapper_109_0,
        input[218],
        input[220],
        v_216,
    )
    v_217, v_218 = CLIPEncoderLayer_58_0(input[215], v_215, input[216], CLIPMLP_57_0_0)
    CLIPAttention_59_0_0 = CLIPAttention_59_0(
        v_217,
        utils_constEvalFuncWrapper_86_0,
        input[212],
        utils_constEvalFuncWrapper_101_0,
        utils_constEvalFuncWrapper_11_0,
    )
    v_219, v_220 = CLIPEncoderLayer_60_0(
        input[209], input[210], v_218, CLIPAttention_59_0_0
    )
    CLIPMLP_61_0_0 = CLIPMLP_61_0(
        input[206],
        input[208],
        utils_constEvalFuncWrapper_18_0,
        v_220,
        utils_constEvalFuncWrapper_141_0,
    )
    v_221, v_222 = CLIPEncoderLayer_62_0(CLIPMLP_61_0_0, input[203], input[204], v_219)
    CLIPAttention_63_0_0 = CLIPAttention_63_0(
        v_221,
        utils_constEvalFuncWrapper_72_0,
        utils_constEvalFuncWrapper_114_0,
        input[200],
        utils_constEvalFuncWrapper_23_0,
    )
    v_223, v_224 = CLIPEncoderLayer_64_0(
        input[197], CLIPAttention_63_0_0, input[198], v_222
    )
    CLIPMLP_65_0_0 = CLIPMLP_65_0(
        input[194],
        utils_constEvalFuncWrapper_83_0,
        utils_constEvalFuncWrapper_154_0,
        input[196],
        v_224,
    )
    v_225, v_226 = CLIPEncoderLayer_66_0(input[191], v_223, CLIPMLP_65_0_0, input[192])
    CLIPAttention_67_0_0 = CLIPAttention_67_0(
        utils_constEvalFuncWrapper_118_0,
        v_226,
        utils_constEvalFuncWrapper_89_0,
        input[188],
        utils_constEvalFuncWrapper_63_0,
    )
    v_227, v_228 = CLIPEncoderLayer_68_0(
        v_225, CLIPAttention_67_0_0, input[185], input[186]
    )
    CLIPMLP_69_0_0 = CLIPMLP_69_0(
        input[184],
        v_228,
        input[182],
        utils_constEvalFuncWrapper_130_0,
        utils_constEvalFuncWrapper_104_0,
    )
    v_229, v_230 = CLIPEncoderLayer_70_0(v_227, CLIPMLP_69_0_0, input[180], input[179])
    CLIPAttention_71_0_0 = CLIPAttention_71_0(
        utils_constEvalFuncWrapper_34_0,
        utils_constEvalFuncWrapper_7_0,
        v_230,
        input[176],
        utils_constEvalFuncWrapper_17_0,
    )
    v_231, v_232 = CLIPEncoderLayer_72_0(
        input[174], input[173], CLIPAttention_71_0_0, v_229
    )
    CLIPMLP_73_0_0 = CLIPMLP_73_0(
        v_231,
        utils_constEvalFuncWrapper_108_0,
        utils_constEvalFuncWrapper_19_0,
        input[172],
        input[170],
    )
    v_233, v_234 = CLIPEncoderLayer_74_0(input[168], input[167], v_232, CLIPMLP_73_0_0)
    CLIPAttention_75_0_0 = CLIPAttention_75_0(
        utils_constEvalFuncWrapper_134_0,
        input[164],
        utils_constEvalFuncWrapper_112_0,
        v_233,
        utils_constEvalFuncWrapper_100_0,
    )
    v_235, v_236 = CLIPEncoderLayer_76_0(
        input[161], CLIPAttention_75_0_0, v_234, input[162]
    )
    CLIPMLP_77_0_0 = CLIPMLP_77_0(
        v_235,
        input[160],
        input[158],
        utils_constEvalFuncWrapper_94_0,
        utils_constEvalFuncWrapper_147_0,
    )
    v_237, v_238 = CLIPEncoderLayer_78_0(input[156], CLIPMLP_77_0_0, v_236, input[155])
    CLIPAttention_79_0_0 = CLIPAttention_79_0(
        input[152],
        v_237,
        utils_constEvalFuncWrapper_12_0,
        utils_constEvalFuncWrapper_50_0,
        utils_constEvalFuncWrapper_52_0,
    )
    v_239, v_240 = CLIPEncoderLayer_80_0(
        input[150], input[149], v_238, CLIPAttention_79_0_0
    )
    CLIPMLP_81_0_0 = CLIPMLP_81_0(
        v_240,
        utils_constEvalFuncWrapper_44_0,
        input[146],
        input[148],
        utils_constEvalFuncWrapper_28_0,
    )
    v_241, v_242 = CLIPEncoderLayer_82_0(v_239, input[144], input[143], CLIPMLP_81_0_0)
    CLIPAttention_83_0_0 = CLIPAttention_83_0(
        utils_constEvalFuncWrapper_78_0,
        utils_constEvalFuncWrapper_60_0,
        input[140],
        utils_constEvalFuncWrapper_65_0,
        v_242,
    )
    v_243, v_244 = CLIPEncoderLayer_84_0(
        CLIPAttention_83_0_0, input[137], v_241, input[138]
    )
    CLIPMLP_85_0_0 = CLIPMLP_85_0(
        utils_constEvalFuncWrapper_82_0,
        input[134],
        input[136],
        utils_constEvalFuncWrapper_107_0,
        v_244,
    )
    v_245, v_246 = CLIPEncoderLayer_86_0(v_243, CLIPMLP_85_0_0, input[132], input[131])
    CLIPAttention_87_0_0 = CLIPAttention_87_0(
        input[128],
        utils_constEvalFuncWrapper_37_0,
        utils_constEvalFuncWrapper_20_0,
        v_246,
        utils_constEvalFuncWrapper_111_0,
    )
    v_247, v_248 = CLIPEncoderLayer_88_0(
        CLIPAttention_87_0_0, v_245, input[125], input[126]
    )
    CLIPMLP_89_0_0 = CLIPMLP_89_0(
        input[122],
        utils_constEvalFuncWrapper_110_0,
        v_247,
        input[124],
        utils_constEvalFuncWrapper_160_0,
    )
    v_249, v_250 = CLIPEncoderLayer_90_0(input[120], CLIPMLP_89_0_0, v_248, input[119])
    CLIPAttention_91_0_0 = CLIPAttention_91_0(
        v_249,
        utils_constEvalFuncWrapper_148_0,
        input[116],
        utils_constEvalFuncWrapper_57_0,
        utils_constEvalFuncWrapper_33_0,
    )
    v_251, v_252 = CLIPEncoderLayer_92_0(
        CLIPAttention_91_0_0, input[113], v_250, input[114]
    )
    CLIPMLP_93_0_0 = CLIPMLP_93_0(
        input[112],
        utils_constEvalFuncWrapper_125_0,
        input[110],
        v_252,
        utils_constEvalFuncWrapper_4_0,
    )
    v_253, v_254 = CLIPEncoderLayer_94_0(input[108], input[107], v_251, CLIPMLP_93_0_0)
    CLIPAttention_95_0_0 = CLIPAttention_95_0(
        input[104],
        v_254,
        utils_constEvalFuncWrapper_36_0,
        utils_constEvalFuncWrapper_32_0,
        utils_constEvalFuncWrapper_51_0,
    )
    v_255, v_256 = CLIPEncoderLayer_96_0(
        input[102], CLIPAttention_95_0_0, v_253, input[101]
    )
    CLIPMLP_97_0_0 = CLIPMLP_97_0(
        input[98],
        utils_constEvalFuncWrapper_0_0,
        input[100],
        utils_constEvalFuncWrapper_22_0,
        v_256,
    )
    v_257, v_258 = CLIPEncoderLayer_98_0(v_255, input[95], CLIPMLP_97_0_0, input[96])
    CLIPAttention_99_0_0 = CLIPAttention_99_0(
        utils_constEvalFuncWrapper_76_0,
        v_258,
        utils_constEvalFuncWrapper_143_0,
        utils_constEvalFuncWrapper_59_0,
        input[92],
    )
    v_259, v_260 = CLIPEncoderLayer_100_0(
        v_257, CLIPAttention_99_0_0, input[89], input[90]
    )
    CLIPMLP_101_0_0 = CLIPMLP_101_0(
        utils_constEvalFuncWrapper_144_0,
        utils_constEvalFuncWrapper_139_0,
        input[88],
        input[86],
        v_260,
    )
    v_261, v_262 = CLIPEncoderLayer_102_0(CLIPMLP_101_0_0, v_259, input[83], input[84])
    CLIPAttention_103_0_0 = CLIPAttention_103_0(
        input[80],
        utils_constEvalFuncWrapper_61_0,
        utils_constEvalFuncWrapper_31_0,
        utils_constEvalFuncWrapper_58_0,
        v_261,
    )
    v_263, v_264 = CLIPEncoderLayer_104_0(
        CLIPAttention_103_0_0, input[78], input[77], v_262
    )
    CLIPMLP_105_0_0 = CLIPMLP_105_0(
        input[76],
        utils_constEvalFuncWrapper_117_0,
        input[74],
        utils_constEvalFuncWrapper_39_0,
        v_264,
    )
    v_265, v_266 = CLIPEncoderLayer_106_0(v_263, CLIPMLP_105_0_0, input[72], input[71])
    CLIPAttention_107_0_0 = CLIPAttention_107_0(
        utils_constEvalFuncWrapper_77_0,
        v_265,
        input[68],
        utils_constEvalFuncWrapper_105_0,
        utils_constEvalFuncWrapper_9_0,
    )
    v_267, v_268 = CLIPEncoderLayer_108_0(
        CLIPAttention_107_0_0, input[66], input[65], v_266
    )
    CLIPMLP_109_0_0 = CLIPMLP_109_0(
        input[62],
        v_267,
        utils_constEvalFuncWrapper_123_0,
        input[64],
        utils_constEvalFuncWrapper_98_0,
    )
    v_269, v_270 = CLIPEncoderLayer_110_0(input[59], input[60], CLIPMLP_109_0_0, v_268)
    CLIPAttention_111_0_0 = CLIPAttention_111_0(
        v_270,
        input[56],
        utils_constEvalFuncWrapper_159_0,
        utils_constEvalFuncWrapper_8_0,
        utils_constEvalFuncWrapper_41_0,
    )
    v_271, v_272 = CLIPEncoderLayer_112_0(
        input[54], v_269, CLIPAttention_111_0_0, input[53]
    )
    CLIPMLP_113_0_0 = CLIPMLP_113_0(
        v_271,
        input[50],
        input[52],
        utils_constEvalFuncWrapper_129_0,
        utils_constEvalFuncWrapper_115_0,
    )
    v_273, v_274 = CLIPEncoderLayer_114_0(input[47], CLIPMLP_113_0_0, v_272, input[48])
    CLIPAttention_115_0_0 = CLIPAttention_115_0(
        v_274,
        input[44],
        utils_constEvalFuncWrapper_16_0,
        utils_constEvalFuncWrapper_3_0,
        utils_constEvalFuncWrapper_121_0,
    )
    v_275, v_276 = CLIPEncoderLayer_116_0(
        v_273, input[41], input[42], CLIPAttention_115_0_0
    )
    CLIPMLP_117_0_0 = CLIPMLP_117_0(
        input[38],
        utils_constEvalFuncWrapper_56_0,
        v_275,
        utils_constEvalFuncWrapper_14_0,
        input[40],
    )
    v_277, v_278 = CLIPEncoderLayer_118_0(input[36], CLIPMLP_117_0_0, input[35], v_276)
    CLIPAttention_119_0_0 = CLIPAttention_119_0(
        utils_constEvalFuncWrapper_45_0,
        utils_constEvalFuncWrapper_79_0,
        v_278,
        input[32],
        utils_constEvalFuncWrapper_75_0,
    )
    v_279, v_280 = CLIPEncoderLayer_120_0(
        v_277, input[29], input[30], CLIPAttention_119_0_0
    )
    CLIPMLP_121_0_0 = CLIPMLP_121_0(
        utils_constEvalFuncWrapper_38_0,
        utils_constEvalFuncWrapper_35_0,
        input[26],
        v_280,
        input[28],
    )
    v_281, v_282 = CLIPEncoderLayer_122_0(input[23], input[24], v_279, CLIPMLP_121_0_0)
    CLIPAttention_123_0_0 = CLIPAttention_123_0(
        input[20],
        utils_constEvalFuncWrapper_48_0,
        v_281,
        utils_constEvalFuncWrapper_138_0,
        utils_constEvalFuncWrapper_131_0,
    )
    v_283, v_284 = CLIPEncoderLayer_124_0(
        CLIPAttention_123_0_0, v_282, input[18], input[17]
    )
    CLIPMLP_125_0_0 = CLIPMLP_125_0(
        v_284,
        input[14],
        input[16],
        utils_constEvalFuncWrapper_135_0,
        utils_constEvalFuncWrapper_54_0,
    )
    CLIPEncoderLayer_126_0_0 = CLIPEncoderLayer_126_0(v_283, CLIPMLP_125_0_0)
    Linear_127_0_0 = Linear_127_0(
        utils_constEvalFuncWrapper_137_0, input[12], CLIPEncoderLayer_126_0_0
    )
    v_285, v_286, v_287, v_288 = IPAdapterPlusImageProjectionBlock_128_0(
        utils_constEvalFuncWrapper_30_0,
        input[539],
        input[551],
        Linear_127_0_0,
        input[527],
        input[550],
        input[10],
        input[526],
        input[9],
        input[538],
    )
    Attention_129_0_0 = Attention_129_0(
        v_286,
        input[5],
        input[516],
        utils_constEvalFuncWrapperZeroArg_0_0,
        input[6],
        utils_constEvalFuncWrapper_30_1,
    )
    v_289, v_290 = IPAdapterPlusImageProjectionBlock_130_0(
        input[521], Attention_129_0_0, input[4], input[520]
    )
    Linear_131_0_0 = Linear_131_0(input[519], v_290)
    Linear_132_0_0 = Linear_132_0(input[518], Linear_131_0_0)
    v_291, v_292, v_293 = IPAdapterPlusImageProjectionBlock_133_0(
        v_289, Linear_132_0_0, input[525], input[524], v_288
    )
    Attention_134_0_0 = Attention_134_0(
        input[529],
        input[528],
        input[522],
        utils_constEvalFuncWrapperZeroArg_0_0,
        input[523],
        v_292,
        v_293,
    )
    v_294, v_295 = IPAdapterPlusImageProjectionBlock_135_0(
        v_291, input[533], Attention_134_0_0, input[532]
    )
    Linear_136_0_0 = Linear_136_0(input[531], v_295)
    Linear_137_0_0 = Linear_137_0(Linear_136_0_0, input[530])
    v_296, v_297, v_298 = IPAdapterPlusImageProjectionBlock_138_0(
        v_285, input[537], v_294, Linear_137_0_0, input[536]
    )
    Attention_139_0_0 = Attention_139_0(
        input[540],
        input[535],
        input[541],
        utils_constEvalFuncWrapperZeroArg_0_0,
        input[534],
        v_296,
        v_298,
    )
    v_299, v_300 = IPAdapterPlusImageProjectionBlock_140_0(
        input[544], Attention_139_0_0, v_297, input[545]
    )
    Linear_141_0_0 = Linear_141_0(input[543], v_300)
    Linear_142_0_0 = Linear_142_0(Linear_141_0_0, input[542])
    v_301, v_302, v_303 = IPAdapterPlusImageProjectionBlock_143_0(
        Linear_142_0_0, v_287, input[548], input[549], v_299
    )
    Attention_144_0_0 = Attention_144_0(
        input[552],
        input[553],
        utils_constEvalFuncWrapperZeroArg_0_0,
        input[547],
        input[546],
        v_301,
        v_302,
    )
    v_304, v_305 = IPAdapterPlusImageProjectionBlock_145_0(
        input[556], input[557], Attention_144_0_0, v_303
    )
    v_306, v_307 = Linear_147_0(v_304, input[555])
    Linear_150_0(v_307)
    Linear_146_0_0 = Linear_146_0(v_305)
    IPAdapterPlusImageProjectionBlock_148_0_0 = IPAdapterPlusImageProjectionBlock_148_0(
        Linear_146_0_0, v_306, input[554]
    )
    IPAdapterPlusImageProjection_149_0_0 = IPAdapterPlusImageProjection_149_0(
        v_305,
        input[1],
        utils_constEvalFuncWrapper_6_0,
        IPAdapterPlusImageProjectionBlock_148_0_0,
        input[3],
        input[0],
    )
    IPAdapterPlusImageProjectionBlock_151_0(v_306)
    util_create_list_385 = [IPAdapterPlusImageProjection_149_0_0]
    return util_create_list_385


def CLIPVisionEmbeddings_0_0(input_0, input_1, input_2, input_3):
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
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


def LayerNorm_1_0(input_0, input_1, input_2):
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


def CLIPEncoderLayer_2_0(input_0, input_1, input_2):
    ttnn_layer_norm_2 = ttnn.layer_norm(
        input_1,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_2


def CLIPAttention_3_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_195 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_195,
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
    ttnn_add_1 = ttnn.add(
        ttnn_matmul_1,
        input_1,
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
    ttnn_add_2 = ttnn.add(
        ttnn_matmul_2,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_2


def CLIPEncoderLayer_4_0(input_0, input_1, input_2, input_3):
    ttnn_add_3 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_3 = ttnn.layer_norm(
        ttnn_add_3,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_3, ttnn_add_3


def CLIPMLP_5_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_200 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_reshape_200,
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
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_3,
        input_2,
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
    ttnn_add_5 = ttnn.add(
        ttnn_matmul_4,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_5


def CLIPEncoderLayer_6_0(input_0, input_1, input_2, input_3):
    ttnn_add_6 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_4 = ttnn.layer_norm(
        ttnn_add_6,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_6, ttnn_layer_norm_4


def CLIPAttention_7_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_202 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_reshape_202,
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
    ttnn_add_7 = ttnn.add(
        ttnn_matmul_5,
        input_0,
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
    ttnn_add_8 = ttnn.add(
        ttnn_matmul_6,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_8


def CLIPEncoderLayer_8_0(input_0, input_1, input_2, input_3):
    ttnn_add_9 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_5 = ttnn.layer_norm(
        ttnn_add_9,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_9, ttnn_layer_norm_5


def CLIPMLP_9_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_207 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_207,
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
    ttnn_add_10 = ttnn.add(
        ttnn_matmul_7,
        input_0,
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
    ttnn_add_11 = ttnn.add(
        ttnn_matmul_8,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_11


def CLIPEncoderLayer_10_0(input_0, input_1, input_2, input_3):
    ttnn_add_12 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_6 = ttnn.layer_norm(
        ttnn_add_12,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_12, ttnn_layer_norm_6


def CLIPAttention_11_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_209 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_209,
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
    ttnn_add_13 = ttnn.add(
        ttnn_matmul_9,
        input_0,
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
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_10,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_14


def CLIPEncoderLayer_12_0(input_0, input_1, input_2, input_3):
    ttnn_add_15 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_7 = ttnn.layer_norm(
        ttnn_add_15,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_15, ttnn_layer_norm_7


def CLIPMLP_13_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_214 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_reshape_214,
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
    ttnn_add_16 = ttnn.add(
        ttnn_matmul_11,
        input_0,
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
    ttnn_add_17 = ttnn.add(
        ttnn_matmul_12,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_17


def CLIPEncoderLayer_14_0(input_0, input_1, input_2, input_3):
    ttnn_add_18 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_8 = ttnn.layer_norm(
        ttnn_add_18,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_18, ttnn_layer_norm_8


def CLIPAttention_15_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_216 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_13 = ttnn.matmul(
        ttnn_reshape_216,
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
    ttnn_add_19 = ttnn.add(
        ttnn_matmul_13,
        input_0,
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
    ttnn_add_20 = ttnn.add(
        ttnn_matmul_14,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_20


def CLIPEncoderLayer_16_0(input_0, input_1, input_2, input_3):
    ttnn_add_21 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_9 = ttnn.layer_norm(
        ttnn_add_21,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_9, ttnn_add_21


def CLIPMLP_17_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_221 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_reshape_221,
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
    ttnn_add_22 = ttnn.add(
        ttnn_matmul_15,
        input_1,
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
    ttnn_add_23 = ttnn.add(
        ttnn_matmul_16,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_23


def CLIPEncoderLayer_18_0(input_0, input_1, input_2, input_3):
    ttnn_add_24 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_10 = ttnn.layer_norm(
        ttnn_add_24,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_10, ttnn_add_24


def CLIPAttention_19_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_223 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_reshape_223,
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
    ttnn_add_25 = ttnn.add(
        ttnn_matmul_17,
        input_2,
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
    ttnn_add_26 = ttnn.add(
        ttnn_matmul_18,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_26


def CLIPEncoderLayer_20_0(input_0, input_1, input_2, input_3):
    ttnn_add_27 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_11 = ttnn.layer_norm(
        ttnn_add_27,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_11, ttnn_add_27


def CLIPMLP_21_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_228 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_19 = ttnn.matmul(
        ttnn_reshape_228,
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
    ttnn_add_28 = ttnn.add(
        ttnn_matmul_19,
        input_0,
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
    ttnn_add_29 = ttnn.add(
        ttnn_matmul_20,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_29


def CLIPEncoderLayer_22_0(input_0, input_1, input_2, input_3):
    ttnn_add_30 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_12 = ttnn.layer_norm(
        ttnn_add_30,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_30, ttnn_layer_norm_12


def CLIPAttention_23_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_230 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_21 = ttnn.matmul(
        ttnn_reshape_230,
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
    ttnn_add_31 = ttnn.add(
        ttnn_matmul_21,
        input_1,
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
    ttnn_add_32 = ttnn.add(
        ttnn_matmul_22,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_32


def CLIPEncoderLayer_24_0(input_0, input_1, input_2, input_3):
    ttnn_add_33 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_13 = ttnn.layer_norm(
        ttnn_add_33,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_13, ttnn_add_33


def CLIPMLP_25_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_235 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_23 = ttnn.matmul(
        ttnn_reshape_235,
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
    ttnn_add_34 = ttnn.add(
        ttnn_matmul_23,
        input_0,
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
    ttnn_add_35 = ttnn.add(
        ttnn_matmul_24,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_35


def CLIPEncoderLayer_26_0(input_0, input_1, input_2, input_3):
    ttnn_add_36 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_14 = ttnn.layer_norm(
        ttnn_add_36,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_14, ttnn_add_36


def CLIPAttention_27_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_237 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_reshape_237,
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
    ttnn_add_37 = ttnn.add(
        ttnn_matmul_25,
        input_0,
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
    ttnn_add_38 = ttnn.add(
        ttnn_matmul_26,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_38


def CLIPEncoderLayer_28_0(input_0, input_1, input_2, input_3):
    ttnn_add_39 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_15 = ttnn.layer_norm(
        ttnn_add_39,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_39, ttnn_layer_norm_15


def CLIPMLP_29_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_242 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_reshape_242,
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
    ttnn_add_40 = ttnn.add(
        ttnn_matmul_27,
        input_0,
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
    ttnn_add_41 = ttnn.add(
        ttnn_matmul_28,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_41


def CLIPEncoderLayer_30_0(input_0, input_1, input_2, input_3):
    ttnn_add_42 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_16 = ttnn.layer_norm(
        ttnn_add_42,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_16, ttnn_add_42


def CLIPAttention_31_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_244 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_29 = ttnn.matmul(
        ttnn_reshape_244,
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
    ttnn_add_43 = ttnn.add(
        ttnn_matmul_29,
        input_1,
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
    ttnn_add_44 = ttnn.add(
        ttnn_matmul_30,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_44


def CLIPEncoderLayer_32_0(input_0, input_1, input_2, input_3):
    ttnn_add_45 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_17 = ttnn.layer_norm(
        ttnn_add_45,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_45, ttnn_layer_norm_17


def CLIPMLP_33_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_249 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_reshape_249,
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
    ttnn_add_46 = ttnn.add(
        ttnn_matmul_31,
        input_1,
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
    ttnn_add_47 = ttnn.add(
        ttnn_matmul_32,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_47


def CLIPEncoderLayer_34_0(input_0, input_1, input_2, input_3):
    ttnn_add_48 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_18 = ttnn.layer_norm(
        ttnn_add_48,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_18, ttnn_add_48


def CLIPAttention_35_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_251 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_reshape_251,
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
    ttnn_add_49 = ttnn.add(
        ttnn_matmul_33,
        input_1,
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
    ttnn_add_50 = ttnn.add(
        ttnn_matmul_34,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_50


def CLIPEncoderLayer_36_0(input_0, input_1, input_2, input_3):
    ttnn_add_51 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_19 = ttnn.layer_norm(
        ttnn_add_51,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_19, ttnn_add_51


def CLIPMLP_37_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_256 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_35 = ttnn.matmul(
        ttnn_reshape_256,
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
    ttnn_add_52 = ttnn.add(
        ttnn_matmul_35,
        input_0,
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
    ttnn_add_53 = ttnn.add(
        ttnn_matmul_36,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_53


def CLIPEncoderLayer_38_0(input_0, input_1, input_2, input_3):
    ttnn_add_54 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_20 = ttnn.layer_norm(
        ttnn_add_54,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_20, ttnn_add_54


def CLIPAttention_39_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_258 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_reshape_258,
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
    ttnn_add_55 = ttnn.add(
        ttnn_matmul_37,
        input_2,
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
    ttnn_add_56 = ttnn.add(
        ttnn_matmul_38,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_56


def CLIPEncoderLayer_40_0(input_0, input_1, input_2, input_3):
    ttnn_add_57 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_21 = ttnn.layer_norm(
        ttnn_add_57,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_57, ttnn_layer_norm_21


def CLIPMLP_41_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_263 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_39 = ttnn.matmul(
        ttnn_reshape_263,
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
    ttnn_add_58 = ttnn.add(
        ttnn_matmul_39,
        input_4,
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
    ttnn_add_59 = ttnn.add(
        ttnn_matmul_40,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_59


def CLIPEncoderLayer_42_0(input_0, input_1, input_2, input_3):
    ttnn_add_60 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_22 = ttnn.layer_norm(
        ttnn_add_60,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_60, ttnn_layer_norm_22


def CLIPAttention_43_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_265 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_41 = ttnn.matmul(
        ttnn_reshape_265,
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
    ttnn_add_61 = ttnn.add(
        ttnn_matmul_41,
        input_4,
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
    ttnn_add_62 = ttnn.add(
        ttnn_matmul_42,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_62


def CLIPEncoderLayer_44_0(input_0, input_1, input_2, input_3):
    ttnn_add_63 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_23 = ttnn.layer_norm(
        ttnn_add_63,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_63, ttnn_layer_norm_23


def CLIPMLP_45_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_270 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_43 = ttnn.matmul(
        ttnn_reshape_270,
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
    ttnn_add_64 = ttnn.add(
        ttnn_matmul_43,
        input_1,
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
    ttnn_add_65 = ttnn.add(
        ttnn_matmul_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_65


def CLIPEncoderLayer_46_0(input_0, input_1, input_2, input_3):
    ttnn_add_66 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_24 = ttnn.layer_norm(
        ttnn_add_66,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_24, ttnn_add_66


def CLIPAttention_47_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_272 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_45 = ttnn.matmul(
        ttnn_reshape_272,
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
    ttnn_add_67 = ttnn.add(
        ttnn_matmul_45,
        input_3,
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
    ttnn_add_68 = ttnn.add(
        ttnn_matmul_46,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_68


def CLIPEncoderLayer_48_0(input_0, input_1, input_2, input_3):
    ttnn_add_69 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_25 = ttnn.layer_norm(
        ttnn_add_69,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_25, ttnn_add_69


def CLIPMLP_49_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_277 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_47 = ttnn.matmul(
        ttnn_reshape_277,
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
    ttnn_add_70 = ttnn.add(
        ttnn_matmul_47,
        input_1,
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
    ttnn_add_71 = ttnn.add(
        ttnn_matmul_48,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_71


def CLIPEncoderLayer_50_0(input_0, input_1, input_2, input_3):
    ttnn_add_72 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_26 = ttnn.layer_norm(
        ttnn_add_72,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_26, ttnn_add_72


def CLIPAttention_51_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_279 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_49 = ttnn.matmul(
        ttnn_reshape_279,
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
    ttnn_add_73 = ttnn.add(
        ttnn_matmul_49,
        input_3,
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
    ttnn_add_74 = ttnn.add(
        ttnn_matmul_50,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_74


def CLIPEncoderLayer_52_0(input_0, input_1, input_2, input_3):
    ttnn_add_75 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_27 = ttnn.layer_norm(
        ttnn_add_75,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_27, ttnn_add_75


def CLIPMLP_53_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_284 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_51 = ttnn.matmul(
        ttnn_reshape_284,
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
    ttnn_add_76 = ttnn.add(
        ttnn_matmul_51,
        input_1,
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
    ttnn_add_77 = ttnn.add(
        ttnn_matmul_52,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_77


def CLIPEncoderLayer_54_0(input_0, input_1, input_2, input_3):
    ttnn_add_78 = ttnn.add(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_28 = ttnn.layer_norm(
        ttnn_add_78,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_78, ttnn_layer_norm_28


def CLIPAttention_55_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_286 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_53 = ttnn.matmul(
        ttnn_reshape_286,
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
    ttnn_add_79 = ttnn.add(
        ttnn_matmul_53,
        input_1,
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
    ttnn_add_80 = ttnn.add(
        ttnn_matmul_54,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_80


def CLIPEncoderLayer_56_0(input_0, input_1, input_2, input_3):
    ttnn_add_81 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_29 = ttnn.layer_norm(
        ttnn_add_81,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_81, ttnn_layer_norm_29


def CLIPMLP_57_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_291 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_55 = ttnn.matmul(
        ttnn_reshape_291,
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
    ttnn_add_82 = ttnn.add(
        ttnn_matmul_55,
        input_0,
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
    ttnn_add_83 = ttnn.add(
        ttnn_matmul_56,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_83


def CLIPEncoderLayer_58_0(input_0, input_1, input_2, input_3):
    ttnn_add_84 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_30 = ttnn.layer_norm(
        ttnn_add_84,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_30, ttnn_add_84


def CLIPAttention_59_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_293 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_57 = ttnn.matmul(
        ttnn_reshape_293,
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
    ttnn_add_85 = ttnn.add(
        ttnn_matmul_57,
        input_3,
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
    ttnn_add_86 = ttnn.add(
        ttnn_matmul_58,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_86


def CLIPEncoderLayer_60_0(input_0, input_1, input_2, input_3):
    ttnn_add_87 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_31 = ttnn.layer_norm(
        ttnn_add_87,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_87, ttnn_layer_norm_31


def CLIPMLP_61_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_298 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_59 = ttnn.matmul(
        ttnn_reshape_298,
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
    ttnn_add_88 = ttnn.add(
        ttnn_matmul_59,
        input_4,
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
    ttnn_add_89 = ttnn.add(
        ttnn_matmul_60,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_89


def CLIPEncoderLayer_62_0(input_0, input_1, input_2, input_3):
    ttnn_add_90 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_32 = ttnn.layer_norm(
        ttnn_add_90,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_32, ttnn_add_90


def CLIPAttention_63_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_300 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_61 = ttnn.matmul(
        ttnn_reshape_300,
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
    ttnn_add_91 = ttnn.add(
        ttnn_matmul_61,
        input_1,
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
    ttnn_add_92 = ttnn.add(
        ttnn_matmul_62,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_92


def CLIPEncoderLayer_64_0(input_0, input_1, input_2, input_3):
    ttnn_add_93 = ttnn.add(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_33 = ttnn.layer_norm(
        ttnn_add_93,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_93, ttnn_layer_norm_33


def CLIPMLP_65_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_305 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_63 = ttnn.matmul(
        ttnn_reshape_305,
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
    ttnn_add_94 = ttnn.add(
        ttnn_matmul_63,
        input_2,
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
    ttnn_add_95 = ttnn.add(
        ttnn_matmul_64,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_95


def CLIPEncoderLayer_66_0(input_0, input_1, input_2, input_3):
    ttnn_add_96 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_34 = ttnn.layer_norm(
        ttnn_add_96,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_96, ttnn_layer_norm_34


def CLIPAttention_67_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_307 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_65 = ttnn.matmul(
        ttnn_reshape_307,
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
    ttnn_add_97 = ttnn.add(
        ttnn_matmul_65,
        input_0,
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
    ttnn_add_98 = ttnn.add(
        ttnn_matmul_66,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_98


def CLIPEncoderLayer_68_0(input_0, input_1, input_2, input_3):
    ttnn_add_99 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_35 = ttnn.layer_norm(
        ttnn_add_99,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_99, ttnn_layer_norm_35


def CLIPMLP_69_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_312 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_67 = ttnn.matmul(
        ttnn_reshape_312,
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
    ttnn_add_100 = ttnn.add(
        ttnn_matmul_67,
        input_3,
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
    ttnn_add_101 = ttnn.add(
        ttnn_matmul_68,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_101


def CLIPEncoderLayer_70_0(input_0, input_1, input_2, input_3):
    ttnn_add_102 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_36 = ttnn.layer_norm(
        ttnn_add_102,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_102, ttnn_layer_norm_36


def CLIPAttention_71_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_314 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_69 = ttnn.matmul(
        ttnn_reshape_314,
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
    ttnn_add_103 = ttnn.add(
        ttnn_matmul_69,
        input_4,
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
    ttnn_add_104 = ttnn.add(
        ttnn_matmul_70,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_104


def CLIPEncoderLayer_72_0(input_0, input_1, input_2, input_3):
    ttnn_add_105 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_37 = ttnn.layer_norm(
        ttnn_add_105,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_37, ttnn_add_105


def CLIPMLP_73_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_319 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_71 = ttnn.matmul(
        ttnn_reshape_319,
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
    ttnn_add_106 = ttnn.add(
        ttnn_matmul_71,
        input_1,
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
    ttnn_add_107 = ttnn.add(
        ttnn_matmul_72,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_107


def CLIPEncoderLayer_74_0(input_0, input_1, input_2, input_3):
    ttnn_add_108 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_38 = ttnn.layer_norm(
        ttnn_add_108,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_38, ttnn_add_108


def CLIPAttention_75_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_321 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_73 = ttnn.matmul(
        ttnn_reshape_321,
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
    ttnn_add_109 = ttnn.add(
        ttnn_matmul_73,
        input_0,
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
    ttnn_add_110 = ttnn.add(
        ttnn_matmul_74,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_110


def CLIPEncoderLayer_76_0(input_0, input_1, input_2, input_3):
    ttnn_add_111 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_39 = ttnn.layer_norm(
        ttnn_add_111,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_39, ttnn_add_111


def CLIPMLP_77_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_326 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_75 = ttnn.matmul(
        ttnn_reshape_326,
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
    ttnn_add_112 = ttnn.add(
        ttnn_matmul_75,
        input_3,
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
    ttnn_add_113 = ttnn.add(
        ttnn_matmul_76,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_113


def CLIPEncoderLayer_78_0(input_0, input_1, input_2, input_3):
    ttnn_add_114 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_40 = ttnn.layer_norm(
        ttnn_add_114,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_40, ttnn_add_114


def CLIPAttention_79_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_328 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_77 = ttnn.matmul(
        ttnn_reshape_328,
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
    ttnn_add_115 = ttnn.add(
        ttnn_matmul_77,
        input_3,
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
    ttnn_add_116 = ttnn.add(
        ttnn_matmul_78,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_116


def CLIPEncoderLayer_80_0(input_0, input_1, input_2, input_3):
    ttnn_add_117 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_41 = ttnn.layer_norm(
        ttnn_add_117,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_117, ttnn_layer_norm_41


def CLIPMLP_81_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_333 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_79 = ttnn.matmul(
        ttnn_reshape_333,
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
    ttnn_add_118 = ttnn.add(
        ttnn_matmul_79,
        input_1,
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
    ttnn_add_119 = ttnn.add(
        ttnn_matmul_80,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_119


def CLIPEncoderLayer_82_0(input_0, input_1, input_2, input_3):
    ttnn_add_120 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_42 = ttnn.layer_norm(
        ttnn_add_120,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_120, ttnn_layer_norm_42


def CLIPAttention_83_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_335 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_81 = ttnn.matmul(
        ttnn_reshape_335,
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
    ttnn_add_121 = ttnn.add(
        ttnn_matmul_81,
        input_1,
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
    ttnn_add_122 = ttnn.add(
        ttnn_matmul_82,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_122


def CLIPEncoderLayer_84_0(input_0, input_1, input_2, input_3):
    ttnn_add_123 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_43 = ttnn.layer_norm(
        ttnn_add_123,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_123, ttnn_layer_norm_43


def CLIPMLP_85_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_340 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_83 = ttnn.matmul(
        ttnn_reshape_340,
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
    ttnn_add_124 = ttnn.add(
        ttnn_matmul_83,
        input_3,
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
    ttnn_add_125 = ttnn.add(
        ttnn_matmul_84,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_125


def CLIPEncoderLayer_86_0(input_0, input_1, input_2, input_3):
    ttnn_add_126 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_44 = ttnn.layer_norm(
        ttnn_add_126,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_126, ttnn_layer_norm_44


def CLIPAttention_87_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_342 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_85 = ttnn.matmul(
        ttnn_reshape_342,
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
    ttnn_add_127 = ttnn.add(
        ttnn_matmul_85,
        input_4,
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
    ttnn_add_128 = ttnn.add(
        ttnn_matmul_86,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_128


def CLIPEncoderLayer_88_0(input_0, input_1, input_2, input_3):
    ttnn_add_129 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_45 = ttnn.layer_norm(
        ttnn_add_129,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_45, ttnn_add_129


def CLIPMLP_89_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_347 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_87 = ttnn.matmul(
        ttnn_reshape_347,
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
    ttnn_add_130 = ttnn.add(
        ttnn_matmul_87,
        input_1,
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
    ttnn_add_131 = ttnn.add(
        ttnn_matmul_88,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_131


def CLIPEncoderLayer_90_0(input_0, input_1, input_2, input_3):
    ttnn_add_132 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_46 = ttnn.layer_norm(
        ttnn_add_132,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_46, ttnn_add_132


def CLIPAttention_91_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_349 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_89 = ttnn.matmul(
        ttnn_reshape_349,
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
    ttnn_add_133 = ttnn.add(
        ttnn_matmul_89,
        input_3,
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
    ttnn_add_134 = ttnn.add(
        ttnn_matmul_90,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_134


def CLIPEncoderLayer_92_0(input_0, input_1, input_2, input_3):
    ttnn_add_135 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_47 = ttnn.layer_norm(
        ttnn_add_135,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_135, ttnn_layer_norm_47


def CLIPMLP_93_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_354 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_91 = ttnn.matmul(
        ttnn_reshape_354,
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
    ttnn_add_136 = ttnn.add(
        ttnn_matmul_91,
        input_4,
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
    ttnn_add_137 = ttnn.add(
        ttnn_matmul_92,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_137


def CLIPEncoderLayer_94_0(input_0, input_1, input_2, input_3):
    ttnn_add_138 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_48 = ttnn.layer_norm(
        ttnn_add_138,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_138, ttnn_layer_norm_48


def CLIPAttention_95_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_356 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_93 = ttnn.matmul(
        ttnn_reshape_356,
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
    ttnn_add_139 = ttnn.add(
        ttnn_matmul_93,
        input_3,
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
    ttnn_add_140 = ttnn.add(
        ttnn_matmul_94,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_140


def CLIPEncoderLayer_96_0(input_0, input_1, input_2, input_3):
    ttnn_add_141 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_49 = ttnn.layer_norm(
        ttnn_add_141,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_141, ttnn_layer_norm_49


def CLIPMLP_97_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_361 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_95 = ttnn.matmul(
        ttnn_reshape_361,
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
    ttnn_add_142 = ttnn.add(
        ttnn_matmul_95,
        input_1,
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
    ttnn_add_143 = ttnn.add(
        ttnn_matmul_96,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_143


def CLIPEncoderLayer_98_0(input_0, input_1, input_2, input_3):
    ttnn_add_144 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_50 = ttnn.layer_norm(
        ttnn_add_144,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_144, ttnn_layer_norm_50


def CLIPAttention_99_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_363 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_97 = ttnn.matmul(
        ttnn_reshape_363,
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
    ttnn_add_145 = ttnn.add(
        ttnn_matmul_97,
        input_3,
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
    ttnn_add_146 = ttnn.add(
        ttnn_matmul_98,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_146


def CLIPEncoderLayer_100_0(input_0, input_1, input_2, input_3):
    ttnn_add_147 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_51 = ttnn.layer_norm(
        ttnn_add_147,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_147, ttnn_layer_norm_51


def CLIPMLP_101_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_368 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_99 = ttnn.matmul(
        ttnn_reshape_368,
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
    ttnn_add_148 = ttnn.add(
        ttnn_matmul_99,
        input_1,
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
    ttnn_add_149 = ttnn.add(
        ttnn_matmul_100,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_149


def CLIPEncoderLayer_102_0(input_0, input_1, input_2, input_3):
    ttnn_add_150 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_52 = ttnn.layer_norm(
        ttnn_add_150,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_52, ttnn_add_150


def CLIPAttention_103_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_370 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_101 = ttnn.matmul(
        ttnn_reshape_370,
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
    ttnn_add_151 = ttnn.add(
        ttnn_matmul_101,
        input_3,
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
    ttnn_add_152 = ttnn.add(
        ttnn_matmul_102,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_152


def CLIPEncoderLayer_104_0(input_0, input_1, input_2, input_3):
    ttnn_add_153 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_53 = ttnn.layer_norm(
        ttnn_add_153,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_153, ttnn_layer_norm_53


def CLIPMLP_105_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_375 = ttnn.reshape(
        input_4,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_103 = ttnn.matmul(
        ttnn_reshape_375,
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
    ttnn_add_154 = ttnn.add(
        ttnn_matmul_103,
        input_1,
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
    ttnn_add_155 = ttnn.add(
        ttnn_matmul_104,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_155


def CLIPEncoderLayer_106_0(input_0, input_1, input_2, input_3):
    ttnn_add_156 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_54 = ttnn.layer_norm(
        ttnn_add_156,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_54, ttnn_add_156


def CLIPAttention_107_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_377 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_105 = ttnn.matmul(
        ttnn_reshape_377,
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
    ttnn_add_157 = ttnn.add(
        ttnn_matmul_105,
        input_0,
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
    ttnn_add_158 = ttnn.add(
        ttnn_matmul_106,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_158


def CLIPEncoderLayer_108_0(input_0, input_1, input_2, input_3):
    ttnn_add_159 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_55 = ttnn.layer_norm(
        ttnn_add_159,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_55, ttnn_add_159


def CLIPMLP_109_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_382 = ttnn.reshape(
        input_1,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_107 = ttnn.matmul(
        ttnn_reshape_382,
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
    ttnn_add_160 = ttnn.add(
        ttnn_matmul_107,
        input_4,
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
    ttnn_add_161 = ttnn.add(
        ttnn_matmul_108,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_161


def CLIPEncoderLayer_110_0(input_0, input_1, input_2, input_3):
    ttnn_add_162 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_56 = ttnn.layer_norm(
        ttnn_add_162,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_162, ttnn_layer_norm_56


def CLIPAttention_111_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_384 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_109 = ttnn.matmul(
        ttnn_reshape_384,
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
    ttnn_add_163 = ttnn.add(
        ttnn_matmul_109,
        input_4,
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
    ttnn_add_164 = ttnn.add(
        ttnn_matmul_110,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_164


def CLIPEncoderLayer_112_0(input_0, input_1, input_2, input_3):
    ttnn_add_165 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_57 = ttnn.layer_norm(
        ttnn_add_165,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_57, ttnn_add_165


def CLIPMLP_113_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_389 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_111 = ttnn.matmul(
        ttnn_reshape_389,
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
    ttnn_add_166 = ttnn.add(
        ttnn_matmul_111,
        input_4,
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
    ttnn_add_167 = ttnn.add(
        ttnn_matmul_112,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_167


def CLIPEncoderLayer_114_0(input_0, input_1, input_2, input_3):
    ttnn_add_168 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_58 = ttnn.layer_norm(
        ttnn_add_168,
        epsilon=9.9999997473787516e-06,
        weight=input_3,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_168, ttnn_layer_norm_58


def CLIPAttention_115_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_391 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_113 = ttnn.matmul(
        ttnn_reshape_391,
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
    ttnn_add_169 = ttnn.add(
        ttnn_matmul_113,
        input_2,
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
    ttnn_add_170 = ttnn.add(
        ttnn_matmul_114,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_170


def CLIPEncoderLayer_116_0(input_0, input_1, input_2, input_3):
    ttnn_add_171 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_59 = ttnn.layer_norm(
        ttnn_add_171,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_59, ttnn_add_171


def CLIPMLP_117_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_396 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_115 = ttnn.matmul(
        ttnn_reshape_396,
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
    ttnn_add_172 = ttnn.add(
        ttnn_matmul_115,
        input_3,
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
    ttnn_add_173 = ttnn.add(
        ttnn_matmul_116,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_173


def CLIPEncoderLayer_118_0(input_0, input_1, input_2, input_3):
    ttnn_add_174 = ttnn.add(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_60 = ttnn.layer_norm(
        ttnn_add_174,
        epsilon=9.9999997473787516e-06,
        weight=input_0,
        bias=input_2,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_174, ttnn_layer_norm_60


def CLIPAttention_119_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_398 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_117 = ttnn.matmul(
        ttnn_reshape_398,
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
    ttnn_add_175 = ttnn.add(
        ttnn_matmul_117,
        input_0,
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
    ttnn_add_176 = ttnn.add(
        ttnn_matmul_118,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_176


def CLIPEncoderLayer_120_0(input_0, input_1, input_2, input_3):
    ttnn_add_177 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_61 = ttnn.layer_norm(
        ttnn_add_177,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_1,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_177, ttnn_layer_norm_61


def CLIPMLP_121_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_403 = ttnn.reshape(
        input_3,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_119 = ttnn.matmul(
        ttnn_reshape_403,
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
    ttnn_add_178 = ttnn.add(
        ttnn_matmul_119,
        input_0,
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
    ttnn_add_179 = ttnn.add(
        ttnn_matmul_120,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_179


def CLIPEncoderLayer_122_0(input_0, input_1, input_2, input_3):
    ttnn_add_180 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_62 = ttnn.layer_norm(
        ttnn_add_180,
        epsilon=9.9999997473787516e-06,
        weight=input_1,
        bias=input_0,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_layer_norm_62, ttnn_add_180


def CLIPAttention_123_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_405 = ttnn.reshape(
        input_2,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_121 = ttnn.matmul(
        ttnn_reshape_405,
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
    ttnn_add_181 = ttnn.add(
        ttnn_matmul_121,
        input_4,
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
    ttnn_add_182 = ttnn.add(
        ttnn_matmul_122,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_182


def CLIPEncoderLayer_124_0(input_0, input_1, input_2, input_3):
    ttnn_add_183 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_layer_norm_63 = ttnn.layer_norm(
        ttnn_add_183,
        epsilon=9.9999997473787516e-06,
        weight=input_2,
        bias=input_3,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    return ttnn_add_183, ttnn_layer_norm_63


def CLIPMLP_125_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_reshape_410 = ttnn.reshape(
        input_0,
        [257, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_123 = ttnn.matmul(
        ttnn_reshape_410,
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
    ttnn_add_184 = ttnn.add(
        ttnn_matmul_123,
        input_3,
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
    ttnn_add_185 = ttnn.add(
        ttnn_matmul_124,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_add_185


def CLIPEncoderLayer_126_0(input_0, input_1):
    ttnn_add_186 = ttnn.add(
        input_0,
        input_1,
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


def Linear_127_0(input_0, input_1, input_2):
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


def Attention_129_0(input_0, input_1, input_2, input_3, input_4, input_5):
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


def IPAdapterPlusImageProjectionBlock_130_0(input_0, input_1, input_2, input_3):
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


def Linear_131_0(input_0, input_1):
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


def Linear_132_0(input_0, input_1):
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
    input_0, input_1, input_2, input_3, input_4
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


def Attention_134_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
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


def IPAdapterPlusImageProjectionBlock_135_0(input_0, input_1, input_2, input_3):
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


def Linear_136_0(input_0, input_1):
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


def Linear_137_0(input_0, input_1):
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
    input_0, input_1, input_2, input_3, input_4
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


def Attention_139_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
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


def IPAdapterPlusImageProjectionBlock_140_0(input_0, input_1, input_2, input_3):
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


def Linear_141_0(input_0, input_1):
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


def Linear_142_0(input_0, input_1):
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
    input_0, input_1, input_2, input_3, input_4
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


def Attention_144_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
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


def IPAdapterPlusImageProjectionBlock_145_0(input_0, input_1, input_2, input_3):
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


def Linear_146_0(input):
    ttnn_reshape_448 = ttnn.reshape(
        input,
        [16, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_448


def Linear_147_0(input_0, input_1):
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


def IPAdapterPlusImageProjectionBlock_148_0(input_0, input_1, input_2):
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
    input_0, input_1, input_2, input_3, input_4, input_5
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


def Linear_150_0(input):
    return


def IPAdapterPlusImageProjectionBlock_151_0(input):
    return


def load_inputs_for__main():
    utils_DeviceGetter_get_device_163 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_139 = utils.load_tensor(
        "./tensors/arg139.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_140 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_141 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_142 = utils.load_tensor(
        "./tensors/arg142.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_143 = utils.load_tensor(
        "./tensors/arg143.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_144 = utils.load_tensor(
        "./tensors/arg144.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_145 = utils.load_tensor(
        "./tensors/arg145.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_146 = utils.load_tensor(
        "./tensors/arg146.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_147 = utils.load_tensor(
        "./tensors/arg147.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_148 = utils.load_tensor(
        "./tensors/arg148.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_149 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_150 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_151 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_152 = utils.load_tensor(
        "./tensors/arg152.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_153 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_154 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_155 = utils.load_tensor(
        "./tensors/arg155.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_156 = utils.load_tensor(
        "./tensors/arg156.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_157 = utils.load_tensor(
        "./tensors/arg157.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_158 = utils.load_tensor(
        "./tensors/arg158.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_159 = utils.load_tensor(
        "./tensors/arg159.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_160 = utils.load_tensor(
        "./tensors/arg160.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_161 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_162 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_163 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_164 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_165 = utils.load_tensor(
        "./tensors/arg165.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_166 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_167 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_168 = utils.load_tensor(
        "./tensors/arg168.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_169 = utils.load_tensor(
        "./tensors/arg169.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_170 = utils.load_tensor(
        "./tensors/arg170.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_171 = utils.load_tensor(
        "./tensors/arg171.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_172 = utils.load_tensor(
        "./tensors/arg172.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_173 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_174 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_175 = utils.load_tensor(
        "./tensors/arg175.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_176 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_177 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_178 = utils.load_tensor(
        "./tensors/arg178.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_179 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_180 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_181 = utils.load_tensor(
        "./tensors/arg181.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_182 = utils.load_tensor(
        "./tensors/arg182.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_183 = utils.load_tensor(
        "./tensors/arg183.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_184 = utils.load_tensor(
        "./tensors/arg184.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_185 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_186 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_187 = utils.load_tensor(
        "./tensors/arg187.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_188 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_189 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_190 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_191 = utils.load_tensor(
        "./tensors/arg191.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_192 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_193 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_194 = utils.load_tensor(
        "./tensors/arg194.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_195 = utils.load_tensor(
        "./tensors/arg195.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_196 = utils.load_tensor(
        "./tensors/arg196.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_197 = utils.load_tensor(
        "./tensors/arg197.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_198 = utils.load_tensor(
        "./tensors/arg198.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_199 = utils.load_tensor(
        "./tensors/arg199.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_200 = utils.load_tensor(
        "./tensors/arg200.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_201 = utils.load_tensor(
        "./tensors/arg201.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_202 = utils.load_tensor(
        "./tensors/arg202.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_203 = utils.load_tensor(
        "./tensors/arg203.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_204 = utils.load_tensor(
        "./tensors/arg204.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_205 = utils.load_tensor(
        "./tensors/arg205.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_206 = utils.load_tensor(
        "./tensors/arg206.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_207 = utils.load_tensor(
        "./tensors/arg207.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_208 = utils.load_tensor(
        "./tensors/arg208.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_209 = utils.load_tensor(
        "./tensors/arg209.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_210 = utils.load_tensor(
        "./tensors/arg210.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_211 = utils.load_tensor(
        "./tensors/arg211.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_212 = utils.load_tensor(
        "./tensors/arg212.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_213 = utils.load_tensor(
        "./tensors/arg213.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_214 = utils.load_tensor(
        "./tensors/arg214.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_215 = utils.load_tensor(
        "./tensors/arg215.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_216 = utils.load_tensor(
        "./tensors/arg216.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_217 = utils.load_tensor(
        "./tensors/arg217.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_218 = utils.load_tensor(
        "./tensors/arg218.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_219 = utils.load_tensor(
        "./tensors/arg219.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_220 = utils.load_tensor(
        "./tensors/arg220.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_221 = utils.load_tensor(
        "./tensors/arg221.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_222 = utils.load_tensor(
        "./tensors/arg222.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_223 = utils.load_tensor(
        "./tensors/arg223.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_224 = utils.load_tensor(
        "./tensors/arg224.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_225 = utils.load_tensor(
        "./tensors/arg225.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_226 = utils.load_tensor(
        "./tensors/arg226.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_227 = utils.load_tensor(
        "./tensors/arg227.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_228 = utils.load_tensor(
        "./tensors/arg228.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_229 = utils.load_tensor(
        "./tensors/arg229.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_230 = utils.load_tensor(
        "./tensors/arg230.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_231 = utils.load_tensor(
        "./tensors/arg231.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_232 = utils.load_tensor(
        "./tensors/arg232.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_233 = utils.load_tensor(
        "./tensors/arg233.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_234 = utils.load_tensor(
        "./tensors/arg234.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_235 = utils.load_tensor(
        "./tensors/arg235.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_236 = utils.load_tensor(
        "./tensors/arg236.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_237 = utils.load_tensor(
        "./tensors/arg237.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_238 = utils.load_tensor(
        "./tensors/arg238.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_239 = utils.load_tensor(
        "./tensors/arg239.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_240 = utils.load_tensor(
        "./tensors/arg240.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_241 = utils.load_tensor(
        "./tensors/arg241.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_242 = utils.load_tensor(
        "./tensors/arg242.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_243 = utils.load_tensor(
        "./tensors/arg243.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_244 = utils.load_tensor(
        "./tensors/arg244.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_245 = utils.load_tensor(
        "./tensors/arg245.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_246 = utils.load_tensor(
        "./tensors/arg246.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_247 = utils.load_tensor(
        "./tensors/arg247.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_248 = utils.load_tensor(
        "./tensors/arg248.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_249 = utils.load_tensor(
        "./tensors/arg249.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_250 = utils.load_tensor(
        "./tensors/arg250.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_251 = utils.load_tensor(
        "./tensors/arg251.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_252 = utils.load_tensor(
        "./tensors/arg252.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_253 = utils.load_tensor(
        "./tensors/arg253.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_254 = utils.load_tensor(
        "./tensors/arg254.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_255 = utils.load_tensor(
        "./tensors/arg255.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_256 = utils.load_tensor(
        "./tensors/arg256.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_257 = utils.load_tensor(
        "./tensors/arg257.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_258 = utils.load_tensor(
        "./tensors/arg258.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_259 = utils.load_tensor(
        "./tensors/arg259.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_260 = utils.load_tensor(
        "./tensors/arg260.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_261 = utils.load_tensor(
        "./tensors/arg261.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_262 = utils.load_tensor(
        "./tensors/arg262.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_263 = utils.load_tensor(
        "./tensors/arg263.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_264 = utils.load_tensor(
        "./tensors/arg264.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_265 = utils.load_tensor(
        "./tensors/arg265.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_266 = utils.load_tensor(
        "./tensors/arg266.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_267 = utils.load_tensor(
        "./tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_268 = utils.load_tensor(
        "./tensors/arg268.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_269 = utils.load_tensor(
        "./tensors/arg269.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_270 = utils.load_tensor(
        "./tensors/arg270.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_271 = utils.load_tensor(
        "./tensors/arg271.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_272 = utils.load_tensor(
        "./tensors/arg272.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_273 = utils.load_tensor(
        "./tensors/arg273.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_274 = utils.load_tensor(
        "./tensors/arg274.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_275 = utils.load_tensor(
        "./tensors/arg275.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_276 = utils.load_tensor(
        "./tensors/arg276.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_277 = utils.load_tensor(
        "./tensors/arg277.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_278 = utils.load_tensor(
        "./tensors/arg278.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_279 = utils.load_tensor(
        "./tensors/arg279.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_280 = utils.load_tensor(
        "./tensors/arg280.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_281 = utils.load_tensor(
        "./tensors/arg281.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_282 = utils.load_tensor(
        "./tensors/arg282.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_283 = utils.load_tensor(
        "./tensors/arg283.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_284 = utils.load_tensor(
        "./tensors/arg284.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_285 = utils.load_tensor(
        "./tensors/arg285.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_286 = utils.load_tensor(
        "./tensors/arg286.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_287 = utils.load_tensor(
        "./tensors/arg287.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_288 = utils.load_tensor(
        "./tensors/arg288.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_289 = utils.load_tensor(
        "./tensors/arg289.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_290 = utils.load_tensor(
        "./tensors/arg290.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_291 = utils.load_tensor(
        "./tensors/arg291.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_292 = utils.load_tensor(
        "./tensors/arg292.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_293 = utils.load_tensor(
        "./tensors/arg293.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_294 = utils.load_tensor(
        "./tensors/arg294.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_295 = utils.load_tensor(
        "./tensors/arg295.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_296 = utils.load_tensor(
        "./tensors/arg296.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_297 = utils.load_tensor(
        "./tensors/arg297.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_298 = utils.load_tensor(
        "./tensors/arg298.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_299 = utils.load_tensor(
        "./tensors/arg299.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_300 = utils.load_tensor(
        "./tensors/arg300.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_301 = utils.load_tensor(
        "./tensors/arg301.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_302 = utils.load_tensor(
        "./tensors/arg302.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_303 = utils.load_tensor(
        "./tensors/arg303.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_304 = utils.load_tensor(
        "./tensors/arg304.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_305 = utils.load_tensor(
        "./tensors/arg305.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_306 = utils.load_tensor(
        "./tensors/arg306.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_307 = utils.load_tensor(
        "./tensors/arg307.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_308 = utils.load_tensor(
        "./tensors/arg308.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_309 = utils.load_tensor(
        "./tensors/arg309.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_310 = utils.load_tensor(
        "./tensors/arg310.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_311 = utils.load_tensor(
        "./tensors/arg311.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_312 = utils.load_tensor(
        "./tensors/arg312.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_313 = utils.load_tensor(
        "./tensors/arg313.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_314 = utils.load_tensor(
        "./tensors/arg314.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_315 = utils.load_tensor(
        "./tensors/arg315.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_316 = utils.load_tensor(
        "./tensors/arg316.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_317 = utils.load_tensor(
        "./tensors/arg317.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_318 = utils.load_tensor(
        "./tensors/arg318.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_319 = utils.load_tensor(
        "./tensors/arg319.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_320 = utils.load_tensor(
        "./tensors/arg320.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_321 = utils.load_tensor(
        "./tensors/arg321.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_322 = utils.load_tensor(
        "./tensors/arg322.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_323 = utils.load_tensor(
        "./tensors/arg323.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_324 = utils.load_tensor(
        "./tensors/arg324.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_325 = utils.load_tensor(
        "./tensors/arg325.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_326 = utils.load_tensor(
        "./tensors/arg326.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_327 = utils.load_tensor(
        "./tensors/arg327.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_328 = utils.load_tensor(
        "./tensors/arg328.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_329 = utils.load_tensor(
        "./tensors/arg329.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_330 = utils.load_tensor(
        "./tensors/arg330.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_331 = utils.load_tensor(
        "./tensors/arg331.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_332 = utils.load_tensor(
        "./tensors/arg332.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_333 = utils.load_tensor(
        "./tensors/arg333.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_334 = utils.load_tensor(
        "./tensors/arg334.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_335 = utils.load_tensor(
        "./tensors/arg335.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_336 = utils.load_tensor(
        "./tensors/arg336.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_337 = utils.load_tensor(
        "./tensors/arg337.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_338 = utils.load_tensor(
        "./tensors/arg338.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_339 = utils.load_tensor(
        "./tensors/arg339.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_340 = utils.load_tensor(
        "./tensors/arg340.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_341 = utils.load_tensor(
        "./tensors/arg341.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_342 = utils.load_tensor(
        "./tensors/arg342.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_343 = utils.load_tensor(
        "./tensors/arg343.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_344 = utils.load_tensor(
        "./tensors/arg344.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_345 = utils.load_tensor(
        "./tensors/arg345.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_346 = utils.load_tensor(
        "./tensors/arg346.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_347 = utils.load_tensor(
        "./tensors/arg347.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_348 = utils.load_tensor(
        "./tensors/arg348.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_349 = utils.load_tensor(
        "./tensors/arg349.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_350 = utils.load_tensor(
        "./tensors/arg350.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_351 = utils.load_tensor(
        "./tensors/arg351.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_352 = utils.load_tensor(
        "./tensors/arg352.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_353 = utils.load_tensor(
        "./tensors/arg353.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_354 = utils.load_tensor(
        "./tensors/arg354.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_355 = utils.load_tensor(
        "./tensors/arg355.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_356 = utils.load_tensor(
        "./tensors/arg356.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_357 = utils.load_tensor(
        "./tensors/arg357.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_358 = utils.load_tensor(
        "./tensors/arg358.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_359 = utils.load_tensor(
        "./tensors/arg359.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_360 = utils.load_tensor(
        "./tensors/arg360.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_361 = utils.load_tensor(
        "./tensors/arg361.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_362 = utils.load_tensor(
        "./tensors/arg362.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_363 = utils.load_tensor(
        "./tensors/arg363.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_364 = utils.load_tensor(
        "./tensors/arg364.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_365 = utils.load_tensor(
        "./tensors/arg365.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_366 = utils.load_tensor(
        "./tensors/arg366.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_367 = utils.load_tensor(
        "./tensors/arg367.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_368 = utils.load_tensor(
        "./tensors/arg368.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_369 = utils.load_tensor(
        "./tensors/arg369.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_370 = utils.load_tensor(
        "./tensors/arg370.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_371 = utils.load_tensor(
        "./tensors/arg371.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_372 = utils.load_tensor(
        "./tensors/arg372.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_373 = utils.load_tensor(
        "./tensors/arg373.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_374 = utils.load_tensor(
        "./tensors/arg374.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_375 = utils.load_tensor(
        "./tensors/arg375.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_376 = utils.load_tensor(
        "./tensors/arg376.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_377 = utils.load_tensor(
        "./tensors/arg377.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_378 = utils.load_tensor(
        "./tensors/arg378.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_379 = utils.load_tensor(
        "./tensors/arg379.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_380 = utils.load_tensor(
        "./tensors/arg380.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_381 = utils.load_tensor(
        "./tensors/arg381.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_382 = utils.load_tensor(
        "./tensors/arg382.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_383 = utils.load_tensor(
        "./tensors/arg383.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_384 = utils.load_tensor(
        "./tensors/arg384.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_385 = utils.load_tensor(
        "./tensors/arg385.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_386 = utils.load_tensor(
        "./tensors/arg386.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_387 = utils.load_tensor(
        "./tensors/arg387.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        None,
        None,
    )
    utils_load_tensor_388 = utils.load_tensor(
        "./tensors/arg388.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_389 = utils.load_tensor(
        "./tensors/arg389.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_390 = utils.load_tensor(
        "./tensors/arg390.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_391 = utils.load_tensor(
        "./tensors/arg391.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_392 = utils.load_tensor(
        "./tensors/arg392.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_393 = utils.load_tensor(
        "./tensors/arg393.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_394 = utils.load_tensor(
        "./tensors/arg394.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_395 = utils.load_tensor(
        "./tensors/arg395.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_396 = utils.load_tensor(
        "./tensors/arg396.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_397 = utils.load_tensor(
        "./tensors/arg397.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_398 = utils.load_tensor(
        "./tensors/arg398.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_399 = utils.load_tensor(
        "./tensors/arg399.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_400 = utils.load_tensor(
        "./tensors/arg400.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_401 = utils.load_tensor(
        "./tensors/arg401.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_402 = utils.load_tensor(
        "./tensors/arg402.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_403 = utils.load_tensor(
        "./tensors/arg403.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_404 = utils.load_tensor(
        "./tensors/arg404.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_405 = utils.load_tensor(
        "./tensors/arg405.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_406 = utils.load_tensor(
        "./tensors/arg406.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_407 = utils.load_tensor(
        "./tensors/arg407.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_408 = utils.load_tensor(
        "./tensors/arg408.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_409 = utils.load_tensor(
        "./tensors/arg409.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_410 = utils.load_tensor(
        "./tensors/arg410.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_411 = utils.load_tensor(
        "./tensors/arg411.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_412 = utils.load_tensor(
        "./tensors/arg412.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_413 = utils.load_tensor(
        "./tensors/arg413.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_414 = utils.load_tensor(
        "./tensors/arg414.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_415 = utils.load_tensor(
        "./tensors/arg415.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_416 = utils.load_tensor(
        "./tensors/arg416.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_417 = utils.load_tensor(
        "./tensors/arg417.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_418 = utils.load_tensor(
        "./tensors/arg418.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_419 = utils.load_tensor(
        "./tensors/arg419.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_420 = utils.load_tensor(
        "./tensors/arg420.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_421 = utils.load_tensor(
        "./tensors/arg421.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_422 = utils.load_tensor(
        "./tensors/arg422.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_423 = utils.load_tensor(
        "./tensors/arg423.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_424 = utils.load_tensor(
        "./tensors/arg424.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_425 = utils.load_tensor(
        "./tensors/arg425.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_426 = utils.load_tensor(
        "./tensors/arg426.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_427 = utils.load_tensor(
        "./tensors/arg427.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_428 = utils.load_tensor(
        "./tensors/arg428.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_429 = utils.load_tensor(
        "./tensors/arg429.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_430 = utils.load_tensor(
        "./tensors/arg430.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_431 = utils.load_tensor(
        "./tensors/arg431.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_432 = utils.load_tensor(
        "./tensors/arg432.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_433 = utils.load_tensor(
        "./tensors/arg433.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_434 = utils.load_tensor(
        "./tensors/arg434.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_435 = utils.load_tensor(
        "./tensors/arg435.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_436 = utils.load_tensor(
        "./tensors/arg436.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_437 = utils.load_tensor(
        "./tensors/arg437.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_438 = utils.load_tensor(
        "./tensors/arg438.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_439 = utils.load_tensor(
        "./tensors/arg439.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_440 = utils.load_tensor(
        "./tensors/arg440.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_441 = utils.load_tensor(
        "./tensors/arg441.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_442 = utils.load_tensor(
        "./tensors/arg442.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_443 = utils.load_tensor(
        "./tensors/arg443.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_444 = utils.load_tensor(
        "./tensors/arg444.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_445 = utils.load_tensor(
        "./tensors/arg445.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_446 = utils.load_tensor(
        "./tensors/arg446.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_447 = utils.load_tensor(
        "./tensors/arg447.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_448 = utils.load_tensor(
        "./tensors/arg448.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_449 = utils.load_tensor(
        "./tensors/arg449.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_450 = utils.load_tensor(
        "./tensors/arg450.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_451 = utils.load_tensor(
        "./tensors/arg451.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_452 = utils.load_tensor(
        "./tensors/arg452.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_453 = utils.load_tensor(
        "./tensors/arg453.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_454 = utils.load_tensor(
        "./tensors/arg454.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_455 = utils.load_tensor(
        "./tensors/arg455.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_456 = utils.load_tensor(
        "./tensors/arg456.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_457 = utils.load_tensor(
        "./tensors/arg457.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_458 = utils.load_tensor(
        "./tensors/arg458.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_459 = utils.load_tensor(
        "./tensors/arg459.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_460 = utils.load_tensor(
        "./tensors/arg460.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_461 = utils.load_tensor(
        "./tensors/arg461.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_462 = utils.load_tensor(
        "./tensors/arg462.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_463 = utils.load_tensor(
        "./tensors/arg463.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_464 = utils.load_tensor(
        "./tensors/arg464.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_465 = utils.load_tensor(
        "./tensors/arg465.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_466 = utils.load_tensor(
        "./tensors/arg466.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_467 = utils.load_tensor(
        "./tensors/arg467.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_468 = utils.load_tensor(
        "./tensors/arg468.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_469 = utils.load_tensor(
        "./tensors/arg469.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_470 = utils.load_tensor(
        "./tensors/arg470.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_471 = utils.load_tensor(
        "./tensors/arg471.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_472 = utils.load_tensor(
        "./tensors/arg472.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_473 = utils.load_tensor(
        "./tensors/arg473.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_474 = utils.load_tensor(
        "./tensors/arg474.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_475 = utils.load_tensor(
        "./tensors/arg475.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_476 = utils.load_tensor(
        "./tensors/arg476.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_477 = utils.load_tensor(
        "./tensors/arg477.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_478 = utils.load_tensor(
        "./tensors/arg478.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_479 = utils.load_tensor(
        "./tensors/arg479.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_480 = utils.load_tensor(
        "./tensors/arg480.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_481 = utils.load_tensor(
        "./tensors/arg481.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_482 = utils.load_tensor(
        "./tensors/arg482.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_483 = utils.load_tensor(
        "./tensors/arg483.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_484 = utils.load_tensor(
        "./tensors/arg484.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_485 = utils.load_tensor(
        "./tensors/arg485.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_486 = utils.load_tensor(
        "./tensors/arg486.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_487 = utils.load_tensor(
        "./tensors/arg487.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_488 = utils.load_tensor(
        "./tensors/arg488.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_489 = utils.load_tensor(
        "./tensors/arg489.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_490 = utils.load_tensor(
        "./tensors/arg490.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_491 = utils.load_tensor(
        "./tensors/arg491.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_492 = utils.load_tensor(
        "./tensors/arg492.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_493 = utils.load_tensor(
        "./tensors/arg493.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_494 = utils.load_tensor(
        "./tensors/arg494.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_495 = utils.load_tensor(
        "./tensors/arg495.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_496 = utils.load_tensor(
        "./tensors/arg496.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_497 = utils.load_tensor(
        "./tensors/arg497.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_498 = utils.load_tensor(
        "./tensors/arg498.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_499 = utils.load_tensor(
        "./tensors/arg499.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_500 = utils.load_tensor(
        "./tensors/arg500.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_501 = utils.load_tensor(
        "./tensors/arg501.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_502 = utils.load_tensor(
        "./tensors/arg502.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_503 = utils.load_tensor(
        "./tensors/arg503.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_504 = utils.load_tensor(
        "./tensors/arg504.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_505 = utils.load_tensor(
        "./tensors/arg505.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_506 = utils.load_tensor(
        "./tensors/arg506.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_507 = utils.load_tensor(
        "./tensors/arg507.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_508 = utils.load_tensor(
        "./tensors/arg508.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_509 = utils.load_tensor(
        "./tensors/arg509.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_510 = utils.load_tensor(
        "./tensors/arg510.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_511 = utils.load_tensor(
        "./tensors/arg511.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_512 = utils.load_tensor(
        "./tensors/arg512.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_513 = utils.load_tensor(
        "./tensors/arg513.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_514 = utils.load_tensor(
        "./tensors/arg514.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_515 = utils.load_tensor(
        "./tensors/arg515.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_516 = utils.load_tensor(
        "./tensors/arg516.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_517 = utils.load_tensor(
        "./tensors/arg517.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_518 = utils.load_tensor(
        "./tensors/arg518.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_519 = utils.load_tensor(
        "./tensors/arg519.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_520 = utils.load_tensor(
        "./tensors/arg520.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_521 = utils.load_tensor(
        "./tensors/arg521.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_522 = utils.load_tensor(
        "./tensors/arg522.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_523 = utils.load_tensor(
        "./tensors/arg523.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_524 = utils.load_tensor(
        "./tensors/arg524.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_525 = utils.load_tensor(
        "./tensors/arg525.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_526 = utils.load_tensor(
        "./tensors/arg526.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_527 = utils.load_tensor(
        "./tensors/arg527.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_528 = utils.load_tensor(
        "./tensors/arg528.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_529 = utils.load_tensor(
        "./tensors/arg529.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_530 = utils.load_tensor(
        "./tensors/arg530.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_531 = utils.load_tensor(
        "./tensors/arg531.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_532 = utils.load_tensor(
        "./tensors/arg532.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_533 = utils.load_tensor(
        "./tensors/arg533.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_534 = utils.load_tensor(
        "./tensors/arg534.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_535 = utils.load_tensor(
        "./tensors/arg535.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_536 = utils.load_tensor(
        "./tensors/arg536.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_537 = utils.load_tensor(
        "./tensors/arg537.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_538 = utils.load_tensor(
        "./tensors/arg538.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_539 = utils.load_tensor(
        "./tensors/arg539.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_540 = utils.load_tensor(
        "./tensors/arg540.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_541 = utils.load_tensor(
        "./tensors/arg541.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_542 = utils.load_tensor(
        "./tensors/arg542.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_543 = utils.load_tensor(
        "./tensors/arg543.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_544 = utils.load_tensor(
        "./tensors/arg544.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_545 = utils.load_tensor(
        "./tensors/arg545.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_546 = utils.load_tensor(
        "./tensors/arg546.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_547 = utils.load_tensor(
        "./tensors/arg547.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_548 = utils.load_tensor(
        "./tensors/arg548.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_549 = utils.load_tensor(
        "./tensors/arg549.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_550 = utils.load_tensor(
        "./tensors/arg550.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_551 = utils.load_tensor(
        "./tensors/arg551.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_552 = utils.load_tensor(
        "./tensors/arg552.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_553 = utils.load_tensor(
        "./tensors/arg553.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_554 = utils.load_tensor(
        "./tensors/arg554.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_555 = utils.load_tensor(
        "./tensors/arg555.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_556 = utils.load_tensor(
        "./tensors/arg556.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_557 = utils.load_tensor(
        "./tensors/arg557.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_391 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
        utils_load_tensor_126,
        utils_load_tensor_127,
        utils_load_tensor_128,
        utils_load_tensor_129,
        utils_load_tensor_130,
        utils_load_tensor_131,
        utils_load_tensor_132,
        utils_load_tensor_133,
        utils_load_tensor_134,
        utils_load_tensor_135,
        utils_load_tensor_136,
        utils_load_tensor_137,
        utils_load_tensor_138,
        utils_load_tensor_139,
        utils_load_tensor_140,
        utils_load_tensor_141,
        utils_load_tensor_142,
        utils_load_tensor_143,
        utils_load_tensor_144,
        utils_load_tensor_145,
        utils_load_tensor_146,
        utils_load_tensor_147,
        utils_load_tensor_148,
        utils_load_tensor_149,
        utils_load_tensor_150,
        utils_load_tensor_151,
        utils_load_tensor_152,
        utils_load_tensor_153,
        utils_load_tensor_154,
        utils_load_tensor_155,
        utils_load_tensor_156,
        utils_load_tensor_157,
        utils_load_tensor_158,
        utils_load_tensor_159,
        utils_load_tensor_160,
        utils_load_tensor_161,
        utils_load_tensor_162,
        utils_load_tensor_163,
        utils_load_tensor_164,
        utils_load_tensor_165,
        utils_load_tensor_166,
        utils_load_tensor_167,
        utils_load_tensor_168,
        utils_load_tensor_169,
        utils_load_tensor_170,
        utils_load_tensor_171,
        utils_load_tensor_172,
        utils_load_tensor_173,
        utils_load_tensor_174,
        utils_load_tensor_175,
        utils_load_tensor_176,
        utils_load_tensor_177,
        utils_load_tensor_178,
        utils_load_tensor_179,
        utils_load_tensor_180,
        utils_load_tensor_181,
        utils_load_tensor_182,
        utils_load_tensor_183,
        utils_load_tensor_184,
        utils_load_tensor_185,
        utils_load_tensor_186,
        utils_load_tensor_187,
        utils_load_tensor_188,
        utils_load_tensor_189,
        utils_load_tensor_190,
        utils_load_tensor_191,
        utils_load_tensor_192,
        utils_load_tensor_193,
        utils_load_tensor_194,
        utils_load_tensor_195,
        utils_load_tensor_196,
        utils_load_tensor_197,
        utils_load_tensor_198,
        utils_load_tensor_199,
        utils_load_tensor_200,
        utils_load_tensor_201,
        utils_load_tensor_202,
        utils_load_tensor_203,
        utils_load_tensor_204,
        utils_load_tensor_205,
        utils_load_tensor_206,
        utils_load_tensor_207,
        utils_load_tensor_208,
        utils_load_tensor_209,
        utils_load_tensor_210,
        utils_load_tensor_211,
        utils_load_tensor_212,
        utils_load_tensor_213,
        utils_load_tensor_214,
        utils_load_tensor_215,
        utils_load_tensor_216,
        utils_load_tensor_217,
        utils_load_tensor_218,
        utils_load_tensor_219,
        utils_load_tensor_220,
        utils_load_tensor_221,
        utils_load_tensor_222,
        utils_load_tensor_223,
        utils_load_tensor_224,
        utils_load_tensor_225,
        utils_load_tensor_226,
        utils_load_tensor_227,
        utils_load_tensor_228,
        utils_load_tensor_229,
        utils_load_tensor_230,
        utils_load_tensor_231,
        utils_load_tensor_232,
        utils_load_tensor_233,
        utils_load_tensor_234,
        utils_load_tensor_235,
        utils_load_tensor_236,
        utils_load_tensor_237,
        utils_load_tensor_238,
        utils_load_tensor_239,
        utils_load_tensor_240,
        utils_load_tensor_241,
        utils_load_tensor_242,
        utils_load_tensor_243,
        utils_load_tensor_244,
        utils_load_tensor_245,
        utils_load_tensor_246,
        utils_load_tensor_247,
        utils_load_tensor_248,
        utils_load_tensor_249,
        utils_load_tensor_250,
        utils_load_tensor_251,
        utils_load_tensor_252,
        utils_load_tensor_253,
        utils_load_tensor_254,
        utils_load_tensor_255,
        utils_load_tensor_256,
        utils_load_tensor_257,
        utils_load_tensor_258,
        utils_load_tensor_259,
        utils_load_tensor_260,
        utils_load_tensor_261,
        utils_load_tensor_262,
        utils_load_tensor_263,
        utils_load_tensor_264,
        utils_load_tensor_265,
        utils_load_tensor_266,
        utils_load_tensor_267,
        utils_load_tensor_268,
        utils_load_tensor_269,
        utils_load_tensor_270,
        utils_load_tensor_271,
        utils_load_tensor_272,
        utils_load_tensor_273,
        utils_load_tensor_274,
        utils_load_tensor_275,
        utils_load_tensor_276,
        utils_load_tensor_277,
        utils_load_tensor_278,
        utils_load_tensor_279,
        utils_load_tensor_280,
        utils_load_tensor_281,
        utils_load_tensor_282,
        utils_load_tensor_283,
        utils_load_tensor_284,
        utils_load_tensor_285,
        utils_load_tensor_286,
        utils_load_tensor_287,
        utils_load_tensor_288,
        utils_load_tensor_289,
        utils_load_tensor_290,
        utils_load_tensor_291,
        utils_load_tensor_292,
        utils_load_tensor_293,
        utils_load_tensor_294,
        utils_load_tensor_295,
        utils_load_tensor_296,
        utils_load_tensor_297,
        utils_load_tensor_298,
        utils_load_tensor_299,
        utils_load_tensor_300,
        utils_load_tensor_301,
        utils_load_tensor_302,
        utils_load_tensor_303,
        utils_load_tensor_304,
        utils_load_tensor_305,
        utils_load_tensor_306,
        utils_load_tensor_307,
        utils_load_tensor_308,
        utils_load_tensor_309,
        utils_load_tensor_310,
        utils_load_tensor_311,
        utils_load_tensor_312,
        utils_load_tensor_313,
        utils_load_tensor_314,
        utils_load_tensor_315,
        utils_load_tensor_316,
        utils_load_tensor_317,
        utils_load_tensor_318,
        utils_load_tensor_319,
        utils_load_tensor_320,
        utils_load_tensor_321,
        utils_load_tensor_322,
        utils_load_tensor_323,
        utils_load_tensor_324,
        utils_load_tensor_325,
        utils_load_tensor_326,
        utils_load_tensor_327,
        utils_load_tensor_328,
        utils_load_tensor_329,
        utils_load_tensor_330,
        utils_load_tensor_331,
        utils_load_tensor_332,
        utils_load_tensor_333,
        utils_load_tensor_334,
        utils_load_tensor_335,
        utils_load_tensor_336,
        utils_load_tensor_337,
        utils_load_tensor_338,
        utils_load_tensor_339,
        utils_load_tensor_340,
        utils_load_tensor_341,
        utils_load_tensor_342,
        utils_load_tensor_343,
        utils_load_tensor_344,
        utils_load_tensor_345,
        utils_load_tensor_346,
        utils_load_tensor_347,
        utils_load_tensor_348,
        utils_load_tensor_349,
        utils_load_tensor_350,
        utils_load_tensor_351,
        utils_load_tensor_352,
        utils_load_tensor_353,
        utils_load_tensor_354,
        utils_load_tensor_355,
        utils_load_tensor_356,
        utils_load_tensor_357,
        utils_load_tensor_358,
        utils_load_tensor_359,
        utils_load_tensor_360,
        utils_load_tensor_361,
        utils_load_tensor_362,
        utils_load_tensor_363,
        utils_load_tensor_364,
        utils_load_tensor_365,
        utils_load_tensor_366,
        utils_load_tensor_367,
        utils_load_tensor_368,
        utils_load_tensor_369,
        utils_load_tensor_370,
        utils_load_tensor_371,
        utils_load_tensor_372,
        utils_load_tensor_373,
        utils_load_tensor_374,
        utils_load_tensor_375,
        utils_load_tensor_376,
        utils_load_tensor_377,
        utils_load_tensor_378,
        utils_load_tensor_379,
        utils_load_tensor_380,
        utils_load_tensor_381,
        utils_load_tensor_382,
        utils_load_tensor_383,
        utils_load_tensor_384,
        utils_load_tensor_385,
        utils_load_tensor_386,
        utils_load_tensor_387,
        utils_load_tensor_388,
        utils_load_tensor_389,
        utils_load_tensor_390,
        utils_load_tensor_391,
        utils_load_tensor_392,
        utils_load_tensor_393,
        utils_load_tensor_394,
        utils_load_tensor_395,
        utils_load_tensor_396,
        utils_load_tensor_397,
        utils_load_tensor_398,
        utils_load_tensor_399,
        utils_load_tensor_400,
        utils_load_tensor_401,
        utils_load_tensor_402,
        utils_load_tensor_403,
        utils_load_tensor_404,
        utils_load_tensor_405,
        utils_load_tensor_406,
        utils_load_tensor_407,
        utils_load_tensor_408,
        utils_load_tensor_409,
        utils_load_tensor_410,
        utils_load_tensor_411,
        utils_load_tensor_412,
        utils_load_tensor_413,
        utils_load_tensor_414,
        utils_load_tensor_415,
        utils_load_tensor_416,
        utils_load_tensor_417,
        utils_load_tensor_418,
        utils_load_tensor_419,
        utils_load_tensor_420,
        utils_load_tensor_421,
        utils_load_tensor_422,
        utils_load_tensor_423,
        utils_load_tensor_424,
        utils_load_tensor_425,
        utils_load_tensor_426,
        utils_load_tensor_427,
        utils_load_tensor_428,
        utils_load_tensor_429,
        utils_load_tensor_430,
        utils_load_tensor_431,
        utils_load_tensor_432,
        utils_load_tensor_433,
        utils_load_tensor_434,
        utils_load_tensor_435,
        utils_load_tensor_436,
        utils_load_tensor_437,
        utils_load_tensor_438,
        utils_load_tensor_439,
        utils_load_tensor_440,
        utils_load_tensor_441,
        utils_load_tensor_442,
        utils_load_tensor_443,
        utils_load_tensor_444,
        utils_load_tensor_445,
        utils_load_tensor_446,
        utils_load_tensor_447,
        utils_load_tensor_448,
        utils_load_tensor_449,
        utils_load_tensor_450,
        utils_load_tensor_451,
        utils_load_tensor_452,
        utils_load_tensor_453,
        utils_load_tensor_454,
        utils_load_tensor_455,
        utils_load_tensor_456,
        utils_load_tensor_457,
        utils_load_tensor_458,
        utils_load_tensor_459,
        utils_load_tensor_460,
        utils_load_tensor_461,
        utils_load_tensor_462,
        utils_load_tensor_463,
        utils_load_tensor_464,
        utils_load_tensor_465,
        utils_load_tensor_466,
        utils_load_tensor_467,
        utils_load_tensor_468,
        utils_load_tensor_469,
        utils_load_tensor_470,
        utils_load_tensor_471,
        utils_load_tensor_472,
        utils_load_tensor_473,
        utils_load_tensor_474,
        utils_load_tensor_475,
        utils_load_tensor_476,
        utils_load_tensor_477,
        utils_load_tensor_478,
        utils_load_tensor_479,
        utils_load_tensor_480,
        utils_load_tensor_481,
        utils_load_tensor_482,
        utils_load_tensor_483,
        utils_load_tensor_484,
        utils_load_tensor_485,
        utils_load_tensor_486,
        utils_load_tensor_487,
        utils_load_tensor_488,
        utils_load_tensor_489,
        utils_load_tensor_490,
        utils_load_tensor_491,
        utils_load_tensor_492,
        utils_load_tensor_493,
        utils_load_tensor_494,
        utils_load_tensor_495,
        utils_load_tensor_496,
        utils_load_tensor_497,
        utils_load_tensor_498,
        utils_load_tensor_499,
        utils_load_tensor_500,
        utils_load_tensor_501,
        utils_load_tensor_502,
        utils_load_tensor_503,
        utils_load_tensor_504,
        utils_load_tensor_505,
        utils_load_tensor_506,
        utils_load_tensor_507,
        utils_load_tensor_508,
        utils_load_tensor_509,
        utils_load_tensor_510,
        utils_load_tensor_511,
        utils_load_tensor_512,
        utils_load_tensor_513,
        utils_load_tensor_514,
        utils_load_tensor_515,
        utils_load_tensor_516,
        utils_load_tensor_517,
        utils_load_tensor_518,
        utils_load_tensor_519,
        utils_load_tensor_520,
        utils_load_tensor_521,
        utils_load_tensor_522,
        utils_load_tensor_523,
        utils_load_tensor_524,
        utils_load_tensor_525,
        utils_load_tensor_526,
        utils_load_tensor_527,
        utils_load_tensor_528,
        utils_load_tensor_529,
        utils_load_tensor_530,
        utils_load_tensor_531,
        utils_load_tensor_532,
        utils_load_tensor_533,
        utils_load_tensor_534,
        utils_load_tensor_535,
        utils_load_tensor_536,
        utils_load_tensor_537,
        utils_load_tensor_538,
        utils_load_tensor_539,
        utils_load_tensor_540,
        utils_load_tensor_541,
        utils_load_tensor_542,
        utils_load_tensor_543,
        utils_load_tensor_544,
        utils_load_tensor_545,
        utils_load_tensor_546,
        utils_load_tensor_547,
        utils_load_tensor_548,
        utils_load_tensor_549,
        utils_load_tensor_550,
        utils_load_tensor_551,
        utils_load_tensor_552,
        utils_load_tensor_553,
        utils_load_tensor_554,
        utils_load_tensor_555,
        utils_load_tensor_556,
        utils_load_tensor_557,
    ]
    return util_create_list_391


def main():
    # Get PyTorch golden output
    pt_input = model_pt.get_input()
    pt_output = model_pt.run_pytorch_inference(input_tensor=pt_input)

    # Load TTNN inputs
    load_inputs_for__main_0 = load_inputs_for__main()

    # Run TTNN model
    for i in range(3):
        start_time = time.time()

        # Run TTNN model
        out_ttnn_device = _main(load_inputs_for__main_0)[0]

        # Get outputs
        out_ttnn_host = ttnn.from_device(out_ttnn_device, blocking=True)
        end_time = time.time()

        # Calculate duration and PCC
        duration = (end_time - start_time) * 1000
        pcc = utils.calculate_pcc(pt_output, ttnn.to_torch(out_ttnn_host))

        # Print results
        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tPCC: {pcc:.6f}")

    return 0


if __name__ == "__main__":
    main()
