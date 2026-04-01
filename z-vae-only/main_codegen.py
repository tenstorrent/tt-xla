import ttnn
import utils

_CONST_EVAL_CACHE = {}


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_0 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
        ],
        [160],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_1 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
            90.0,
            90.5,
            91.0,
            91.5,
            92.0,
            92.5,
            93.0,
            93.5,
            94.0,
            94.5,
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            100.5,
            101.0,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.0,
            104.5,
            105.0,
            105.5,
            106.0,
            106.5,
            107.0,
            107.5,
            108.0,
            108.5,
            109.0,
            109.5,
            110.0,
            110.5,
            111.0,
            111.5,
            112.0,
            112.5,
            113.0,
            113.5,
            114.0,
            114.5,
            115.0,
            115.5,
            116.0,
            116.5,
            117.0,
            117.5,
            118.0,
            118.5,
            119.0,
            119.5,
            120.0,
            120.5,
            121.0,
            121.5,
            122.0,
            122.5,
            123.0,
            123.5,
            124.0,
            124.5,
            125.0,
            125.5,
            126.0,
            126.5,
            127.0,
            127.5,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            130.5,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            135.5,
            136.0,
            136.5,
            137.0,
            137.5,
            138.0,
            138.5,
            139.0,
            139.5,
            140.0,
            140.5,
            141.0,
            141.5,
            142.0,
            142.5,
            143.0,
            143.5,
            144.0,
            144.5,
            145.0,
            145.5,
            146.0,
            146.5,
            147.0,
            147.5,
            148.0,
            148.5,
            149.0,
            149.5,
            150.0,
            150.5,
            151.0,
            151.5,
            152.0,
            152.5,
            153.0,
            153.5,
            154.0,
            154.5,
            155.0,
            155.5,
            156.0,
            156.5,
            157.0,
            157.5,
            158.0,
            158.5,
            159.0,
            159.5,
        ],
        [320],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_0 = ttnn.floor(
        ttnn_Tensor_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_1, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_floor_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_typecast_0,
        [320, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_0,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_Tensor_0,
        [1, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_1,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_eq_0 = ttnn.eq(
        ttnn_permute_0,
        ttnn_permute_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn.deallocate(ttnn_permute_0, False)
    util_create_list_0 = [ttnn_eq_0]
    return util_create_list_0


def main_const_eval_1(input):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_1,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_1 = [ttnn_prepare_conv_weights_0]
    return util_create_list_1


def main_const_eval_2(input):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_permute_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_to_layout_1)
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_prepare_conv_bias_0 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_0,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=16,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_2,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_0, False)
    util_create_list_2 = [ttnn_prepare_conv_bias_0]
    return util_create_list_2


def main_const_eval_3(input):
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_1, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_reshape_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_permute_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_typecast_1,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_permute_4,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_4, False)
    util_create_list_3 = [ttnn_reshape_4]
    return util_create_list_3


def main_const_eval_4(input):
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_2 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_2, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_reshape_5,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_permute_5,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn_permute_6 = ttnn.permute(
        ttnn_typecast_2,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_permute_6,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_6, False)
    util_create_list_4 = [ttnn_reshape_6]
    return util_create_list_4


def main_const_eval_5():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_5 = [ttnn_full_0]
    return util_create_list_5


def main_const_eval_6(input):
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_3 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_3, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_reshape_7,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_permute_7, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_7, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_to_layout_5)
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_prepare_conv_bias_1 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_1,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_6,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_1, False)
    util_create_list_6 = [ttnn_prepare_conv_bias_1]
    return util_create_list_6


def main_const_eval_7(input):
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_4 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_4, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_to_layout_6,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_8,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_permute_8,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_8, False)
    ttnn_permute_9 = ttnn.permute(
        ttnn_typecast_3,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_permute_9,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_9, False)
    util_create_list_7 = [ttnn_reshape_9]
    return util_create_list_7


def main_const_eval_8(input):
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_1 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_8,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_8 = [ttnn_prepare_conv_weights_1]
    return util_create_list_8


def main_const_eval_9(input):
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_5 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_5, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_5, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_to_layout_7,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_permute_10 = ttnn.permute(
        ttnn_reshape_10,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_permute_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_10, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_typecast_4,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_permute_11,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_11, False)
    util_create_list_9 = [ttnn_reshape_11]
    return util_create_list_9


def main_const_eval_10(input):
    utils_DeviceGetter_get_device_10 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_6 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_to_device_6, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_6, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_to_layout_8,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_8, False)
    ttnn_permute_12 = ttnn.permute(
        ttnn_reshape_12,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_permute_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_12, False)
    ttnn_permute_13 = ttnn.permute(
        ttnn_typecast_5,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_permute_13,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_13, False)
    util_create_list_10 = [ttnn_reshape_13]
    return util_create_list_10


def main_const_eval_11(input):
    utils_DeviceGetter_get_device_11 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_7 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_7, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_7, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_to_layout_9,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    ttnn_permute_14 = ttnn.permute(
        ttnn_reshape_14,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_permute_14,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_14, False)
    ttnn_permute_15 = ttnn.permute(
        ttnn_typecast_6,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_permute_15,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_15, False)
    util_create_list_11 = [ttnn_reshape_15]
    return util_create_list_11


def main_const_eval_12():
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_2 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            298,
            299,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            316,
            317,
            318,
            319,
            320,
            321,
            322,
            323,
            324,
            325,
            326,
            327,
            328,
            329,
            330,
            331,
            332,
            333,
            334,
            335,
            336,
            337,
            338,
            339,
            340,
            341,
            342,
            343,
            344,
            345,
            346,
            347,
            348,
            349,
            350,
            351,
            352,
            353,
            354,
            355,
            356,
            357,
            358,
            359,
            360,
            361,
            362,
            363,
            364,
            365,
            366,
            367,
            368,
            369,
            370,
            371,
            372,
            373,
            374,
            375,
            376,
            377,
            378,
            379,
            380,
            381,
            382,
            383,
            384,
            385,
            386,
            387,
            388,
            389,
            390,
            391,
            392,
            393,
            394,
            395,
            396,
            397,
            398,
            399,
            400,
            401,
            402,
            403,
            404,
            405,
            406,
            407,
            408,
            409,
            410,
            411,
            412,
            413,
            414,
            415,
            416,
            417,
            418,
            419,
            420,
            421,
            422,
            423,
            424,
            425,
            426,
            427,
            428,
            429,
            430,
            431,
            432,
            433,
            434,
            435,
            436,
            437,
            438,
            439,
            440,
            441,
            442,
            443,
            444,
            445,
            446,
            447,
            448,
            449,
            450,
            451,
            452,
            453,
            454,
            455,
            456,
            457,
            458,
            459,
            460,
            461,
            462,
            463,
            464,
            465,
            466,
            467,
            468,
            469,
            470,
            471,
            472,
            473,
            474,
            475,
            476,
            477,
            478,
            479,
            480,
            481,
            482,
            483,
            484,
            485,
            486,
            487,
            488,
            489,
            490,
            491,
            492,
            493,
            494,
            495,
            496,
            497,
            498,
            499,
            500,
            501,
            502,
            503,
            504,
            505,
            506,
            507,
            508,
            509,
            510,
            511,
            512,
            513,
            514,
            515,
            516,
            517,
            518,
            519,
            520,
            521,
            522,
            523,
            524,
            525,
            526,
            527,
            528,
            529,
            530,
            531,
            532,
            533,
            534,
            535,
            536,
            537,
            538,
            539,
            540,
            541,
            542,
            543,
            544,
            545,
            546,
            547,
            548,
            549,
            550,
            551,
            552,
            553,
            554,
            555,
            556,
            557,
            558,
            559,
            560,
            561,
            562,
            563,
            564,
            565,
            566,
            567,
            568,
            569,
            570,
            571,
            572,
            573,
            574,
            575,
            576,
            577,
            578,
            579,
            580,
            581,
            582,
            583,
            584,
            585,
            586,
            587,
            588,
            589,
            590,
            591,
            592,
            593,
            594,
            595,
            596,
            597,
            598,
            599,
            600,
            601,
            602,
            603,
            604,
            605,
            606,
            607,
            608,
            609,
            610,
            611,
            612,
            613,
            614,
            615,
            616,
            617,
            618,
            619,
            620,
            621,
            622,
            623,
            624,
            625,
            626,
            627,
            628,
            629,
            630,
            631,
            632,
            633,
            634,
            635,
            636,
            637,
            638,
            639,
        ],
        [640],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_3 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
            90.0,
            90.5,
            91.0,
            91.5,
            92.0,
            92.5,
            93.0,
            93.5,
            94.0,
            94.5,
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            100.5,
            101.0,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.0,
            104.5,
            105.0,
            105.5,
            106.0,
            106.5,
            107.0,
            107.5,
            108.0,
            108.5,
            109.0,
            109.5,
            110.0,
            110.5,
            111.0,
            111.5,
            112.0,
            112.5,
            113.0,
            113.5,
            114.0,
            114.5,
            115.0,
            115.5,
            116.0,
            116.5,
            117.0,
            117.5,
            118.0,
            118.5,
            119.0,
            119.5,
            120.0,
            120.5,
            121.0,
            121.5,
            122.0,
            122.5,
            123.0,
            123.5,
            124.0,
            124.5,
            125.0,
            125.5,
            126.0,
            126.5,
            127.0,
            127.5,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            130.5,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            135.5,
            136.0,
            136.5,
            137.0,
            137.5,
            138.0,
            138.5,
            139.0,
            139.5,
            140.0,
            140.5,
            141.0,
            141.5,
            142.0,
            142.5,
            143.0,
            143.5,
            144.0,
            144.5,
            145.0,
            145.5,
            146.0,
            146.5,
            147.0,
            147.5,
            148.0,
            148.5,
            149.0,
            149.5,
            150.0,
            150.5,
            151.0,
            151.5,
            152.0,
            152.5,
            153.0,
            153.5,
            154.0,
            154.5,
            155.0,
            155.5,
            156.0,
            156.5,
            157.0,
            157.5,
            158.0,
            158.5,
            159.0,
            159.5,
            160.0,
            160.5,
            161.0,
            161.5,
            162.0,
            162.5,
            163.0,
            163.5,
            164.0,
            164.5,
            165.0,
            165.5,
            166.0,
            166.5,
            167.0,
            167.5,
            168.0,
            168.5,
            169.0,
            169.5,
            170.0,
            170.5,
            171.0,
            171.5,
            172.0,
            172.5,
            173.0,
            173.5,
            174.0,
            174.5,
            175.0,
            175.5,
            176.0,
            176.5,
            177.0,
            177.5,
            178.0,
            178.5,
            179.0,
            179.5,
            180.0,
            180.5,
            181.0,
            181.5,
            182.0,
            182.5,
            183.0,
            183.5,
            184.0,
            184.5,
            185.0,
            185.5,
            186.0,
            186.5,
            187.0,
            187.5,
            188.0,
            188.5,
            189.0,
            189.5,
            190.0,
            190.5,
            191.0,
            191.5,
            192.0,
            192.5,
            193.0,
            193.5,
            194.0,
            194.5,
            195.0,
            195.5,
            196.0,
            196.5,
            197.0,
            197.5,
            198.0,
            198.5,
            199.0,
            199.5,
            200.0,
            200.5,
            201.0,
            201.5,
            202.0,
            202.5,
            203.0,
            203.5,
            204.0,
            204.5,
            205.0,
            205.5,
            206.0,
            206.5,
            207.0,
            207.5,
            208.0,
            208.5,
            209.0,
            209.5,
            210.0,
            210.5,
            211.0,
            211.5,
            212.0,
            212.5,
            213.0,
            213.5,
            214.0,
            214.5,
            215.0,
            215.5,
            216.0,
            216.5,
            217.0,
            217.5,
            218.0,
            218.5,
            219.0,
            219.5,
            220.0,
            220.5,
            221.0,
            221.5,
            222.0,
            222.5,
            223.0,
            223.5,
            224.0,
            224.5,
            225.0,
            225.5,
            226.0,
            226.5,
            227.0,
            227.5,
            228.0,
            228.5,
            229.0,
            229.5,
            230.0,
            230.5,
            231.0,
            231.5,
            232.0,
            232.5,
            233.0,
            233.5,
            234.0,
            234.5,
            235.0,
            235.5,
            236.0,
            236.5,
            237.0,
            237.5,
            238.0,
            238.5,
            239.0,
            239.5,
            240.0,
            240.5,
            241.0,
            241.5,
            242.0,
            242.5,
            243.0,
            243.5,
            244.0,
            244.5,
            245.0,
            245.5,
            246.0,
            246.5,
            247.0,
            247.5,
            248.0,
            248.5,
            249.0,
            249.5,
            250.0,
            250.5,
            251.0,
            251.5,
            252.0,
            252.5,
            253.0,
            253.5,
            254.0,
            254.5,
            255.0,
            255.5,
            256.0,
            256.5,
            257.0,
            257.5,
            258.0,
            258.5,
            259.0,
            259.5,
            260.0,
            260.5,
            261.0,
            261.5,
            262.0,
            262.5,
            263.0,
            263.5,
            264.0,
            264.5,
            265.0,
            265.5,
            266.0,
            266.5,
            267.0,
            267.5,
            268.0,
            268.5,
            269.0,
            269.5,
            270.0,
            270.5,
            271.0,
            271.5,
            272.0,
            272.5,
            273.0,
            273.5,
            274.0,
            274.5,
            275.0,
            275.5,
            276.0,
            276.5,
            277.0,
            277.5,
            278.0,
            278.5,
            279.0,
            279.5,
            280.0,
            280.5,
            281.0,
            281.5,
            282.0,
            282.5,
            283.0,
            283.5,
            284.0,
            284.5,
            285.0,
            285.5,
            286.0,
            286.5,
            287.0,
            287.5,
            288.0,
            288.5,
            289.0,
            289.5,
            290.0,
            290.5,
            291.0,
            291.5,
            292.0,
            292.5,
            293.0,
            293.5,
            294.0,
            294.5,
            295.0,
            295.5,
            296.0,
            296.5,
            297.0,
            297.5,
            298.0,
            298.5,
            299.0,
            299.5,
            300.0,
            300.5,
            301.0,
            301.5,
            302.0,
            302.5,
            303.0,
            303.5,
            304.0,
            304.5,
            305.0,
            305.5,
            306.0,
            306.5,
            307.0,
            307.5,
            308.0,
            308.5,
            309.0,
            309.5,
            310.0,
            310.5,
            311.0,
            311.5,
            312.0,
            312.5,
            313.0,
            313.5,
            314.0,
            314.5,
            315.0,
            315.5,
            316.0,
            316.5,
            317.0,
            317.5,
            318.0,
            318.5,
            319.0,
            319.5,
            320.0,
            320.5,
            321.0,
            321.5,
            322.0,
            322.5,
            323.0,
            323.5,
            324.0,
            324.5,
            325.0,
            325.5,
            326.0,
            326.5,
            327.0,
            327.5,
            328.0,
            328.5,
            329.0,
            329.5,
            330.0,
            330.5,
            331.0,
            331.5,
            332.0,
            332.5,
            333.0,
            333.5,
            334.0,
            334.5,
            335.0,
            335.5,
            336.0,
            336.5,
            337.0,
            337.5,
            338.0,
            338.5,
            339.0,
            339.5,
            340.0,
            340.5,
            341.0,
            341.5,
            342.0,
            342.5,
            343.0,
            343.5,
            344.0,
            344.5,
            345.0,
            345.5,
            346.0,
            346.5,
            347.0,
            347.5,
            348.0,
            348.5,
            349.0,
            349.5,
            350.0,
            350.5,
            351.0,
            351.5,
            352.0,
            352.5,
            353.0,
            353.5,
            354.0,
            354.5,
            355.0,
            355.5,
            356.0,
            356.5,
            357.0,
            357.5,
            358.0,
            358.5,
            359.0,
            359.5,
            360.0,
            360.5,
            361.0,
            361.5,
            362.0,
            362.5,
            363.0,
            363.5,
            364.0,
            364.5,
            365.0,
            365.5,
            366.0,
            366.5,
            367.0,
            367.5,
            368.0,
            368.5,
            369.0,
            369.5,
            370.0,
            370.5,
            371.0,
            371.5,
            372.0,
            372.5,
            373.0,
            373.5,
            374.0,
            374.5,
            375.0,
            375.5,
            376.0,
            376.5,
            377.0,
            377.5,
            378.0,
            378.5,
            379.0,
            379.5,
            380.0,
            380.5,
            381.0,
            381.5,
            382.0,
            382.5,
            383.0,
            383.5,
            384.0,
            384.5,
            385.0,
            385.5,
            386.0,
            386.5,
            387.0,
            387.5,
            388.0,
            388.5,
            389.0,
            389.5,
            390.0,
            390.5,
            391.0,
            391.5,
            392.0,
            392.5,
            393.0,
            393.5,
            394.0,
            394.5,
            395.0,
            395.5,
            396.0,
            396.5,
            397.0,
            397.5,
            398.0,
            398.5,
            399.0,
            399.5,
            400.0,
            400.5,
            401.0,
            401.5,
            402.0,
            402.5,
            403.0,
            403.5,
            404.0,
            404.5,
            405.0,
            405.5,
            406.0,
            406.5,
            407.0,
            407.5,
            408.0,
            408.5,
            409.0,
            409.5,
            410.0,
            410.5,
            411.0,
            411.5,
            412.0,
            412.5,
            413.0,
            413.5,
            414.0,
            414.5,
            415.0,
            415.5,
            416.0,
            416.5,
            417.0,
            417.5,
            418.0,
            418.5,
            419.0,
            419.5,
            420.0,
            420.5,
            421.0,
            421.5,
            422.0,
            422.5,
            423.0,
            423.5,
            424.0,
            424.5,
            425.0,
            425.5,
            426.0,
            426.5,
            427.0,
            427.5,
            428.0,
            428.5,
            429.0,
            429.5,
            430.0,
            430.5,
            431.0,
            431.5,
            432.0,
            432.5,
            433.0,
            433.5,
            434.0,
            434.5,
            435.0,
            435.5,
            436.0,
            436.5,
            437.0,
            437.5,
            438.0,
            438.5,
            439.0,
            439.5,
            440.0,
            440.5,
            441.0,
            441.5,
            442.0,
            442.5,
            443.0,
            443.5,
            444.0,
            444.5,
            445.0,
            445.5,
            446.0,
            446.5,
            447.0,
            447.5,
            448.0,
            448.5,
            449.0,
            449.5,
            450.0,
            450.5,
            451.0,
            451.5,
            452.0,
            452.5,
            453.0,
            453.5,
            454.0,
            454.5,
            455.0,
            455.5,
            456.0,
            456.5,
            457.0,
            457.5,
            458.0,
            458.5,
            459.0,
            459.5,
            460.0,
            460.5,
            461.0,
            461.5,
            462.0,
            462.5,
            463.0,
            463.5,
            464.0,
            464.5,
            465.0,
            465.5,
            466.0,
            466.5,
            467.0,
            467.5,
            468.0,
            468.5,
            469.0,
            469.5,
            470.0,
            470.5,
            471.0,
            471.5,
            472.0,
            472.5,
            473.0,
            473.5,
            474.0,
            474.5,
            475.0,
            475.5,
            476.0,
            476.5,
            477.0,
            477.5,
            478.0,
            478.5,
            479.0,
            479.5,
            480.0,
            480.5,
            481.0,
            481.5,
            482.0,
            482.5,
            483.0,
            483.5,
            484.0,
            484.5,
            485.0,
            485.5,
            486.0,
            486.5,
            487.0,
            487.5,
            488.0,
            488.5,
            489.0,
            489.5,
            490.0,
            490.5,
            491.0,
            491.5,
            492.0,
            492.5,
            493.0,
            493.5,
            494.0,
            494.5,
            495.0,
            495.5,
            496.0,
            496.5,
            497.0,
            497.5,
            498.0,
            498.5,
            499.0,
            499.5,
            500.0,
            500.5,
            501.0,
            501.5,
            502.0,
            502.5,
            503.0,
            503.5,
            504.0,
            504.5,
            505.0,
            505.5,
            506.0,
            506.5,
            507.0,
            507.5,
            508.0,
            508.5,
            509.0,
            509.5,
            510.0,
            510.5,
            511.0,
            511.5,
            512.0,
            512.5,
            513.0,
            513.5,
            514.0,
            514.5,
            515.0,
            515.5,
            516.0,
            516.5,
            517.0,
            517.5,
            518.0,
            518.5,
            519.0,
            519.5,
            520.0,
            520.5,
            521.0,
            521.5,
            522.0,
            522.5,
            523.0,
            523.5,
            524.0,
            524.5,
            525.0,
            525.5,
            526.0,
            526.5,
            527.0,
            527.5,
            528.0,
            528.5,
            529.0,
            529.5,
            530.0,
            530.5,
            531.0,
            531.5,
            532.0,
            532.5,
            533.0,
            533.5,
            534.0,
            534.5,
            535.0,
            535.5,
            536.0,
            536.5,
            537.0,
            537.5,
            538.0,
            538.5,
            539.0,
            539.5,
            540.0,
            540.5,
            541.0,
            541.5,
            542.0,
            542.5,
            543.0,
            543.5,
            544.0,
            544.5,
            545.0,
            545.5,
            546.0,
            546.5,
            547.0,
            547.5,
            548.0,
            548.5,
            549.0,
            549.5,
            550.0,
            550.5,
            551.0,
            551.5,
            552.0,
            552.5,
            553.0,
            553.5,
            554.0,
            554.5,
            555.0,
            555.5,
            556.0,
            556.5,
            557.0,
            557.5,
            558.0,
            558.5,
            559.0,
            559.5,
            560.0,
            560.5,
            561.0,
            561.5,
            562.0,
            562.5,
            563.0,
            563.5,
            564.0,
            564.5,
            565.0,
            565.5,
            566.0,
            566.5,
            567.0,
            567.5,
            568.0,
            568.5,
            569.0,
            569.5,
            570.0,
            570.5,
            571.0,
            571.5,
            572.0,
            572.5,
            573.0,
            573.5,
            574.0,
            574.5,
            575.0,
            575.5,
            576.0,
            576.5,
            577.0,
            577.5,
            578.0,
            578.5,
            579.0,
            579.5,
            580.0,
            580.5,
            581.0,
            581.5,
            582.0,
            582.5,
            583.0,
            583.5,
            584.0,
            584.5,
            585.0,
            585.5,
            586.0,
            586.5,
            587.0,
            587.5,
            588.0,
            588.5,
            589.0,
            589.5,
            590.0,
            590.5,
            591.0,
            591.5,
            592.0,
            592.5,
            593.0,
            593.5,
            594.0,
            594.5,
            595.0,
            595.5,
            596.0,
            596.5,
            597.0,
            597.5,
            598.0,
            598.5,
            599.0,
            599.5,
            600.0,
            600.5,
            601.0,
            601.5,
            602.0,
            602.5,
            603.0,
            603.5,
            604.0,
            604.5,
            605.0,
            605.5,
            606.0,
            606.5,
            607.0,
            607.5,
            608.0,
            608.5,
            609.0,
            609.5,
            610.0,
            610.5,
            611.0,
            611.5,
            612.0,
            612.5,
            613.0,
            613.5,
            614.0,
            614.5,
            615.0,
            615.5,
            616.0,
            616.5,
            617.0,
            617.5,
            618.0,
            618.5,
            619.0,
            619.5,
            620.0,
            620.5,
            621.0,
            621.5,
            622.0,
            622.5,
            623.0,
            623.5,
            624.0,
            624.5,
            625.0,
            625.5,
            626.0,
            626.5,
            627.0,
            627.5,
            628.0,
            628.5,
            629.0,
            629.5,
            630.0,
            630.5,
            631.0,
            631.5,
            632.0,
            632.5,
            633.0,
            633.5,
            634.0,
            634.5,
            635.0,
            635.5,
            636.0,
            636.5,
            637.0,
            637.5,
            638.0,
            638.5,
            639.0,
            639.5,
        ],
        [1280],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_1 = ttnn.floor(
        ttnn_Tensor_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_3, False)
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_floor_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_1, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_typecast_7,
        [1280, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn_permute_16 = ttnn.permute(
        ttnn_reshape_16,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_Tensor_2,
        [1, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_2, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_reshape_17,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_17, False)
    ttnn_eq_1 = ttnn.eq(
        ttnn_permute_16,
        ttnn_permute_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_17, False)
    ttnn.deallocate(ttnn_permute_16, False)
    util_create_list_12 = [ttnn_eq_1]
    return util_create_list_12


def main_const_eval_13(input):
    utils_DeviceGetter_get_device_13 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_8 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_8, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_8, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_to_layout_10,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_permute_18 = ttnn.permute(
        ttnn_reshape_18,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_permute_18,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_18, False)
    ttnn_permute_19 = ttnn.permute(
        ttnn_typecast_8,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_permute_19,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_19, False)
    util_create_list_13 = [ttnn_reshape_19]
    return util_create_list_13


def main_const_eval_14(input):
    utils_DeviceGetter_get_device_14 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_9 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_9, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_9, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_to_layout_11,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn_permute_20 = ttnn.permute(
        ttnn_reshape_20,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_permute_20,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_20, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_typecast_9,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_permute_21,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_21, False)
    util_create_list_14 = [ttnn_reshape_21]
    return util_create_list_14


def main_const_eval_15(input):
    utils_DeviceGetter_get_device_15 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_2 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_15,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_15 = [ttnn_prepare_conv_weights_2]
    return util_create_list_15


def main_const_eval_16():
    utils_DeviceGetter_get_device_16 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 256, 640, 360]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_22 = ttnn.permute(
        ttnn_full_1,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_16 = [ttnn_full_1, ttnn_permute_22]
    return util_create_list_16


def main_const_eval_17(input):
    utils_DeviceGetter_get_device_17 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_3 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_17,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_17 = [ttnn_prepare_conv_weights_3]
    return util_create_list_17


def main_const_eval_18():
    utils_DeviceGetter_get_device_18 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.11572265625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_18 = [ttnn_full_2]
    return util_create_list_18


def main_const_eval_19(input):
    utils_DeviceGetter_get_device_19 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_10 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_19,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_10, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_to_layout_12,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn_permute_23 = ttnn.permute(
        ttnn_reshape_22,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_permute_23,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_23, False)
    ttnn_permute_24 = ttnn.permute(
        ttnn_typecast_10,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_permute_24,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_24, False)
    util_create_list_19 = [ttnn_reshape_23]
    return util_create_list_19


def main_const_eval_20(input):
    utils_DeviceGetter_get_device_20 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_4 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_20,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_20 = [ttnn_prepare_conv_weights_4]
    return util_create_list_20


def main_const_eval_21(input):
    utils_DeviceGetter_get_device_21 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_11 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_to_device_11, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_11, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_to_layout_13,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_reshape_24,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_permute_25,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_25, False)
    ttnn_permute_26 = ttnn.permute(
        ttnn_typecast_11,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_permute_26,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_26, False)
    util_create_list_21 = [ttnn_reshape_25]
    return util_create_list_21


def main_const_eval_22():
    utils_DeviceGetter_get_device_22 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.361328125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_22 = [ttnn_full_3]
    return util_create_list_22


def main_const_eval_23(input):
    utils_DeviceGetter_get_device_23 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_12 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_23,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_to_device_12, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_to_layout_14,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    ttnn_permute_27 = ttnn.permute(
        ttnn_reshape_26,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_permute_27, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_27, False)
    ttnn_from_device_2 = ttnn.from_device(ttnn_to_layout_15)
    ttnn.deallocate(ttnn_to_layout_15, False)
    ttnn_prepare_conv_bias_2 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_2,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_23,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_2, False)
    util_create_list_23 = [ttnn_prepare_conv_bias_2]
    return util_create_list_23


def main_const_eval_24(input):
    utils_DeviceGetter_get_device_24 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_13 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_13, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_13, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_to_layout_16,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    ttnn_permute_28 = ttnn.permute(
        ttnn_reshape_27,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_27, False)
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_permute_28, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_28, False)
    ttnn_from_device_3 = ttnn.from_device(ttnn_to_layout_17)
    ttnn.deallocate(ttnn_to_layout_17, False)
    ttnn_prepare_conv_bias_3 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_3,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_24,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_3, False)
    util_create_list_24 = [ttnn_prepare_conv_bias_3]
    return util_create_list_24


def main_const_eval_25(input):
    utils_DeviceGetter_get_device_25 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_14 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_14, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_to_layout_18,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    ttnn_permute_29 = ttnn.permute(
        ttnn_reshape_28,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_28, False)
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_permute_29, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_from_device_4 = ttnn.from_device(ttnn_to_layout_19)
    ttnn.deallocate(ttnn_to_layout_19, False)
    ttnn_prepare_conv_bias_4 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_4,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_25,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_4, False)
    util_create_list_25 = [ttnn_prepare_conv_bias_4]
    return util_create_list_25


def main_const_eval_26(input):
    utils_DeviceGetter_get_device_26 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_15 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_15, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_15, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_to_layout_20,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_reshape_29,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_permute_30,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_30, False)
    ttnn_permute_31 = ttnn.permute(
        ttnn_typecast_12,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_permute_31,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_31, False)
    util_create_list_26 = [ttnn_reshape_30]
    return util_create_list_26


def main_const_eval_27(input):
    utils_DeviceGetter_get_device_27 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_5 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_27,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_27 = [ttnn_prepare_conv_weights_5]
    return util_create_list_27


def main_const_eval_28(input):
    utils_DeviceGetter_get_device_28 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_6 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_28,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_28 = [ttnn_prepare_conv_weights_6]
    return util_create_list_28


def main_const_eval_29(input):
    utils_DeviceGetter_get_device_29 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_16 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_16, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_16, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_to_layout_21,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn_permute_32 = ttnn.permute(
        ttnn_reshape_31,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_31, False)
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_permute_32,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_32, False)
    ttnn_permute_33 = ttnn.permute(
        ttnn_typecast_13,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_permute_33,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_33, False)
    util_create_list_29 = [ttnn_reshape_32]
    return util_create_list_29


def main_const_eval_30(input):
    utils_DeviceGetter_get_device_30 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_17 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_to_device_17, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_to_layout_22,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    ttnn_permute_34 = ttnn.permute(
        ttnn_reshape_33,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_33, False)
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_permute_34,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_34, False)
    ttnn_permute_35 = ttnn.permute(
        ttnn_typecast_14,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_14, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_permute_35,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_35, False)
    util_create_list_30 = [ttnn_reshape_34]
    return util_create_list_30


def main_const_eval_31(input):
    utils_DeviceGetter_get_device_31 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_18 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_to_device_18, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_18, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_to_layout_23,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    ttnn_permute_36 = ttnn.permute(
        ttnn_reshape_35,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_35, False)
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_permute_36, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_36, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_to_layout_24)
    ttnn.deallocate(ttnn_to_layout_24, False)
    ttnn_prepare_conv_bias_5 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_5,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_31,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_5, False)
    util_create_list_31 = [ttnn_prepare_conv_bias_5]
    return util_create_list_31


def main_const_eval_32(input):
    utils_DeviceGetter_get_device_32 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_19 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_to_device_19, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_19, False)
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_to_layout_25,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_permute_37 = ttnn.permute(
        ttnn_reshape_36,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_36, False)
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_permute_37,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_37, False)
    ttnn_permute_38 = ttnn.permute(
        ttnn_typecast_15,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_permute_38,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_38, False)
    util_create_list_32 = [ttnn_reshape_37]
    return util_create_list_32


def main_const_eval_33(input):
    utils_DeviceGetter_get_device_33 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_7 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_33,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_33 = [ttnn_prepare_conv_weights_7]
    return util_create_list_33


def main_const_eval_34(input):
    utils_DeviceGetter_get_device_34 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_8 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_34,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_34 = [ttnn_prepare_conv_weights_8]
    return util_create_list_34


def main_const_eval_35(input):
    utils_DeviceGetter_get_device_35 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_20 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_35,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_20, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_to_layout_26,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    ttnn_permute_39 = ttnn.permute(
        ttnn_reshape_38,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_38, False)
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_permute_39, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_39, False)
    ttnn_from_device_6 = ttnn.from_device(ttnn_to_layout_27)
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_prepare_conv_bias_6 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_6,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_35,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_6, False)
    util_create_list_35 = [ttnn_prepare_conv_bias_6]
    return util_create_list_35


def main_const_eval_36(input):
    utils_DeviceGetter_get_device_36 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_21 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_36,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_21, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_21, False)
    ttnn_reshape_39 = ttnn.reshape(
        ttnn_to_layout_28,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    ttnn_permute_40 = ttnn.permute(
        ttnn_reshape_39,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_39, False)
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_permute_40,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_40, False)
    ttnn_permute_41 = ttnn.permute(
        ttnn_typecast_16,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn_reshape_40 = ttnn.reshape(
        ttnn_permute_41,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_41, False)
    util_create_list_36 = [ttnn_reshape_40]
    return util_create_list_36


def main_const_eval_37(input):
    utils_DeviceGetter_get_device_37 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_22 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_37,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_22, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_22, False)
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_to_layout_29,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_permute_42 = ttnn.permute(
        ttnn_reshape_41,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_41, False)
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_permute_42,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_42, False)
    ttnn_permute_43 = ttnn.permute(
        ttnn_typecast_17,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_permute_43,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_43, False)
    util_create_list_37 = [ttnn_reshape_42]
    return util_create_list_37


def main_const_eval_38(input):
    utils_DeviceGetter_get_device_38 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_23 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_38,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_23, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_23, False)
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_to_layout_30,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn_permute_44 = ttnn.permute(
        ttnn_reshape_43,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_43, False)
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_permute_44,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_44, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_typecast_18,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_permute_45,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    util_create_list_38 = [ttnn_reshape_44]
    return util_create_list_38


def main_const_eval_39(input):
    utils_DeviceGetter_get_device_39 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_24 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_39,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_24, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_to_layout_31,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_permute_46 = ttnn.permute(
        ttnn_reshape_45,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_45, False)
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_permute_46,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_46, False)
    ttnn_permute_47 = ttnn.permute(
        ttnn_typecast_19,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_19, False)
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_permute_47,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_47, False)
    util_create_list_39 = [ttnn_reshape_46]
    return util_create_list_39


def main_const_eval_40(input):
    utils_DeviceGetter_get_device_40 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_25 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_40,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_to_device_25, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_to_layout_32,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_permute_48 = ttnn.permute(
        ttnn_reshape_47,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_47, False)
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_permute_48, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_48, False)
    ttnn_from_device_7 = ttnn.from_device(ttnn_to_layout_33)
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_prepare_conv_bias_7 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_7,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_40,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_7, False)
    util_create_list_40 = [ttnn_prepare_conv_bias_7]
    return util_create_list_40


def main_const_eval_41(input):
    utils_DeviceGetter_get_device_41 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_26 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_41,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_26, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_reshape_48 = ttnn.reshape(
        ttnn_to_layout_34,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_permute_49 = ttnn.permute(
        ttnn_reshape_48,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_48, False)
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_permute_49,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_49, False)
    ttnn_permute_50 = ttnn.permute(
        ttnn_typecast_20,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_permute_50,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_50, False)
    util_create_list_41 = [ttnn_reshape_49]
    return util_create_list_41


def main_const_eval_42(input):
    utils_DeviceGetter_get_device_42 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_27 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_to_device_27, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_to_layout_35,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    ttnn_permute_51 = ttnn.permute(
        ttnn_reshape_50,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_50, False)
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_permute_51,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_51, False)
    ttnn_permute_52 = ttnn.permute(
        ttnn_typecast_21,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn_reshape_51 = ttnn.reshape(
        ttnn_permute_52,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_52, False)
    util_create_list_42 = [ttnn_reshape_51]
    return util_create_list_42


def main_const_eval_43(input):
    utils_DeviceGetter_get_device_43 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_9 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_43,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_43 = [ttnn_prepare_conv_weights_9]
    return util_create_list_43


def main_const_eval_44(input):
    utils_DeviceGetter_get_device_44 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_28 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_44,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_28, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_to_layout_36,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_36, False)
    ttnn_permute_53 = ttnn.permute(
        ttnn_reshape_52,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_52, False)
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_permute_53, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_53, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_to_layout_37)
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn_prepare_conv_bias_8 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_8,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_44,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_8, False)
    util_create_list_44 = [ttnn_prepare_conv_bias_8]
    return util_create_list_44


def main_const_eval_45(input):
    utils_DeviceGetter_get_device_45 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_29 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_45,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_29, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_29, False)
    ttnn_reshape_53 = ttnn.reshape(
        ttnn_to_layout_38,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn_permute_54 = ttnn.permute(
        ttnn_reshape_53,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_53, False)
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_permute_54,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_54, False)
    ttnn_permute_55 = ttnn.permute(
        ttnn_typecast_22,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_permute_55,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_55, False)
    util_create_list_45 = [ttnn_reshape_54]
    return util_create_list_45


def main_const_eval_46(input):
    utils_DeviceGetter_get_device_46 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_10 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_46,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_46 = [ttnn_prepare_conv_weights_10]
    return util_create_list_46


def main_const_eval_47(input):
    utils_DeviceGetter_get_device_47 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_11 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_47,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_47 = [ttnn_prepare_conv_weights_11]
    return util_create_list_47


def main_const_eval_48(input):
    utils_DeviceGetter_get_device_48 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_30 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_48,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_30, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_reshape_55 = ttnn.reshape(
        ttnn_to_layout_39,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_permute_56 = ttnn.permute(
        ttnn_reshape_55,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_55, False)
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_permute_56, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_56, False)
    ttnn_from_device_9 = ttnn.from_device(ttnn_to_layout_40)
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_prepare_conv_bias_9 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_9,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_48,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_9, False)
    util_create_list_48 = [ttnn_prepare_conv_bias_9]
    return util_create_list_48


def main_const_eval_49(input):
    utils_DeviceGetter_get_device_49 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_31 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_49,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_31, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_to_layout_41,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_permute_57 = ttnn.permute(
        ttnn_reshape_56,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_56, False)
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_permute_57,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_57, False)
    ttnn_permute_58 = ttnn.permute(
        ttnn_typecast_23,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn_reshape_57 = ttnn.reshape(
        ttnn_permute_58,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_58, False)
    util_create_list_49 = [ttnn_reshape_57]
    return util_create_list_49


def main_const_eval_50(input):
    utils_DeviceGetter_get_device_50 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_32 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_50,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_32, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_to_layout_42,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn_permute_59 = ttnn.permute(
        ttnn_reshape_58,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_58, False)
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_permute_59, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_59, False)
    ttnn_from_device_10 = ttnn.from_device(ttnn_to_layout_43)
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn_prepare_conv_bias_10 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_10,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_50,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_10, False)
    util_create_list_50 = [ttnn_prepare_conv_bias_10]
    return util_create_list_50


def main_const_eval_51():
    utils_DeviceGetter_get_device_51 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_4 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            298,
            299,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            316,
            317,
            318,
            319,
            320,
            321,
            322,
            323,
            324,
            325,
            326,
            327,
            328,
            329,
            330,
            331,
            332,
            333,
            334,
            335,
            336,
            337,
            338,
            339,
            340,
            341,
            342,
            343,
            344,
            345,
            346,
            347,
            348,
            349,
            350,
            351,
            352,
            353,
            354,
            355,
            356,
            357,
            358,
            359,
        ],
        [360],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_5 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
            90.0,
            90.5,
            91.0,
            91.5,
            92.0,
            92.5,
            93.0,
            93.5,
            94.0,
            94.5,
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            100.5,
            101.0,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.0,
            104.5,
            105.0,
            105.5,
            106.0,
            106.5,
            107.0,
            107.5,
            108.0,
            108.5,
            109.0,
            109.5,
            110.0,
            110.5,
            111.0,
            111.5,
            112.0,
            112.5,
            113.0,
            113.5,
            114.0,
            114.5,
            115.0,
            115.5,
            116.0,
            116.5,
            117.0,
            117.5,
            118.0,
            118.5,
            119.0,
            119.5,
            120.0,
            120.5,
            121.0,
            121.5,
            122.0,
            122.5,
            123.0,
            123.5,
            124.0,
            124.5,
            125.0,
            125.5,
            126.0,
            126.5,
            127.0,
            127.5,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            130.5,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            135.5,
            136.0,
            136.5,
            137.0,
            137.5,
            138.0,
            138.5,
            139.0,
            139.5,
            140.0,
            140.5,
            141.0,
            141.5,
            142.0,
            142.5,
            143.0,
            143.5,
            144.0,
            144.5,
            145.0,
            145.5,
            146.0,
            146.5,
            147.0,
            147.5,
            148.0,
            148.5,
            149.0,
            149.5,
            150.0,
            150.5,
            151.0,
            151.5,
            152.0,
            152.5,
            153.0,
            153.5,
            154.0,
            154.5,
            155.0,
            155.5,
            156.0,
            156.5,
            157.0,
            157.5,
            158.0,
            158.5,
            159.0,
            159.5,
            160.0,
            160.5,
            161.0,
            161.5,
            162.0,
            162.5,
            163.0,
            163.5,
            164.0,
            164.5,
            165.0,
            165.5,
            166.0,
            166.5,
            167.0,
            167.5,
            168.0,
            168.5,
            169.0,
            169.5,
            170.0,
            170.5,
            171.0,
            171.5,
            172.0,
            172.5,
            173.0,
            173.5,
            174.0,
            174.5,
            175.0,
            175.5,
            176.0,
            176.5,
            177.0,
            177.5,
            178.0,
            178.5,
            179.0,
            179.5,
            180.0,
            180.5,
            181.0,
            181.5,
            182.0,
            182.5,
            183.0,
            183.5,
            184.0,
            184.5,
            185.0,
            185.5,
            186.0,
            186.5,
            187.0,
            187.5,
            188.0,
            188.5,
            189.0,
            189.5,
            190.0,
            190.5,
            191.0,
            191.5,
            192.0,
            192.5,
            193.0,
            193.5,
            194.0,
            194.5,
            195.0,
            195.5,
            196.0,
            196.5,
            197.0,
            197.5,
            198.0,
            198.5,
            199.0,
            199.5,
            200.0,
            200.5,
            201.0,
            201.5,
            202.0,
            202.5,
            203.0,
            203.5,
            204.0,
            204.5,
            205.0,
            205.5,
            206.0,
            206.5,
            207.0,
            207.5,
            208.0,
            208.5,
            209.0,
            209.5,
            210.0,
            210.5,
            211.0,
            211.5,
            212.0,
            212.5,
            213.0,
            213.5,
            214.0,
            214.5,
            215.0,
            215.5,
            216.0,
            216.5,
            217.0,
            217.5,
            218.0,
            218.5,
            219.0,
            219.5,
            220.0,
            220.5,
            221.0,
            221.5,
            222.0,
            222.5,
            223.0,
            223.5,
            224.0,
            224.5,
            225.0,
            225.5,
            226.0,
            226.5,
            227.0,
            227.5,
            228.0,
            228.5,
            229.0,
            229.5,
            230.0,
            230.5,
            231.0,
            231.5,
            232.0,
            232.5,
            233.0,
            233.5,
            234.0,
            234.5,
            235.0,
            235.5,
            236.0,
            236.5,
            237.0,
            237.5,
            238.0,
            238.5,
            239.0,
            239.5,
            240.0,
            240.5,
            241.0,
            241.5,
            242.0,
            242.5,
            243.0,
            243.5,
            244.0,
            244.5,
            245.0,
            245.5,
            246.0,
            246.5,
            247.0,
            247.5,
            248.0,
            248.5,
            249.0,
            249.5,
            250.0,
            250.5,
            251.0,
            251.5,
            252.0,
            252.5,
            253.0,
            253.5,
            254.0,
            254.5,
            255.0,
            255.5,
            256.0,
            256.5,
            257.0,
            257.5,
            258.0,
            258.5,
            259.0,
            259.5,
            260.0,
            260.5,
            261.0,
            261.5,
            262.0,
            262.5,
            263.0,
            263.5,
            264.0,
            264.5,
            265.0,
            265.5,
            266.0,
            266.5,
            267.0,
            267.5,
            268.0,
            268.5,
            269.0,
            269.5,
            270.0,
            270.5,
            271.0,
            271.5,
            272.0,
            272.5,
            273.0,
            273.5,
            274.0,
            274.5,
            275.0,
            275.5,
            276.0,
            276.5,
            277.0,
            277.5,
            278.0,
            278.5,
            279.0,
            279.5,
            280.0,
            280.5,
            281.0,
            281.5,
            282.0,
            282.5,
            283.0,
            283.5,
            284.0,
            284.5,
            285.0,
            285.5,
            286.0,
            286.5,
            287.0,
            287.5,
            288.0,
            288.5,
            289.0,
            289.5,
            290.0,
            290.5,
            291.0,
            291.5,
            292.0,
            292.5,
            293.0,
            293.5,
            294.0,
            294.5,
            295.0,
            295.5,
            296.0,
            296.5,
            297.0,
            297.5,
            298.0,
            298.5,
            299.0,
            299.5,
            300.0,
            300.5,
            301.0,
            301.5,
            302.0,
            302.5,
            303.0,
            303.5,
            304.0,
            304.5,
            305.0,
            305.5,
            306.0,
            306.5,
            307.0,
            307.5,
            308.0,
            308.5,
            309.0,
            309.5,
            310.0,
            310.5,
            311.0,
            311.5,
            312.0,
            312.5,
            313.0,
            313.5,
            314.0,
            314.5,
            315.0,
            315.5,
            316.0,
            316.5,
            317.0,
            317.5,
            318.0,
            318.5,
            319.0,
            319.5,
            320.0,
            320.5,
            321.0,
            321.5,
            322.0,
            322.5,
            323.0,
            323.5,
            324.0,
            324.5,
            325.0,
            325.5,
            326.0,
            326.5,
            327.0,
            327.5,
            328.0,
            328.5,
            329.0,
            329.5,
            330.0,
            330.5,
            331.0,
            331.5,
            332.0,
            332.5,
            333.0,
            333.5,
            334.0,
            334.5,
            335.0,
            335.5,
            336.0,
            336.5,
            337.0,
            337.5,
            338.0,
            338.5,
            339.0,
            339.5,
            340.0,
            340.5,
            341.0,
            341.5,
            342.0,
            342.5,
            343.0,
            343.5,
            344.0,
            344.5,
            345.0,
            345.5,
            346.0,
            346.5,
            347.0,
            347.5,
            348.0,
            348.5,
            349.0,
            349.5,
            350.0,
            350.5,
            351.0,
            351.5,
            352.0,
            352.5,
            353.0,
            353.5,
            354.0,
            354.5,
            355.0,
            355.5,
            356.0,
            356.5,
            357.0,
            357.5,
            358.0,
            358.5,
            359.0,
            359.5,
        ],
        [720],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_51,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_2 = ttnn.floor(
        ttnn_Tensor_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_5, False)
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_floor_2,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_2, False)
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_typecast_24,
        [720, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn_permute_60 = ttnn.permute(
        ttnn_reshape_59,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_59, False)
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_Tensor_4,
        [1, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_4, False)
    ttnn_permute_61 = ttnn.permute(
        ttnn_reshape_60,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_60, False)
    ttnn_eq_2 = ttnn.eq(
        ttnn_permute_60,
        ttnn_permute_61,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_61, False)
    ttnn.deallocate(ttnn_permute_60, False)
    util_create_list_51 = [ttnn_eq_2]
    return util_create_list_51


def main_const_eval_52(input):
    utils_DeviceGetter_get_device_52 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_33 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_52,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_33, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_to_layout_44,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_permute_62 = ttnn.permute(
        ttnn_reshape_61,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_61, False)
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_permute_62, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_62, False)
    ttnn_from_device_11 = ttnn.from_device(ttnn_to_layout_45)
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_prepare_conv_bias_11 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_11,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_52,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_11, False)
    util_create_list_52 = [ttnn_prepare_conv_bias_11]
    return util_create_list_52


def main_const_eval_53(input):
    utils_DeviceGetter_get_device_53 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_34 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_53,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_to_device_34, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_to_layout_46,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_permute_63 = ttnn.permute(
        ttnn_reshape_62,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_62, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_permute_63,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_63, False)
    ttnn_permute_64 = ttnn.permute(
        ttnn_typecast_25,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_permute_64,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_64, False)
    util_create_list_53 = [ttnn_reshape_63]
    return util_create_list_53


def main_const_eval_54(input):
    utils_DeviceGetter_get_device_54 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_35 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_54,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_35, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_35, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_to_layout_47,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_permute_65 = ttnn.permute(
        ttnn_reshape_64,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_64, False)
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_permute_65, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_65, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_to_layout_48)
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_prepare_conv_bias_12 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_12,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_54,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_12, False)
    util_create_list_54 = [ttnn_prepare_conv_bias_12]
    return util_create_list_54


def main_const_eval_55(input):
    utils_DeviceGetter_get_device_55 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_36 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_55,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_to_device_36, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_to_layout_49,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_49, False)
    ttnn_permute_66 = ttnn.permute(
        ttnn_reshape_65,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_65, False)
    ttnn_typecast_26 = ttnn.typecast(
        ttnn_permute_66,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_66, False)
    ttnn_permute_67 = ttnn.permute(
        ttnn_typecast_26,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_permute_67,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_67, False)
    util_create_list_55 = [ttnn_reshape_66]
    return util_create_list_55


def main_const_eval_56(input):
    utils_DeviceGetter_get_device_56 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_37 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_56,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_37, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_to_layout_50,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_50, False)
    ttnn_permute_68 = ttnn.permute(
        ttnn_reshape_67,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_67, False)
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_permute_68, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_68, False)
    ttnn_from_device_13 = ttnn.from_device(ttnn_to_layout_51)
    ttnn.deallocate(ttnn_to_layout_51, False)
    ttnn_prepare_conv_bias_13 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_13,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_56,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_13, False)
    util_create_list_56 = [ttnn_prepare_conv_bias_13]
    return util_create_list_56


def main_const_eval_57(input):
    utils_DeviceGetter_get_device_57 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_38 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_57,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_38, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_to_layout_52,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_52, False)
    ttnn_permute_69 = ttnn.permute(
        ttnn_reshape_68,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_68, False)
    ttnn_typecast_27 = ttnn.typecast(
        ttnn_permute_69,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_69, False)
    ttnn_permute_70 = ttnn.permute(
        ttnn_typecast_27,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_27, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_permute_70,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_70, False)
    util_create_list_57 = [ttnn_reshape_69]
    return util_create_list_57


def main_const_eval_58(input):
    utils_DeviceGetter_get_device_58 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_12 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_58,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_58 = [ttnn_prepare_conv_weights_12]
    return util_create_list_58


def main_const_eval_59(input):
    utils_DeviceGetter_get_device_59 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_39 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_59,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_to_device_39, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_to_layout_53,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_53, False)
    ttnn_repeat_0 = ttnn.repeat(
        ttnn_reshape_70,
        ttnn.Shape([1, 14400, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_70, False)
    util_create_list_59 = [ttnn_repeat_0]
    return util_create_list_59


def main_const_eval_60(input):
    utils_DeviceGetter_get_device_60 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_40 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_60,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_to_device_40, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_to_layout_54,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_54, False)
    ttnn_permute_71 = ttnn.permute(
        ttnn_reshape_71,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_71, False)
    ttnn_typecast_28 = ttnn.typecast(
        ttnn_permute_71,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_71, False)
    ttnn_permute_72 = ttnn.permute(
        ttnn_typecast_28,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_28, False)
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_permute_72,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_72, False)
    util_create_list_60 = [ttnn_reshape_72]
    return util_create_list_60


def main_const_eval_61(input):
    utils_DeviceGetter_get_device_61 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_41 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_61,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_to_device_41, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_to_layout_55,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_55, False)
    ttnn_permute_73 = ttnn.permute(
        ttnn_reshape_73,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_73, False)
    ttnn_typecast_29 = ttnn.typecast(
        ttnn_permute_73,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_73, False)
    ttnn_permute_74 = ttnn.permute(
        ttnn_typecast_29,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_permute_74,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_74, False)
    util_create_list_61 = [ttnn_reshape_74]
    return util_create_list_61


def main_const_eval_62():
    utils_DeviceGetter_get_device_62 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_6 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            298,
            299,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            316,
            317,
            318,
            319,
        ],
        [320],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_7 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
            90.0,
            90.5,
            91.0,
            91.5,
            92.0,
            92.5,
            93.0,
            93.5,
            94.0,
            94.5,
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            100.5,
            101.0,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.0,
            104.5,
            105.0,
            105.5,
            106.0,
            106.5,
            107.0,
            107.5,
            108.0,
            108.5,
            109.0,
            109.5,
            110.0,
            110.5,
            111.0,
            111.5,
            112.0,
            112.5,
            113.0,
            113.5,
            114.0,
            114.5,
            115.0,
            115.5,
            116.0,
            116.5,
            117.0,
            117.5,
            118.0,
            118.5,
            119.0,
            119.5,
            120.0,
            120.5,
            121.0,
            121.5,
            122.0,
            122.5,
            123.0,
            123.5,
            124.0,
            124.5,
            125.0,
            125.5,
            126.0,
            126.5,
            127.0,
            127.5,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            130.5,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            135.5,
            136.0,
            136.5,
            137.0,
            137.5,
            138.0,
            138.5,
            139.0,
            139.5,
            140.0,
            140.5,
            141.0,
            141.5,
            142.0,
            142.5,
            143.0,
            143.5,
            144.0,
            144.5,
            145.0,
            145.5,
            146.0,
            146.5,
            147.0,
            147.5,
            148.0,
            148.5,
            149.0,
            149.5,
            150.0,
            150.5,
            151.0,
            151.5,
            152.0,
            152.5,
            153.0,
            153.5,
            154.0,
            154.5,
            155.0,
            155.5,
            156.0,
            156.5,
            157.0,
            157.5,
            158.0,
            158.5,
            159.0,
            159.5,
            160.0,
            160.5,
            161.0,
            161.5,
            162.0,
            162.5,
            163.0,
            163.5,
            164.0,
            164.5,
            165.0,
            165.5,
            166.0,
            166.5,
            167.0,
            167.5,
            168.0,
            168.5,
            169.0,
            169.5,
            170.0,
            170.5,
            171.0,
            171.5,
            172.0,
            172.5,
            173.0,
            173.5,
            174.0,
            174.5,
            175.0,
            175.5,
            176.0,
            176.5,
            177.0,
            177.5,
            178.0,
            178.5,
            179.0,
            179.5,
            180.0,
            180.5,
            181.0,
            181.5,
            182.0,
            182.5,
            183.0,
            183.5,
            184.0,
            184.5,
            185.0,
            185.5,
            186.0,
            186.5,
            187.0,
            187.5,
            188.0,
            188.5,
            189.0,
            189.5,
            190.0,
            190.5,
            191.0,
            191.5,
            192.0,
            192.5,
            193.0,
            193.5,
            194.0,
            194.5,
            195.0,
            195.5,
            196.0,
            196.5,
            197.0,
            197.5,
            198.0,
            198.5,
            199.0,
            199.5,
            200.0,
            200.5,
            201.0,
            201.5,
            202.0,
            202.5,
            203.0,
            203.5,
            204.0,
            204.5,
            205.0,
            205.5,
            206.0,
            206.5,
            207.0,
            207.5,
            208.0,
            208.5,
            209.0,
            209.5,
            210.0,
            210.5,
            211.0,
            211.5,
            212.0,
            212.5,
            213.0,
            213.5,
            214.0,
            214.5,
            215.0,
            215.5,
            216.0,
            216.5,
            217.0,
            217.5,
            218.0,
            218.5,
            219.0,
            219.5,
            220.0,
            220.5,
            221.0,
            221.5,
            222.0,
            222.5,
            223.0,
            223.5,
            224.0,
            224.5,
            225.0,
            225.5,
            226.0,
            226.5,
            227.0,
            227.5,
            228.0,
            228.5,
            229.0,
            229.5,
            230.0,
            230.5,
            231.0,
            231.5,
            232.0,
            232.5,
            233.0,
            233.5,
            234.0,
            234.5,
            235.0,
            235.5,
            236.0,
            236.5,
            237.0,
            237.5,
            238.0,
            238.5,
            239.0,
            239.5,
            240.0,
            240.5,
            241.0,
            241.5,
            242.0,
            242.5,
            243.0,
            243.5,
            244.0,
            244.5,
            245.0,
            245.5,
            246.0,
            246.5,
            247.0,
            247.5,
            248.0,
            248.5,
            249.0,
            249.5,
            250.0,
            250.5,
            251.0,
            251.5,
            252.0,
            252.5,
            253.0,
            253.5,
            254.0,
            254.5,
            255.0,
            255.5,
            256.0,
            256.5,
            257.0,
            257.5,
            258.0,
            258.5,
            259.0,
            259.5,
            260.0,
            260.5,
            261.0,
            261.5,
            262.0,
            262.5,
            263.0,
            263.5,
            264.0,
            264.5,
            265.0,
            265.5,
            266.0,
            266.5,
            267.0,
            267.5,
            268.0,
            268.5,
            269.0,
            269.5,
            270.0,
            270.5,
            271.0,
            271.5,
            272.0,
            272.5,
            273.0,
            273.5,
            274.0,
            274.5,
            275.0,
            275.5,
            276.0,
            276.5,
            277.0,
            277.5,
            278.0,
            278.5,
            279.0,
            279.5,
            280.0,
            280.5,
            281.0,
            281.5,
            282.0,
            282.5,
            283.0,
            283.5,
            284.0,
            284.5,
            285.0,
            285.5,
            286.0,
            286.5,
            287.0,
            287.5,
            288.0,
            288.5,
            289.0,
            289.5,
            290.0,
            290.5,
            291.0,
            291.5,
            292.0,
            292.5,
            293.0,
            293.5,
            294.0,
            294.5,
            295.0,
            295.5,
            296.0,
            296.5,
            297.0,
            297.5,
            298.0,
            298.5,
            299.0,
            299.5,
            300.0,
            300.5,
            301.0,
            301.5,
            302.0,
            302.5,
            303.0,
            303.5,
            304.0,
            304.5,
            305.0,
            305.5,
            306.0,
            306.5,
            307.0,
            307.5,
            308.0,
            308.5,
            309.0,
            309.5,
            310.0,
            310.5,
            311.0,
            311.5,
            312.0,
            312.5,
            313.0,
            313.5,
            314.0,
            314.5,
            315.0,
            315.5,
            316.0,
            316.5,
            317.0,
            317.5,
            318.0,
            318.5,
            319.0,
            319.5,
        ],
        [640],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_62,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_3 = ttnn.floor(
        ttnn_Tensor_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_7, False)
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_floor_3,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_3, False)
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_typecast_30,
        [640, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    ttnn_permute_75 = ttnn.permute(
        ttnn_reshape_75,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_75, False)
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_Tensor_6,
        [1, 320],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_6, False)
    ttnn_permute_76 = ttnn.permute(
        ttnn_reshape_76,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_76, False)
    ttnn_eq_3 = ttnn.eq(
        ttnn_permute_75,
        ttnn_permute_76,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_76, False)
    ttnn.deallocate(ttnn_permute_75, False)
    util_create_list_62 = [ttnn_eq_3]
    return util_create_list_62


def main_const_eval_63(input):
    utils_DeviceGetter_get_device_63 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_42 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_63,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_to_device_42, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_reshape_77 = ttnn.reshape(
        ttnn_to_layout_56,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_56, False)
    ttnn_permute_77 = ttnn.permute(
        ttnn_reshape_77,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_77, False)
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_permute_77, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_77, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_to_layout_57)
    ttnn.deallocate(ttnn_to_layout_57, False)
    ttnn_prepare_conv_bias_14 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_14,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_63,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_14, False)
    util_create_list_63 = [ttnn_prepare_conv_bias_14]
    return util_create_list_63


def main_const_eval_64(input):
    utils_DeviceGetter_get_device_64 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_43 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_64,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_to_device_43, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_43, False)
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_to_layout_58,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_58, False)
    ttnn_permute_78 = ttnn.permute(
        ttnn_reshape_78,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_78, False)
    ttnn_typecast_31 = ttnn.typecast(
        ttnn_permute_78,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_78, False)
    ttnn_permute_79 = ttnn.permute(
        ttnn_typecast_31,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_permute_79,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_79, False)
    util_create_list_64 = [ttnn_reshape_79]
    return util_create_list_64


def main_const_eval_65(input):
    utils_DeviceGetter_get_device_65 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_44 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_65,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_to_device_44, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_to_layout_59,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_59, False)
    ttnn_permute_80 = ttnn.permute(
        ttnn_reshape_80,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_80, False)
    ttnn_typecast_32 = ttnn.typecast(
        ttnn_permute_80,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_80, False)
    ttnn_permute_81 = ttnn.permute(
        ttnn_typecast_32,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_permute_81,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_81, False)
    util_create_list_65 = [ttnn_reshape_81]
    return util_create_list_65


def main_const_eval_66(input):
    utils_DeviceGetter_get_device_66 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_45 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_66,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_to_device_45, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_45, False)
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_to_layout_60,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_60, False)
    ttnn_permute_82 = ttnn.permute(
        ttnn_reshape_82,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_82, False)
    ttnn_typecast_33 = ttnn.typecast(
        ttnn_permute_82,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_82, False)
    ttnn_permute_83 = ttnn.permute(
        ttnn_typecast_33,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn_reshape_83 = ttnn.reshape(
        ttnn_permute_83,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_83, False)
    util_create_list_66 = [ttnn_reshape_83]
    return util_create_list_66


def main_const_eval_67(input):
    utils_DeviceGetter_get_device_67 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_46 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_67,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_to_device_46, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_to_layout_61,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_61, False)
    ttnn_permute_84 = ttnn.permute(
        ttnn_reshape_84,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_84, False)
    ttnn_typecast_34 = ttnn.typecast(
        ttnn_permute_84,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_84, False)
    ttnn_permute_85 = ttnn.permute(
        ttnn_typecast_34,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_reshape_85 = ttnn.reshape(
        ttnn_permute_85,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_85, False)
    util_create_list_67 = [ttnn_reshape_85]
    return util_create_list_67


def main_const_eval_68(input):
    utils_DeviceGetter_get_device_68 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_47 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_68,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_to_device_47, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_47, False)
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_to_layout_62,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_62, False)
    ttnn_permute_86 = ttnn.permute(
        ttnn_reshape_86,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_86, False)
    ttnn_typecast_35 = ttnn.typecast(
        ttnn_permute_86,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_86, False)
    ttnn_permute_87 = ttnn.permute(
        ttnn_typecast_35,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn_reshape_87 = ttnn.reshape(
        ttnn_permute_87,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_87, False)
    util_create_list_68 = [ttnn_reshape_87]
    return util_create_list_68


def main_const_eval_69(input):
    utils_DeviceGetter_get_device_69 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_48 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_69,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_to_device_48, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_reshape_88 = ttnn.reshape(
        ttnn_to_layout_63,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_63, False)
    ttnn_permute_88 = ttnn.permute(
        ttnn_reshape_88,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_88, False)
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_permute_88, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_88, False)
    ttnn_from_device_15 = ttnn.from_device(ttnn_to_layout_64)
    ttnn.deallocate(ttnn_to_layout_64, False)
    ttnn_prepare_conv_bias_15 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_15,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_69,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_15, False)
    util_create_list_69 = [ttnn_prepare_conv_bias_15]
    return util_create_list_69


def main_const_eval_70(input):
    utils_DeviceGetter_get_device_70 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_13 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_70,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_70 = [ttnn_prepare_conv_weights_13]
    return util_create_list_70


def main_const_eval_71(input):
    utils_DeviceGetter_get_device_71 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_14 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_71,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_71 = [ttnn_prepare_conv_weights_14]
    return util_create_list_71


def main_const_eval_72(input):
    utils_DeviceGetter_get_device_72 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_49 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_72,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_to_device_49, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_to_layout_65,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_65, False)
    ttnn_permute_89 = ttnn.permute(
        ttnn_reshape_89,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_89, False)
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_permute_89, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_89, False)
    ttnn_from_device_16 = ttnn.from_device(ttnn_to_layout_66)
    ttnn.deallocate(ttnn_to_layout_66, False)
    ttnn_prepare_conv_bias_16 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_16,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_72,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_16, False)
    util_create_list_72 = [ttnn_prepare_conv_bias_16]
    return util_create_list_72


def main_const_eval_73(input):
    utils_DeviceGetter_get_device_73 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_15 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_73,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_73 = [ttnn_prepare_conv_weights_15]
    return util_create_list_73


def main_const_eval_74(input):
    utils_DeviceGetter_get_device_74 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_50 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_74,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_to_device_50, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_50, False)
    ttnn_reshape_90 = ttnn.reshape(
        ttnn_to_layout_67,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    ttnn_permute_90 = ttnn.permute(
        ttnn_reshape_90,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_90, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_permute_90,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_90, False)
    ttnn_permute_91 = ttnn.permute(
        ttnn_typecast_36,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_permute_91,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_91, False)
    util_create_list_74 = [ttnn_reshape_91]
    return util_create_list_74


def main_const_eval_75(input):
    utils_DeviceGetter_get_device_75 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_51 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_75,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_to_device_51, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_reshape_92 = ttnn.reshape(
        ttnn_to_layout_68,
        [1, 512, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    ttnn_permute_92 = ttnn.permute(
        ttnn_reshape_92,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_92, False)
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_permute_92,
        [1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_92, False)
    ttnn_typecast_37 = ttnn.typecast(
        ttnn_reshape_93,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_93, False)
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_typecast_37,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn_permute_93 = ttnn.permute(
        ttnn_reshape_94,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_94, False)
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_permute_93,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_93, False)
    util_create_list_75 = [ttnn_reshape_95]
    return util_create_list_75


def main_const_eval_76(input):
    utils_DeviceGetter_get_device_76 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_16 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=16,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_76,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_76 = [ttnn_prepare_conv_weights_16]
    return util_create_list_76


def main_const_eval_77(input):
    utils_DeviceGetter_get_device_77 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_52 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_77,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_to_device_52, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_52, False)
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_to_layout_69,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    ttnn_permute_94 = ttnn.permute(
        ttnn_reshape_96,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_96, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_permute_94,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_94, False)
    ttnn_permute_95 = ttnn.permute(
        ttnn_typecast_38,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_permute_95,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_95, False)
    util_create_list_77 = [ttnn_reshape_97]
    return util_create_list_77


def main_const_eval_78(input):
    utils_DeviceGetter_get_device_78 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_53 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_78,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_to_device_53, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_to_layout_70,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_70, False)
    ttnn_permute_96 = ttnn.permute(
        ttnn_reshape_98,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_98, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_permute_96,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_96, False)
    ttnn_permute_97 = ttnn.permute(
        ttnn_typecast_39,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_permute_97,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_97, False)
    util_create_list_78 = [ttnn_reshape_99]
    return util_create_list_78


def main_const_eval_79(input):
    utils_DeviceGetter_get_device_79 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_54 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_79,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_to_device_54, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_54, False)
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_to_layout_71,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_71, False)
    ttnn_permute_98 = ttnn.permute(
        ttnn_reshape_100,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_100, False)
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_permute_98, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_98, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_to_layout_72)
    ttnn.deallocate(ttnn_to_layout_72, False)
    ttnn_prepare_conv_bias_17 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_17,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_79,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_17, False)
    util_create_list_79 = [ttnn_prepare_conv_bias_17]
    return util_create_list_79


def main_const_eval_80(input):
    utils_DeviceGetter_get_device_80 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_55 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_80,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_to_device_55, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_55, False)
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_to_layout_73,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_73, False)
    ttnn_permute_99 = ttnn.permute(
        ttnn_reshape_101,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_101, False)
    ttnn_typecast_40 = ttnn.typecast(
        ttnn_permute_99,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_99, False)
    ttnn_permute_100 = ttnn.permute(
        ttnn_typecast_40,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn_reshape_102 = ttnn.reshape(
        ttnn_permute_100,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_100, False)
    util_create_list_80 = [ttnn_reshape_102]
    return util_create_list_80


def main_const_eval_81(input):
    utils_DeviceGetter_get_device_81 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_17 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_81,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_81 = [ttnn_prepare_conv_weights_17]
    return util_create_list_81


def main_const_eval_82(input):
    utils_DeviceGetter_get_device_82 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_56 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_82,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_74 = ttnn.to_layout(
        ttnn_to_device_56, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_56, False)
    ttnn_reshape_103 = ttnn.reshape(
        ttnn_to_layout_74,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_74, False)
    ttnn_permute_101 = ttnn.permute(
        ttnn_reshape_103,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_103, False)
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_permute_101,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_101, False)
    ttnn_permute_102 = ttnn.permute(
        ttnn_typecast_41,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn_reshape_104 = ttnn.reshape(
        ttnn_permute_102,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_102, False)
    util_create_list_82 = [ttnn_reshape_104]
    return util_create_list_82


def main_const_eval_83(input):
    utils_DeviceGetter_get_device_83 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_18 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_83,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_83 = [ttnn_prepare_conv_weights_18]
    return util_create_list_83


def main_const_eval_84():
    utils_DeviceGetter_get_device_84 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_4 = ttnn.full(
        shape=ttnn.Shape([1, 512, 320, 180]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_84,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_103 = ttnn.permute(
        ttnn_full_4,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_84 = [ttnn_full_4, ttnn_permute_103]
    return util_create_list_84


def main_const_eval_85(input):
    utils_DeviceGetter_get_device_85 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_57 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_85,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_75 = ttnn.to_layout(
        ttnn_to_device_57, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_57, False)
    ttnn_reshape_105 = ttnn.reshape(
        ttnn_to_layout_75,
        [1, 512, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_75, False)
    ttnn_permute_104 = ttnn.permute(
        ttnn_reshape_105,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_105, False)
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_permute_104,
        [1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_104, False)
    ttnn_typecast_42 = ttnn.typecast(
        ttnn_reshape_106,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_106, False)
    ttnn_reshape_107 = ttnn.reshape(
        ttnn_typecast_42,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_permute_105 = ttnn.permute(
        ttnn_reshape_107,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_107, False)
    ttnn_reshape_108 = ttnn.reshape(
        ttnn_permute_105,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_105, False)
    util_create_list_85 = [ttnn_reshape_108]
    return util_create_list_85


def main_const_eval_86(input):
    utils_DeviceGetter_get_device_86 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_58 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_86,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_to_device_58, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_58, False)
    ttnn_reshape_109 = ttnn.reshape(
        ttnn_to_layout_76,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_76, False)
    ttnn_permute_106 = ttnn.permute(
        ttnn_reshape_109,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_109, False)
    ttnn_typecast_43 = ttnn.typecast(
        ttnn_permute_106,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_106, False)
    ttnn_permute_107 = ttnn.permute(
        ttnn_typecast_43,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn_reshape_110 = ttnn.reshape(
        ttnn_permute_107,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_107, False)
    util_create_list_86 = [ttnn_reshape_110]
    return util_create_list_86


def main_const_eval_87(input):
    utils_DeviceGetter_get_device_87 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_59 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_87,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_to_device_59, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_59, False)
    ttnn_reshape_111 = ttnn.reshape(
        ttnn_to_layout_77,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_77, False)
    ttnn_permute_108 = ttnn.permute(
        ttnn_reshape_111,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_111, False)
    ttnn_typecast_44 = ttnn.typecast(
        ttnn_permute_108,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_108, False)
    ttnn_permute_109 = ttnn.permute(
        ttnn_typecast_44,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn_reshape_112 = ttnn.reshape(
        ttnn_permute_109,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_109, False)
    util_create_list_87 = [ttnn_reshape_112]
    return util_create_list_87


def main_const_eval_88(input):
    utils_DeviceGetter_get_device_88 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_19 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_88,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_88 = [ttnn_prepare_conv_weights_19]
    return util_create_list_88


def main_const_eval_89(input):
    utils_DeviceGetter_get_device_89 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_60 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_89,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_78 = ttnn.to_layout(
        ttnn_to_device_60, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_60, False)
    ttnn_reshape_113 = ttnn.reshape(
        ttnn_to_layout_78,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_78, False)
    ttnn_permute_110 = ttnn.permute(
        ttnn_reshape_113,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_113, False)
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_permute_110,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_110, False)
    ttnn_permute_111 = ttnn.permute(
        ttnn_typecast_45,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_permute_111,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_111, False)
    util_create_list_89 = [ttnn_reshape_114]
    return util_create_list_89


def main_const_eval_90(input):
    utils_DeviceGetter_get_device_90 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_20 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_90,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_90 = [ttnn_prepare_conv_weights_20]
    return util_create_list_90


def main_const_eval_91(input):
    utils_DeviceGetter_get_device_91 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_61 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_91,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_to_device_61, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_61, False)
    ttnn_reshape_115 = ttnn.reshape(
        ttnn_to_layout_79,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_79, False)
    ttnn_permute_112 = ttnn.permute(
        ttnn_reshape_115,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_115, False)
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_permute_112, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_112, False)
    ttnn_from_device_18 = ttnn.from_device(ttnn_to_layout_80)
    ttnn.deallocate(ttnn_to_layout_80, False)
    ttnn_prepare_conv_bias_18 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_18,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_91,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_18, False)
    util_create_list_91 = [ttnn_prepare_conv_bias_18]
    return util_create_list_91


def main_const_eval_92(input):
    utils_DeviceGetter_get_device_92 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_62 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_92,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_to_device_62, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_62, False)
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_to_layout_81,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_81, False)
    ttnn_permute_113 = ttnn.permute(
        ttnn_reshape_116,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_116, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_permute_113,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_113, False)
    ttnn_permute_114 = ttnn.permute(
        ttnn_typecast_46,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_permute_114,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_114, False)
    util_create_list_92 = [ttnn_reshape_117]
    return util_create_list_92


def main_const_eval_93(input):
    utils_DeviceGetter_get_device_93 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_63 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_93,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_82 = ttnn.to_layout(
        ttnn_to_device_63, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_to_layout_82,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_82, False)
    ttnn_permute_115 = ttnn.permute(
        ttnn_reshape_118,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_118, False)
    ttnn_typecast_47 = ttnn.typecast(
        ttnn_permute_115,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_115, False)
    ttnn_permute_116 = ttnn.permute(
        ttnn_typecast_47,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_permute_116,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_116, False)
    util_create_list_93 = [ttnn_reshape_119]
    return util_create_list_93


def main_const_eval_94(input):
    utils_DeviceGetter_get_device_94 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_64 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_94,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_83 = ttnn.to_layout(
        ttnn_to_device_64, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_64, False)
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_to_layout_83,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_83, False)
    ttnn_permute_117 = ttnn.permute(
        ttnn_reshape_120,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_120, False)
    ttnn_typecast_48 = ttnn.typecast(
        ttnn_permute_117,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_117, False)
    ttnn_permute_118 = ttnn.permute(
        ttnn_typecast_48,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_permute_118,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_118, False)
    util_create_list_94 = [ttnn_reshape_121]
    return util_create_list_94


def main_const_eval_95(input):
    utils_DeviceGetter_get_device_95 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_21 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_95,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_95 = [ttnn_prepare_conv_weights_21]
    return util_create_list_95


def main_const_eval_96(input):
    utils_DeviceGetter_get_device_96 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_65 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_96,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_84 = ttnn.to_layout(
        ttnn_to_device_65, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_to_layout_84,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_84, False)
    ttnn_permute_119 = ttnn.permute(
        ttnn_reshape_122,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_122, False)
    ttnn_to_layout_85 = ttnn.to_layout(
        ttnn_permute_119, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_119, False)
    ttnn_from_device_19 = ttnn.from_device(ttnn_to_layout_85)
    ttnn.deallocate(ttnn_to_layout_85, False)
    ttnn_prepare_conv_bias_19 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_19,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_96,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_19, False)
    util_create_list_96 = [ttnn_prepare_conv_bias_19]
    return util_create_list_96


def main_const_eval_97(input):
    utils_DeviceGetter_get_device_97 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_66 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_97,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_86 = ttnn.to_layout(
        ttnn_to_device_66, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_to_layout_86,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_86, False)
    ttnn_permute_120 = ttnn.permute(
        ttnn_reshape_123,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_123, False)
    ttnn_to_layout_87 = ttnn.to_layout(
        ttnn_permute_120, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_120, False)
    ttnn_from_device_20 = ttnn.from_device(ttnn_to_layout_87)
    ttnn.deallocate(ttnn_to_layout_87, False)
    ttnn_prepare_conv_bias_20 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_20,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_97,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_20, False)
    util_create_list_97 = [ttnn_prepare_conv_bias_20]
    return util_create_list_97


def main_const_eval_98(input):
    utils_DeviceGetter_get_device_98 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_67 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_98,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_88 = ttnn.to_layout(
        ttnn_to_device_67, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn_reshape_124 = ttnn.reshape(
        ttnn_to_layout_88,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_88, False)
    ttnn_permute_121 = ttnn.permute(
        ttnn_reshape_124,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_124, False)
    ttnn_typecast_49 = ttnn.typecast(
        ttnn_permute_121,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_121, False)
    ttnn_permute_122 = ttnn.permute(
        ttnn_typecast_49,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn_reshape_125 = ttnn.reshape(
        ttnn_permute_122,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_122, False)
    util_create_list_98 = [ttnn_reshape_125]
    return util_create_list_98


def main_const_eval_99(input):
    utils_DeviceGetter_get_device_99 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_22 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_99,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_99 = [ttnn_prepare_conv_weights_22]
    return util_create_list_99


def main_const_eval_100(input):
    utils_DeviceGetter_get_device_100 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_68 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_100,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_89 = ttnn.to_layout(
        ttnn_to_device_68, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_68, False)
    ttnn_reshape_126 = ttnn.reshape(
        ttnn_to_layout_89,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_89, False)
    ttnn_permute_123 = ttnn.permute(
        ttnn_reshape_126,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_126, False)
    ttnn_typecast_50 = ttnn.typecast(
        ttnn_permute_123,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_123, False)
    ttnn_permute_124 = ttnn.permute(
        ttnn_typecast_50,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn_reshape_127 = ttnn.reshape(
        ttnn_permute_124,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_124, False)
    util_create_list_100 = [ttnn_reshape_127]
    return util_create_list_100


def main_const_eval_101(input):
    utils_DeviceGetter_get_device_101 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_69 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_101,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_90 = ttnn.to_layout(
        ttnn_to_device_69, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_69, False)
    ttnn_reshape_128 = ttnn.reshape(
        ttnn_to_layout_90,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_90, False)
    ttnn_permute_125 = ttnn.permute(
        ttnn_reshape_128,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_128, False)
    ttnn_to_layout_91 = ttnn.to_layout(
        ttnn_permute_125, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_125, False)
    ttnn_from_device_21 = ttnn.from_device(ttnn_to_layout_91)
    ttnn.deallocate(ttnn_to_layout_91, False)
    ttnn_prepare_conv_bias_21 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_21,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_101,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_21, False)
    util_create_list_101 = [ttnn_prepare_conv_bias_21]
    return util_create_list_101


def main_const_eval_102(input):
    utils_DeviceGetter_get_device_102 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_70 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_102,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_92 = ttnn.to_layout(
        ttnn_to_device_70, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_70, False)
    ttnn_reshape_129 = ttnn.reshape(
        ttnn_to_layout_92,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_92, False)
    ttnn_permute_126 = ttnn.permute(
        ttnn_reshape_129,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_129, False)
    ttnn_to_layout_93 = ttnn.to_layout(
        ttnn_permute_126, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_126, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_to_layout_93)
    ttnn.deallocate(ttnn_to_layout_93, False)
    ttnn_prepare_conv_bias_22 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_22,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_102,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_22, False)
    util_create_list_102 = [ttnn_prepare_conv_bias_22]
    return util_create_list_102


def main_const_eval_103(input):
    utils_DeviceGetter_get_device_103 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_71 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_103,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_94 = ttnn.to_layout(
        ttnn_to_device_71, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_71, False)
    ttnn_reshape_130 = ttnn.reshape(
        ttnn_to_layout_94,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_94, False)
    ttnn_repeat_1 = ttnn.repeat(
        ttnn_reshape_130,
        ttnn.Shape([1, 14400, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_130, False)
    util_create_list_103 = [ttnn_repeat_1]
    return util_create_list_103


def main_const_eval_104(input):
    utils_DeviceGetter_get_device_104 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_72 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_104,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_95 = ttnn.to_layout(
        ttnn_to_device_72, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_72, False)
    ttnn_reshape_131 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_95, False)
    ttnn_repeat_2 = ttnn.repeat(
        ttnn_reshape_131,
        ttnn.Shape([1, 14400, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_131, False)
    util_create_list_104 = [ttnn_repeat_2]
    return util_create_list_104


def main_const_eval_105(input):
    utils_DeviceGetter_get_device_105 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_73 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_105,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_96 = ttnn.to_layout(
        ttnn_to_device_73, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_73, False)
    ttnn_reshape_132 = ttnn.reshape(
        ttnn_to_layout_96,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_96, False)
    ttnn_permute_127 = ttnn.permute(
        ttnn_reshape_132,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_132, False)
    ttnn_typecast_51 = ttnn.typecast(
        ttnn_permute_127,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_127, False)
    ttnn_permute_128 = ttnn.permute(
        ttnn_typecast_51,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_reshape_133 = ttnn.reshape(
        ttnn_permute_128,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_128, False)
    util_create_list_105 = [ttnn_reshape_133]
    return util_create_list_105


def main_const_eval_106(input):
    utils_DeviceGetter_get_device_106 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_74 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_106,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_97 = ttnn.to_layout(
        ttnn_to_device_74, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_74, False)
    ttnn_reshape_134 = ttnn.reshape(
        ttnn_to_layout_97,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_97, False)
    ttnn_permute_129 = ttnn.permute(
        ttnn_reshape_134,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_134, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_permute_129,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_129, False)
    ttnn_permute_130 = ttnn.permute(
        ttnn_typecast_52,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_permute_130,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_130, False)
    util_create_list_106 = [ttnn_reshape_135]
    return util_create_list_106


def main_const_eval_107(input):
    utils_DeviceGetter_get_device_107 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_75 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_107,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_98 = ttnn.to_layout(
        ttnn_to_device_75, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_75, False)
    ttnn_reshape_136 = ttnn.reshape(
        ttnn_to_layout_98,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_98, False)
    ttnn_permute_131 = ttnn.permute(
        ttnn_reshape_136,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_136, False)
    ttnn_to_layout_99 = ttnn.to_layout(
        ttnn_permute_131, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_131, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_to_layout_99)
    ttnn.deallocate(ttnn_to_layout_99, False)
    ttnn_prepare_conv_bias_23 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_23,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_107,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_23, False)
    util_create_list_107 = [ttnn_prepare_conv_bias_23]
    return util_create_list_107


def main_const_eval_108():
    utils_DeviceGetter_get_device_108 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_5 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=9.9999999747524271e-07,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_108,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_108 = [ttnn_full_5]
    return util_create_list_108


def main_const_eval_109(input):
    utils_DeviceGetter_get_device_109 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_76 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_109,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_100 = ttnn.to_layout(
        ttnn_to_device_76, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_76, False)
    ttnn_reshape_137 = ttnn.reshape(
        ttnn_to_layout_100,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_100, False)
    ttnn_permute_132 = ttnn.permute(
        ttnn_reshape_137,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_137, False)
    ttnn_to_layout_101 = ttnn.to_layout(
        ttnn_permute_132, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_132, False)
    ttnn_from_device_24 = ttnn.from_device(ttnn_to_layout_101)
    ttnn.deallocate(ttnn_to_layout_101, False)
    ttnn_prepare_conv_bias_24 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_24,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_109,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_24, False)
    util_create_list_109 = [ttnn_prepare_conv_bias_24]
    return util_create_list_109


def main_const_eval_110(input):
    utils_DeviceGetter_get_device_110 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_77 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_110,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_102 = ttnn.to_layout(
        ttnn_to_device_77, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_77, False)
    ttnn_reshape_138 = ttnn.reshape(
        ttnn_to_layout_102,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_102, False)
    ttnn_permute_133 = ttnn.permute(
        ttnn_reshape_138,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_138, False)
    ttnn_to_layout_103 = ttnn.to_layout(
        ttnn_permute_133, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_133, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_to_layout_103)
    ttnn.deallocate(ttnn_to_layout_103, False)
    ttnn_prepare_conv_bias_25 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_110,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_25, False)
    util_create_list_110 = [ttnn_prepare_conv_bias_25]
    return util_create_list_110


def main_const_eval_111():
    utils_DeviceGetter_get_device_111 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_6 = ttnn.full(
        shape=ttnn.Shape([1, 512, 160, 90]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_111,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_134 = ttnn.permute(
        ttnn_full_6,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_111 = [ttnn_full_6, ttnn_permute_134]
    return util_create_list_111


def main_const_eval_112(input):
    utils_DeviceGetter_get_device_112 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_78 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_112,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_104 = ttnn.to_layout(
        ttnn_to_device_78, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_78, False)
    ttnn_reshape_139 = ttnn.reshape(
        ttnn_to_layout_104,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_104, False)
    ttnn_permute_135 = ttnn.permute(
        ttnn_reshape_139,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_139, False)
    ttnn_to_layout_105 = ttnn.to_layout(
        ttnn_permute_135, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_135, False)
    ttnn_from_device_26 = ttnn.from_device(ttnn_to_layout_105)
    ttnn.deallocate(ttnn_to_layout_105, False)
    ttnn_prepare_conv_bias_26 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_26,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_112,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_26, False)
    util_create_list_112 = [ttnn_prepare_conv_bias_26]
    return util_create_list_112


def main_const_eval_113(input):
    utils_DeviceGetter_get_device_113 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_23 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_113,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_113 = [ttnn_prepare_conv_weights_23]
    return util_create_list_113


def main_const_eval_114(input):
    utils_DeviceGetter_get_device_114 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_79 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_114,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_106 = ttnn.to_layout(
        ttnn_to_device_79, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_79, False)
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_to_layout_106,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_106, False)
    ttnn_permute_136 = ttnn.permute(
        ttnn_reshape_140,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_140, False)
    ttnn_to_layout_107 = ttnn.to_layout(
        ttnn_permute_136, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_136, False)
    ttnn_from_device_27 = ttnn.from_device(ttnn_to_layout_107)
    ttnn.deallocate(ttnn_to_layout_107, False)
    ttnn_prepare_conv_bias_27 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_114,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_27, False)
    util_create_list_114 = [ttnn_prepare_conv_bias_27]
    return util_create_list_114


def main_const_eval_115(input):
    utils_DeviceGetter_get_device_115 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_80 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_115,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_108 = ttnn.to_layout(
        ttnn_to_device_80, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_80, False)
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_to_layout_108,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_108, False)
    ttnn_permute_137 = ttnn.permute(
        ttnn_reshape_141,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_141, False)
    ttnn_typecast_53 = ttnn.typecast(
        ttnn_permute_137,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_137, False)
    ttnn_permute_138 = ttnn.permute(
        ttnn_typecast_53,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_permute_138,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_138, False)
    util_create_list_115 = [ttnn_reshape_142]
    return util_create_list_115


def main_const_eval_116(input):
    utils_DeviceGetter_get_device_116 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_81 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_116,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_109 = ttnn.to_layout(
        ttnn_to_device_81, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_81, False)
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_to_layout_109,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_109, False)
    ttnn_permute_139 = ttnn.permute(
        ttnn_reshape_143,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn_typecast_54 = ttnn.typecast(
        ttnn_permute_139,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_139, False)
    ttnn_permute_140 = ttnn.permute(
        ttnn_typecast_54,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_permute_140,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_140, False)
    util_create_list_116 = [ttnn_reshape_144]
    return util_create_list_116


def main_const_eval_117(input):
    utils_DeviceGetter_get_device_117 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_24 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_117,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_117 = [ttnn_prepare_conv_weights_24]
    return util_create_list_117


def main_const_eval_118(input):
    utils_DeviceGetter_get_device_118 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_25 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_118,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_118 = [ttnn_prepare_conv_weights_25]
    return util_create_list_118


def main_const_eval_119(input):
    utils_DeviceGetter_get_device_119 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_82 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_119,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_110 = ttnn.to_layout(
        ttnn_to_device_82, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_82, False)
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_to_layout_110,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_110, False)
    ttnn_permute_141 = ttnn.permute(
        ttnn_reshape_145,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_145, False)
    ttnn_typecast_55 = ttnn.typecast(
        ttnn_permute_141,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_141, False)
    ttnn_permute_142 = ttnn.permute(
        ttnn_typecast_55,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_permute_142,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_142, False)
    util_create_list_119 = [ttnn_reshape_146]
    return util_create_list_119


def main_const_eval_120(input):
    utils_DeviceGetter_get_device_120 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_26 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_120,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_120 = [ttnn_prepare_conv_weights_26]
    return util_create_list_120


def main_const_eval_121(input):
    utils_DeviceGetter_get_device_121 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_27 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_121,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_121 = [ttnn_prepare_conv_weights_27]
    return util_create_list_121


def main_const_eval_122(input):
    utils_DeviceGetter_get_device_122 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_28 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_122,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_122 = [ttnn_prepare_conv_weights_28]
    return util_create_list_122


def main_const_eval_123():
    utils_DeviceGetter_get_device_123 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_8 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
        ],
        [90],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_123,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_9 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
        ],
        [180],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_123,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_4 = ttnn.floor(
        ttnn_Tensor_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_9, False)
    ttnn_typecast_56 = ttnn.typecast(
        ttnn_floor_4,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_4, False)
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_typecast_56,
        [180, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn_permute_143 = ttnn.permute(
        ttnn_reshape_147,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_reshape_148 = ttnn.reshape(
        ttnn_Tensor_8,
        [1, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_8, False)
    ttnn_permute_144 = ttnn.permute(
        ttnn_reshape_148,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_148, False)
    ttnn_eq_4 = ttnn.eq(
        ttnn_permute_143,
        ttnn_permute_144,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_144, False)
    ttnn.deallocate(ttnn_permute_143, False)
    util_create_list_123 = [ttnn_eq_4]
    return util_create_list_123


def main_const_eval_124(input):
    utils_DeviceGetter_get_device_124 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_29 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_124,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_124 = [ttnn_prepare_conv_weights_29]
    return util_create_list_124


def main_const_eval_125(input):
    utils_DeviceGetter_get_device_125 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_83 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_125,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_111 = ttnn.to_layout(
        ttnn_to_device_83, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_83, False)
    ttnn_reshape_149 = ttnn.reshape(
        ttnn_to_layout_111,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_111, False)
    ttnn_permute_145 = ttnn.permute(
        ttnn_reshape_149,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_149, False)
    ttnn_typecast_57 = ttnn.typecast(
        ttnn_permute_145,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_145, False)
    ttnn_permute_146 = ttnn.permute(
        ttnn_typecast_57,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_permute_146,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_146, False)
    util_create_list_125 = [ttnn_reshape_150]
    return util_create_list_125


def main_const_eval_126(input):
    utils_DeviceGetter_get_device_126 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_84 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_126,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_112 = ttnn.to_layout(
        ttnn_to_device_84, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_84, False)
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_to_layout_112,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_112, False)
    ttnn_permute_147 = ttnn.permute(
        ttnn_reshape_151,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_to_layout_113 = ttnn.to_layout(
        ttnn_permute_147, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_147, False)
    ttnn_from_device_28 = ttnn.from_device(ttnn_to_layout_113)
    ttnn.deallocate(ttnn_to_layout_113, False)
    ttnn_prepare_conv_bias_28 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_28,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_126,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_28, False)
    util_create_list_126 = [ttnn_prepare_conv_bias_28]
    return util_create_list_126


def main_const_eval_127(input):
    utils_DeviceGetter_get_device_127 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_30 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_127,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_127 = [ttnn_prepare_conv_weights_30]
    return util_create_list_127


def main_const_eval_128(input):
    utils_DeviceGetter_get_device_128 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_85 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_128,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_114 = ttnn.to_layout(
        ttnn_to_device_85, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_85, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_to_layout_114,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_114, False)
    ttnn_permute_148 = ttnn.permute(
        ttnn_reshape_152,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_152, False)
    ttnn_typecast_58 = ttnn.typecast(
        ttnn_permute_148,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_148, False)
    ttnn_permute_149 = ttnn.permute(
        ttnn_typecast_58,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_permute_149,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_149, False)
    util_create_list_128 = [ttnn_reshape_153]
    return util_create_list_128


def main_const_eval_129(input):
    utils_DeviceGetter_get_device_129 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_86 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_129,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_115 = ttnn.to_layout(
        ttnn_to_device_86, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_86, False)
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_to_layout_115,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_115, False)
    ttnn_permute_150 = ttnn.permute(
        ttnn_reshape_154,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_154, False)
    ttnn_to_layout_116 = ttnn.to_layout(
        ttnn_permute_150, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_150, False)
    ttnn_from_device_29 = ttnn.from_device(ttnn_to_layout_116)
    ttnn.deallocate(ttnn_to_layout_116, False)
    ttnn_prepare_conv_bias_29 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_29,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_129,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_29, False)
    util_create_list_129 = [ttnn_prepare_conv_bias_29]
    return util_create_list_129


def main_const_eval_130(input):
    utils_DeviceGetter_get_device_130 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_87 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_130,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_117 = ttnn.to_layout(
        ttnn_to_device_87, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_87, False)
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_to_layout_117,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_117, False)
    ttnn_permute_151 = ttnn.permute(
        ttnn_reshape_155,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_155, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_permute_151,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_151, False)
    ttnn_permute_152 = ttnn.permute(
        ttnn_typecast_59,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_permute_152,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_152, False)
    util_create_list_130 = [ttnn_reshape_156]
    return util_create_list_130


def main_const_eval_131(input):
    utils_DeviceGetter_get_device_131 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_88 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_131,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_118 = ttnn.to_layout(
        ttnn_to_device_88, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_88, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_to_layout_118,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_118, False)
    ttnn_permute_153 = ttnn.permute(
        ttnn_reshape_157,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_157, False)
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_permute_153,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_153, False)
    ttnn_permute_154 = ttnn.permute(
        ttnn_typecast_60,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_permute_154,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_154, False)
    util_create_list_131 = [ttnn_reshape_158]
    return util_create_list_131


def main_const_eval_132(input):
    utils_DeviceGetter_get_device_132 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_89 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_132,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_119 = ttnn.to_layout(
        ttnn_to_device_89, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_89, False)
    ttnn_reshape_159 = ttnn.reshape(
        ttnn_to_layout_119,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_119, False)
    ttnn_permute_155 = ttnn.permute(
        ttnn_reshape_159,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_159, False)
    ttnn_to_layout_120 = ttnn.to_layout(
        ttnn_permute_155, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_155, False)
    ttnn_from_device_30 = ttnn.from_device(ttnn_to_layout_120)
    ttnn.deallocate(ttnn_to_layout_120, False)
    ttnn_prepare_conv_bias_30 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_30,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_132,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_30, False)
    util_create_list_132 = [ttnn_prepare_conv_bias_30]
    return util_create_list_132


def main_const_eval_133(input):
    utils_DeviceGetter_get_device_133 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_90 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_133,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_121 = ttnn.to_layout(
        ttnn_to_device_90, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_90, False)
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_to_layout_121,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_121, False)
    ttnn_repeat_3 = ttnn.repeat(
        ttnn_reshape_160,
        ttnn.Shape([1, 14400, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_160, False)
    util_create_list_133 = [ttnn_repeat_3]
    return util_create_list_133


def main_const_eval_134(input):
    utils_DeviceGetter_get_device_134 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_91 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_134,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_122 = ttnn.to_layout(
        ttnn_to_device_91, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_91, False)
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_to_layout_122,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_122, False)
    ttnn_permute_156 = ttnn.permute(
        ttnn_reshape_161,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn_to_layout_123 = ttnn.to_layout(
        ttnn_permute_156, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_156, False)
    ttnn_from_device_31 = ttnn.from_device(ttnn_to_layout_123)
    ttnn.deallocate(ttnn_to_layout_123, False)
    ttnn_prepare_conv_bias_31 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_31,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_134,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_31, False)
    util_create_list_134 = [ttnn_prepare_conv_bias_31]
    return util_create_list_134


def main_const_eval_135(input):
    utils_DeviceGetter_get_device_135 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_31 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_135,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_135 = [ttnn_prepare_conv_weights_31]
    return util_create_list_135


def main_const_eval_136(input):
    utils_DeviceGetter_get_device_136 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_92 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_136,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_124 = ttnn.to_layout(
        ttnn_to_device_92, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_92, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_to_layout_124,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_124, False)
    ttnn_permute_157 = ttnn.permute(
        ttnn_reshape_162,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_162, False)
    ttnn_typecast_61 = ttnn.typecast(
        ttnn_permute_157,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_157, False)
    ttnn_permute_158 = ttnn.permute(
        ttnn_typecast_61,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn_reshape_163 = ttnn.reshape(
        ttnn_permute_158,
        [1, 32, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_158, False)
    util_create_list_136 = [ttnn_reshape_163]
    return util_create_list_136


def main_const_eval_137(input):
    utils_DeviceGetter_get_device_137 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_93 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_137,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_125 = ttnn.to_layout(
        ttnn_to_device_93, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_93, False)
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_to_layout_125,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_125, False)
    ttnn_permute_159 = ttnn.permute(
        ttnn_reshape_164,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_164, False)
    ttnn_typecast_62 = ttnn.typecast(
        ttnn_permute_159,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_159, False)
    ttnn_permute_160 = ttnn.permute(
        ttnn_typecast_62,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_62, False)
    ttnn_reshape_165 = ttnn.reshape(
        ttnn_permute_160,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_160, False)
    util_create_list_137 = [ttnn_reshape_165]
    return util_create_list_137


def main_const_eval_138(input):
    utils_DeviceGetter_get_device_138 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_94 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_138,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_126 = ttnn.to_layout(
        ttnn_to_device_94, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_94, False)
    ttnn_reshape_166 = ttnn.reshape(
        ttnn_to_layout_126,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_126, False)
    ttnn_permute_161 = ttnn.permute(
        ttnn_reshape_166,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_166, False)
    ttnn_typecast_63 = ttnn.typecast(
        ttnn_permute_161,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_161, False)
    ttnn_permute_162 = ttnn.permute(
        ttnn_typecast_63,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_63, False)
    ttnn_reshape_167 = ttnn.reshape(
        ttnn_permute_162,
        [1, 32, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_162, False)
    util_create_list_138 = [ttnn_reshape_167]
    return util_create_list_138


def main_const_eval_139(input):
    utils_DeviceGetter_get_device_139 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_95 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_139,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_127 = ttnn.to_layout(
        ttnn_to_device_95, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_95, False)
    ttnn_reshape_168 = ttnn.reshape(
        ttnn_to_layout_127,
        [1, 3, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_127, False)
    ttnn_permute_163 = ttnn.permute(
        ttnn_reshape_168,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_168, False)
    ttnn_to_layout_128 = ttnn.to_layout(
        ttnn_permute_163, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_163, False)
    ttnn_from_device_32 = ttnn.from_device(ttnn_to_layout_128)
    ttnn.deallocate(ttnn_to_layout_128, False)
    ttnn_prepare_conv_bias_32 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_32,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=3,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_139,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_32, False)
    util_create_list_139 = [ttnn_prepare_conv_bias_32]
    return util_create_list_139


def main_const_eval_140(input):
    utils_DeviceGetter_get_device_140 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_32 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_140,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_140 = [ttnn_prepare_conv_weights_32]
    return util_create_list_140


def main_const_eval_141(input):
    utils_DeviceGetter_get_device_141 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_96 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_141,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_129 = ttnn.to_layout(
        ttnn_to_device_96, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_96, False)
    ttnn_reshape_169 = ttnn.reshape(
        ttnn_to_layout_129,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_129, False)
    ttnn_permute_164 = ttnn.permute(
        ttnn_reshape_169,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_169, False)
    ttnn_typecast_64 = ttnn.typecast(
        ttnn_permute_164,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_164, False)
    ttnn_permute_165 = ttnn.permute(
        ttnn_typecast_64,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn_reshape_170 = ttnn.reshape(
        ttnn_permute_165,
        [1, 32, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_165, False)
    util_create_list_141 = [ttnn_reshape_170]
    return util_create_list_141


def main_const_eval_142(input):
    utils_DeviceGetter_get_device_142 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_33 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=3,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_142,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_142 = [ttnn_prepare_conv_weights_33]
    return util_create_list_142


def main_const_eval_143(input):
    utils_DeviceGetter_get_device_143 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_97 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_143,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_130 = ttnn.to_layout(
        ttnn_to_device_97, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_97, False)
    ttnn_reshape_171 = ttnn.reshape(
        ttnn_to_layout_130,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_130, False)
    ttnn_permute_166 = ttnn.permute(
        ttnn_reshape_171,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_171, False)
    ttnn_to_layout_131 = ttnn.to_layout(
        ttnn_permute_166, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_166, False)
    ttnn_from_device_33 = ttnn.from_device(ttnn_to_layout_131)
    ttnn.deallocate(ttnn_to_layout_131, False)
    ttnn_prepare_conv_bias_33 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_33,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_143,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_33, False)
    util_create_list_143 = [ttnn_prepare_conv_bias_33]
    return util_create_list_143


def main_const_eval_144(input):
    utils_DeviceGetter_get_device_144 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_98 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_144,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_132 = ttnn.to_layout(
        ttnn_to_device_98, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_98, False)
    ttnn_reshape_172 = ttnn.reshape(
        ttnn_to_layout_132,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_132, False)
    ttnn_permute_167 = ttnn.permute(
        ttnn_reshape_172,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_172, False)
    ttnn_to_layout_133 = ttnn.to_layout(
        ttnn_permute_167, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_permute_167, False)
    ttnn_from_device_34 = ttnn.from_device(ttnn_to_layout_133)
    ttnn.deallocate(ttnn_to_layout_133, False)
    ttnn_prepare_conv_bias_34 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_from_device_34,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=utils_DeviceGetter_get_device_144,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_from_device_34, False)
    util_create_list_144 = [ttnn_prepare_conv_bias_34]
    return util_create_list_144


def main_const_eval_145(input):
    utils_DeviceGetter_get_device_145 = utils.DeviceGetter.get_device((1, 1))
    ttnn_prepare_conv_weights_34 = ttnn.prepare_conv_weights(
        weight_tensor=input[0],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=utils_DeviceGetter_get_device_145,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
    )
    util_create_list_145 = [ttnn_prepare_conv_weights_34]
    return util_create_list_145


def main_const_eval_146():
    utils_DeviceGetter_get_device_146 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_10 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
        ],
        [180],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_146,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_Tensor_11 = ttnn.Tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
            13.5,
            14.0,
            14.5,
            15.0,
            15.5,
            16.0,
            16.5,
            17.0,
            17.5,
            18.0,
            18.5,
            19.0,
            19.5,
            20.0,
            20.5,
            21.0,
            21.5,
            22.0,
            22.5,
            23.0,
            23.5,
            24.0,
            24.5,
            25.0,
            25.5,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.5,
            29.0,
            29.5,
            30.0,
            30.5,
            31.0,
            31.5,
            32.0,
            32.5,
            33.0,
            33.5,
            34.0,
            34.5,
            35.0,
            35.5,
            36.0,
            36.5,
            37.0,
            37.5,
            38.0,
            38.5,
            39.0,
            39.5,
            40.0,
            40.5,
            41.0,
            41.5,
            42.0,
            42.5,
            43.0,
            43.5,
            44.0,
            44.5,
            45.0,
            45.5,
            46.0,
            46.5,
            47.0,
            47.5,
            48.0,
            48.5,
            49.0,
            49.5,
            50.0,
            50.5,
            51.0,
            51.5,
            52.0,
            52.5,
            53.0,
            53.5,
            54.0,
            54.5,
            55.0,
            55.5,
            56.0,
            56.5,
            57.0,
            57.5,
            58.0,
            58.5,
            59.0,
            59.5,
            60.0,
            60.5,
            61.0,
            61.5,
            62.0,
            62.5,
            63.0,
            63.5,
            64.0,
            64.5,
            65.0,
            65.5,
            66.0,
            66.5,
            67.0,
            67.5,
            68.0,
            68.5,
            69.0,
            69.5,
            70.0,
            70.5,
            71.0,
            71.5,
            72.0,
            72.5,
            73.0,
            73.5,
            74.0,
            74.5,
            75.0,
            75.5,
            76.0,
            76.5,
            77.0,
            77.5,
            78.0,
            78.5,
            79.0,
            79.5,
            80.0,
            80.5,
            81.0,
            81.5,
            82.0,
            82.5,
            83.0,
            83.5,
            84.0,
            84.5,
            85.0,
            85.5,
            86.0,
            86.5,
            87.0,
            87.5,
            88.0,
            88.5,
            89.0,
            89.5,
            90.0,
            90.5,
            91.0,
            91.5,
            92.0,
            92.5,
            93.0,
            93.5,
            94.0,
            94.5,
            95.0,
            95.5,
            96.0,
            96.5,
            97.0,
            97.5,
            98.0,
            98.5,
            99.0,
            99.5,
            100.0,
            100.5,
            101.0,
            101.5,
            102.0,
            102.5,
            103.0,
            103.5,
            104.0,
            104.5,
            105.0,
            105.5,
            106.0,
            106.5,
            107.0,
            107.5,
            108.0,
            108.5,
            109.0,
            109.5,
            110.0,
            110.5,
            111.0,
            111.5,
            112.0,
            112.5,
            113.0,
            113.5,
            114.0,
            114.5,
            115.0,
            115.5,
            116.0,
            116.5,
            117.0,
            117.5,
            118.0,
            118.5,
            119.0,
            119.5,
            120.0,
            120.5,
            121.0,
            121.5,
            122.0,
            122.5,
            123.0,
            123.5,
            124.0,
            124.5,
            125.0,
            125.5,
            126.0,
            126.5,
            127.0,
            127.5,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            130.5,
            131.0,
            131.5,
            132.0,
            132.5,
            133.0,
            133.5,
            134.0,
            134.5,
            135.0,
            135.5,
            136.0,
            136.5,
            137.0,
            137.5,
            138.0,
            138.5,
            139.0,
            139.5,
            140.0,
            140.5,
            141.0,
            141.5,
            142.0,
            142.5,
            143.0,
            143.5,
            144.0,
            144.5,
            145.0,
            145.5,
            146.0,
            146.5,
            147.0,
            147.5,
            148.0,
            148.5,
            149.0,
            149.5,
            150.0,
            150.5,
            151.0,
            151.5,
            152.0,
            152.5,
            153.0,
            153.5,
            154.0,
            154.5,
            155.0,
            155.5,
            156.0,
            156.5,
            157.0,
            157.5,
            158.0,
            158.5,
            159.0,
            159.5,
            160.0,
            160.5,
            161.0,
            161.5,
            162.0,
            162.5,
            163.0,
            163.5,
            164.0,
            164.5,
            165.0,
            165.5,
            166.0,
            166.5,
            167.0,
            167.5,
            168.0,
            168.5,
            169.0,
            169.5,
            170.0,
            170.5,
            171.0,
            171.5,
            172.0,
            172.5,
            173.0,
            173.5,
            174.0,
            174.5,
            175.0,
            175.5,
            176.0,
            176.5,
            177.0,
            177.5,
            178.0,
            178.5,
            179.0,
            179.5,
        ],
        [360],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_146,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_floor_5 = ttnn.floor(
        ttnn_Tensor_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_11, False)
    ttnn_typecast_65 = ttnn.typecast(
        ttnn_floor_5,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_floor_5, False)
    ttnn_reshape_173 = ttnn.reshape(
        ttnn_typecast_65,
        [360, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn_permute_168 = ttnn.permute(
        ttnn_reshape_173,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_173, False)
    ttnn_reshape_174 = ttnn.reshape(
        ttnn_Tensor_10,
        [1, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_Tensor_10, False)
    ttnn_permute_169 = ttnn.permute(
        ttnn_reshape_174,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_174, False)
    ttnn_eq_5 = ttnn.eq(
        ttnn_permute_168,
        ttnn_permute_169,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_169, False)
    ttnn.deallocate(ttnn_permute_168, False)
    util_create_list_146 = [ttnn_eq_5]
    return util_create_list_146


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, _CONST_EVAL_CACHE, const_1
    )
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_2 = main_const_eval_1
    util_create_list_147 = [input[136]]
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_2, util_create_list_147, _CONST_EVAL_CACHE, const_3
    )
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_4 = main_const_eval_2
    util_create_list_148 = [input[122]]
    const_5 = "main_const_eval_2"
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_4, util_create_list_148, _CONST_EVAL_CACHE, const_5
    )
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_6 = main_const_eval_3
    util_create_list_149 = [input[33]]
    const_7 = "main_const_eval_3"
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_6, util_create_list_149, _CONST_EVAL_CACHE, const_7
    )
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_8 = main_const_eval_4
    util_create_list_150 = [input[11]]
    const_9 = "main_const_eval_4"
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_8, util_create_list_150, _CONST_EVAL_CACHE, const_9
    )
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_10 = main_const_eval_5
    const_11 = "main_const_eval_5"
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(
        const_10, _CONST_EVAL_CACHE, const_11
    )
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_12 = main_const_eval_6
    util_create_list_151 = [input[106]]
    const_13 = "main_const_eval_6"
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_12, util_create_list_151, _CONST_EVAL_CACHE, const_13
    )
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_14 = main_const_eval_7
    util_create_list_152 = [input[63]]
    const_15 = "main_const_eval_7"
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(
        const_14, util_create_list_152, _CONST_EVAL_CACHE, const_15
    )
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_16 = main_const_eval_8
    util_create_list_153 = [input[138]]
    const_17 = "main_const_eval_8"
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(
        const_16, util_create_list_153, _CONST_EVAL_CACHE, const_17
    )
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_18 = main_const_eval_9
    util_create_list_154 = [input[62]]
    const_19 = "main_const_eval_9"
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(
        const_18, util_create_list_154, _CONST_EVAL_CACHE, const_19
    )
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_20 = main_const_eval_10
    util_create_list_155 = [input[89]]
    const_21 = "main_const_eval_10"
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(
        const_20, util_create_list_155, _CONST_EVAL_CACHE, const_21
    )
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_22 = main_const_eval_11
    util_create_list_156 = [input[121]]
    const_23 = "main_const_eval_11"
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(
        const_22, util_create_list_156, _CONST_EVAL_CACHE, const_23
    )
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_24 = main_const_eval_12
    const_25 = "main_const_eval_12"
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(
        const_24, _CONST_EVAL_CACHE, const_25
    )
    utils_constEvalFuncWrapperZeroArg_2_0 = utils_constEvalFuncWrapperZeroArg_2[0]
    const_26 = main_const_eval_13
    util_create_list_157 = [input[78]]
    const_27 = "main_const_eval_13"
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_26, util_create_list_157, _CONST_EVAL_CACHE, const_27
    )
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_28 = main_const_eval_14
    util_create_list_158 = [input[101]]
    const_29 = "main_const_eval_14"
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_28, util_create_list_158, _CONST_EVAL_CACHE, const_29
    )
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_30 = main_const_eval_15
    util_create_list_159 = [input[29]]
    const_31 = "main_const_eval_15"
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_30, util_create_list_159, _CONST_EVAL_CACHE, const_31
    )
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_32 = main_const_eval_16
    const_33 = "main_const_eval_16"
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(
        const_32, _CONST_EVAL_CACHE, const_33
    )
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    utils_constEvalFuncWrapperZeroArg_3_1 = utils_constEvalFuncWrapperZeroArg_3[1]
    const_34 = main_const_eval_17
    util_create_list_160 = [input[111]]
    const_35 = "main_const_eval_17"
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_34, util_create_list_160, _CONST_EVAL_CACHE, const_35
    )
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_36 = main_const_eval_18
    const_37 = "main_const_eval_18"
    utils_constEvalFuncWrapperZeroArg_4 = utils.constEvalFuncWrapperZeroArg(
        const_36, _CONST_EVAL_CACHE, const_37
    )
    utils_constEvalFuncWrapperZeroArg_4_0 = utils_constEvalFuncWrapperZeroArg_4[0]
    const_38 = main_const_eval_19
    util_create_list_161 = [input[105]]
    const_39 = "main_const_eval_19"
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_38, util_create_list_161, _CONST_EVAL_CACHE, const_39
    )
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_40 = main_const_eval_20
    util_create_list_162 = [input[55]]
    const_41 = "main_const_eval_20"
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_40, util_create_list_162, _CONST_EVAL_CACHE, const_41
    )
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_42 = main_const_eval_21
    util_create_list_163 = [input[88]]
    const_43 = "main_const_eval_21"
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_42, util_create_list_163, _CONST_EVAL_CACHE, const_43
    )
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_44 = main_const_eval_22
    const_45 = "main_const_eval_22"
    utils_constEvalFuncWrapperZeroArg_5 = utils.constEvalFuncWrapperZeroArg(
        const_44, _CONST_EVAL_CACHE, const_45
    )
    utils_constEvalFuncWrapperZeroArg_5_0 = utils_constEvalFuncWrapperZeroArg_5[0]
    const_46 = main_const_eval_23
    util_create_list_164 = [input[102]]
    const_47 = "main_const_eval_23"
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_46, util_create_list_164, _CONST_EVAL_CACHE, const_47
    )
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_48 = main_const_eval_24
    util_create_list_165 = [input[68]]
    const_49 = "main_const_eval_24"
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_48, util_create_list_165, _CONST_EVAL_CACHE, const_49
    )
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_50 = main_const_eval_25
    util_create_list_166 = [input[12]]
    const_51 = "main_const_eval_25"
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_50, util_create_list_166, _CONST_EVAL_CACHE, const_51
    )
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_52 = main_const_eval_26
    util_create_list_167 = [input[109]]
    const_53 = "main_const_eval_26"
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_52, util_create_list_167, _CONST_EVAL_CACHE, const_53
    )
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_54 = main_const_eval_27
    util_create_list_168 = [input[103]]
    const_55 = "main_const_eval_27"
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_54, util_create_list_168, _CONST_EVAL_CACHE, const_55
    )
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_56 = main_const_eval_28
    util_create_list_169 = [input[77]]
    const_57 = "main_const_eval_28"
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_56, util_create_list_169, _CONST_EVAL_CACHE, const_57
    )
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_58 = main_const_eval_29
    util_create_list_170 = [input[49]]
    const_59 = "main_const_eval_29"
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_58, util_create_list_170, _CONST_EVAL_CACHE, const_59
    )
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_60 = main_const_eval_30
    util_create_list_171 = [input[52]]
    const_61 = "main_const_eval_30"
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_60, util_create_list_171, _CONST_EVAL_CACHE, const_61
    )
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_62 = main_const_eval_31
    util_create_list_172 = [input[34]]
    const_63 = "main_const_eval_31"
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_62, util_create_list_172, _CONST_EVAL_CACHE, const_63
    )
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_64 = main_const_eval_32
    util_create_list_173 = [input[23]]
    const_65 = "main_const_eval_32"
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_64, util_create_list_173, _CONST_EVAL_CACHE, const_65
    )
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_66 = main_const_eval_33
    util_create_list_174 = [input[107]]
    const_67 = "main_const_eval_33"
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_66, util_create_list_174, _CONST_EVAL_CACHE, const_67
    )
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_68 = main_const_eval_34
    util_create_list_175 = [input[81]]
    const_69 = "main_const_eval_34"
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_68, util_create_list_175, _CONST_EVAL_CACHE, const_69
    )
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_70 = main_const_eval_35
    util_create_list_176 = [input[86]]
    const_71 = "main_const_eval_35"
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_70, util_create_list_176, _CONST_EVAL_CACHE, const_71
    )
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_72 = main_const_eval_36
    util_create_list_177 = [input[18]]
    const_73 = "main_const_eval_36"
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_72, util_create_list_177, _CONST_EVAL_CACHE, const_73
    )
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    const_74 = main_const_eval_37
    util_create_list_178 = [input[85]]
    const_75 = "main_const_eval_37"
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_74, util_create_list_178, _CONST_EVAL_CACHE, const_75
    )
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_76 = main_const_eval_38
    util_create_list_179 = [input[96]]
    const_77 = "main_const_eval_38"
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_76, util_create_list_179, _CONST_EVAL_CACHE, const_77
    )
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_78 = main_const_eval_39
    util_create_list_180 = [input[71]]
    const_79 = "main_const_eval_39"
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_78, util_create_list_180, _CONST_EVAL_CACHE, const_79
    )
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_80 = main_const_eval_40
    util_create_list_181 = [input[80]]
    const_81 = "main_const_eval_40"
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_80, util_create_list_181, _CONST_EVAL_CACHE, const_81
    )
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_82 = main_const_eval_41
    util_create_list_182 = [input[48]]
    const_83 = "main_const_eval_41"
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_82, util_create_list_182, _CONST_EVAL_CACHE, const_83
    )
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_84 = main_const_eval_42
    util_create_list_183 = [input[14]]
    const_85 = "main_const_eval_42"
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_84, util_create_list_183, _CONST_EVAL_CACHE, const_85
    )
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_86 = main_const_eval_43
    util_create_list_184 = [input[115]]
    const_87 = "main_const_eval_43"
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_86, util_create_list_184, _CONST_EVAL_CACHE, const_87
    )
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_88 = main_const_eval_44
    util_create_list_185 = [input[137]]
    const_89 = "main_const_eval_44"
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_88, util_create_list_185, _CONST_EVAL_CACHE, const_89
    )
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_90 = main_const_eval_45
    util_create_list_186 = [input[79]]
    const_91 = "main_const_eval_45"
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_90, util_create_list_186, _CONST_EVAL_CACHE, const_91
    )
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_92 = main_const_eval_46
    util_create_list_187 = [input[5]]
    const_93 = "main_const_eval_46"
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_92, util_create_list_187, _CONST_EVAL_CACHE, const_93
    )
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_94 = main_const_eval_47
    util_create_list_188 = [input[87]]
    const_95 = "main_const_eval_47"
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_94, util_create_list_188, _CONST_EVAL_CACHE, const_95
    )
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_96 = main_const_eval_48
    util_create_list_189 = [input[46]]
    const_97 = "main_const_eval_48"
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_96, util_create_list_189, _CONST_EVAL_CACHE, const_97
    )
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_98 = main_const_eval_49
    util_create_list_190 = [input[116]]
    const_99 = "main_const_eval_49"
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_98, util_create_list_190, _CONST_EVAL_CACHE, const_99
    )
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_100 = main_const_eval_50
    util_create_list_191 = [input[56]]
    const_101 = "main_const_eval_50"
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_100, util_create_list_191, _CONST_EVAL_CACHE, const_101
    )
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_102 = main_const_eval_51
    const_103 = "main_const_eval_51"
    utils_constEvalFuncWrapperZeroArg_6 = utils.constEvalFuncWrapperZeroArg(
        const_102, _CONST_EVAL_CACHE, const_103
    )
    utils_constEvalFuncWrapperZeroArg_6_0 = utils_constEvalFuncWrapperZeroArg_6[0]
    const_104 = main_const_eval_52
    util_create_list_192 = [input[118]]
    const_105 = "main_const_eval_52"
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_104, util_create_list_192, _CONST_EVAL_CACHE, const_105
    )
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_106 = main_const_eval_53
    util_create_list_193 = [input[104]]
    const_107 = "main_const_eval_53"
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_106, util_create_list_193, _CONST_EVAL_CACHE, const_107
    )
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_108 = main_const_eval_54
    util_create_list_194 = [input[28]]
    const_109 = "main_const_eval_54"
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_108, util_create_list_194, _CONST_EVAL_CACHE, const_109
    )
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_110 = main_const_eval_55
    util_create_list_195 = [input[100]]
    const_111 = "main_const_eval_55"
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_110, util_create_list_195, _CONST_EVAL_CACHE, const_111
    )
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_112 = main_const_eval_56
    util_create_list_196 = [input[114]]
    const_113 = "main_const_eval_56"
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_112, util_create_list_196, _CONST_EVAL_CACHE, const_113
    )
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_114 = main_const_eval_57
    util_create_list_197 = [input[37]]
    const_115 = "main_const_eval_57"
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_114, util_create_list_197, _CONST_EVAL_CACHE, const_115
    )
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_116 = main_const_eval_58
    util_create_list_198 = [input[69]]
    const_117 = "main_const_eval_58"
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_116, util_create_list_198, _CONST_EVAL_CACHE, const_117
    )
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_118 = main_const_eval_59
    util_create_list_199 = [input[125]]
    const_119 = "main_const_eval_59"
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_118, util_create_list_199, _CONST_EVAL_CACHE, const_119
    )
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_120 = main_const_eval_60
    util_create_list_200 = [input[44]]
    const_121 = "main_const_eval_60"
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_120, util_create_list_200, _CONST_EVAL_CACHE, const_121
    )
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_122 = main_const_eval_61
    util_create_list_201 = [input[2]]
    const_123 = "main_const_eval_61"
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_122, util_create_list_201, _CONST_EVAL_CACHE, const_123
    )
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_124 = main_const_eval_62
    const_125 = "main_const_eval_62"
    utils_constEvalFuncWrapperZeroArg_7 = utils.constEvalFuncWrapperZeroArg(
        const_124, _CONST_EVAL_CACHE, const_125
    )
    utils_constEvalFuncWrapperZeroArg_7_0 = utils_constEvalFuncWrapperZeroArg_7[0]
    const_126 = main_const_eval_63
    util_create_list_202 = [input[76]]
    const_127 = "main_const_eval_63"
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_126, util_create_list_202, _CONST_EVAL_CACHE, const_127
    )
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_128 = main_const_eval_64
    util_create_list_203 = [input[10]]
    const_129 = "main_const_eval_64"
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_128, util_create_list_203, _CONST_EVAL_CACHE, const_129
    )
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_130 = main_const_eval_65
    util_create_list_204 = [input[93]]
    const_131 = "main_const_eval_65"
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_130, util_create_list_204, _CONST_EVAL_CACHE, const_131
    )
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_132 = main_const_eval_66
    util_create_list_205 = [input[75]]
    const_133 = "main_const_eval_66"
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_132, util_create_list_205, _CONST_EVAL_CACHE, const_133
    )
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_134 = main_const_eval_67
    util_create_list_206 = [input[41]]
    const_135 = "main_const_eval_67"
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_134, util_create_list_206, _CONST_EVAL_CACHE, const_135
    )
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_136 = main_const_eval_68
    util_create_list_207 = [input[120]]
    const_137 = "main_const_eval_68"
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_136, util_create_list_207, _CONST_EVAL_CACHE, const_137
    )
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_138 = main_const_eval_69
    util_create_list_208 = [input[90]]
    const_139 = "main_const_eval_69"
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_138, util_create_list_208, _CONST_EVAL_CACHE, const_139
    )
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_140 = main_const_eval_70
    util_create_list_209 = [input[25]]
    const_141 = "main_const_eval_70"
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_140, util_create_list_209, _CONST_EVAL_CACHE, const_141
    )
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_142 = main_const_eval_71
    util_create_list_210 = [input[47]]
    const_143 = "main_const_eval_71"
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_142, util_create_list_210, _CONST_EVAL_CACHE, const_143
    )
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_144 = main_const_eval_72
    util_create_list_211 = [input[24]]
    const_145 = "main_const_eval_72"
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_144, util_create_list_211, _CONST_EVAL_CACHE, const_145
    )
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_146 = main_const_eval_73
    util_create_list_212 = [input[65]]
    const_147 = "main_const_eval_73"
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_146, util_create_list_212, _CONST_EVAL_CACHE, const_147
    )
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_148 = main_const_eval_74
    util_create_list_213 = [input[113]]
    const_149 = "main_const_eval_74"
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_148, util_create_list_213, _CONST_EVAL_CACHE, const_149
    )
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_150 = main_const_eval_75
    util_create_list_214 = [input[130]]
    const_151 = "main_const_eval_75"
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_150, util_create_list_214, _CONST_EVAL_CACHE, const_151
    )
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_152 = main_const_eval_76
    util_create_list_215 = [input[123]]
    const_153 = "main_const_eval_76"
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_152, util_create_list_215, _CONST_EVAL_CACHE, const_153
    )
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_154 = main_const_eval_77
    util_create_list_216 = [input[112]]
    const_155 = "main_const_eval_77"
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_154, util_create_list_216, _CONST_EVAL_CACHE, const_155
    )
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_156 = main_const_eval_78
    util_create_list_217 = [input[117]]
    const_157 = "main_const_eval_78"
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_156, util_create_list_217, _CONST_EVAL_CACHE, const_157
    )
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_158 = main_const_eval_79
    util_create_list_218 = [input[64]]
    const_159 = "main_const_eval_79"
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_158, util_create_list_218, _CONST_EVAL_CACHE, const_159
    )
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_160 = main_const_eval_80
    util_create_list_219 = [input[66]]
    const_161 = "main_const_eval_80"
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_160, util_create_list_219, _CONST_EVAL_CACHE, const_161
    )
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_162 = main_const_eval_81
    util_create_list_220 = [input[95]]
    const_163 = "main_const_eval_81"
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_162, util_create_list_220, _CONST_EVAL_CACHE, const_163
    )
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_164 = main_const_eval_82
    util_create_list_221 = [input[40]]
    const_165 = "main_const_eval_82"
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_164, util_create_list_221, _CONST_EVAL_CACHE, const_165
    )
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_166 = main_const_eval_83
    util_create_list_222 = [input[17]]
    const_167 = "main_const_eval_83"
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_166, util_create_list_222, _CONST_EVAL_CACHE, const_167
    )
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_168 = main_const_eval_84
    const_169 = "main_const_eval_84"
    utils_constEvalFuncWrapperZeroArg_8 = utils.constEvalFuncWrapperZeroArg(
        const_168, _CONST_EVAL_CACHE, const_169
    )
    utils_constEvalFuncWrapperZeroArg_8_0 = utils_constEvalFuncWrapperZeroArg_8[0]
    utils_constEvalFuncWrapperZeroArg_8_1 = utils_constEvalFuncWrapperZeroArg_8[1]
    const_170 = main_const_eval_85
    util_create_list_223 = [input[129]]
    const_171 = "main_const_eval_85"
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_170, util_create_list_223, _CONST_EVAL_CACHE, const_171
    )
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_172 = main_const_eval_86
    util_create_list_224 = [input[45]]
    const_173 = "main_const_eval_86"
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_172, util_create_list_224, _CONST_EVAL_CACHE, const_173
    )
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_174 = main_const_eval_87
    util_create_list_225 = [input[58]]
    const_175 = "main_const_eval_87"
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_174, util_create_list_225, _CONST_EVAL_CACHE, const_175
    )
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_176 = main_const_eval_88
    util_create_list_226 = [input[61]]
    const_177 = "main_const_eval_88"
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_176, util_create_list_226, _CONST_EVAL_CACHE, const_177
    )
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_178 = main_const_eval_89
    util_create_list_227 = [input[32]]
    const_179 = "main_const_eval_89"
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_178, util_create_list_227, _CONST_EVAL_CACHE, const_179
    )
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_180 = main_const_eval_90
    util_create_list_228 = [input[39]]
    const_181 = "main_const_eval_90"
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_180, util_create_list_228, _CONST_EVAL_CACHE, const_181
    )
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_182 = main_const_eval_91
    util_create_list_229 = [input[135]]
    const_183 = "main_const_eval_91"
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_182, util_create_list_229, _CONST_EVAL_CACHE, const_183
    )
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_184 = main_const_eval_92
    util_create_list_230 = [input[22]]
    const_185 = "main_const_eval_92"
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_184, util_create_list_230, _CONST_EVAL_CACHE, const_185
    )
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_186 = main_const_eval_93
    util_create_list_231 = [input[7]]
    const_187 = "main_const_eval_93"
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_186, util_create_list_231, _CONST_EVAL_CACHE, const_187
    )
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_188 = main_const_eval_94
    util_create_list_232 = [input[92]]
    const_189 = "main_const_eval_94"
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_188, util_create_list_232, _CONST_EVAL_CACHE, const_189
    )
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_190 = main_const_eval_95
    util_create_list_233 = [input[35]]
    const_191 = "main_const_eval_95"
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_190, util_create_list_233, _CONST_EVAL_CACHE, const_191
    )
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_192 = main_const_eval_96
    util_create_list_234 = [input[20]]
    const_193 = "main_const_eval_96"
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_192, util_create_list_234, _CONST_EVAL_CACHE, const_193
    )
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_194 = main_const_eval_97
    util_create_list_235 = [input[54]]
    const_195 = "main_const_eval_97"
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_194, util_create_list_235, _CONST_EVAL_CACHE, const_195
    )
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_196 = main_const_eval_98
    util_create_list_236 = [input[108]]
    const_197 = "main_const_eval_98"
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_196, util_create_list_236, _CONST_EVAL_CACHE, const_197
    )
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_198 = main_const_eval_99
    util_create_list_237 = [input[13]]
    const_199 = "main_const_eval_99"
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_198, util_create_list_237, _CONST_EVAL_CACHE, const_199
    )
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_200 = main_const_eval_100
    util_create_list_238 = [input[67]]
    const_201 = "main_const_eval_100"
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_200, util_create_list_238, _CONST_EVAL_CACHE, const_201
    )
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_202 = main_const_eval_101
    util_create_list_239 = [input[98]]
    const_203 = "main_const_eval_101"
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_202, util_create_list_239, _CONST_EVAL_CACHE, const_203
    )
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_204 = main_const_eval_102
    util_create_list_240 = [input[82]]
    const_205 = "main_const_eval_102"
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_204, util_create_list_240, _CONST_EVAL_CACHE, const_205
    )
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_206 = main_const_eval_103
    util_create_list_241 = [input[127]]
    const_207 = "main_const_eval_103"
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_206, util_create_list_241, _CONST_EVAL_CACHE, const_207
    )
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_208 = main_const_eval_104
    util_create_list_242 = [input[133]]
    const_209 = "main_const_eval_104"
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_208, util_create_list_242, _CONST_EVAL_CACHE, const_209
    )
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_210 = main_const_eval_105
    util_create_list_243 = [input[3]]
    const_211 = "main_const_eval_105"
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_210, util_create_list_243, _CONST_EVAL_CACHE, const_211
    )
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_212 = main_const_eval_106
    util_create_list_244 = [input[70]]
    const_213 = "main_const_eval_106"
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_212, util_create_list_244, _CONST_EVAL_CACHE, const_213
    )
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_214 = main_const_eval_107
    util_create_list_245 = [input[38]]
    const_215 = "main_const_eval_107"
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_214, util_create_list_245, _CONST_EVAL_CACHE, const_215
    )
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_216 = main_const_eval_108
    const_217 = "main_const_eval_108"
    utils_constEvalFuncWrapperZeroArg_9 = utils.constEvalFuncWrapperZeroArg(
        const_216, _CONST_EVAL_CACHE, const_217
    )
    utils_constEvalFuncWrapperZeroArg_9_0 = utils_constEvalFuncWrapperZeroArg_9[0]
    const_218 = main_const_eval_109
    util_create_list_246 = [input[72]]
    const_219 = "main_const_eval_109"
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_218, util_create_list_246, _CONST_EVAL_CACHE, const_219
    )
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_220 = main_const_eval_110
    util_create_list_247 = [input[50]]
    const_221 = "main_const_eval_110"
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_220, util_create_list_247, _CONST_EVAL_CACHE, const_221
    )
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_222 = main_const_eval_111
    const_223 = "main_const_eval_111"
    utils_constEvalFuncWrapperZeroArg_10 = utils.constEvalFuncWrapperZeroArg(
        const_222, _CONST_EVAL_CACHE, const_223
    )
    utils_constEvalFuncWrapperZeroArg_10_0 = utils_constEvalFuncWrapperZeroArg_10[0]
    utils_constEvalFuncWrapperZeroArg_10_1 = utils_constEvalFuncWrapperZeroArg_10[1]
    const_224 = main_const_eval_112
    util_create_list_248 = [input[16]]
    const_225 = "main_const_eval_112"
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_224, util_create_list_248, _CONST_EVAL_CACHE, const_225
    )
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_226 = main_const_eval_113
    util_create_list_249 = [input[31]]
    const_227 = "main_const_eval_113"
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_226, util_create_list_249, _CONST_EVAL_CACHE, const_227
    )
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_228 = main_const_eval_114
    util_create_list_250 = [input[8]]
    const_229 = "main_const_eval_114"
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_228, util_create_list_250, _CONST_EVAL_CACHE, const_229
    )
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_230 = main_const_eval_115
    util_create_list_251 = [input[15]]
    const_231 = "main_const_eval_115"
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_230, util_create_list_251, _CONST_EVAL_CACHE, const_231
    )
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_232 = main_const_eval_116
    util_create_list_252 = [input[53]]
    const_233 = "main_const_eval_116"
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_232, util_create_list_252, _CONST_EVAL_CACHE, const_233
    )
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_234 = main_const_eval_117
    util_create_list_253 = [input[99]]
    const_235 = "main_const_eval_117"
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_234, util_create_list_253, _CONST_EVAL_CACHE, const_235
    )
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_236 = main_const_eval_118
    util_create_list_254 = [input[57]]
    const_237 = "main_const_eval_118"
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_236, util_create_list_254, _CONST_EVAL_CACHE, const_237
    )
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_238 = main_const_eval_119
    util_create_list_255 = [input[36]]
    const_239 = "main_const_eval_119"
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_238, util_create_list_255, _CONST_EVAL_CACHE, const_239
    )
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_240 = main_const_eval_120
    util_create_list_256 = [input[21]]
    const_241 = "main_const_eval_120"
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_240, util_create_list_256, _CONST_EVAL_CACHE, const_241
    )
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_242 = main_const_eval_121
    util_create_list_257 = [input[83]]
    const_243 = "main_const_eval_121"
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_242, util_create_list_257, _CONST_EVAL_CACHE, const_243
    )
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_244 = main_const_eval_122
    util_create_list_258 = [input[9]]
    const_245 = "main_const_eval_122"
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_244, util_create_list_258, _CONST_EVAL_CACHE, const_245
    )
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_246 = main_const_eval_123
    const_247 = "main_const_eval_123"
    utils_constEvalFuncWrapperZeroArg_11 = utils.constEvalFuncWrapperZeroArg(
        const_246, _CONST_EVAL_CACHE, const_247
    )
    utils_constEvalFuncWrapperZeroArg_11_0 = utils_constEvalFuncWrapperZeroArg_11[0]
    const_248 = main_const_eval_124
    util_create_list_259 = [input[73]]
    const_249 = "main_const_eval_124"
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_248, util_create_list_259, _CONST_EVAL_CACHE, const_249
    )
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_250 = main_const_eval_125
    util_create_list_260 = [input[6]]
    const_251 = "main_const_eval_125"
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_250, util_create_list_260, _CONST_EVAL_CACHE, const_251
    )
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_252 = main_const_eval_126
    util_create_list_261 = [input[60]]
    const_253 = "main_const_eval_126"
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_252, util_create_list_261, _CONST_EVAL_CACHE, const_253
    )
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_254 = main_const_eval_127
    util_create_list_262 = [input[51]]
    const_255 = "main_const_eval_127"
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_254, util_create_list_262, _CONST_EVAL_CACHE, const_255
    )
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_256 = main_const_eval_128
    util_create_list_263 = [input[59]]
    const_257 = "main_const_eval_128"
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_256, util_create_list_263, _CONST_EVAL_CACHE, const_257
    )
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_258 = main_const_eval_129
    util_create_list_264 = [input[42]]
    const_259 = "main_const_eval_129"
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_258, util_create_list_264, _CONST_EVAL_CACHE, const_259
    )
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_260 = main_const_eval_130
    util_create_list_265 = [input[26]]
    const_261 = "main_const_eval_130"
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_260, util_create_list_265, _CONST_EVAL_CACHE, const_261
    )
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_262 = main_const_eval_131
    util_create_list_266 = [input[74]]
    const_263 = "main_const_eval_131"
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_262, util_create_list_266, _CONST_EVAL_CACHE, const_263
    )
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_264 = main_const_eval_132
    util_create_list_267 = [input[110]]
    const_265 = "main_const_eval_132"
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_264, util_create_list_267, _CONST_EVAL_CACHE, const_265
    )
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_266 = main_const_eval_133
    util_create_list_268 = [input[131]]
    const_267 = "main_const_eval_133"
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_266, util_create_list_268, _CONST_EVAL_CACHE, const_267
    )
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_268 = main_const_eval_134
    util_create_list_269 = [input[4]]
    const_269 = "main_const_eval_134"
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_268, util_create_list_269, _CONST_EVAL_CACHE, const_269
    )
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_270 = main_const_eval_135
    util_create_list_270 = [input[43]]
    const_271 = "main_const_eval_135"
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_270, util_create_list_270, _CONST_EVAL_CACHE, const_271
    )
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_272 = main_const_eval_136
    util_create_list_271 = [input[19]]
    const_273 = "main_const_eval_136"
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_272, util_create_list_271, _CONST_EVAL_CACHE, const_273
    )
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_274 = main_const_eval_137
    util_create_list_272 = [input[97]]
    const_275 = "main_const_eval_137"
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_274, util_create_list_272, _CONST_EVAL_CACHE, const_275
    )
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    const_276 = main_const_eval_138
    util_create_list_273 = [input[27]]
    const_277 = "main_const_eval_138"
    utils_constEvalFuncWrapper_126 = utils.constEvalFuncWrapper(
        const_276, util_create_list_273, _CONST_EVAL_CACHE, const_277
    )
    utils_constEvalFuncWrapper_126_0 = utils_constEvalFuncWrapper_126[0]
    const_278 = main_const_eval_139
    util_create_list_274 = [input[0]]
    const_279 = "main_const_eval_139"
    utils_constEvalFuncWrapper_127 = utils.constEvalFuncWrapper(
        const_278, util_create_list_274, _CONST_EVAL_CACHE, const_279
    )
    utils_constEvalFuncWrapper_127_0 = utils_constEvalFuncWrapper_127[0]
    const_280 = main_const_eval_140
    util_create_list_275 = [input[91]]
    const_281 = "main_const_eval_140"
    utils_constEvalFuncWrapper_128 = utils.constEvalFuncWrapper(
        const_280, util_create_list_275, _CONST_EVAL_CACHE, const_281
    )
    utils_constEvalFuncWrapper_128_0 = utils_constEvalFuncWrapper_128[0]
    const_282 = main_const_eval_141
    util_create_list_276 = [input[84]]
    const_283 = "main_const_eval_141"
    utils_constEvalFuncWrapper_129 = utils.constEvalFuncWrapper(
        const_282, util_create_list_276, _CONST_EVAL_CACHE, const_283
    )
    utils_constEvalFuncWrapper_129_0 = utils_constEvalFuncWrapper_129[0]
    const_284 = main_const_eval_142
    util_create_list_277 = [input[1]]
    const_285 = "main_const_eval_142"
    utils_constEvalFuncWrapper_130 = utils.constEvalFuncWrapper(
        const_284, util_create_list_277, _CONST_EVAL_CACHE, const_285
    )
    utils_constEvalFuncWrapper_130_0 = utils_constEvalFuncWrapper_130[0]
    const_286 = main_const_eval_143
    util_create_list_278 = [input[30]]
    const_287 = "main_const_eval_143"
    utils_constEvalFuncWrapper_131 = utils.constEvalFuncWrapper(
        const_286, util_create_list_278, _CONST_EVAL_CACHE, const_287
    )
    utils_constEvalFuncWrapper_131_0 = utils_constEvalFuncWrapper_131[0]
    const_288 = main_const_eval_144
    util_create_list_279 = [input[94]]
    const_289 = "main_const_eval_144"
    utils_constEvalFuncWrapper_132 = utils.constEvalFuncWrapper(
        const_288, util_create_list_279, _CONST_EVAL_CACHE, const_289
    )
    utils_constEvalFuncWrapper_132_0 = utils_constEvalFuncWrapper_132[0]
    const_290 = main_const_eval_145
    util_create_list_280 = [input[119]]
    const_291 = "main_const_eval_145"
    utils_constEvalFuncWrapper_133 = utils.constEvalFuncWrapper(
        const_290, util_create_list_280, _CONST_EVAL_CACHE, const_291
    )
    utils_constEvalFuncWrapper_133_0 = utils_constEvalFuncWrapper_133[0]
    const_292 = main_const_eval_146
    const_293 = "main_const_eval_146"
    utils_constEvalFuncWrapperZeroArg_12 = utils.constEvalFuncWrapperZeroArg(
        const_292, _CONST_EVAL_CACHE, const_293
    )
    utils_constEvalFuncWrapperZeroArg_12_0 = utils_constEvalFuncWrapperZeroArg_12[0]
    utils_DeviceGetter_get_device_147 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_134 = ttnn.to_layout(
        input[124], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(input[124], False)
    ttnn_divide_0 = ttnn.divide(
        ttnn_to_layout_134,
        utils_constEvalFuncWrapperZeroArg_5_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_134, False)
    ttnn_add_0 = ttnn.add(
        ttnn_divide_0,
        utils_constEvalFuncWrapperZeroArg_4_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_0, False)
    ttnn_permute_170 = ttnn.permute(
        ttnn_add_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_reshape_175 = ttnn.reshape(
        ttnn_permute_170,
        [1, 1, 14400, 16],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_170, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_reshape_175,
        weight_tensor=utils_constEvalFuncWrapper_68_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=16,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_1_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_175, False)
    ttnn_typecast_66 = ttnn.typecast(
        ttnn_conv2d_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_176 = ttnn.reshape(
        ttnn_typecast_66,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn_permute_171 = ttnn.permute(
        ttnn_reshape_176,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_176, False)
    ttnn_reshape_177 = ttnn.reshape(
        ttnn_permute_171,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_171, False)
    ttnn_mean_0 = ttnn.mean(
        ttnn_reshape_177,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_0 = ttnn.neg(
        ttnn_mean_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_0, False)
    ttnn_add_1 = ttnn.add(
        ttnn_reshape_177,
        ttnn_neg_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_0, False)
    ttnn.deallocate(ttnn_reshape_177, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_add_1,
        ttnn_add_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_1 = ttnn.mean(
        ttnn_multiply_0,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_add_2 = ttnn.add(
        ttnn_mean_1,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_1, False)
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_add_2,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_add_1,
        ttnn_rsqrt_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_0, False)
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_multiply_1,
        utils_constEvalFuncWrapper_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_add_3 = ttnn.add(
        ttnn_multiply_2,
        utils_constEvalFuncWrapper_60_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_silu_0 = ttnn.silu(
        ttnn_add_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_typecast_67 = ttnn.typecast(
        ttnn_silu_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_0, False)
    ttnn_reshape_178 = ttnn.reshape(
        ttnn_typecast_67,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn_permute_172 = ttnn.permute(
        ttnn_reshape_178,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_178, False)
    ttnn_reshape_179 = ttnn.reshape(
        ttnn_permute_172,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_172, False)
    ttnn_conv2d_1 = ttnn.conv2d(
        input_tensor=ttnn_reshape_179,
        weight_tensor=utils_constEvalFuncWrapper_133_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_45_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_179, False)
    ttnn_typecast_68 = ttnn.typecast(
        ttnn_conv2d_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_1, False)
    ttnn_reshape_180 = ttnn.reshape(
        ttnn_typecast_68,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_68, False)
    ttnn_permute_173 = ttnn.permute(
        ttnn_reshape_180,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_180, False)
    ttnn_reshape_181 = ttnn.reshape(
        ttnn_permute_173,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_173, False)
    ttnn_mean_2 = ttnn.mean(
        ttnn_reshape_181,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_1 = ttnn.neg(
        ttnn_mean_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_2, False)
    ttnn_add_4 = ttnn.add(
        ttnn_reshape_181,
        ttnn_neg_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_1, False)
    ttnn.deallocate(ttnn_reshape_181, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_add_4,
        ttnn_add_4,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_3 = ttnn.mean(
        ttnn_multiply_3,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_add_5 = ttnn.add(
        ttnn_mean_3,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_3, False)
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_add_5,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_add_4,
        ttnn_rsqrt_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_1, False)
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_multiply_4,
        utils_constEvalFuncWrapper_70_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_add_6 = ttnn.add(
        ttnn_multiply_5,
        utils_constEvalFuncWrapper_43_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn_silu_1 = ttnn.silu(
        ttnn_add_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_typecast_69 = ttnn.typecast(
        ttnn_silu_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_1, False)
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_typecast_69,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_69, False)
    ttnn_permute_174 = ttnn.permute(
        ttnn_reshape_182,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_182, False)
    ttnn_reshape_183 = ttnn.reshape(
        ttnn_permute_174,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_174, False)
    ttnn_conv2d_2 = ttnn.conv2d(
        input_tensor=ttnn_reshape_183,
        weight_tensor=utils_constEvalFuncWrapper_37_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_49_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_183, False)
    ttnn_add_7 = ttnn.add(
        ttnn_conv2d_0,
        ttnn_conv2d_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_2, False)
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_add_7,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_permute_175 = ttnn.permute(
        ttnn_reshape_184,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_184, False)
    ttnn_divide_1 = ttnn.divide(
        ttnn_permute_175,
        utils_constEvalFuncWrapperZeroArg_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_175, False)
    ttnn_typecast_70 = ttnn.typecast(
        ttnn_divide_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_185 = ttnn.reshape(
        ttnn_typecast_70,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_70, False)
    ttnn_mean_4 = ttnn.mean(
        ttnn_reshape_185,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_2 = ttnn.neg(
        ttnn_mean_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_4, False)
    ttnn_add_8 = ttnn.add(
        ttnn_reshape_185,
        ttnn_neg_2,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_2, False)
    ttnn.deallocate(ttnn_reshape_185, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_add_8,
        ttnn_add_8,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_5 = ttnn.mean(
        ttnn_multiply_6,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_add_9 = ttnn.add(
        ttnn_mean_5,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_5, False)
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_add_9,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_add_8,
        ttnn_rsqrt_2,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_2, False)
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_multiply_7,
        utils_constEvalFuncWrapper_67_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn_add_10 = ttnn.add(
        ttnn_multiply_8,
        utils_constEvalFuncWrapper_76_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn_typecast_71 = ttnn.typecast(
        ttnn_add_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_reshape_186 = ttnn.reshape(
        ttnn_typecast_71,
        [1, 512, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_71, False)
    ttnn_permute_176 = ttnn.permute(
        ttnn_reshape_186,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_186, False)
    ttnn_reshape_187 = ttnn.reshape(
        ttnn_permute_176,
        [14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_176, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_187,
        input[134],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_add_11 = ttnn.add(
        ttnn_matmul_0,
        utils_constEvalFuncWrapper_95_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_typecast_72 = ttnn.typecast(
        ttnn_add_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_reshape_188 = ttnn.reshape(
        ttnn_typecast_72,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_72, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_187,
        input[132],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_add_12 = ttnn.add(
        ttnn_matmul_1,
        utils_constEvalFuncWrapper_121_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_typecast_73 = ttnn.typecast(
        ttnn_add_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_reshape_189 = ttnn.reshape(
        ttnn_typecast_73,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_73, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_reshape_187,
        input[128],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_187, False)
    ttnn_add_13 = ttnn.add(
        ttnn_matmul_2,
        utils_constEvalFuncWrapper_94_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_typecast_74 = ttnn.typecast(
        ttnn_add_13,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_typecast_74,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_74, False)
    ttnn_typecast_75 = ttnn.typecast(
        ttnn_reshape_188,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_188, False)
    ttnn_typecast_76 = ttnn.typecast(
        ttnn_reshape_189,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_189, False)
    ttnn_typecast_77 = ttnn.typecast(
        ttnn_reshape_190,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_190, False)
    ttnn_transformer_scaled_dot_product_attention_0 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_75,
            ttnn_typecast_76,
            ttnn_typecast_77,
            attn_mask=None,
            is_causal=False,
            scale=0.04419417679309845,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    ttnn.deallocate(ttnn_typecast_76, False)
    ttnn.deallocate(ttnn_typecast_75, False)
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_transformer_scaled_dot_product_attention_0,
        [14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_reshape_191,
        input[126],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_191, False)
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_3,
        utils_constEvalFuncWrapper_52_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn_permute_177 = ttnn.permute(
        ttnn_add_14,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_reshape_192 = ttnn.reshape(
        ttnn_permute_177,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_177, False)
    ttnn_add_15 = ttnn.add(
        ttnn_reshape_192,
        ttnn_divide_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_192, False)
    ttnn.deallocate(ttnn_divide_1, False)
    ttnn_divide_2 = ttnn.divide(
        ttnn_add_15,
        utils_constEvalFuncWrapperZeroArg_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_15, False)
    ttnn_typecast_78 = ttnn.typecast(
        ttnn_divide_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_193 = ttnn.reshape(
        ttnn_typecast_78,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    ttnn_mean_6 = ttnn.mean(
        ttnn_reshape_193,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_3 = ttnn.neg(
        ttnn_mean_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_6, False)
    ttnn_add_16 = ttnn.add(
        ttnn_reshape_193,
        ttnn_neg_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_3, False)
    ttnn.deallocate(ttnn_reshape_193, False)
    ttnn_multiply_9 = ttnn.multiply(
        ttnn_add_16,
        ttnn_add_16,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_7 = ttnn.mean(
        ttnn_multiply_9,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn_add_17 = ttnn.add(
        ttnn_mean_7,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_7, False)
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_add_17,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_17, False)
    ttnn_multiply_10 = ttnn.multiply(
        ttnn_add_16,
        ttnn_rsqrt_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_3, False)
    ttnn.deallocate(ttnn_add_16, False)
    ttnn_multiply_11 = ttnn.multiply(
        ttnn_multiply_10,
        utils_constEvalFuncWrapper_66_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_10, False)
    ttnn_add_18 = ttnn.add(
        ttnn_multiply_11,
        utils_constEvalFuncWrapper_69_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_11, False)
    ttnn_silu_2 = ttnn.silu(
        ttnn_add_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_18, False)
    ttnn_typecast_79 = ttnn.typecast(
        ttnn_silu_2,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_2, False)
    ttnn_reshape_194 = ttnn.reshape(
        ttnn_typecast_79,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn_permute_178 = ttnn.permute(
        ttnn_reshape_194,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_194, False)
    ttnn_reshape_195 = ttnn.reshape(
        ttnn_permute_178,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_178, False)
    ttnn_conv2d_3 = ttnn.conv2d(
        input_tensor=ttnn_reshape_195,
        weight_tensor=utils_constEvalFuncWrapper_13_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_120_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_195, False)
    ttnn_typecast_80 = ttnn.typecast(
        ttnn_conv2d_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_3, False)
    ttnn_reshape_196 = ttnn.reshape(
        ttnn_typecast_80,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_80, False)
    ttnn_permute_179 = ttnn.permute(
        ttnn_reshape_196,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_196, False)
    ttnn_reshape_197 = ttnn.reshape(
        ttnn_permute_179,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_179, False)
    ttnn_mean_8 = ttnn.mean(
        ttnn_reshape_197,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_4 = ttnn.neg(
        ttnn_mean_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_8, False)
    ttnn_add_19 = ttnn.add(
        ttnn_reshape_197,
        ttnn_neg_4,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_4, False)
    ttnn.deallocate(ttnn_reshape_197, False)
    ttnn_multiply_12 = ttnn.multiply(
        ttnn_add_19,
        ttnn_add_19,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_9 = ttnn.mean(
        ttnn_multiply_12,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_12, False)
    ttnn_add_20 = ttnn.add(
        ttnn_mean_9,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_9, False)
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_add_20,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_20, False)
    ttnn_multiply_13 = ttnn.multiply(
        ttnn_add_19,
        ttnn_rsqrt_4,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_4, False)
    ttnn.deallocate(ttnn_add_19, False)
    ttnn_multiply_14 = ttnn.multiply(
        ttnn_multiply_13,
        utils_constEvalFuncWrapper_20_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_13, False)
    ttnn_add_21 = ttnn.add(
        ttnn_multiply_14,
        utils_constEvalFuncWrapper_89_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_14, False)
    ttnn_silu_3 = ttnn.silu(
        ttnn_add_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_21, False)
    ttnn_typecast_81 = ttnn.typecast(
        ttnn_silu_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_3, False)
    ttnn_reshape_198 = ttnn.reshape(
        ttnn_typecast_81,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_81, False)
    ttnn_permute_180 = ttnn.permute(
        ttnn_reshape_198,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_198, False)
    ttnn_reshape_199 = ttnn.reshape(
        ttnn_permute_180,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_180, False)
    ttnn_conv2d_4 = ttnn.conv2d(
        input_tensor=ttnn_reshape_199,
        weight_tensor=utils_constEvalFuncWrapper_27_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_4_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_199, False)
    ttnn_reshape_200 = ttnn.reshape(
        ttnn_conv2d_4,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_4, False)
    ttnn_permute_181 = ttnn.permute(
        ttnn_reshape_200,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_200, False)
    ttnn_add_22 = ttnn.add(
        ttnn_divide_2,
        ttnn_permute_181,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_181, False)
    ttnn.deallocate(ttnn_divide_2, False)
    ttnn_divide_3 = ttnn.divide(
        ttnn_add_22,
        utils_constEvalFuncWrapperZeroArg_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_22, False)
    ttnn_typecast_82 = ttnn.typecast(
        ttnn_divide_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_201 = ttnn.reshape(
        ttnn_typecast_82,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_82, False)
    ttnn_mean_10 = ttnn.mean(
        ttnn_reshape_201,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_5 = ttnn.neg(
        ttnn_mean_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_10, False)
    ttnn_add_23 = ttnn.add(
        ttnn_reshape_201,
        ttnn_neg_5,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_5, False)
    ttnn.deallocate(ttnn_reshape_201, False)
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_add_23,
        ttnn_add_23,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_11 = ttnn.mean(
        ttnn_multiply_15,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_15, False)
    ttnn_add_24 = ttnn.add(
        ttnn_mean_11,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_11, False)
    ttnn_rsqrt_5 = ttnn.rsqrt(
        ttnn_add_24,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_24, False)
    ttnn_multiply_16 = ttnn.multiply(
        ttnn_add_23,
        ttnn_rsqrt_5,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_5, False)
    ttnn.deallocate(ttnn_add_23, False)
    ttnn_multiply_17 = ttnn.multiply(
        ttnn_multiply_16,
        utils_constEvalFuncWrapper_14_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_16, False)
    ttnn_add_25 = ttnn.add(
        ttnn_multiply_17,
        utils_constEvalFuncWrapper_46_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_17, False)
    ttnn_silu_4 = ttnn.silu(
        ttnn_add_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_25, False)
    ttnn_typecast_83 = ttnn.typecast(
        ttnn_silu_4,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_4, False)
    ttnn_reshape_202 = ttnn.reshape(
        ttnn_typecast_83,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_83, False)
    ttnn_permute_182 = ttnn.permute(
        ttnn_reshape_202,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_202, False)
    ttnn_reshape_203 = ttnn.reshape(
        ttnn_permute_182,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_182, False)
    ttnn_conv2d_5 = ttnn.conv2d(
        input_tensor=ttnn_reshape_203,
        weight_tensor=utils_constEvalFuncWrapper_21_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_17_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_203, False)
    ttnn_typecast_84 = ttnn.typecast(
        ttnn_conv2d_5,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_5, False)
    ttnn_reshape_204 = ttnn.reshape(
        ttnn_typecast_84,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_84, False)
    ttnn_permute_183 = ttnn.permute(
        ttnn_reshape_204,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_204, False)
    ttnn_reshape_205 = ttnn.reshape(
        ttnn_permute_183,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_183, False)
    ttnn_mean_12 = ttnn.mean(
        ttnn_reshape_205,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_6 = ttnn.neg(
        ttnn_mean_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_12, False)
    ttnn_add_26 = ttnn.add(
        ttnn_reshape_205,
        ttnn_neg_6,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_6, False)
    ttnn.deallocate(ttnn_reshape_205, False)
    ttnn_multiply_18 = ttnn.multiply(
        ttnn_add_26,
        ttnn_add_26,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_13 = ttnn.mean(
        ttnn_multiply_18,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_18, False)
    ttnn_add_27 = ttnn.add(
        ttnn_mean_13,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_13, False)
    ttnn_rsqrt_6 = ttnn.rsqrt(
        ttnn_add_27,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_27, False)
    ttnn_multiply_19 = ttnn.multiply(
        ttnn_add_26,
        ttnn_rsqrt_6,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_6, False)
    ttnn.deallocate(ttnn_add_26, False)
    ttnn_multiply_20 = ttnn.multiply(
        ttnn_multiply_19,
        utils_constEvalFuncWrapper_11_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_19, False)
    ttnn_add_28 = ttnn.add(
        ttnn_multiply_20,
        utils_constEvalFuncWrapper_48_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_20, False)
    ttnn_silu_5 = ttnn.silu(
        ttnn_add_28,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_28, False)
    ttnn_typecast_85 = ttnn.typecast(
        ttnn_silu_5,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_5, False)
    ttnn_reshape_206 = ttnn.reshape(
        ttnn_typecast_85,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_85, False)
    ttnn_permute_184 = ttnn.permute(
        ttnn_reshape_206,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_206, False)
    ttnn_reshape_207 = ttnn.reshape(
        ttnn_permute_184,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_184, False)
    ttnn_conv2d_6 = ttnn.conv2d(
        input_tensor=ttnn_reshape_207,
        weight_tensor=utils_constEvalFuncWrapper_106_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_92_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_207, False)
    ttnn_reshape_208 = ttnn.reshape(
        ttnn_conv2d_6,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_6, False)
    ttnn_permute_185 = ttnn.permute(
        ttnn_reshape_208,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_208, False)
    ttnn_add_29 = ttnn.add(
        ttnn_divide_3,
        ttnn_permute_185,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_185, False)
    ttnn.deallocate(ttnn_divide_3, False)
    ttnn_divide_4 = ttnn.divide(
        ttnn_add_29,
        utils_constEvalFuncWrapperZeroArg_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_29, False)
    ttnn_typecast_86 = ttnn.typecast(
        ttnn_divide_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_209 = ttnn.reshape(
        ttnn_typecast_86,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn_mean_14 = ttnn.mean(
        ttnn_reshape_209,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_7 = ttnn.neg(
        ttnn_mean_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_14, False)
    ttnn_add_30 = ttnn.add(
        ttnn_reshape_209,
        ttnn_neg_7,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_7, False)
    ttnn.deallocate(ttnn_reshape_209, False)
    ttnn_multiply_21 = ttnn.multiply(
        ttnn_add_30,
        ttnn_add_30,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_15 = ttnn.mean(
        ttnn_multiply_21,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_21, False)
    ttnn_add_31 = ttnn.add(
        ttnn_mean_15,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_15, False)
    ttnn_rsqrt_7 = ttnn.rsqrt(
        ttnn_add_31,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_31, False)
    ttnn_multiply_22 = ttnn.multiply(
        ttnn_add_30,
        ttnn_rsqrt_7,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_7, False)
    ttnn.deallocate(ttnn_add_30, False)
    ttnn_multiply_23 = ttnn.multiply(
        ttnn_multiply_22,
        utils_constEvalFuncWrapper_125_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_22, False)
    ttnn_add_32 = ttnn.add(
        ttnn_multiply_23,
        utils_constEvalFuncWrapper_32_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_23, False)
    ttnn_silu_6 = ttnn.silu(
        ttnn_add_32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_32, False)
    ttnn_typecast_87 = ttnn.typecast(
        ttnn_silu_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_6, False)
    ttnn_reshape_210 = ttnn.reshape(
        ttnn_typecast_87,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_87, False)
    ttnn_permute_186 = ttnn.permute(
        ttnn_reshape_210,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_210, False)
    ttnn_reshape_211 = ttnn.reshape(
        ttnn_permute_186,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_186, False)
    ttnn_conv2d_7 = ttnn.conv2d(
        input_tensor=ttnn_reshape_211,
        weight_tensor=utils_constEvalFuncWrapper_73_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_132_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_211, False)
    ttnn_typecast_88 = ttnn.typecast(
        ttnn_conv2d_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_7, False)
    ttnn_reshape_212 = ttnn.reshape(
        ttnn_typecast_88,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_88, False)
    ttnn_permute_187 = ttnn.permute(
        ttnn_reshape_212,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_212, False)
    ttnn_reshape_213 = ttnn.reshape(
        ttnn_permute_187,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_187, False)
    ttnn_mean_16 = ttnn.mean(
        ttnn_reshape_213,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_8 = ttnn.neg(
        ttnn_mean_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_16, False)
    ttnn_add_33 = ttnn.add(
        ttnn_reshape_213,
        ttnn_neg_8,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_8, False)
    ttnn.deallocate(ttnn_reshape_213, False)
    ttnn_multiply_24 = ttnn.multiply(
        ttnn_add_33,
        ttnn_add_33,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_17 = ttnn.mean(
        ttnn_multiply_24,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_24, False)
    ttnn_add_34 = ttnn.add(
        ttnn_mean_17,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_17, False)
    ttnn_rsqrt_8 = ttnn.rsqrt(
        ttnn_add_34,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_34, False)
    ttnn_multiply_25 = ttnn.multiply(
        ttnn_add_33,
        ttnn_rsqrt_8,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_8, False)
    ttnn.deallocate(ttnn_add_33, False)
    ttnn_multiply_26 = ttnn.multiply(
        ttnn_multiply_25,
        utils_constEvalFuncWrapper_57_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_25, False)
    ttnn_add_35 = ttnn.add(
        ttnn_multiply_26,
        utils_constEvalFuncWrapper_85_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_26, False)
    ttnn_silu_7 = ttnn.silu(
        ttnn_add_35,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_35, False)
    ttnn_typecast_89 = ttnn.typecast(
        ttnn_silu_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_7, False)
    ttnn_reshape_214 = ttnn.reshape(
        ttnn_typecast_89,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_89, False)
    ttnn_permute_188 = ttnn.permute(
        ttnn_reshape_214,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_214, False)
    ttnn_reshape_215 = ttnn.reshape(
        ttnn_permute_188,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_188, False)
    ttnn_conv2d_8 = ttnn.conv2d(
        input_tensor=ttnn_reshape_215,
        weight_tensor=utils_constEvalFuncWrapper_128_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_61_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_215, False)
    ttnn_reshape_216 = ttnn.reshape(
        ttnn_conv2d_8,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_8, False)
    ttnn_permute_189 = ttnn.permute(
        ttnn_reshape_216,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_216, False)
    ttnn_add_36 = ttnn.add(
        ttnn_divide_4,
        ttnn_permute_189,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_189, False)
    ttnn.deallocate(ttnn_divide_4, False)
    ttnn_divide_5 = ttnn.divide(
        ttnn_add_36,
        utils_constEvalFuncWrapperZeroArg_10_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_36, False)
    ttnn_typecast_90 = ttnn.typecast(
        ttnn_divide_5,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_217 = ttnn.reshape(
        ttnn_typecast_90,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_90, False)
    ttnn_mean_18 = ttnn.mean(
        ttnn_reshape_217,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_9 = ttnn.neg(
        ttnn_mean_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_18, False)
    ttnn_add_37 = ttnn.add(
        ttnn_reshape_217,
        ttnn_neg_9,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_9, False)
    ttnn.deallocate(ttnn_reshape_217, False)
    ttnn_multiply_27 = ttnn.multiply(
        ttnn_add_37,
        ttnn_add_37,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_19 = ttnn.mean(
        ttnn_multiply_27,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_27, False)
    ttnn_add_38 = ttnn.add(
        ttnn_mean_19,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_19, False)
    ttnn_rsqrt_9 = ttnn.rsqrt(
        ttnn_add_38,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_38, False)
    ttnn_multiply_28 = ttnn.multiply(
        ttnn_add_37,
        ttnn_rsqrt_9,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_9, False)
    ttnn.deallocate(ttnn_add_37, False)
    ttnn_multiply_29 = ttnn.multiply(
        ttnn_multiply_28,
        utils_constEvalFuncWrapper_8_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_28, False)
    ttnn_add_39 = ttnn.add(
        ttnn_multiply_29,
        utils_constEvalFuncWrapper_16_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_29, False)
    ttnn_silu_8 = ttnn.silu(
        ttnn_add_39,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_39, False)
    ttnn_typecast_91 = ttnn.typecast(
        ttnn_silu_8,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_8, False)
    ttnn_reshape_218 = ttnn.reshape(
        ttnn_typecast_91,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_91, False)
    ttnn_permute_190 = ttnn.permute(
        ttnn_reshape_218,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_218, False)
    ttnn_reshape_219 = ttnn.reshape(
        ttnn_permute_190,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_190, False)
    ttnn_conv2d_9 = ttnn.conv2d(
        input_tensor=ttnn_reshape_219,
        weight_tensor=utils_constEvalFuncWrapper_41_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_29_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_219, False)
    ttnn_typecast_92 = ttnn.typecast(
        ttnn_conv2d_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_9, False)
    ttnn_reshape_220 = ttnn.reshape(
        ttnn_typecast_92,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn_permute_191 = ttnn.permute(
        ttnn_reshape_220,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_220, False)
    ttnn_reshape_221 = ttnn.reshape(
        ttnn_permute_191,
        [1, 32, 16, 14400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_191, False)
    ttnn_mean_20 = ttnn.mean(
        ttnn_reshape_221,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_10 = ttnn.neg(
        ttnn_mean_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_20, False)
    ttnn_add_40 = ttnn.add(
        ttnn_reshape_221,
        ttnn_neg_10,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_10, False)
    ttnn.deallocate(ttnn_reshape_221, False)
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_add_40,
        ttnn_add_40,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_21 = ttnn.mean(
        ttnn_multiply_30,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_30, False)
    ttnn_add_41 = ttnn.add(
        ttnn_mean_21,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_21, False)
    ttnn_rsqrt_10 = ttnn.rsqrt(
        ttnn_add_41,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_41, False)
    ttnn_multiply_31 = ttnn.multiply(
        ttnn_add_40,
        ttnn_rsqrt_10,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_10, False)
    ttnn.deallocate(ttnn_add_40, False)
    ttnn_multiply_32 = ttnn.multiply(
        ttnn_multiply_31,
        utils_constEvalFuncWrapper_31_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_31, False)
    ttnn_add_42 = ttnn.add(
        ttnn_multiply_32,
        utils_constEvalFuncWrapper_129_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_32, False)
    ttnn_silu_9 = ttnn.silu(
        ttnn_add_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_42, False)
    ttnn_typecast_93 = ttnn.typecast(
        ttnn_silu_9,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_9, False)
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_typecast_93,
        [1, 512, 160, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_93, False)
    ttnn_permute_192 = ttnn.permute(
        ttnn_reshape_222,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_222, False)
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_permute_192,
        [1, 1, 14400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_192, False)
    ttnn_conv2d_10 = ttnn.conv2d(
        input_tensor=ttnn_reshape_223,
        weight_tensor=utils_constEvalFuncWrapper_110_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=160,
        input_width=90,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_93_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_223, False)
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_conv2d_10,
        [1, 160, 90, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_10, False)
    ttnn_permute_193 = ttnn.permute(
        ttnn_divide_5,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_divide_5, False)
    ttnn_permute_194 = ttnn.permute(
        ttnn_reshape_224,
        [0, 3, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_224, False)
    ttnn_add_43 = ttnn.add(
        ttnn_permute_193,
        ttnn_permute_194,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_194, False)
    ttnn.deallocate(ttnn_permute_193, False)
    ttnn_divide_6 = ttnn.divide(
        ttnn_add_43,
        utils_constEvalFuncWrapperZeroArg_10_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_43, False)
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_divide_6,
        [46080, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_6, False)
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_reshape_225,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_225, False)
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_matmul_4,
        [1, 512, 90, 320],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn_permute_195 = ttnn.permute(
        ttnn_reshape_226,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_226, False)
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_permute_195,
        [163840, 90],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_195, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_reshape_227,
        utils_constEvalFuncWrapperZeroArg_11_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_227, False)
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_matmul_5,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_permute_196 = ttnn.permute(
        ttnn_reshape_228,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_228, False)
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_permute_196,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_196, False)
    ttnn_conv2d_11 = ttnn.conv2d(
        input_tensor=ttnn_reshape_229,
        weight_tensor=utils_constEvalFuncWrapper_28_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_34_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_229, False)
    ttnn_typecast_94 = ttnn.typecast(
        ttnn_conv2d_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_typecast_94,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    ttnn_permute_197 = ttnn.permute(
        ttnn_reshape_230,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_230, False)
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_permute_197,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_197, False)
    ttnn_mean_22 = ttnn.mean(
        ttnn_reshape_231,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_11 = ttnn.neg(
        ttnn_mean_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_22, False)
    ttnn_add_44 = ttnn.add(
        ttnn_reshape_231,
        ttnn_neg_11,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_11, False)
    ttnn.deallocate(ttnn_reshape_231, False)
    ttnn_multiply_33 = ttnn.multiply(
        ttnn_add_44,
        ttnn_add_44,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_23 = ttnn.mean(
        ttnn_multiply_33,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_33, False)
    ttnn_add_45 = ttnn.add(
        ttnn_mean_23,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_23, False)
    ttnn_rsqrt_11 = ttnn.rsqrt(
        ttnn_add_45,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_45, False)
    ttnn_multiply_34 = ttnn.multiply(
        ttnn_add_44,
        ttnn_rsqrt_11,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_11, False)
    ttnn.deallocate(ttnn_add_44, False)
    ttnn_multiply_35 = ttnn.multiply(
        ttnn_multiply_34,
        utils_constEvalFuncWrapper_39_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_34, False)
    ttnn_add_46 = ttnn.add(
        ttnn_multiply_35,
        utils_constEvalFuncWrapper_10_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_35, False)
    ttnn_silu_10 = ttnn.silu(
        ttnn_add_46,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_46, False)
    ttnn_typecast_95 = ttnn.typecast(
        ttnn_silu_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_10, False)
    ttnn_reshape_232 = ttnn.reshape(
        ttnn_typecast_95,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_95, False)
    ttnn_permute_198 = ttnn.permute(
        ttnn_reshape_232,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_232, False)
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_permute_198,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_198, False)
    ttnn_conv2d_12 = ttnn.conv2d(
        input_tensor=ttnn_reshape_233,
        weight_tensor=utils_constEvalFuncWrapper_22_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_55_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_233, False)
    ttnn_typecast_96 = ttnn.typecast(
        ttnn_conv2d_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_12, False)
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_typecast_96,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_96, False)
    ttnn_permute_199 = ttnn.permute(
        ttnn_reshape_234,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_234, False)
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_permute_199,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_199, False)
    ttnn_mean_24 = ttnn.mean(
        ttnn_reshape_235,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_12 = ttnn.neg(
        ttnn_mean_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_24, False)
    ttnn_add_47 = ttnn.add(
        ttnn_reshape_235,
        ttnn_neg_12,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_12, False)
    ttnn.deallocate(ttnn_reshape_235, False)
    ttnn_multiply_36 = ttnn.multiply(
        ttnn_add_47,
        ttnn_add_47,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_25 = ttnn.mean(
        ttnn_multiply_36,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_36, False)
    ttnn_add_48 = ttnn.add(
        ttnn_mean_25,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_25, False)
    ttnn_rsqrt_12 = ttnn.rsqrt(
        ttnn_add_48,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_48, False)
    ttnn_multiply_37 = ttnn.multiply(
        ttnn_add_47,
        ttnn_rsqrt_12,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_12, False)
    ttnn.deallocate(ttnn_add_47, False)
    ttnn_multiply_38 = ttnn.multiply(
        ttnn_multiply_37,
        utils_constEvalFuncWrapper_58_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_37, False)
    ttnn_add_49 = ttnn.add(
        ttnn_multiply_38,
        utils_constEvalFuncWrapper_119_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_38, False)
    ttnn_silu_11 = ttnn.silu(
        ttnn_add_49,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_49, False)
    ttnn_typecast_97 = ttnn.typecast(
        ttnn_silu_11,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_11, False)
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_typecast_97,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn_permute_200 = ttnn.permute(
        ttnn_reshape_236,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_236, False)
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_permute_200,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_200, False)
    ttnn_conv2d_13 = ttnn.conv2d(
        input_tensor=ttnn_reshape_237,
        weight_tensor=utils_constEvalFuncWrapper_112_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_99_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_237, False)
    ttnn_add_50 = ttnn.add(
        ttnn_conv2d_11,
        ttnn_conv2d_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_13, False)
    ttnn.deallocate(ttnn_conv2d_11, False)
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_add_50,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_50, False)
    ttnn_permute_201 = ttnn.permute(
        ttnn_reshape_238,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_238, False)
    ttnn_divide_7 = ttnn.divide(
        ttnn_permute_201,
        utils_constEvalFuncWrapperZeroArg_8_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_201, False)
    ttnn_typecast_98 = ttnn.typecast(
        ttnn_divide_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_239 = ttnn.reshape(
        ttnn_typecast_98,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn_mean_26 = ttnn.mean(
        ttnn_reshape_239,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_13 = ttnn.neg(
        ttnn_mean_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_26, False)
    ttnn_add_51 = ttnn.add(
        ttnn_reshape_239,
        ttnn_neg_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_13, False)
    ttnn.deallocate(ttnn_reshape_239, False)
    ttnn_multiply_39 = ttnn.multiply(
        ttnn_add_51,
        ttnn_add_51,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_27 = ttnn.mean(
        ttnn_multiply_39,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_39, False)
    ttnn_add_52 = ttnn.add(
        ttnn_mean_27,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_27, False)
    ttnn_rsqrt_13 = ttnn.rsqrt(
        ttnn_add_52,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_52, False)
    ttnn_multiply_40 = ttnn.multiply(
        ttnn_add_51,
        ttnn_rsqrt_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_13, False)
    ttnn.deallocate(ttnn_add_51, False)
    ttnn_multiply_41 = ttnn.multiply(
        ttnn_multiply_40,
        utils_constEvalFuncWrapper_33_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_40, False)
    ttnn_add_53 = ttnn.add(
        ttnn_multiply_41,
        utils_constEvalFuncWrapper_97_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_41, False)
    ttnn_silu_12 = ttnn.silu(
        ttnn_add_53,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_53, False)
    ttnn_typecast_99 = ttnn.typecast(
        ttnn_silu_12,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_12, False)
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_typecast_99,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_99, False)
    ttnn_permute_202 = ttnn.permute(
        ttnn_reshape_240,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_240, False)
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_permute_202,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_202, False)
    ttnn_conv2d_14 = ttnn.conv2d(
        input_tensor=ttnn_reshape_241,
        weight_tensor=utils_constEvalFuncWrapper_51_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_18_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_241, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_conv2d_14,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_14, False)
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_typecast_100,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn_permute_203 = ttnn.permute(
        ttnn_reshape_242,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_242, False)
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_permute_203,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_203, False)
    ttnn_mean_28 = ttnn.mean(
        ttnn_reshape_243,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_14 = ttnn.neg(
        ttnn_mean_28,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_28, False)
    ttnn_add_54 = ttnn.add(
        ttnn_reshape_243,
        ttnn_neg_14,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_14, False)
    ttnn.deallocate(ttnn_reshape_243, False)
    ttnn_multiply_42 = ttnn.multiply(
        ttnn_add_54,
        ttnn_add_54,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_29 = ttnn.mean(
        ttnn_multiply_42,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_42, False)
    ttnn_add_55 = ttnn.add(
        ttnn_mean_29,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_29, False)
    ttnn_rsqrt_14 = ttnn.rsqrt(
        ttnn_add_55,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_55, False)
    ttnn_multiply_43 = ttnn.multiply(
        ttnn_add_54,
        ttnn_rsqrt_14,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_14, False)
    ttnn.deallocate(ttnn_add_54, False)
    ttnn_multiply_44 = ttnn.multiply(
        ttnn_multiply_43,
        utils_constEvalFuncWrapper_91_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_43, False)
    ttnn_add_56 = ttnn.add(
        ttnn_multiply_44,
        utils_constEvalFuncWrapper_72_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_44, False)
    ttnn_silu_13 = ttnn.silu(
        ttnn_add_56,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_56, False)
    ttnn_typecast_101 = ttnn.typecast(
        ttnn_silu_13,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_13, False)
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_typecast_101,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn_permute_204 = ttnn.permute(
        ttnn_reshape_244,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_244, False)
    ttnn_reshape_245 = ttnn.reshape(
        ttnn_permute_204,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_204, False)
    ttnn_conv2d_15 = ttnn.conv2d(
        input_tensor=ttnn_reshape_245,
        weight_tensor=utils_constEvalFuncWrapper_65_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_71_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_245, False)
    ttnn_reshape_246 = ttnn.reshape(
        ttnn_conv2d_15,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_15, False)
    ttnn_permute_205 = ttnn.permute(
        ttnn_reshape_246,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_246, False)
    ttnn_add_57 = ttnn.add(
        ttnn_divide_7,
        ttnn_permute_205,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_205, False)
    ttnn.deallocate(ttnn_divide_7, False)
    ttnn_divide_8 = ttnn.divide(
        ttnn_add_57,
        utils_constEvalFuncWrapperZeroArg_8_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_57, False)
    ttnn_typecast_102 = ttnn.typecast(
        ttnn_divide_8,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_247 = ttnn.reshape(
        ttnn_typecast_102,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn_mean_30 = ttnn.mean(
        ttnn_reshape_247,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_15 = ttnn.neg(
        ttnn_mean_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_30, False)
    ttnn_add_58 = ttnn.add(
        ttnn_reshape_247,
        ttnn_neg_15,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_15, False)
    ttnn.deallocate(ttnn_reshape_247, False)
    ttnn_multiply_45 = ttnn.multiply(
        ttnn_add_58,
        ttnn_add_58,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_31 = ttnn.mean(
        ttnn_multiply_45,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_45, False)
    ttnn_add_59 = ttnn.add(
        ttnn_mean_31,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_31, False)
    ttnn_rsqrt_15 = ttnn.rsqrt(
        ttnn_add_59,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_59, False)
    ttnn_multiply_46 = ttnn.multiply(
        ttnn_add_58,
        ttnn_rsqrt_15,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_15, False)
    ttnn.deallocate(ttnn_add_58, False)
    ttnn_multiply_47 = ttnn.multiply(
        ttnn_multiply_46,
        utils_constEvalFuncWrapper_5_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_46, False)
    ttnn_add_60 = ttnn.add(
        ttnn_multiply_47,
        utils_constEvalFuncWrapper_7_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_47, False)
    ttnn_silu_14 = ttnn.silu(
        ttnn_add_60,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_60, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_silu_14,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_14, False)
    ttnn_reshape_248 = ttnn.reshape(
        ttnn_typecast_103,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn_permute_206 = ttnn.permute(
        ttnn_reshape_248,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_248, False)
    ttnn_reshape_249 = ttnn.reshape(
        ttnn_permute_206,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_206, False)
    ttnn_conv2d_16 = ttnn.conv2d(
        input_tensor=ttnn_reshape_249,
        weight_tensor=utils_constEvalFuncWrapper_79_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_114_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_249, False)
    ttnn_typecast_104 = ttnn.typecast(
        ttnn_conv2d_16,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_16, False)
    ttnn_reshape_250 = ttnn.reshape(
        ttnn_typecast_104,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_104, False)
    ttnn_permute_207 = ttnn.permute(
        ttnn_reshape_250,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_250, False)
    ttnn_reshape_251 = ttnn.reshape(
        ttnn_permute_207,
        [1, 32, 16, 57600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_207, False)
    ttnn_mean_32 = ttnn.mean(
        ttnn_reshape_251,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_16 = ttnn.neg(
        ttnn_mean_32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_32, False)
    ttnn_add_61 = ttnn.add(
        ttnn_reshape_251,
        ttnn_neg_16,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_16, False)
    ttnn.deallocate(ttnn_reshape_251, False)
    ttnn_multiply_48 = ttnn.multiply(
        ttnn_add_61,
        ttnn_add_61,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_33 = ttnn.mean(
        ttnn_multiply_48,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_48, False)
    ttnn_add_62 = ttnn.add(
        ttnn_mean_33,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_33, False)
    ttnn_rsqrt_16 = ttnn.rsqrt(
        ttnn_add_62,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_62, False)
    ttnn_multiply_49 = ttnn.multiply(
        ttnn_add_61,
        ttnn_rsqrt_16,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_16, False)
    ttnn.deallocate(ttnn_add_61, False)
    ttnn_multiply_50 = ttnn.multiply(
        ttnn_multiply_49,
        utils_constEvalFuncWrapper_116_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_49, False)
    ttnn_add_63 = ttnn.add(
        ttnn_multiply_50,
        utils_constEvalFuncWrapper_78_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_50, False)
    ttnn_silu_15 = ttnn.silu(
        ttnn_add_63,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_63, False)
    ttnn_typecast_105 = ttnn.typecast(
        ttnn_silu_15,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_15, False)
    ttnn_reshape_252 = ttnn.reshape(
        ttnn_typecast_105,
        [1, 512, 320, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_105, False)
    ttnn_permute_208 = ttnn.permute(
        ttnn_reshape_252,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_252, False)
    ttnn_reshape_253 = ttnn.reshape(
        ttnn_permute_208,
        [1, 1, 57600, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_208, False)
    ttnn_conv2d_17 = ttnn.conv2d(
        input_tensor=ttnn_reshape_253,
        weight_tensor=utils_constEvalFuncWrapper_107_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=320,
        input_width=180,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_44_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_253, False)
    ttnn_reshape_254 = ttnn.reshape(
        ttnn_conv2d_17,
        [1, 320, 180, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_17, False)
    ttnn_permute_209 = ttnn.permute(
        ttnn_divide_8,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_divide_8, False)
    ttnn_permute_210 = ttnn.permute(
        ttnn_reshape_254,
        [0, 3, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_254, False)
    ttnn_add_64 = ttnn.add(
        ttnn_permute_209,
        ttnn_permute_210,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_210, False)
    ttnn.deallocate(ttnn_permute_209, False)
    ttnn_divide_9 = ttnn.divide(
        ttnn_add_64,
        utils_constEvalFuncWrapperZeroArg_8_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_64, False)
    ttnn_reshape_255 = ttnn.reshape(
        ttnn_divide_9,
        [92160, 320],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_9, False)
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_reshape_255,
        utils_constEvalFuncWrapperZeroArg_7_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_255, False)
    ttnn_reshape_256 = ttnn.reshape(
        ttnn_matmul_6,
        [1, 512, 180, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_permute_211 = ttnn.permute(
        ttnn_reshape_256,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_256, False)
    ttnn_reshape_257 = ttnn.reshape(
        ttnn_permute_211,
        [327680, 180],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_211, False)
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_257,
        utils_constEvalFuncWrapperZeroArg_12_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_257, False)
    ttnn_reshape_258 = ttnn.reshape(
        ttnn_matmul_7,
        [1, 512, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn_permute_212 = ttnn.permute(
        ttnn_reshape_258,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_258, False)
    ttnn_reshape_259 = ttnn.reshape(
        ttnn_permute_212,
        [1, 1, 230400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_212, False)
    ttnn_conv2d_18 = ttnn.conv2d(
        input_tensor=ttnn_reshape_259,
        weight_tensor=utils_constEvalFuncWrapper_15_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_88_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_259, False)
    ttnn_conv2d_19 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_18,
        weight_tensor=utils_constEvalFuncWrapper_0_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_82_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_106 = ttnn.typecast(
        ttnn_conv2d_18,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_18, False)
    ttnn_reshape_260 = ttnn.reshape(
        ttnn_typecast_106,
        [1, 640, 360, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_106, False)
    ttnn_permute_213 = ttnn.permute(
        ttnn_reshape_260,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_260, False)
    ttnn_reshape_261 = ttnn.reshape(
        ttnn_permute_213,
        [1, 32, 16, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_213, False)
    ttnn_mean_34 = ttnn.mean(
        ttnn_reshape_261,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_17 = ttnn.neg(
        ttnn_mean_34,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_34, False)
    ttnn_add_65 = ttnn.add(
        ttnn_reshape_261,
        ttnn_neg_17,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_17, False)
    ttnn.deallocate(ttnn_reshape_261, False)
    ttnn_multiply_51 = ttnn.multiply(
        ttnn_add_65,
        ttnn_add_65,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_35 = ttnn.mean(
        ttnn_multiply_51,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_51, False)
    ttnn_add_66 = ttnn.add(
        ttnn_mean_35,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_35, False)
    ttnn_rsqrt_17 = ttnn.rsqrt(
        ttnn_add_66,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_66, False)
    ttnn_multiply_52 = ttnn.multiply(
        ttnn_add_65,
        ttnn_rsqrt_17,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_17, False)
    ttnn.deallocate(ttnn_add_65, False)
    ttnn_multiply_53 = ttnn.multiply(
        ttnn_multiply_52,
        utils_constEvalFuncWrapper_105_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_52, False)
    ttnn_add_67 = ttnn.add(
        ttnn_multiply_53,
        utils_constEvalFuncWrapper_24_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_53, False)
    ttnn_silu_16 = ttnn.silu(
        ttnn_add_67,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_67, False)
    ttnn_typecast_107 = ttnn.typecast(
        ttnn_silu_16,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_16, False)
    ttnn_reshape_262 = ttnn.reshape(
        ttnn_typecast_107,
        [1, 512, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_107, False)
    ttnn_permute_214 = ttnn.permute(
        ttnn_reshape_262,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_262, False)
    ttnn_reshape_263 = ttnn.reshape(
        ttnn_permute_214,
        [1, 1, 230400, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_214, False)
    ttnn_conv2d_20 = ttnn.conv2d(
        input_tensor=ttnn_reshape_263,
        weight_tensor=utils_constEvalFuncWrapper_115_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_100_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_263, False)
    ttnn_typecast_108 = ttnn.typecast(
        ttnn_conv2d_20,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_20, False)
    ttnn_reshape_264 = ttnn.reshape(
        ttnn_typecast_108,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_108, False)
    ttnn_permute_215 = ttnn.permute(
        ttnn_reshape_264,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_264, False)
    ttnn_reshape_265 = ttnn.reshape(
        ttnn_permute_215,
        [1, 32, 8, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_215, False)
    ttnn_mean_36 = ttnn.mean(
        ttnn_reshape_265,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_18 = ttnn.neg(
        ttnn_mean_36,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_36, False)
    ttnn_add_68 = ttnn.add(
        ttnn_reshape_265,
        ttnn_neg_18,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_18, False)
    ttnn.deallocate(ttnn_reshape_265, False)
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_add_68,
        ttnn_add_68,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_37 = ttnn.mean(
        ttnn_multiply_54,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_54, False)
    ttnn_add_69 = ttnn.add(
        ttnn_mean_37,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_37, False)
    ttnn_rsqrt_18 = ttnn.rsqrt(
        ttnn_add_69,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_69, False)
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_add_68,
        ttnn_rsqrt_18,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_18, False)
    ttnn.deallocate(ttnn_add_68, False)
    ttnn_multiply_56 = ttnn.multiply(
        ttnn_multiply_55,
        utils_constEvalFuncWrapper_23_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_55, False)
    ttnn_add_70 = ttnn.add(
        ttnn_multiply_56,
        utils_constEvalFuncWrapper_35_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_56, False)
    ttnn_silu_17 = ttnn.silu(
        ttnn_add_70,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_70, False)
    ttnn_typecast_109 = ttnn.typecast(
        ttnn_silu_17,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_17, False)
    ttnn_reshape_266 = ttnn.reshape(
        ttnn_typecast_109,
        [1, 256, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_109, False)
    ttnn_permute_216 = ttnn.permute(
        ttnn_reshape_266,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_266, False)
    ttnn_reshape_267 = ttnn.reshape(
        ttnn_permute_216,
        [1, 1, 230400, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_216, False)
    ttnn_conv2d_21 = ttnn.conv2d(
        input_tensor=ttnn_reshape_267,
        weight_tensor=utils_constEvalFuncWrapper_63_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_42_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_267, False)
    ttnn_add_71 = ttnn.add(
        ttnn_conv2d_19,
        ttnn_conv2d_21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_21, False)
    ttnn.deallocate(ttnn_conv2d_19, False)
    ttnn_reshape_268 = ttnn.reshape(
        ttnn_add_71,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_71, False)
    ttnn_permute_217 = ttnn.permute(
        ttnn_reshape_268,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_268, False)
    ttnn_divide_10 = ttnn.divide(
        ttnn_permute_217,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_217, False)
    ttnn_typecast_110 = ttnn.typecast(
        ttnn_divide_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_269 = ttnn.reshape(
        ttnn_typecast_110,
        [1, 32, 8, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_110, False)
    ttnn_mean_38 = ttnn.mean(
        ttnn_reshape_269,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_19 = ttnn.neg(
        ttnn_mean_38,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_38, False)
    ttnn_add_72 = ttnn.add(
        ttnn_reshape_269,
        ttnn_neg_19,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_19, False)
    ttnn.deallocate(ttnn_reshape_269, False)
    ttnn_multiply_57 = ttnn.multiply(
        ttnn_add_72,
        ttnn_add_72,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_39 = ttnn.mean(
        ttnn_multiply_57,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_57, False)
    ttnn_add_73 = ttnn.add(
        ttnn_mean_39,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_39, False)
    ttnn_rsqrt_19 = ttnn.rsqrt(
        ttnn_add_73,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_73, False)
    ttnn_multiply_58 = ttnn.multiply(
        ttnn_add_72,
        ttnn_rsqrt_19,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_19, False)
    ttnn.deallocate(ttnn_add_72, False)
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_multiply_58,
        utils_constEvalFuncWrapper_77_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_58, False)
    ttnn_add_74 = ttnn.add(
        ttnn_multiply_59,
        utils_constEvalFuncWrapper_53_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_59, False)
    ttnn_silu_18 = ttnn.silu(
        ttnn_add_74,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_74, False)
    ttnn_typecast_111 = ttnn.typecast(
        ttnn_silu_18,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_18, False)
    ttnn_reshape_270 = ttnn.reshape(
        ttnn_typecast_111,
        [1, 256, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_111, False)
    ttnn_permute_218 = ttnn.permute(
        ttnn_reshape_270,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_270, False)
    ttnn_reshape_271 = ttnn.reshape(
        ttnn_permute_218,
        [1, 1, 230400, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_218, False)
    ttnn_conv2d_22 = ttnn.conv2d(
        input_tensor=ttnn_reshape_271,
        weight_tensor=utils_constEvalFuncWrapper_123_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_117_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_271, False)
    ttnn_typecast_112 = ttnn.typecast(
        ttnn_conv2d_22,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_22, False)
    ttnn_reshape_272 = ttnn.reshape(
        ttnn_typecast_112,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_112, False)
    ttnn_permute_219 = ttnn.permute(
        ttnn_reshape_272,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_272, False)
    ttnn_reshape_273 = ttnn.reshape(
        ttnn_permute_219,
        [1, 32, 8, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_219, False)
    ttnn_mean_40 = ttnn.mean(
        ttnn_reshape_273,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_20 = ttnn.neg(
        ttnn_mean_40,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_40, False)
    ttnn_add_75 = ttnn.add(
        ttnn_reshape_273,
        ttnn_neg_20,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_20, False)
    ttnn.deallocate(ttnn_reshape_273, False)
    ttnn_multiply_60 = ttnn.multiply(
        ttnn_add_75,
        ttnn_add_75,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_41 = ttnn.mean(
        ttnn_multiply_60,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_60, False)
    ttnn_add_76 = ttnn.add(
        ttnn_mean_41,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_41, False)
    ttnn_rsqrt_20 = ttnn.rsqrt(
        ttnn_add_76,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_76, False)
    ttnn_multiply_61 = ttnn.multiply(
        ttnn_add_75,
        ttnn_rsqrt_20,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_20, False)
    ttnn.deallocate(ttnn_add_75, False)
    ttnn_multiply_62 = ttnn.multiply(
        ttnn_multiply_61,
        utils_constEvalFuncWrapper_59_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_61, False)
    ttnn_add_77 = ttnn.add(
        ttnn_multiply_62,
        utils_constEvalFuncWrapper_74_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_62, False)
    ttnn_silu_19 = ttnn.silu(
        ttnn_add_77,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_77, False)
    ttnn_typecast_113 = ttnn.typecast(
        ttnn_silu_19,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_19, False)
    ttnn_reshape_274 = ttnn.reshape(
        ttnn_typecast_113,
        [1, 256, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_113, False)
    ttnn_permute_220 = ttnn.permute(
        ttnn_reshape_274,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_274, False)
    ttnn_reshape_275 = ttnn.reshape(
        ttnn_permute_220,
        [1, 1, 230400, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_220, False)
    ttnn_conv2d_23 = ttnn.conv2d(
        input_tensor=ttnn_reshape_275,
        weight_tensor=utils_constEvalFuncWrapper_81_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_98_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_275, False)
    ttnn_reshape_276 = ttnn.reshape(
        ttnn_conv2d_23,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_23, False)
    ttnn_permute_221 = ttnn.permute(
        ttnn_reshape_276,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_276, False)
    ttnn_add_78 = ttnn.add(
        ttnn_divide_10,
        ttnn_permute_221,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_221, False)
    ttnn.deallocate(ttnn_divide_10, False)
    ttnn_divide_11 = ttnn.divide(
        ttnn_add_78,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_78, False)
    ttnn_typecast_114 = ttnn.typecast(
        ttnn_divide_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_277 = ttnn.reshape(
        ttnn_typecast_114,
        [1, 32, 8, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_114, False)
    ttnn_mean_42 = ttnn.mean(
        ttnn_reshape_277,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_21 = ttnn.neg(
        ttnn_mean_42,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_42, False)
    ttnn_add_79 = ttnn.add(
        ttnn_reshape_277,
        ttnn_neg_21,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_21, False)
    ttnn.deallocate(ttnn_reshape_277, False)
    ttnn_multiply_63 = ttnn.multiply(
        ttnn_add_79,
        ttnn_add_79,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_43 = ttnn.mean(
        ttnn_multiply_63,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_63, False)
    ttnn_add_80 = ttnn.add(
        ttnn_mean_43,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_43, False)
    ttnn_rsqrt_21 = ttnn.rsqrt(
        ttnn_add_80,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_80, False)
    ttnn_multiply_64 = ttnn.multiply(
        ttnn_add_79,
        ttnn_rsqrt_21,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_21, False)
    ttnn.deallocate(ttnn_add_79, False)
    ttnn_multiply_65 = ttnn.multiply(
        ttnn_multiply_64,
        utils_constEvalFuncWrapper_50_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_64, False)
    ttnn_add_81 = ttnn.add(
        ttnn_multiply_65,
        utils_constEvalFuncWrapper_108_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_65, False)
    ttnn_silu_20 = ttnn.silu(
        ttnn_add_81,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_81, False)
    ttnn_typecast_115 = ttnn.typecast(
        ttnn_silu_20,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_20, False)
    ttnn_reshape_278 = ttnn.reshape(
        ttnn_typecast_115,
        [1, 256, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_115, False)
    ttnn_permute_222 = ttnn.permute(
        ttnn_reshape_278,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_278, False)
    ttnn_reshape_279 = ttnn.reshape(
        ttnn_permute_222,
        [1, 1, 230400, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_222, False)
    ttnn_conv2d_24 = ttnn.conv2d(
        input_tensor=ttnn_reshape_279,
        weight_tensor=utils_constEvalFuncWrapper_86_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_25_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_279, False)
    ttnn_typecast_116 = ttnn.typecast(
        ttnn_conv2d_24,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_24, False)
    ttnn_reshape_280 = ttnn.reshape(
        ttnn_typecast_116,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_116, False)
    ttnn_permute_223 = ttnn.permute(
        ttnn_reshape_280,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_280, False)
    ttnn_reshape_281 = ttnn.reshape(
        ttnn_permute_223,
        [1, 32, 8, 230400],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_223, False)
    ttnn_mean_44 = ttnn.mean(
        ttnn_reshape_281,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_22 = ttnn.neg(
        ttnn_mean_44,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_44, False)
    ttnn_add_82 = ttnn.add(
        ttnn_reshape_281,
        ttnn_neg_22,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_22, False)
    ttnn.deallocate(ttnn_reshape_281, False)
    ttnn_multiply_66 = ttnn.multiply(
        ttnn_add_82,
        ttnn_add_82,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_45 = ttnn.mean(
        ttnn_multiply_66,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_66, False)
    ttnn_add_83 = ttnn.add(
        ttnn_mean_45,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_45, False)
    ttnn_rsqrt_22 = ttnn.rsqrt(
        ttnn_add_83,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_83, False)
    ttnn_multiply_67 = ttnn.multiply(
        ttnn_add_82,
        ttnn_rsqrt_22,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_22, False)
    ttnn.deallocate(ttnn_add_82, False)
    ttnn_multiply_68 = ttnn.multiply(
        ttnn_multiply_67,
        utils_constEvalFuncWrapper_2_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_67, False)
    ttnn_add_84 = ttnn.add(
        ttnn_multiply_68,
        utils_constEvalFuncWrapper_80_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_68, False)
    ttnn_silu_21 = ttnn.silu(
        ttnn_add_84,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_84, False)
    ttnn_typecast_117 = ttnn.typecast(
        ttnn_silu_21,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_21, False)
    ttnn_reshape_282 = ttnn.reshape(
        ttnn_typecast_117,
        [1, 256, 640, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_117, False)
    ttnn_permute_224 = ttnn.permute(
        ttnn_reshape_282,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_282, False)
    ttnn_reshape_283 = ttnn.reshape(
        ttnn_permute_224,
        [1, 1, 230400, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_224, False)
    ttnn_conv2d_25 = ttnn.conv2d(
        input_tensor=ttnn_reshape_283,
        weight_tensor=utils_constEvalFuncWrapper_102_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=640,
        input_width=360,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_131_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_283, False)
    ttnn_reshape_284 = ttnn.reshape(
        ttnn_conv2d_25,
        [1, 640, 360, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_25, False)
    ttnn_permute_225 = ttnn.permute(
        ttnn_divide_11,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_divide_11, False)
    ttnn_permute_226 = ttnn.permute(
        ttnn_reshape_284,
        [0, 3, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_284, False)
    ttnn_add_85 = ttnn.add(
        ttnn_permute_225,
        ttnn_permute_226,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_226, False)
    ttnn.deallocate(ttnn_permute_225, False)
    ttnn_divide_12 = ttnn.divide(
        ttnn_add_85,
        utils_constEvalFuncWrapperZeroArg_3_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_85, False)
    ttnn_reshape_285 = ttnn.reshape(
        ttnn_divide_12,
        [92160, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_12, False)
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_reshape_285,
        utils_constEvalFuncWrapperZeroArg_2_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_285, False)
    ttnn_reshape_286 = ttnn.reshape(
        ttnn_matmul_8,
        [1, 256, 360, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn_permute_227 = ttnn.permute(
        ttnn_reshape_286,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_286, False)
    ttnn_reshape_287 = ttnn.reshape(
        ttnn_permute_227,
        [327680, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_227, False)
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_287,
        utils_constEvalFuncWrapperZeroArg_6_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_287, False)
    ttnn_reshape_288 = ttnn.reshape(
        ttnn_matmul_9,
        [1, 256, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn_permute_228 = ttnn.permute(
        ttnn_reshape_288,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_288, False)
    ttnn_reshape_289 = ttnn.reshape(
        ttnn_permute_228,
        [1, 1, 921600, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_228, False)
    ttnn_conv2d_26 = ttnn.conv2d(
        input_tensor=ttnn_reshape_289,
        weight_tensor=utils_constEvalFuncWrapper_12_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_47_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_289, False)
    ttnn_conv2d_27 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_26,
        weight_tensor=utils_constEvalFuncWrapper_6_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_38_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_118 = ttnn.typecast(
        ttnn_conv2d_26,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_26, False)
    ttnn_reshape_290 = ttnn.reshape(
        ttnn_typecast_118,
        [1, 1280, 720, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_118, False)
    ttnn_permute_229 = ttnn.permute(
        ttnn_reshape_290,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_290, False)
    ttnn_reshape_291 = ttnn.reshape(
        ttnn_permute_229,
        [1, 32, 8, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_229, False)
    ttnn_mean_46 = ttnn.mean(
        ttnn_reshape_291,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_23 = ttnn.neg(
        ttnn_mean_46,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_46, False)
    ttnn_add_86 = ttnn.add(
        ttnn_reshape_291,
        ttnn_neg_23,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_23, False)
    ttnn.deallocate(ttnn_reshape_291, False)
    ttnn_multiply_69 = ttnn.multiply(
        ttnn_add_86,
        ttnn_add_86,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_47 = ttnn.mean(
        ttnn_multiply_69,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_69, False)
    ttnn_add_87 = ttnn.add(
        ttnn_mean_47,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_47, False)
    ttnn_rsqrt_23 = ttnn.rsqrt(
        ttnn_add_87,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_87, False)
    ttnn_multiply_70 = ttnn.multiply(
        ttnn_add_86,
        ttnn_rsqrt_23,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_23, False)
    ttnn.deallocate(ttnn_add_86, False)
    ttnn_multiply_71 = ttnn.multiply(
        ttnn_multiply_70,
        utils_constEvalFuncWrapper_126_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_70, False)
    ttnn_add_88 = ttnn.add(
        ttnn_multiply_71,
        utils_constEvalFuncWrapper_118_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_71, False)
    ttnn_silu_22 = ttnn.silu(
        ttnn_add_88,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_88, False)
    ttnn_typecast_119 = ttnn.typecast(
        ttnn_silu_22,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_22, False)
    ttnn_reshape_292 = ttnn.reshape(
        ttnn_typecast_119,
        [1, 256, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_119, False)
    ttnn_permute_230 = ttnn.permute(
        ttnn_reshape_292,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_292, False)
    ttnn_reshape_293 = ttnn.reshape(
        ttnn_permute_230,
        [1, 1, 921600, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_230, False)
    ttnn_conv2d_28 = ttnn.conv2d(
        input_tensor=ttnn_reshape_293,
        weight_tensor=utils_constEvalFuncWrapper_62_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_64_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_293, False)
    ttnn_typecast_120 = ttnn.typecast(
        ttnn_conv2d_28,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_28, False)
    ttnn_reshape_294 = ttnn.reshape(
        ttnn_typecast_120,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_120, False)
    ttnn_permute_231 = ttnn.permute(
        ttnn_reshape_294,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_294, False)
    ttnn_reshape_295 = ttnn.reshape(
        ttnn_permute_231,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_231, False)
    ttnn_mean_48 = ttnn.mean(
        ttnn_reshape_295,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_24 = ttnn.neg(
        ttnn_mean_48,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_48, False)
    ttnn_add_89 = ttnn.add(
        ttnn_reshape_295,
        ttnn_neg_24,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_24, False)
    ttnn.deallocate(ttnn_reshape_295, False)
    ttnn_multiply_72 = ttnn.multiply(
        ttnn_add_89,
        ttnn_add_89,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_49 = ttnn.mean(
        ttnn_multiply_72,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_72, False)
    ttnn_add_90 = ttnn.add(
        ttnn_mean_49,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_49, False)
    ttnn_rsqrt_24 = ttnn.rsqrt(
        ttnn_add_90,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_90, False)
    ttnn_multiply_73 = ttnn.multiply(
        ttnn_add_89,
        ttnn_rsqrt_24,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_24, False)
    ttnn.deallocate(ttnn_add_89, False)
    ttnn_multiply_74 = ttnn.multiply(
        ttnn_multiply_73,
        utils_constEvalFuncWrapper_26_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_73, False)
    ttnn_add_91 = ttnn.add(
        ttnn_multiply_74,
        utils_constEvalFuncWrapper_83_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_74, False)
    ttnn_silu_23 = ttnn.silu(
        ttnn_add_91,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_91, False)
    ttnn_typecast_121 = ttnn.typecast(
        ttnn_silu_23,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_23, False)
    ttnn_reshape_296 = ttnn.reshape(
        ttnn_typecast_121,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_121, False)
    ttnn_permute_232 = ttnn.permute(
        ttnn_reshape_296,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_296, False)
    ttnn_reshape_297 = ttnn.reshape(
        ttnn_permute_232,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_232, False)
    ttnn_conv2d_29 = ttnn.conv2d(
        input_tensor=ttnn_reshape_297,
        weight_tensor=utils_constEvalFuncWrapper_109_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_87_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_297, False)
    ttnn_add_92 = ttnn.add(
        ttnn_conv2d_27,
        ttnn_conv2d_29,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_29, False)
    ttnn.deallocate(ttnn_conv2d_27, False)
    ttnn_reshape_298 = ttnn.reshape(
        ttnn_add_92,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_92, False)
    ttnn_permute_233 = ttnn.permute(
        ttnn_reshape_298,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_298, False)
    ttnn_divide_13 = ttnn.divide(
        ttnn_permute_233,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_233, False)
    ttnn_typecast_122 = ttnn.typecast(
        ttnn_divide_13,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_299 = ttnn.reshape(
        ttnn_typecast_122,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_122, False)
    ttnn_mean_50 = ttnn.mean(
        ttnn_reshape_299,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_25 = ttnn.neg(
        ttnn_mean_50,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_50, False)
    ttnn_add_93 = ttnn.add(
        ttnn_reshape_299,
        ttnn_neg_25,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_25, False)
    ttnn.deallocate(ttnn_reshape_299, False)
    ttnn_multiply_75 = ttnn.multiply(
        ttnn_add_93,
        ttnn_add_93,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_51 = ttnn.mean(
        ttnn_multiply_75,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_75, False)
    ttnn_add_94 = ttnn.add(
        ttnn_mean_51,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_51, False)
    ttnn_rsqrt_25 = ttnn.rsqrt(
        ttnn_add_94,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_94, False)
    ttnn_multiply_76 = ttnn.multiply(
        ttnn_add_93,
        ttnn_rsqrt_25,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_25, False)
    ttnn.deallocate(ttnn_add_93, False)
    ttnn_multiply_77 = ttnn.multiply(
        ttnn_multiply_76,
        utils_constEvalFuncWrapper_124_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_76, False)
    ttnn_add_95 = ttnn.add(
        ttnn_multiply_77,
        utils_constEvalFuncWrapper_30_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_77, False)
    ttnn_silu_24 = ttnn.silu(
        ttnn_add_95,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_95, False)
    ttnn_typecast_123 = ttnn.typecast(
        ttnn_silu_24,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_24, False)
    ttnn_reshape_300 = ttnn.reshape(
        ttnn_typecast_123,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_123, False)
    ttnn_permute_234 = ttnn.permute(
        ttnn_reshape_300,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_300, False)
    ttnn_reshape_301 = ttnn.reshape(
        ttnn_permute_234,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_234, False)
    ttnn_conv2d_30 = ttnn.conv2d(
        input_tensor=ttnn_reshape_301,
        weight_tensor=utils_constEvalFuncWrapper_75_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_101_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_301, False)
    ttnn_typecast_124 = ttnn.typecast(
        ttnn_conv2d_30,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_30, False)
    ttnn_reshape_302 = ttnn.reshape(
        ttnn_typecast_124,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_124, False)
    ttnn_permute_235 = ttnn.permute(
        ttnn_reshape_302,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_302, False)
    ttnn_reshape_303 = ttnn.reshape(
        ttnn_permute_235,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_235, False)
    ttnn_mean_52 = ttnn.mean(
        ttnn_reshape_303,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_26 = ttnn.neg(
        ttnn_mean_52,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_52, False)
    ttnn_add_96 = ttnn.add(
        ttnn_reshape_303,
        ttnn_neg_26,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_26, False)
    ttnn.deallocate(ttnn_reshape_303, False)
    ttnn_multiply_78 = ttnn.multiply(
        ttnn_add_96,
        ttnn_add_96,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_53 = ttnn.mean(
        ttnn_multiply_78,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_78, False)
    ttnn_add_97 = ttnn.add(
        ttnn_mean_53,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_53, False)
    ttnn_rsqrt_26 = ttnn.rsqrt(
        ttnn_add_97,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_97, False)
    ttnn_multiply_79 = ttnn.multiply(
        ttnn_add_96,
        ttnn_rsqrt_26,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_26, False)
    ttnn.deallocate(ttnn_add_96, False)
    ttnn_multiply_80 = ttnn.multiply(
        ttnn_multiply_79,
        utils_constEvalFuncWrapper_104_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_79, False)
    ttnn_add_98 = ttnn.add(
        ttnn_multiply_80,
        utils_constEvalFuncWrapper_36_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_80, False)
    ttnn_silu_25 = ttnn.silu(
        ttnn_add_98,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_98, False)
    ttnn_typecast_125 = ttnn.typecast(
        ttnn_silu_25,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_25, False)
    ttnn_reshape_304 = ttnn.reshape(
        ttnn_typecast_125,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_125, False)
    ttnn_permute_236 = ttnn.permute(
        ttnn_reshape_304,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_304, False)
    ttnn_reshape_305 = ttnn.reshape(
        ttnn_permute_236,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_236, False)
    ttnn_conv2d_31 = ttnn.conv2d(
        input_tensor=ttnn_reshape_305,
        weight_tensor=utils_constEvalFuncWrapper_90_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_19_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_305, False)
    ttnn_reshape_306 = ttnn.reshape(
        ttnn_conv2d_31,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_31, False)
    ttnn_permute_237 = ttnn.permute(
        ttnn_reshape_306,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_306, False)
    ttnn_add_99 = ttnn.add(
        ttnn_divide_13,
        ttnn_permute_237,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_237, False)
    ttnn.deallocate(ttnn_divide_13, False)
    ttnn_divide_14 = ttnn.divide(
        ttnn_add_99,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_99, False)
    ttnn_typecast_126 = ttnn.typecast(
        ttnn_divide_14,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_307 = ttnn.reshape(
        ttnn_typecast_126,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_126, False)
    ttnn_mean_54 = ttnn.mean(
        ttnn_reshape_307,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_27 = ttnn.neg(
        ttnn_mean_54,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_54, False)
    ttnn_add_100 = ttnn.add(
        ttnn_reshape_307,
        ttnn_neg_27,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_27, False)
    ttnn.deallocate(ttnn_reshape_307, False)
    ttnn_multiply_81 = ttnn.multiply(
        ttnn_add_100,
        ttnn_add_100,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_55 = ttnn.mean(
        ttnn_multiply_81,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_81, False)
    ttnn_add_101 = ttnn.add(
        ttnn_mean_55,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_55, False)
    ttnn_rsqrt_27 = ttnn.rsqrt(
        ttnn_add_101,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_101, False)
    ttnn_multiply_82 = ttnn.multiply(
        ttnn_add_100,
        ttnn_rsqrt_27,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_27, False)
    ttnn.deallocate(ttnn_add_100, False)
    ttnn_multiply_83 = ttnn.multiply(
        ttnn_multiply_82,
        utils_constEvalFuncWrapper_3_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_82, False)
    ttnn_add_102 = ttnn.add(
        ttnn_multiply_83,
        utils_constEvalFuncWrapper_56_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_83, False)
    ttnn_silu_26 = ttnn.silu(
        ttnn_add_102,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_102, False)
    ttnn_typecast_127 = ttnn.typecast(
        ttnn_silu_26,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_26, False)
    ttnn_reshape_308 = ttnn.reshape(
        ttnn_typecast_127,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_127, False)
    ttnn_permute_238 = ttnn.permute(
        ttnn_reshape_308,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_308, False)
    ttnn_reshape_309 = ttnn.reshape(
        ttnn_permute_238,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_238, False)
    ttnn_conv2d_32 = ttnn.conv2d(
        input_tensor=ttnn_reshape_309,
        weight_tensor=utils_constEvalFuncWrapper_111_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_103_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_309, False)
    ttnn_typecast_128 = ttnn.typecast(
        ttnn_conv2d_32,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_32, False)
    ttnn_reshape_310 = ttnn.reshape(
        ttnn_typecast_128,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_128, False)
    ttnn_permute_239 = ttnn.permute(
        ttnn_reshape_310,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_310, False)
    ttnn_reshape_311 = ttnn.reshape(
        ttnn_permute_239,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_239, False)
    ttnn_mean_56 = ttnn.mean(
        ttnn_reshape_311,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_28 = ttnn.neg(
        ttnn_mean_56,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_56, False)
    ttnn_add_103 = ttnn.add(
        ttnn_reshape_311,
        ttnn_neg_28,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_28, False)
    ttnn.deallocate(ttnn_reshape_311, False)
    ttnn_multiply_84 = ttnn.multiply(
        ttnn_add_103,
        ttnn_add_103,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_57 = ttnn.mean(
        ttnn_multiply_84,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_84, False)
    ttnn_add_104 = ttnn.add(
        ttnn_mean_57,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_57, False)
    ttnn_rsqrt_28 = ttnn.rsqrt(
        ttnn_add_104,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_104, False)
    ttnn_multiply_85 = ttnn.multiply(
        ttnn_add_103,
        ttnn_rsqrt_28,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_28, False)
    ttnn.deallocate(ttnn_add_103, False)
    ttnn_multiply_86 = ttnn.multiply(
        ttnn_multiply_85,
        utils_constEvalFuncWrapper_84_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_85, False)
    ttnn_add_105 = ttnn.add(
        ttnn_multiply_86,
        utils_constEvalFuncWrapper_113_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_86, False)
    ttnn_silu_27 = ttnn.silu(
        ttnn_add_105,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_105, False)
    ttnn_typecast_129 = ttnn.typecast(
        ttnn_silu_27,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_27, False)
    ttnn_reshape_312 = ttnn.reshape(
        ttnn_typecast_129,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_129, False)
    ttnn_permute_240 = ttnn.permute(
        ttnn_reshape_312,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_312, False)
    ttnn_reshape_313 = ttnn.reshape(
        ttnn_permute_240,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_240, False)
    ttnn_conv2d_33 = ttnn.conv2d(
        input_tensor=ttnn_reshape_313,
        weight_tensor=utils_constEvalFuncWrapper_40_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_122_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_313, False)
    ttnn_reshape_314 = ttnn.reshape(
        ttnn_conv2d_33,
        [1, 1280, 720, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_33, False)
    ttnn_permute_241 = ttnn.permute(
        ttnn_reshape_314,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_314, False)
    ttnn_add_106 = ttnn.add(
        ttnn_divide_14,
        ttnn_permute_241,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_241, False)
    ttnn.deallocate(ttnn_divide_14, False)
    ttnn_divide_15 = ttnn.divide(
        ttnn_add_106,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_106, False)
    ttnn_typecast_130 = ttnn.typecast(
        ttnn_divide_15,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_15, False)
    ttnn_reshape_315 = ttnn.reshape(
        ttnn_typecast_130,
        [1, 32, 4, 921600],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_130, False)
    ttnn_mean_58 = ttnn.mean(
        ttnn_reshape_315,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_neg_29 = ttnn.neg(
        ttnn_mean_58,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_58, False)
    ttnn_add_107 = ttnn.add(
        ttnn_reshape_315,
        ttnn_neg_29,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_neg_29, False)
    ttnn.deallocate(ttnn_reshape_315, False)
    ttnn_multiply_87 = ttnn.multiply(
        ttnn_add_107,
        ttnn_add_107,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mean_59 = ttnn.mean(
        ttnn_multiply_87,
        [2, 3],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_87, False)
    ttnn_add_108 = ttnn.add(
        ttnn_mean_59,
        utils_constEvalFuncWrapperZeroArg_9_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_59, False)
    ttnn_rsqrt_29 = ttnn.rsqrt(
        ttnn_add_108,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_108, False)
    ttnn_multiply_88 = ttnn.multiply(
        ttnn_add_107,
        ttnn_rsqrt_29,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rsqrt_29, False)
    ttnn.deallocate(ttnn_add_107, False)
    ttnn_multiply_89 = ttnn.multiply(
        ttnn_multiply_88,
        utils_constEvalFuncWrapper_96_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_88, False)
    ttnn_add_109 = ttnn.add(
        ttnn_multiply_89,
        utils_constEvalFuncWrapper_54_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_89, False)
    ttnn_silu_28 = ttnn.silu(
        ttnn_add_109,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_109, False)
    ttnn_typecast_131 = ttnn.typecast(
        ttnn_silu_28,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_silu_28, False)
    ttnn_reshape_316 = ttnn.reshape(
        ttnn_typecast_131,
        [1, 128, 1280, 720],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_131, False)
    ttnn_permute_242 = ttnn.permute(
        ttnn_reshape_316,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_316, False)
    ttnn_reshape_317 = ttnn.reshape(
        ttnn_permute_242,
        [1, 1, 921600, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_242, False)
    ttnn_conv2d_34 = ttnn.conv2d(
        input_tensor=ttnn_reshape_317,
        weight_tensor=utils_constEvalFuncWrapper_130_0,
        device=utils_DeviceGetter_get_device_147,
        in_channels=128,
        out_channels=3,
        batch_size=1,
        input_height=1280,
        input_width=720,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_127_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=1024,
            enable_kernel_stride_folding=False,
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
        ),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_317, False)
    ttnn_reshape_318 = ttnn.reshape(
        ttnn_conv2d_34,
        [1, 1280, 720, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_34, False)
    ttnn_permute_243 = ttnn.permute(
        ttnn_reshape_318,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_318, False)
    util_create_list_281 = [ttnn_permute_243]
    return util_create_list_281


def load_inputs_for__main():
    utils_DeviceGetter_get_device_148 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_148,
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
        utils_DeviceGetter_get_device_148,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_148,
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
        utils_DeviceGetter_get_device_148,
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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    util_create_list_282 = [
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
    ]
    return util_create_list_282


def main():
    import os
    from pathlib import Path
    import torch

    TENSOR_DUMP_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / "tensor_dump_vae"
    if not TENSOR_DUMP_DIR.exists():
        TENSOR_DUMP_DIR.mkdir(parents=True, exist_ok=True)

    load_inputs_for__main_0 = load_inputs_for__main()

    # # Save inputs
    # ins = [load_inputs_for__main_0[210], load_inputs_for__main_0[212], load_inputs_for__main_0[213]]
    # [ttnn.dump_tensor(f"{TENSOR_DUMP_DIR}/in{i}.tensorbin", inp) for i, inp in enumerate(ins)]
    # ins_pt = [inp.to_torch() for inp in ins]
    # [torch.save(inp, f"{TENSOR_DUMP_DIR}/in{i}.pt") for i, inp in enumerate(ins_pt)]

    outs = _main(load_inputs_for__main_0)

    # Save outputs
    [ttnn.dump_tensor(f"{TENSOR_DUMP_DIR}/out{i}.tensorbin", out) for i, out in enumerate(outs)]
    outs_pt = [out.to_torch() for out in outs]
    [torch.save(out, f"{TENSOR_DUMP_DIR}/out{i}.pt") for i, out in enumerate(outs_pt)]

    return 0


if __name__ == "__main__":
    main()
