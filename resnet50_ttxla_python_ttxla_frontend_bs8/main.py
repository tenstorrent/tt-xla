import ttnn
import utils


def main_const_eval_0():
    v1 = utils.DeviceGetter.get_device((1, 1))
    v2 = ttnn.full(
        shape=ttnn.Shape([8, 2048]),
        fill_value=0.0203857421875,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v3 = [v2]
    return v3


def main_const_eval_1(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_2(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_3(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_4(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_5(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_6(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_7(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_8(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_9(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_10(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_11(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_12(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_13(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_14(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_15(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_16(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_17(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_18(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_19(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_20(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_21(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_22(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_23(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.permute(
        v15,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(v15, False)
    v17 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v18 = ttnn.to_layout(
        v17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    v19 = ttnn.permute(
        v18,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(v18, False)
    v20 = ttnn.multiply(
        v19,
        v16,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v19, False)
    ttnn.deallocate(v16, False)
    v21 = ttnn.permute(
        v20,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(v20, False)
    v22 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v24 = ttnn.multiply(
        v22,
        v23,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    ttnn.deallocate(v22, False)
    v25 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v26 = ttnn.subtract(
        v25,
        v24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v25, False)
    ttnn.deallocate(v24, False)
    v27 = ttnn.to_layout(
        v21,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v21, False)
    v28 = ttnn.from_device(v27)
    ttnn.deallocate(v27, False)
    v29 = ttnn.to_layout(
        v26,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v26, False)
    v30 = ttnn.from_device(v29)
    ttnn.deallocate(v29, False)
    v31 = ttnn.prepare_conv_weights(
        weight_tensor=v28,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=2048,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v28, False)
    v32 = ttnn.prepare_conv_bias(
        bias_tensor=v30,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=2048,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v30, False)
    v33 = [v31, v32]
    return v33


def main_const_eval_24(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_25(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [512, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [512, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_26(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_27(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_28(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_29(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_30(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=64,
        batch_size=8,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=3,
        out_channels=64,
        batch_size=8,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_31(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_32(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_33(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_34(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_35(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_36(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_37(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_38(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_39(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [3136, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=512,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [3136, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=512,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_40(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_41(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_42(v1):
    v2 = v1[0]
    v3 = ttnn.reshape(
        v2,
        [1, 1000],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v4 = ttnn.repeat(v3, ttnn.Shape([8, 1]))
    ttnn.deallocate(v3, False)
    v5 = [v4]
    return v5


def main_const_eval_43(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_44(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_45(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_46(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_47(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_48(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_49(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_50(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_51(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_52(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_53(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=1024,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=1024,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_54(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = utils.DeviceGetter.get_device((1, 1))
    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = ttnn.to_device(
        v6,
        device=v7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def _main(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = v1[5]
    v8 = v1[6]
    v9 = v1[7]
    v10 = v1[8]
    v11 = v1[9]
    v12 = v1[10]
    v13 = v1[11]
    v14 = v1[12]
    v15 = v1[13]
    v16 = v1[14]
    v17 = v1[15]
    v18 = v1[16]
    v19 = v1[17]
    v20 = v1[18]
    v21 = v1[19]
    v22 = v1[20]
    v23 = v1[21]
    v24 = v1[22]
    v25 = v1[23]
    v26 = v1[24]
    v27 = v1[25]
    v28 = v1[26]
    v29 = v1[27]
    v30 = v1[28]
    v31 = v1[29]
    v32 = v1[30]
    v33 = v1[31]
    v34 = v1[32]
    v35 = v1[33]
    v36 = v1[34]
    v37 = v1[35]
    v38 = v1[36]
    v39 = v1[37]
    v40 = v1[38]
    v41 = v1[39]
    v42 = v1[40]
    v43 = v1[41]
    v44 = v1[42]
    v45 = v1[43]
    v46 = v1[44]
    v47 = v1[45]
    v48 = v1[46]
    v49 = v1[47]
    v50 = v1[48]
    v51 = v1[49]
    v52 = v1[50]
    v53 = v1[51]
    v54 = v1[52]
    v55 = v1[53]
    v56 = v1[54]
    v57 = v1[55]
    v58 = v1[56]
    v59 = v1[57]
    v60 = v1[58]
    v61 = v1[59]
    v62 = v1[60]
    v63 = v1[61]
    v64 = v1[62]
    v65 = v1[63]
    v66 = v1[64]
    v67 = v1[65]
    v68 = v1[66]
    v69 = v1[67]
    v70 = v1[68]
    v71 = v1[69]
    v72 = v1[70]
    v73 = v1[71]
    v74 = v1[72]
    v75 = v1[73]
    v76 = v1[74]
    v77 = v1[75]
    v78 = v1[76]
    v79 = v1[77]
    v80 = v1[78]
    v81 = v1[79]
    v82 = v1[80]
    v83 = v1[81]
    v84 = v1[82]
    v85 = v1[83]
    v86 = v1[84]
    v87 = v1[85]
    v88 = v1[86]
    v89 = v1[87]
    v90 = v1[88]
    v91 = v1[89]
    v92 = v1[90]
    v93 = v1[91]
    v94 = v1[92]
    v95 = v1[93]
    v96 = v1[94]
    v97 = v1[95]
    v98 = v1[96]
    v99 = v1[97]
    v100 = v1[98]
    v101 = v1[99]
    v102 = v1[100]
    v103 = v1[101]
    v104 = v1[102]
    v105 = v1[103]
    v106 = v1[104]
    v107 = v1[105]
    v108 = v1[106]
    v109 = v1[107]
    v110 = v1[108]
    v111 = v1[109]
    v112 = v1[110]
    v113 = v1[111]
    v114 = v1[112]
    v115 = v1[113]
    v116 = v1[114]
    v117 = v1[115]
    v118 = v1[116]
    v119 = v1[117]
    v120 = v1[118]
    v121 = v1[119]
    v122 = v1[120]
    v123 = v1[121]
    v124 = v1[122]
    v125 = v1[123]
    v126 = v1[124]
    v127 = v1[125]
    v128 = v1[126]
    v129 = v1[127]
    v130 = v1[128]
    v131 = v1[129]
    v132 = v1[130]
    v133 = v1[131]
    v134 = v1[132]
    v135 = v1[133]
    v136 = v1[134]
    v137 = v1[135]
    v138 = v1[136]
    v139 = v1[137]
    v140 = v1[138]
    v141 = v1[139]
    v142 = v1[140]
    v143 = v1[141]
    v144 = v1[142]
    v145 = v1[143]
    v146 = v1[144]
    v147 = v1[145]
    v148 = v1[146]
    v149 = v1[147]
    v150 = v1[148]
    v151 = v1[149]
    v152 = v1[150]
    v153 = v1[151]
    v154 = v1[152]
    v155 = v1[153]
    v156 = v1[154]
    v157 = v1[155]
    v158 = v1[156]
    v159 = v1[157]
    v160 = v1[158]
    v161 = v1[159]
    v162 = v1[160]
    v163 = v1[161]
    v164 = v1[162]
    v165 = v1[163]
    v166 = v1[164]
    v167 = v1[165]
    v168 = v1[166]
    v169 = v1[167]
    v170 = v1[168]
    v171 = v1[169]
    v172 = v1[170]
    v173 = v1[171]
    v174 = v1[172]
    v175 = v1[173]
    v176 = v1[174]
    v177 = v1[175]
    v178 = v1[176]
    v179 = v1[177]
    v180 = v1[178]
    v181 = v1[179]
    v182 = v1[180]
    v183 = v1[181]
    v184 = v1[182]
    v185 = v1[183]
    v186 = v1[184]
    v187 = v1[185]
    v188 = v1[186]
    v189 = v1[187]
    v190 = v1[188]
    v191 = v1[189]
    v192 = v1[190]
    v193 = v1[191]
    v194 = v1[192]
    v195 = v1[193]
    v196 = v1[194]
    v197 = v1[195]
    v198 = v1[196]
    v199 = v1[197]
    v200 = v1[198]
    v201 = v1[199]
    v202 = v1[200]
    v203 = v1[201]
    v204 = v1[202]
    v205 = v1[203]
    v206 = v1[204]
    v207 = v1[205]
    v208 = v1[206]
    v209 = v1[207]
    v210 = v1[208]
    v211 = v1[209]
    v212 = v1[210]
    v213 = v1[211]
    v214 = v1[212]
    v215 = v1[213]
    v216 = v1[214]
    v217 = v1[215]
    v218 = v1[216]
    v219 = v1[217]
    v220 = v1[218]
    v221 = v1[219]
    v222 = v1[220]
    v223 = v1[221]
    v224 = v1[222]
    v225 = v1[223]
    v226 = v1[224]
    v227 = v1[225]
    v228 = v1[226]
    v229 = v1[227]
    v230 = v1[228]
    v231 = v1[229]
    v232 = v1[230]
    v233 = v1[231]
    v234 = v1[232]
    v235 = v1[233]
    v236 = v1[234]
    v237 = v1[235]
    v238 = v1[236]
    v239 = v1[237]
    v240 = v1[238]
    v241 = v1[239]
    v242 = v1[240]
    v243 = v1[241]
    v244 = v1[242]
    v245 = v1[243]
    v246 = v1[244]
    v247 = v1[245]
    v248 = v1[246]
    v249 = v1[247]
    v250 = v1[248]
    v251 = v1[249]
    v252 = v1[250]
    v253 = v1[251]
    v254 = v1[252]
    v255 = v1[253]
    v256 = v1[254]
    v257 = v1[255]
    v258 = v1[256]
    v259 = v1[257]
    v260 = v1[258]
    v261 = v1[259]
    v262 = v1[260]
    v263 = v1[261]
    v264 = v1[262]
    v265 = v1[263]
    v266 = v1[264]
    v267 = v1[265]
    v268 = v1[266]
    v269 = v1[267]
    v270 = main_const_eval_0()
    v271 = v270[0]
    v272 = [v235, v236, v237, v238, v239]
    v273 = main_const_eval_1(v272)
    v274 = v273[0]
    v275 = v273[1]
    v276 = [v50, v51, v52, v53, v54]
    v277 = main_const_eval_2(v276)
    v278 = v277[0]
    v279 = v277[1]
    v280 = [v105, v106, v107, v108, v109]
    v281 = main_const_eval_3(v280)
    v282 = v281[0]
    v283 = v281[1]
    v284 = [v150, v151, v152, v153, v154]
    v285 = main_const_eval_4(v284)
    v286 = v285[0]
    v287 = v285[1]
    v288 = [v55, v56, v57, v58, v59]
    v289 = main_const_eval_5(v288)
    v290 = v289[0]
    v291 = v289[1]
    v292 = [v250, v251, v252, v253, v254]
    v293 = main_const_eval_6(v292)
    v294 = v293[0]
    v295 = v293[1]
    v296 = [v140, v141, v142, v143, v144]
    v297 = main_const_eval_7(v296)
    v298 = v297[0]
    v299 = v297[1]
    v300 = [v125, v126, v127, v128, v129]
    v301 = main_const_eval_8(v300)
    v302 = v301[0]
    v303 = v301[1]
    v304 = [v45, v46, v47, v48, v49]
    v305 = main_const_eval_9(v304)
    v306 = v305[0]
    v307 = v305[1]
    v308 = [v215, v216, v217, v218, v219]
    v309 = main_const_eval_10(v308)
    v310 = v309[0]
    v311 = v309[1]
    v312 = [v200, v201, v202, v203, v204]
    v313 = main_const_eval_11(v312)
    v314 = v313[0]
    v315 = v313[1]
    v316 = [v115, v116, v117, v118, v119]
    v317 = main_const_eval_12(v316)
    v318 = v317[0]
    v319 = v317[1]
    v320 = [v85, v86, v87, v88, v89]
    v321 = main_const_eval_13(v320)
    v322 = v321[0]
    v323 = v321[1]
    v324 = [v65, v66, v67, v68, v69]
    v325 = main_const_eval_14(v324)
    v326 = v325[0]
    v327 = v325[1]
    v328 = [v75, v76, v77, v78, v79]
    v329 = main_const_eval_15(v328)
    v330 = v329[0]
    v331 = v329[1]
    v332 = [v40, v41, v42, v43, v44]
    v333 = main_const_eval_16(v332)
    v334 = v333[0]
    v335 = v333[1]
    v336 = [v19, v20, v21, v22, v23]
    v337 = main_const_eval_17(v336)
    v338 = v337[0]
    v339 = v337[1]
    v340 = [v265, v266, v267, v268, v269]
    v341 = main_const_eval_18(v340)
    v342 = v341[0]
    v343 = v341[1]
    v344 = [v160, v161, v162, v163, v164]
    v345 = main_const_eval_19(v344)
    v346 = v345[0]
    v347 = v345[1]
    v348 = [v100, v101, v102, v103, v104]
    v349 = main_const_eval_20(v348)
    v350 = v349[0]
    v351 = v349[1]
    v352 = [v120, v121, v122, v123, v124]
    v353 = main_const_eval_21(v352)
    v354 = v353[0]
    v355 = v353[1]
    v356 = [v70, v71, v72, v73, v74]
    v357 = main_const_eval_22(v356)
    v358 = v357[0]
    v359 = v357[1]
    v360 = [v4, v5, v6, v7, v8]
    v361 = main_const_eval_23(v360)
    v362 = v361[0]
    v363 = v361[1]
    v364 = [v175, v176, v177, v178, v179]
    v365 = main_const_eval_24(v364)
    v366 = v365[0]
    v367 = v365[1]
    v368 = [v80, v81, v82, v83, v84]
    v369 = main_const_eval_25(v368)
    v370 = v369[0]
    v371 = v369[1]
    v372 = [v255, v256, v257, v258, v259]
    v373 = main_const_eval_26(v372)
    v374 = v373[0]
    v375 = v373[1]
    v376 = [v185, v186, v187, v188, v189]
    v377 = main_const_eval_27(v376)
    v378 = v377[0]
    v379 = v377[1]
    v380 = [v210, v211, v212, v213, v214]
    v381 = main_const_eval_28(v380)
    v382 = v381[0]
    v383 = v381[1]
    v384 = [v60, v61, v62, v63, v64]
    v385 = main_const_eval_29(v384)
    v386 = v385[0]
    v387 = v385[1]
    v388 = [v24, v25, v26, v27, v28]
    v389 = main_const_eval_30(v388)
    v390 = v389[0]
    v391 = v389[1]
    v392 = [v225, v226, v227, v228, v229]
    v393 = main_const_eval_31(v392)
    v394 = v393[0]
    v395 = v393[1]
    v396 = [v35, v36, v37, v38, v39]
    v397 = main_const_eval_32(v396)
    v398 = v397[0]
    v399 = v397[1]
    v400 = [v170, v171, v172, v173, v174]
    v401 = main_const_eval_33(v400)
    v402 = v401[0]
    v403 = v401[1]
    v404 = [v180, v181, v182, v183, v184]
    v405 = main_const_eval_34(v404)
    v406 = v405[0]
    v407 = v405[1]
    v408 = [v205, v206, v207, v208, v209]
    v409 = main_const_eval_35(v408)
    v410 = v409[0]
    v411 = v409[1]
    v412 = [v95, v96, v97, v98, v99]
    v413 = main_const_eval_36(v412)
    v414 = v413[0]
    v415 = v413[1]
    v416 = [v190, v191, v192, v193, v194]
    v417 = main_const_eval_37(v416)
    v418 = v417[0]
    v419 = v417[1]
    v420 = [v145, v146, v147, v148, v149]
    v421 = main_const_eval_38(v420)
    v422 = v421[0]
    v423 = v421[1]
    v424 = [v14, v15, v16, v17, v18]
    v425 = main_const_eval_39(v424)
    v426 = v425[0]
    v427 = v425[1]
    v428 = [v30, v31, v32, v33, v34]
    v429 = main_const_eval_40(v428)
    v430 = v429[0]
    v431 = v429[1]
    v432 = [v195, v196, v197, v198, v199]
    v433 = main_const_eval_41(v432)
    v434 = v433[0]
    v435 = v433[1]
    v436 = [v2]
    v437 = main_const_eval_42(v436)
    v438 = v437[0]
    v439 = [v155, v156, v157, v158, v159]
    v440 = main_const_eval_43(v439)
    v441 = v440[0]
    v442 = v440[1]
    v443 = [v110, v111, v112, v113, v114]
    v444 = main_const_eval_44(v443)
    v445 = v444[0]
    v446 = v444[1]
    v447 = [v220, v221, v222, v223, v224]
    v448 = main_const_eval_45(v447)
    v449 = v448[0]
    v450 = v448[1]
    v451 = [v130, v131, v132, v133, v134]
    v452 = main_const_eval_46(v451)
    v453 = v452[0]
    v454 = v452[1]
    v455 = [v230, v231, v232, v233, v234]
    v456 = main_const_eval_47(v455)
    v457 = v456[0]
    v458 = v456[1]
    v459 = [v135, v136, v137, v138, v139]
    v460 = main_const_eval_48(v459)
    v461 = v460[0]
    v462 = v460[1]
    v463 = [v245, v246, v247, v248, v249]
    v464 = main_const_eval_49(v463)
    v465 = v464[0]
    v466 = v464[1]
    v467 = [v165, v166, v167, v168, v169]
    v468 = main_const_eval_50(v467)
    v469 = v468[0]
    v470 = v468[1]
    v471 = [v90, v91, v92, v93, v94]
    v472 = main_const_eval_51(v471)
    v473 = v472[0]
    v474 = v472[1]
    v475 = [v240, v241, v242, v243, v244]
    v476 = main_const_eval_52(v475)
    v477 = v476[0]
    v478 = v476[1]
    v479 = [v9, v10, v11, v12, v13]
    v480 = main_const_eval_53(v479)
    v481 = v480[0]
    v482 = v480[1]
    v483 = [v260, v261, v262, v263, v264]
    v484 = main_const_eval_54(v483)
    v485 = v484[0]
    v486 = v484[1]
    v487 = utils.DeviceGetter.get_device((1, 1))
    v488 = ttnn.permute(
        v29,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(v29, False)
    v489 = ttnn.reshape(
        v488,
        [1, 1, 401408, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v488, False)
    v490 = ttnn.conv2d(
        input_tensor=v489,
        weight_tensor=v390,
        device=v487,
        in_channels=3,
        out_channels=64,
        batch_size=8,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v391,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v489, False)
    ttnn.deallocate(v391, False)
    ttnn.deallocate(v390, False)
    v491 = ttnn.max_pool2d(
        v490,
        8,
        112,
        112,
        64,
        [3, 3],
        [2, 2],
        [1, 1],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        applied_shard_scheme=None,
        ceil_mode=False,
        in_place_halo=False,
    )
    ttnn.deallocate(v490, False)
    v492 = ttnn.to_layout(
        v491,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v491, False)
    v493 = ttnn.to_memory_config(
        v492,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    v494 = ttnn.conv2d(
        input_tensor=v493,
        weight_tensor=v338,
        device=v487,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v339,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v493, False)
    ttnn.deallocate(v339, False)
    ttnn.deallocate(v338, False)
    v495 = ttnn.to_memory_config(
        v494,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v494, False)
    v496 = ttnn.to_memory_config(
        v492,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v492, False)
    v497 = ttnn.conv2d(
        input_tensor=v496,
        weight_tensor=v334,
        device=v487,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v335,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v496, False)
    ttnn.deallocate(v335, False)
    ttnn.deallocate(v334, False)
    v498 = ttnn.conv2d(
        input_tensor=v497,
        weight_tensor=v398,
        device=v487,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v399,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v497, False)
    ttnn.deallocate(v399, False)
    ttnn.deallocate(v398, False)
    v499 = ttnn.conv2d(
        input_tensor=v498,
        weight_tensor=v430,
        device=v487,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v431,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v498, False)
    ttnn.deallocate(v431, False)
    ttnn.deallocate(v430, False)
    v500 = ttnn.add(
        v499,
        v495,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v499, False)
    ttnn.deallocate(v495, False)
    v501 = ttnn.relu(
        v500,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v500, False)
    v502 = ttnn.to_memory_config(
        v501,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v503 = ttnn.conv2d(
        input_tensor=v501,
        weight_tensor=v290,
        device=v487,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v291,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v501, False)
    ttnn.deallocate(v291, False)
    ttnn.deallocate(v290, False)
    v504 = ttnn.conv2d(
        input_tensor=v503,
        weight_tensor=v278,
        device=v487,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v279,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v503, False)
    ttnn.deallocate(v279, False)
    ttnn.deallocate(v278, False)
    v505 = ttnn.conv2d(
        input_tensor=v504,
        weight_tensor=v306,
        device=v487,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v307,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v504, False)
    ttnn.deallocate(v307, False)
    ttnn.deallocate(v306, False)
    v506 = ttnn.add(
        v505,
        v502,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v505, False)
    ttnn.deallocate(v502, False)
    v507 = ttnn.relu(
        v506,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v506, False)
    v508 = ttnn.to_memory_config(
        v507,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v509 = ttnn.conv2d(
        input_tensor=v507,
        weight_tensor=v358,
        device=v487,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v359,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v507, False)
    ttnn.deallocate(v359, False)
    ttnn.deallocate(v358, False)
    v510 = ttnn.conv2d(
        input_tensor=v509,
        weight_tensor=v326,
        device=v487,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v327,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v509, False)
    ttnn.deallocate(v327, False)
    ttnn.deallocate(v326, False)
    v511 = ttnn.conv2d(
        input_tensor=v510,
        weight_tensor=v386,
        device=v487,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v387,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v510, False)
    ttnn.deallocate(v387, False)
    ttnn.deallocate(v386, False)
    v512 = ttnn.add(
        v511,
        v508,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v511, False)
    ttnn.deallocate(v508, False)
    v513 = ttnn.relu(
        v512,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v512, False)
    v514 = ttnn.to_memory_config(
        v513,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [3136, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    v515 = ttnn.conv2d(
        input_tensor=v514,
        weight_tensor=v426,
        device=v487,
        in_channels=256,
        out_channels=512,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v427,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v514, False)
    ttnn.deallocate(v427, False)
    ttnn.deallocate(v426, False)
    v516 = ttnn.to_memory_config(
        v515,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v515, False)
    v517 = ttnn.conv2d(
        input_tensor=v513,
        weight_tensor=v322,
        device=v487,
        in_channels=256,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v323,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(4, 7)),
                    ]
                ),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v513, False)
    ttnn.deallocate(v323, False)
    ttnn.deallocate(v322, False)
    v518 = ttnn.to_memory_config(
        v517,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [512, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v517, False)
    v519 = ttnn.conv2d(
        input_tensor=v518,
        weight_tensor=v370,
        device=v487,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v371,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v518, False)
    ttnn.deallocate(v371, False)
    ttnn.deallocate(v370, False)
    v520 = ttnn.conv2d(
        input_tensor=v519,
        weight_tensor=v330,
        device=v487,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v331,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v519, False)
    ttnn.deallocate(v331, False)
    ttnn.deallocate(v330, False)
    v521 = ttnn.add(
        v520,
        v516,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v520, False)
    ttnn.deallocate(v516, False)
    v522 = ttnn.relu(
        v521,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v521, False)
    v523 = ttnn.to_memory_config(
        v522,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v524 = ttnn.conv2d(
        input_tensor=v522,
        weight_tensor=v350,
        device=v487,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v351,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v522, False)
    ttnn.deallocate(v351, False)
    ttnn.deallocate(v350, False)
    v525 = ttnn.conv2d(
        input_tensor=v524,
        weight_tensor=v414,
        device=v487,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v415,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v524, False)
    ttnn.deallocate(v415, False)
    ttnn.deallocate(v414, False)
    v526 = ttnn.conv2d(
        input_tensor=v525,
        weight_tensor=v473,
        device=v487,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v474,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v525, False)
    ttnn.deallocate(v474, False)
    ttnn.deallocate(v473, False)
    v527 = ttnn.add(
        v526,
        v523,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v526, False)
    ttnn.deallocate(v523, False)
    v528 = ttnn.relu(
        v527,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v527, False)
    v529 = ttnn.to_memory_config(
        v528,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v530 = ttnn.conv2d(
        input_tensor=v528,
        weight_tensor=v318,
        device=v487,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v319,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v528, False)
    ttnn.deallocate(v319, False)
    ttnn.deallocate(v318, False)
    v531 = ttnn.conv2d(
        input_tensor=v530,
        weight_tensor=v445,
        device=v487,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v446,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v530, False)
    ttnn.deallocate(v446, False)
    ttnn.deallocate(v445, False)
    v532 = ttnn.conv2d(
        input_tensor=v531,
        weight_tensor=v282,
        device=v487,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v283,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v531, False)
    ttnn.deallocate(v283, False)
    ttnn.deallocate(v282, False)
    v533 = ttnn.add(
        v532,
        v529,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v532, False)
    ttnn.deallocate(v529, False)
    v534 = ttnn.relu(
        v533,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v533, False)
    v535 = ttnn.to_memory_config(
        v534,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v536 = ttnn.conv2d(
        input_tensor=v534,
        weight_tensor=v453,
        device=v487,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v454,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v534, False)
    ttnn.deallocate(v454, False)
    ttnn.deallocate(v453, False)
    v537 = ttnn.conv2d(
        input_tensor=v536,
        weight_tensor=v302,
        device=v487,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v303,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v536, False)
    ttnn.deallocate(v303, False)
    ttnn.deallocate(v302, False)
    v538 = ttnn.conv2d(
        input_tensor=v537,
        weight_tensor=v354,
        device=v487,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v355,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v537, False)
    ttnn.deallocate(v355, False)
    ttnn.deallocate(v354, False)
    v539 = ttnn.add(
        v538,
        v535,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v538, False)
    ttnn.deallocate(v535, False)
    v540 = ttnn.relu(
        v539,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(0, 6)),
                    ]
                ),
                [128, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v539, False)
    v541 = ttnn.to_memory_config(
        v540,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    v542 = ttnn.conv2d(
        input_tensor=v541,
        weight_tensor=v481,
        device=v487,
        in_channels=512,
        out_channels=1024,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v482,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v541, False)
    ttnn.deallocate(v482, False)
    ttnn.deallocate(v481, False)
    v543 = ttnn.to_memory_config(
        v542,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v542, False)
    v544 = ttnn.to_memory_config(
        v540,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v540, False)
    v545 = ttnn.conv2d(
        input_tensor=v544,
        weight_tensor=v422,
        device=v487,
        in_channels=512,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v423,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v544, False)
    ttnn.deallocate(v423, False)
    ttnn.deallocate(v422, False)
    v546 = ttnn.to_memory_config(
        v545,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [896, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v545, False)
    v547 = ttnn.conv2d(
        input_tensor=v546,
        weight_tensor=v298,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v299,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v546, False)
    ttnn.deallocate(v299, False)
    ttnn.deallocate(v298, False)
    v548 = ttnn.conv2d(
        input_tensor=v547,
        weight_tensor=v461,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v462,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v547, False)
    ttnn.deallocate(v462, False)
    ttnn.deallocate(v461, False)
    v549 = ttnn.add(
        v548,
        v543,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v548, False)
    ttnn.deallocate(v543, False)
    v550 = ttnn.relu(
        v549,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v549, False)
    v551 = ttnn.to_memory_config(
        v550,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v552 = ttnn.conv2d(
        input_tensor=v550,
        weight_tensor=v346,
        device=v487,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v347,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v550, False)
    ttnn.deallocate(v347, False)
    ttnn.deallocate(v346, False)
    v553 = ttnn.conv2d(
        input_tensor=v552,
        weight_tensor=v441,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v442,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v552, False)
    ttnn.deallocate(v442, False)
    ttnn.deallocate(v441, False)
    v554 = ttnn.conv2d(
        input_tensor=v553,
        weight_tensor=v286,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v287,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v553, False)
    ttnn.deallocate(v287, False)
    ttnn.deallocate(v286, False)
    v555 = ttnn.add(
        v554,
        v551,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v554, False)
    ttnn.deallocate(v551, False)
    v556 = ttnn.relu(
        v555,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v555, False)
    v557 = ttnn.to_memory_config(
        v556,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v558 = ttnn.conv2d(
        input_tensor=v556,
        weight_tensor=v366,
        device=v487,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v367,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v556, False)
    ttnn.deallocate(v367, False)
    ttnn.deallocate(v366, False)
    v559 = ttnn.conv2d(
        input_tensor=v558,
        weight_tensor=v402,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v403,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v558, False)
    ttnn.deallocate(v403, False)
    ttnn.deallocate(v402, False)
    v560 = ttnn.conv2d(
        input_tensor=v559,
        weight_tensor=v469,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v470,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v559, False)
    ttnn.deallocate(v470, False)
    ttnn.deallocate(v469, False)
    v561 = ttnn.add(
        v560,
        v557,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v560, False)
    ttnn.deallocate(v557, False)
    v562 = ttnn.relu(
        v561,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v561, False)
    v563 = ttnn.to_memory_config(
        v562,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v564 = ttnn.conv2d(
        input_tensor=v562,
        weight_tensor=v418,
        device=v487,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v419,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v562, False)
    ttnn.deallocate(v419, False)
    ttnn.deallocate(v418, False)
    v565 = ttnn.conv2d(
        input_tensor=v564,
        weight_tensor=v378,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v379,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v564, False)
    ttnn.deallocate(v379, False)
    ttnn.deallocate(v378, False)
    v566 = ttnn.conv2d(
        input_tensor=v565,
        weight_tensor=v406,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v407,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v565, False)
    ttnn.deallocate(v407, False)
    ttnn.deallocate(v406, False)
    v567 = ttnn.add(
        v566,
        v563,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v566, False)
    ttnn.deallocate(v563, False)
    v568 = ttnn.relu(
        v567,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v567, False)
    v569 = ttnn.to_memory_config(
        v568,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v570 = ttnn.conv2d(
        input_tensor=v568,
        weight_tensor=v410,
        device=v487,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v411,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v568, False)
    ttnn.deallocate(v411, False)
    ttnn.deallocate(v410, False)
    v571 = ttnn.conv2d(
        input_tensor=v570,
        weight_tensor=v314,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v315,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v570, False)
    ttnn.deallocate(v315, False)
    ttnn.deallocate(v314, False)
    v572 = ttnn.conv2d(
        input_tensor=v571,
        weight_tensor=v434,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v435,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v571, False)
    ttnn.deallocate(v435, False)
    ttnn.deallocate(v434, False)
    v573 = ttnn.add(
        v572,
        v569,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v572, False)
    ttnn.deallocate(v569, False)
    v574 = ttnn.relu(
        v573,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v573, False)
    v575 = ttnn.to_memory_config(
        v574,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v576 = ttnn.conv2d(
        input_tensor=v574,
        weight_tensor=v449,
        device=v487,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v450,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v574, False)
    ttnn.deallocate(v450, False)
    ttnn.deallocate(v449, False)
    v577 = ttnn.conv2d(
        input_tensor=v576,
        weight_tensor=v310,
        device=v487,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v311,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v576, False)
    ttnn.deallocate(v311, False)
    ttnn.deallocate(v310, False)
    v578 = ttnn.conv2d(
        input_tensor=v577,
        weight_tensor=v382,
        device=v487,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v383,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v577, False)
    ttnn.deallocate(v383, False)
    ttnn.deallocate(v382, False)
    v579 = ttnn.add(
        v578,
        v575,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v578, False)
    ttnn.deallocate(v575, False)
    v580 = ttnn.relu(
        v579,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v579, False)
    v581 = ttnn.conv2d(
        input_tensor=v580,
        weight_tensor=v362,
        device=v487,
        in_channels=1024,
        out_channels=2048,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v363,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v363, False)
    ttnn.deallocate(v362, False)
    v582 = ttnn.to_memory_config(
        v581,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v581, False)
    v583 = ttnn.conv2d(
        input_tensor=v580,
        weight_tensor=v274,
        device=v487,
        in_channels=1024,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v275,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [224, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v580, False)
    ttnn.deallocate(v275, False)
    ttnn.deallocate(v274, False)
    v584 = ttnn.conv2d(
        input_tensor=v583,
        weight_tensor=v457,
        device=v487,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v458,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v583, False)
    ttnn.deallocate(v458, False)
    ttnn.deallocate(v457, False)
    v585 = ttnn.conv2d(
        input_tensor=v584,
        weight_tensor=v394,
        device=v487,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v395,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v584, False)
    ttnn.deallocate(v395, False)
    ttnn.deallocate(v394, False)
    v586 = ttnn.add(
        v585,
        v582,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v585, False)
    ttnn.deallocate(v582, False)
    v587 = ttnn.relu(
        v586,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v586, False)
    v588 = ttnn.to_memory_config(
        v587,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v589 = ttnn.conv2d(
        input_tensor=v587,
        weight_tensor=v294,
        device=v487,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v295,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v587, False)
    ttnn.deallocate(v295, False)
    ttnn.deallocate(v294, False)
    v590 = ttnn.conv2d(
        input_tensor=v589,
        weight_tensor=v465,
        device=v487,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v466,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v589, False)
    ttnn.deallocate(v466, False)
    ttnn.deallocate(v465, False)
    v591 = ttnn.conv2d(
        input_tensor=v590,
        weight_tensor=v477,
        device=v487,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v478,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v590, False)
    ttnn.deallocate(v478, False)
    ttnn.deallocate(v477, False)
    v592 = ttnn.add(
        v591,
        v588,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v591, False)
    ttnn.deallocate(v588, False)
    v593 = ttnn.relu(
        v592,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v592, False)
    v594 = ttnn.to_memory_config(
        v593,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v595 = ttnn.conv2d(
        input_tensor=v593,
        weight_tensor=v342,
        device=v487,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v343,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v593, False)
    ttnn.deallocate(v343, False)
    ttnn.deallocate(v342, False)
    v596 = ttnn.conv2d(
        input_tensor=v595,
        weight_tensor=v485,
        device=v487,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v486,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v595, False)
    ttnn.deallocate(v486, False)
    ttnn.deallocate(v485, False)
    v597 = ttnn.conv2d(
        input_tensor=v596,
        weight_tensor=v374,
        device=v487,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v375,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v596, False)
    ttnn.deallocate(v375, False)
    ttnn.deallocate(v374, False)
    v598 = ttnn.add(
        v597,
        v594,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v597, False)
    ttnn.deallocate(v594, False)
    v599 = ttnn.relu(
        v598,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v598, False)
    v600 = ttnn.to_memory_config(
        v599,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v599, False)
    v601 = ttnn.reshape(
        v600,
        [8, 7, 7, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v600, False)
    v602 = ttnn.permute(
        v601,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(v601, False)
    v603 = ttnn.sum(
        v602,
        [2, 3],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v602, False)
    v604 = ttnn.multiply(
        v603,
        v271,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v603, False)
    ttnn.deallocate(v271, False)
    v605 = ttnn.to_memory_config(
        v604,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v604, False)
    v606 = ttnn.linear(
        v605,
        v3,
        bias=v438,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v605, False)
    ttnn.deallocate(v438, False)
    v607 = [v606]
    return v607


def load_inputs_for__main():
    v1 = utils.DeviceGetter.get_device((1, 1))
    v2 = utils.load_tensor(
        "./input_tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v3 = utils.load_tensor(
        "./input_tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v4 = utils.load_tensor(
        "./input_tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v5 = utils.load_tensor(
        "./input_tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v6 = utils.load_tensor(
        "./input_tensors/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v7 = utils.load_tensor(
        "./input_tensors/arg5.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v8 = utils.load_tensor(
        "./input_tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v9 = utils.load_tensor(
        "./input_tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v10 = utils.load_tensor(
        "./input_tensors/arg8.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v11 = utils.load_tensor(
        "./input_tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v12 = utils.load_tensor(
        "./input_tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v13 = utils.load_tensor(
        "./input_tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v14 = utils.load_tensor(
        "./input_tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v15 = utils.load_tensor(
        "./input_tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v16 = utils.load_tensor(
        "./input_tensors/arg14.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v17 = utils.load_tensor(
        "./input_tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v18 = utils.load_tensor(
        "./input_tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v19 = utils.load_tensor(
        "./input_tensors/arg17.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v20 = utils.load_tensor(
        "./input_tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v21 = utils.load_tensor(
        "./input_tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v22 = utils.load_tensor(
        "./input_tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v23 = utils.load_tensor(
        "./input_tensors/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v24 = utils.load_tensor(
        "./input_tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v25 = utils.load_tensor(
        "./input_tensors/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v26 = utils.load_tensor(
        "./input_tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v27 = utils.load_tensor(
        "./input_tensors/arg25.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v28 = utils.load_tensor(
        "./input_tensors/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v29 = utils.load_tensor(
        "./input_tensors/arg27.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v30 = utils.load_tensor(
        "./input_tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v31 = utils.load_tensor(
        "./input_tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v32 = utils.load_tensor(
        "./input_tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v33 = utils.load_tensor(
        "./input_tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v34 = utils.load_tensor(
        "./input_tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v35 = utils.load_tensor(
        "./input_tensors/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v36 = utils.load_tensor(
        "./input_tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v37 = utils.load_tensor(
        "./input_tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v38 = utils.load_tensor(
        "./input_tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v39 = utils.load_tensor(
        "./input_tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v40 = utils.load_tensor(
        "./input_tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v41 = utils.load_tensor(
        "./input_tensors/arg39.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v42 = utils.load_tensor(
        "./input_tensors/arg40.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v43 = utils.load_tensor(
        "./input_tensors/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v44 = utils.load_tensor(
        "./input_tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v45 = utils.load_tensor(
        "./input_tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v46 = utils.load_tensor(
        "./input_tensors/arg44.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v47 = utils.load_tensor(
        "./input_tensors/arg45.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v48 = utils.load_tensor(
        "./input_tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v49 = utils.load_tensor(
        "./input_tensors/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v50 = utils.load_tensor(
        "./input_tensors/arg48.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v51 = utils.load_tensor(
        "./input_tensors/arg49.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v52 = utils.load_tensor(
        "./input_tensors/arg50.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v53 = utils.load_tensor(
        "./input_tensors/arg51.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v54 = utils.load_tensor(
        "./input_tensors/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v55 = utils.load_tensor(
        "./input_tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v56 = utils.load_tensor(
        "./input_tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v57 = utils.load_tensor(
        "./input_tensors/arg55.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v58 = utils.load_tensor(
        "./input_tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v59 = utils.load_tensor(
        "./input_tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v60 = utils.load_tensor(
        "./input_tensors/arg58.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v61 = utils.load_tensor(
        "./input_tensors/arg59.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v62 = utils.load_tensor(
        "./input_tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v63 = utils.load_tensor(
        "./input_tensors/arg61.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v64 = utils.load_tensor(
        "./input_tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v65 = utils.load_tensor(
        "./input_tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v66 = utils.load_tensor(
        "./input_tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v67 = utils.load_tensor(
        "./input_tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v68 = utils.load_tensor(
        "./input_tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v69 = utils.load_tensor(
        "./input_tensors/arg67.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v70 = utils.load_tensor(
        "./input_tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v71 = utils.load_tensor(
        "./input_tensors/arg69.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v72 = utils.load_tensor(
        "./input_tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v73 = utils.load_tensor(
        "./input_tensors/arg71.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v74 = utils.load_tensor(
        "./input_tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v75 = utils.load_tensor(
        "./input_tensors/arg73.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v76 = utils.load_tensor(
        "./input_tensors/arg74.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v77 = utils.load_tensor(
        "./input_tensors/arg75.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v78 = utils.load_tensor(
        "./input_tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v79 = utils.load_tensor(
        "./input_tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v80 = utils.load_tensor(
        "./input_tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v81 = utils.load_tensor(
        "./input_tensors/arg79.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v82 = utils.load_tensor(
        "./input_tensors/arg80.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v83 = utils.load_tensor(
        "./input_tensors/arg81.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v84 = utils.load_tensor(
        "./input_tensors/arg82.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v85 = utils.load_tensor(
        "./input_tensors/arg83.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v86 = utils.load_tensor(
        "./input_tensors/arg84.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v87 = utils.load_tensor(
        "./input_tensors/arg85.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v88 = utils.load_tensor(
        "./input_tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v89 = utils.load_tensor(
        "./input_tensors/arg87.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v90 = utils.load_tensor(
        "./input_tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v91 = utils.load_tensor(
        "./input_tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v92 = utils.load_tensor(
        "./input_tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v93 = utils.load_tensor(
        "./input_tensors/arg91.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v94 = utils.load_tensor(
        "./input_tensors/arg92.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v95 = utils.load_tensor(
        "./input_tensors/arg93.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v96 = utils.load_tensor(
        "./input_tensors/arg94.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v97 = utils.load_tensor(
        "./input_tensors/arg95.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v98 = utils.load_tensor(
        "./input_tensors/arg96.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v99 = utils.load_tensor(
        "./input_tensors/arg97.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v100 = utils.load_tensor(
        "./input_tensors/arg98.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v101 = utils.load_tensor(
        "./input_tensors/arg99.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v102 = utils.load_tensor(
        "./input_tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v103 = utils.load_tensor(
        "./input_tensors/arg101.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v104 = utils.load_tensor(
        "./input_tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v105 = utils.load_tensor(
        "./input_tensors/arg103.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v106 = utils.load_tensor(
        "./input_tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v107 = utils.load_tensor(
        "./input_tensors/arg105.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v108 = utils.load_tensor(
        "./input_tensors/arg106.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v109 = utils.load_tensor(
        "./input_tensors/arg107.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v110 = utils.load_tensor(
        "./input_tensors/arg108.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v111 = utils.load_tensor(
        "./input_tensors/arg109.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v112 = utils.load_tensor(
        "./input_tensors/arg110.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v113 = utils.load_tensor(
        "./input_tensors/arg111.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v114 = utils.load_tensor(
        "./input_tensors/arg112.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v115 = utils.load_tensor(
        "./input_tensors/arg113.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v116 = utils.load_tensor(
        "./input_tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v117 = utils.load_tensor(
        "./input_tensors/arg115.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v118 = utils.load_tensor(
        "./input_tensors/arg116.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v119 = utils.load_tensor(
        "./input_tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v120 = utils.load_tensor(
        "./input_tensors/arg118.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v121 = utils.load_tensor(
        "./input_tensors/arg119.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v122 = utils.load_tensor(
        "./input_tensors/arg120.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v123 = utils.load_tensor(
        "./input_tensors/arg121.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v124 = utils.load_tensor(
        "./input_tensors/arg122.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v125 = utils.load_tensor(
        "./input_tensors/arg123.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v126 = utils.load_tensor(
        "./input_tensors/arg124.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v127 = utils.load_tensor(
        "./input_tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v128 = utils.load_tensor(
        "./input_tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v129 = utils.load_tensor(
        "./input_tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v130 = utils.load_tensor(
        "./input_tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v131 = utils.load_tensor(
        "./input_tensors/arg129.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v132 = utils.load_tensor(
        "./input_tensors/arg130.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v133 = utils.load_tensor(
        "./input_tensors/arg131.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v134 = utils.load_tensor(
        "./input_tensors/arg132.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v135 = utils.load_tensor(
        "./input_tensors/arg133.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v136 = utils.load_tensor(
        "./input_tensors/arg134.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v137 = utils.load_tensor(
        "./input_tensors/arg135.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v138 = utils.load_tensor(
        "./input_tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v139 = utils.load_tensor(
        "./input_tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v140 = utils.load_tensor(
        "./input_tensors/arg138.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v141 = utils.load_tensor(
        "./input_tensors/arg139.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v142 = utils.load_tensor(
        "./input_tensors/arg140.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v143 = utils.load_tensor(
        "./input_tensors/arg141.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v144 = utils.load_tensor(
        "./input_tensors/arg142.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v145 = utils.load_tensor(
        "./input_tensors/arg143.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v146 = utils.load_tensor(
        "./input_tensors/arg144.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v147 = utils.load_tensor(
        "./input_tensors/arg145.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v148 = utils.load_tensor(
        "./input_tensors/arg146.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v149 = utils.load_tensor(
        "./input_tensors/arg147.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v150 = utils.load_tensor(
        "./input_tensors/arg148.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v151 = utils.load_tensor(
        "./input_tensors/arg149.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v152 = utils.load_tensor(
        "./input_tensors/arg150.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v153 = utils.load_tensor(
        "./input_tensors/arg151.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v154 = utils.load_tensor(
        "./input_tensors/arg152.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v155 = utils.load_tensor(
        "./input_tensors/arg153.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v156 = utils.load_tensor(
        "./input_tensors/arg154.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v157 = utils.load_tensor(
        "./input_tensors/arg155.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v158 = utils.load_tensor(
        "./input_tensors/arg156.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v159 = utils.load_tensor(
        "./input_tensors/arg157.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v160 = utils.load_tensor(
        "./input_tensors/arg158.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v161 = utils.load_tensor(
        "./input_tensors/arg159.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v162 = utils.load_tensor(
        "./input_tensors/arg160.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v163 = utils.load_tensor(
        "./input_tensors/arg161.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v164 = utils.load_tensor(
        "./input_tensors/arg162.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v165 = utils.load_tensor(
        "./input_tensors/arg163.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v166 = utils.load_tensor(
        "./input_tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v167 = utils.load_tensor(
        "./input_tensors/arg165.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v168 = utils.load_tensor(
        "./input_tensors/arg166.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v169 = utils.load_tensor(
        "./input_tensors/arg167.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v170 = utils.load_tensor(
        "./input_tensors/arg168.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v171 = utils.load_tensor(
        "./input_tensors/arg169.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v172 = utils.load_tensor(
        "./input_tensors/arg170.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v173 = utils.load_tensor(
        "./input_tensors/arg171.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v174 = utils.load_tensor(
        "./input_tensors/arg172.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v175 = utils.load_tensor(
        "./input_tensors/arg173.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v176 = utils.load_tensor(
        "./input_tensors/arg174.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v177 = utils.load_tensor(
        "./input_tensors/arg175.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v178 = utils.load_tensor(
        "./input_tensors/arg176.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v179 = utils.load_tensor(
        "./input_tensors/arg177.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v180 = utils.load_tensor(
        "./input_tensors/arg178.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v181 = utils.load_tensor(
        "./input_tensors/arg179.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v182 = utils.load_tensor(
        "./input_tensors/arg180.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v183 = utils.load_tensor(
        "./input_tensors/arg181.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v184 = utils.load_tensor(
        "./input_tensors/arg182.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v185 = utils.load_tensor(
        "./input_tensors/arg183.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v186 = utils.load_tensor(
        "./input_tensors/arg184.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v187 = utils.load_tensor(
        "./input_tensors/arg185.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v188 = utils.load_tensor(
        "./input_tensors/arg186.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v189 = utils.load_tensor(
        "./input_tensors/arg187.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v190 = utils.load_tensor(
        "./input_tensors/arg188.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v191 = utils.load_tensor(
        "./input_tensors/arg189.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v192 = utils.load_tensor(
        "./input_tensors/arg190.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v193 = utils.load_tensor(
        "./input_tensors/arg191.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v194 = utils.load_tensor(
        "./input_tensors/arg192.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v195 = utils.load_tensor(
        "./input_tensors/arg193.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v196 = utils.load_tensor(
        "./input_tensors/arg194.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v197 = utils.load_tensor(
        "./input_tensors/arg195.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v198 = utils.load_tensor(
        "./input_tensors/arg196.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v199 = utils.load_tensor(
        "./input_tensors/arg197.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v200 = utils.load_tensor(
        "./input_tensors/arg198.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v201 = utils.load_tensor(
        "./input_tensors/arg199.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v202 = utils.load_tensor(
        "./input_tensors/arg200.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v203 = utils.load_tensor(
        "./input_tensors/arg201.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v204 = utils.load_tensor(
        "./input_tensors/arg202.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v205 = utils.load_tensor(
        "./input_tensors/arg203.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v206 = utils.load_tensor(
        "./input_tensors/arg204.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v207 = utils.load_tensor(
        "./input_tensors/arg205.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v208 = utils.load_tensor(
        "./input_tensors/arg206.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v209 = utils.load_tensor(
        "./input_tensors/arg207.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v210 = utils.load_tensor(
        "./input_tensors/arg208.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v211 = utils.load_tensor(
        "./input_tensors/arg209.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v212 = utils.load_tensor(
        "./input_tensors/arg210.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v213 = utils.load_tensor(
        "./input_tensors/arg211.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v214 = utils.load_tensor(
        "./input_tensors/arg212.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v215 = utils.load_tensor(
        "./input_tensors/arg213.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v216 = utils.load_tensor(
        "./input_tensors/arg214.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v217 = utils.load_tensor(
        "./input_tensors/arg215.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v218 = utils.load_tensor(
        "./input_tensors/arg216.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v219 = utils.load_tensor(
        "./input_tensors/arg217.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v220 = utils.load_tensor(
        "./input_tensors/arg218.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v221 = utils.load_tensor(
        "./input_tensors/arg219.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v222 = utils.load_tensor(
        "./input_tensors/arg220.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v223 = utils.load_tensor(
        "./input_tensors/arg221.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v224 = utils.load_tensor(
        "./input_tensors/arg222.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v225 = utils.load_tensor(
        "./input_tensors/arg223.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v226 = utils.load_tensor(
        "./input_tensors/arg224.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v227 = utils.load_tensor(
        "./input_tensors/arg225.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v228 = utils.load_tensor(
        "./input_tensors/arg226.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v229 = utils.load_tensor(
        "./input_tensors/arg227.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v230 = utils.load_tensor(
        "./input_tensors/arg228.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v231 = utils.load_tensor(
        "./input_tensors/arg229.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v232 = utils.load_tensor(
        "./input_tensors/arg230.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v233 = utils.load_tensor(
        "./input_tensors/arg231.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v234 = utils.load_tensor(
        "./input_tensors/arg232.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v235 = utils.load_tensor(
        "./input_tensors/arg233.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v236 = utils.load_tensor(
        "./input_tensors/arg234.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v237 = utils.load_tensor(
        "./input_tensors/arg235.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v238 = utils.load_tensor(
        "./input_tensors/arg236.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v239 = utils.load_tensor(
        "./input_tensors/arg237.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v240 = utils.load_tensor(
        "./input_tensors/arg238.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v241 = utils.load_tensor(
        "./input_tensors/arg239.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v242 = utils.load_tensor(
        "./input_tensors/arg240.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v243 = utils.load_tensor(
        "./input_tensors/arg241.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v244 = utils.load_tensor(
        "./input_tensors/arg242.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v245 = utils.load_tensor(
        "./input_tensors/arg243.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v246 = utils.load_tensor(
        "./input_tensors/arg244.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v247 = utils.load_tensor(
        "./input_tensors/arg245.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v248 = utils.load_tensor(
        "./input_tensors/arg246.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v249 = utils.load_tensor(
        "./input_tensors/arg247.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v250 = utils.load_tensor(
        "./input_tensors/arg248.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v251 = utils.load_tensor(
        "./input_tensors/arg249.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v252 = utils.load_tensor(
        "./input_tensors/arg250.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v253 = utils.load_tensor(
        "./input_tensors/arg251.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v254 = utils.load_tensor(
        "./input_tensors/arg252.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v255 = utils.load_tensor(
        "./input_tensors/arg253.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v256 = utils.load_tensor(
        "./input_tensors/arg254.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v257 = utils.load_tensor(
        "./input_tensors/arg255.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v258 = utils.load_tensor(
        "./input_tensors/arg256.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v259 = utils.load_tensor(
        "./input_tensors/arg257.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v260 = utils.load_tensor(
        "./input_tensors/arg258.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v261 = utils.load_tensor(
        "./input_tensors/arg259.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v262 = utils.load_tensor(
        "./input_tensors/arg260.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v263 = utils.load_tensor(
        "./input_tensors/arg261.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v264 = utils.load_tensor(
        "./input_tensors/arg262.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v265 = utils.load_tensor(
        "./input_tensors/arg263.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v266 = utils.load_tensor(
        "./input_tensors/arg264.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v267 = utils.load_tensor(
        "./input_tensors/arg265.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v268 = utils.load_tensor(
        "./input_tensors/arg266.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    v269 = utils.load_tensor(
        "./input_tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v270 = [
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
        v21,
        v22,
        v23,
        v24,
        v25,
        v26,
        v27,
        v28,
        v29,
        v30,
        v31,
        v32,
        v33,
        v34,
        v35,
        v36,
        v37,
        v38,
        v39,
        v40,
        v41,
        v42,
        v43,
        v44,
        v45,
        v46,
        v47,
        v48,
        v49,
        v50,
        v51,
        v52,
        v53,
        v54,
        v55,
        v56,
        v57,
        v58,
        v59,
        v60,
        v61,
        v62,
        v63,
        v64,
        v65,
        v66,
        v67,
        v68,
        v69,
        v70,
        v71,
        v72,
        v73,
        v74,
        v75,
        v76,
        v77,
        v78,
        v79,
        v80,
        v81,
        v82,
        v83,
        v84,
        v85,
        v86,
        v87,
        v88,
        v89,
        v90,
        v91,
        v92,
        v93,
        v94,
        v95,
        v96,
        v97,
        v98,
        v99,
        v100,
        v101,
        v102,
        v103,
        v104,
        v105,
        v106,
        v107,
        v108,
        v109,
        v110,
        v111,
        v112,
        v113,
        v114,
        v115,
        v116,
        v117,
        v118,
        v119,
        v120,
        v121,
        v122,
        v123,
        v124,
        v125,
        v126,
        v127,
        v128,
        v129,
        v130,
        v131,
        v132,
        v133,
        v134,
        v135,
        v136,
        v137,
        v138,
        v139,
        v140,
        v141,
        v142,
        v143,
        v144,
        v145,
        v146,
        v147,
        v148,
        v149,
        v150,
        v151,
        v152,
        v153,
        v154,
        v155,
        v156,
        v157,
        v158,
        v159,
        v160,
        v161,
        v162,
        v163,
        v164,
        v165,
        v166,
        v167,
        v168,
        v169,
        v170,
        v171,
        v172,
        v173,
        v174,
        v175,
        v176,
        v177,
        v178,
        v179,
        v180,
        v181,
        v182,
        v183,
        v184,
        v185,
        v186,
        v187,
        v188,
        v189,
        v190,
        v191,
        v192,
        v193,
        v194,
        v195,
        v196,
        v197,
        v198,
        v199,
        v200,
        v201,
        v202,
        v203,
        v204,
        v205,
        v206,
        v207,
        v208,
        v209,
        v210,
        v211,
        v212,
        v213,
        v214,
        v215,
        v216,
        v217,
        v218,
        v219,
        v220,
        v221,
        v222,
        v223,
        v224,
        v225,
        v226,
        v227,
        v228,
        v229,
        v230,
        v231,
        v232,
        v233,
        v234,
        v235,
        v236,
        v237,
        v238,
        v239,
        v240,
        v241,
        v242,
        v243,
        v244,
        v245,
        v246,
        v247,
        v248,
        v249,
        v250,
        v251,
        v252,
        v253,
        v254,
        v255,
        v256,
        v257,
        v258,
        v259,
        v260,
        v261,
        v262,
        v263,
        v264,
        v265,
        v266,
        v267,
        v268,
        v269,
    ]
    return v270


def main():
    v1 = load_inputs_for__main()
    v2 = _main(v1)
    v3 = 0
    return v3


if __name__ == "__main__":
    main()
