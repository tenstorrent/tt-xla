# This file contains the model code for the CLIPVision model

import ttnn


def Linear_141_0(input):
    ttnn_reshape_221 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_221


def LayerNorm_22_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_0 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_multiply_0,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_multiply_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn.deallocate(ttnn_multiply_1, False)
    return ttnn_add_0


def CLIPAttention_126_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_to_memory_config_0,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_matmul_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_to_memory_config_1,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_reshape_222,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_multiply_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_typecast_1,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_softmax_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_typecast_2,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_reshape_223,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_to_memory_config_2,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_matmul_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_to_memory_config_3,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_0, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_1, False)
    ttnn.deallocate(ttnn_reshape_222, False)
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn.deallocate(ttnn_softmax_0, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_reshape_223, False)
    ttnn.deallocate(ttnn_to_memory_config_2, False)
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn.deallocate(ttnn_to_memory_config_3, False)
    return ttnn_reshape_224


def CLIPEncoderLayer_56_0(input_0, input_1, input_2, input_3):
    ttnn_add_1 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_add_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_0 = ttnn.sum(
        ttnn_to_memory_config_4,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_sum_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_multiply_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_to_memory_config_5,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_0 = ttnn.neg(
        ttnn_reshape_225,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_2 = ttnn.add(
        ttnn_to_memory_config_4,
        ttnn_neg_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_6 = ttnn.to_memory_config(
        ttnn_add_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_to_memory_config_6,
        ttnn_to_memory_config_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_7 = ttnn.to_memory_config(
        ttnn_multiply_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_1 = ttnn.sum(
        ttnn_to_memory_config_7,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_sum_1,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_3 = ttnn.add(
        ttnn_multiply_5,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_8 = ttnn.to_memory_config(
        ttnn_add_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_to_memory_config_8,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_rsqrt_0,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_1, False)
    # ttnn.deallocate(ttnn_to_memory_config_4, False)
    ttnn.deallocate(ttnn_sum_0, False)
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn.deallocate(ttnn_to_memory_config_5, False)
    ttnn.deallocate(ttnn_reshape_225, False)
    ttnn.deallocate(ttnn_neg_0, False)
    ttnn.deallocate(ttnn_add_2, False)
    # ttnn.deallocate(ttnn_to_memory_config_6, False)
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn.deallocate(ttnn_to_memory_config_7, False)
    ttnn.deallocate(ttnn_sum_1, False)
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn.deallocate(ttnn_add_3, False)
    ttnn.deallocate(ttnn_to_memory_config_8, False)
    ttnn.deallocate(ttnn_rsqrt_0, False)
    # ttnn.deallocate(ttnn_reshape_226, False)
    return ttnn_to_memory_config_6, ttnn_to_memory_config_4, ttnn_reshape_226


def Linear_37_0(input_0, input_1, input_2):
    ttnn_matmul_2 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_9 = ttnn.to_memory_config(
        ttnn_add_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_to_memory_config_9,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn.deallocate(ttnn_add_4, False)
    ttnn.deallocate(ttnn_to_memory_config_9, False)
    return ttnn_reshape_227


def CLIPEncoderLayer_38_0(input_0, input_1, input_2, input_3):
    ttnn_add_5 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_10 = ttnn.to_memory_config(
        ttnn_add_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_2 = ttnn.sum(
        ttnn_to_memory_config_10,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_sum_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_11 = ttnn.to_memory_config(
        ttnn_multiply_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_to_memory_config_11,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_1 = ttnn.neg(
        ttnn_reshape_228,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_6 = ttnn.add(
        ttnn_to_memory_config_10,
        ttnn_neg_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_12 = ttnn.to_memory_config(
        ttnn_add_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_to_memory_config_12,
        ttnn_to_memory_config_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_13 = ttnn.to_memory_config(
        ttnn_multiply_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_3 = ttnn.sum(
        ttnn_to_memory_config_13,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_sum_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_7 = ttnn.add(
        ttnn_multiply_8,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_14 = ttnn.to_memory_config(
        ttnn_add_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_to_memory_config_14,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_rsqrt_1,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_5, False)
    # ttnn.deallocate(ttnn_to_memory_config_10, False)
    ttnn.deallocate(ttnn_sum_2, False)
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn.deallocate(ttnn_to_memory_config_11, False)
    ttnn.deallocate(ttnn_reshape_228, False)
    ttnn.deallocate(ttnn_neg_1, False)
    ttnn.deallocate(ttnn_add_6, False)
    # ttnn.deallocate(ttnn_to_memory_config_12, False)
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn.deallocate(ttnn_to_memory_config_13, False)
    ttnn.deallocate(ttnn_sum_3, False)
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn.deallocate(ttnn_add_7, False)
    ttnn.deallocate(ttnn_to_memory_config_14, False)
    ttnn.deallocate(ttnn_rsqrt_1, False)
    # ttnn.deallocate(ttnn_reshape_229, False)
    return ttnn_to_memory_config_10, ttnn_to_memory_config_12, ttnn_reshape_229


def QuickGELUActivation_60_0(input_0, input_1):
    ttnn_multiply_9 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_15 = ttnn.to_memory_config(
        ttnn_multiply_9,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_0 = ttnn.sigmoid(
        ttnn_to_memory_config_15,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_10 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn.deallocate(ttnn_to_memory_config_15, False)
    ttnn.deallocate(ttnn_sigmoid_0, False)
    return ttnn_multiply_10


def LayerNorm_70_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_11 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_12 = ttnn.multiply(
        ttnn_multiply_11,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_8 = ttnn.add(
        ttnn_multiply_12,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_11, False)
    ttnn.deallocate(ttnn_multiply_12, False)
    return ttnn_add_8


def LayerNorm_34_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_13 = ttnn.multiply(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_14 = ttnn.multiply(
        ttnn_multiply_13,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_9 = ttnn.add(
        ttnn_multiply_14,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_13, False)
    ttnn.deallocate(ttnn_multiply_14, False)
    return ttnn_add_9


def CLIPEncoderLayer_2_0(input_0, input_1, input_2):
    ttnn_sum_4 = ttnn.sum(
        input_0,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_sum_4,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_16 = ttnn.to_memory_config(
        ttnn_multiply_15,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_to_memory_config_16,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_2 = ttnn.neg(
        ttnn_reshape_230,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_10 = ttnn.add(
        input_0,
        ttnn_neg_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_17 = ttnn.to_memory_config(
        ttnn_add_10,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_16 = ttnn.multiply(
        ttnn_to_memory_config_17,
        ttnn_to_memory_config_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_18 = ttnn.to_memory_config(
        ttnn_multiply_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_5 = ttnn.sum(
        ttnn_to_memory_config_18,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_17 = ttnn.multiply(
        ttnn_sum_5,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_11 = ttnn.add(
        ttnn_multiply_17,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_19 = ttnn.to_memory_config(
        ttnn_add_11,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_to_memory_config_19,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_rsqrt_2,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    # ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_sum_4, False)
    ttnn.deallocate(ttnn_multiply_15, False)
    ttnn.deallocate(ttnn_to_memory_config_16, False)
    ttnn.deallocate(ttnn_reshape_230, False)
    ttnn.deallocate(ttnn_neg_2, False)
    ttnn.deallocate(ttnn_add_10, False)
    # ttnn.deallocate(ttnn_to_memory_config_17, False)
    ttnn.deallocate(ttnn_multiply_16, False)
    ttnn.deallocate(ttnn_to_memory_config_18, False)
    ttnn.deallocate(ttnn_sum_5, False)
    ttnn.deallocate(ttnn_multiply_17, False)
    ttnn.deallocate(ttnn_add_11, False)
    ttnn.deallocate(ttnn_to_memory_config_19, False)
    ttnn.deallocate(ttnn_rsqrt_2, False)
    # ttnn.deallocate(ttnn_reshape_231, False)
    return ttnn_to_memory_config_17, ttnn_reshape_231


def Linear_33_0(input):
    ttnn_reshape_232 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_232


def LayerNorm_28_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_18 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_19 = ttnn.multiply(
        ttnn_multiply_18,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_12 = ttnn.add(
        ttnn_multiply_19,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_20 = ttnn.to_memory_config(
        ttnn_add_12,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_18, False)
    ttnn.deallocate(ttnn_multiply_19, False)
    return ttnn_to_memory_config_20, ttnn_add_12


def Linear_119_0(input_0, input_1, input_2):
    ttnn_matmul_3 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_13 = ttnn.add(
        ttnn_matmul_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_21 = ttnn.to_memory_config(
        ttnn_add_13,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn.deallocate(ttnn_add_13, False)
    return ttnn_to_memory_config_21


def Linear_65_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_4 = ttnn.matmul(
        input_1,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_22 = ttnn.to_memory_config(
        input_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_4,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_23 = ttnn.to_memory_config(
        input_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_to_memory_config_22,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_24 = ttnn.to_memory_config(
        ttnn_add_14,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_to_memory_config_23,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_15 = ttnn.add(
        ttnn_matmul_5,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_16 = ttnn.add(
        ttnn_matmul_6,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_to_memory_config_24,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_25 = ttnn.to_memory_config(
        ttnn_add_15,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_26 = ttnn.to_memory_config(
        ttnn_add_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_50 = ttnn.permute(
        ttnn_reshape_233,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_to_memory_config_25,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_to_memory_config_26,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_permute_50,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_51 = ttnn.permute(
        ttnn_reshape_234,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_52 = ttnn.permute(
        ttnn_reshape_235,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_permute_51,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_permute_52,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(input_6, False)
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn.deallocate(ttnn_add_14, False)
    ttnn.deallocate(ttnn_to_memory_config_22, False)
    ttnn.deallocate(ttnn_to_memory_config_23, False)
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn.deallocate(ttnn_to_memory_config_24, False)
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn.deallocate(ttnn_reshape_233, False)
    ttnn.deallocate(ttnn_add_16, False)
    ttnn.deallocate(ttnn_add_15, False)
    ttnn.deallocate(ttnn_to_memory_config_25, False)
    ttnn.deallocate(ttnn_permute_50, False)
    ttnn.deallocate(ttnn_to_memory_config_26, False)
    ttnn.deallocate(ttnn_reshape_234, False)
    ttnn.deallocate(ttnn_reshape_235, False)
    ttnn.deallocate(ttnn_permute_51, False)
    ttnn.deallocate(ttnn_permute_52, False)
    return ttnn_reshape_238, ttnn_reshape_237, ttnn_reshape_236


def LayerNorm_4_0(input_0, input_1, input_2, input_3):
    input_1.is_allocated()
    input_2.is_allocated()
    ttnn_multiply_20 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_21 = ttnn.multiply(
        ttnn_multiply_20,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_17 = ttnn.add(
        ttnn_multiply_21,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_27 = ttnn.to_memory_config(
        ttnn_add_17,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_20, False)
    ttnn.deallocate(ttnn_multiply_21, False)
    return ttnn_to_memory_config_27, ttnn_add_17


def QuickGELUActivation_12_0(input_0, input_1):
    ttnn_multiply_22 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_28 = ttnn.to_memory_config(
        ttnn_multiply_22,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_1 = ttnn.sigmoid(
        ttnn_to_memory_config_28,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_23 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_22, False)
    ttnn.deallocate(ttnn_to_memory_config_28, False)
    ttnn.deallocate(ttnn_sigmoid_1, False)
    return ttnn_multiply_23


def Linear_137_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_7 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_29 = ttnn.to_memory_config(
        input_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_18 = ttnn.add(
        ttnn_matmul_7,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_30 = ttnn.to_memory_config(
        input_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_to_memory_config_29,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_31 = ttnn.to_memory_config(
        ttnn_add_18,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_to_memory_config_30,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_19 = ttnn.add(
        ttnn_matmul_8,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_20 = ttnn.add(
        ttnn_matmul_9,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_239 = ttnn.reshape(
        ttnn_to_memory_config_31,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_32 = ttnn.to_memory_config(
        ttnn_add_19,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_33 = ttnn.to_memory_config(
        ttnn_add_20,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_53 = ttnn.permute(
        ttnn_reshape_239,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_to_memory_config_32,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_to_memory_config_33,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_permute_53,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_54 = ttnn.permute(
        ttnn_reshape_240,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_55 = ttnn.permute(
        ttnn_reshape_241,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_permute_54,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_permute_55,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_7, False)
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn.deallocate(ttnn_add_18, False)
    ttnn.deallocate(ttnn_to_memory_config_29, False)
    ttnn.deallocate(ttnn_to_memory_config_30, False)
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn.deallocate(ttnn_to_memory_config_31, False)
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn.deallocate(ttnn_reshape_239, False)
    ttnn.deallocate(ttnn_add_20, False)
    ttnn.deallocate(ttnn_add_19, False)
    ttnn.deallocate(ttnn_to_memory_config_32, False)
    ttnn.deallocate(ttnn_permute_53, False)
    ttnn.deallocate(ttnn_to_memory_config_33, False)
    ttnn.deallocate(ttnn_reshape_240, False)
    ttnn.deallocate(ttnn_reshape_241, False)
    ttnn.deallocate(ttnn_permute_54, False)
    ttnn.deallocate(ttnn_permute_55, False)
    return ttnn_reshape_244, ttnn_reshape_242, ttnn_reshape_243


def Linear_139_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_24 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_245 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_24,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_34 = ttnn.to_memory_config(
        ttnn_reshape_245,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_to_memory_config_34,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_21 = ttnn.add(
        ttnn_matmul_10,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_35 = ttnn.to_memory_config(
        ttnn_add_21,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_246 = ttnn.reshape(
        ttnn_to_memory_config_35,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_24, False)
    ttnn.deallocate(ttnn_reshape_245, False)
    ttnn.deallocate(ttnn_to_memory_config_34, False)
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn.deallocate(ttnn_add_21, False)
    ttnn.deallocate(ttnn_to_memory_config_35, False)
    return ttnn_reshape_246


def CLIPEncoderLayer_122_0(input_0, input_1, input_2, input_3):
    ttnn_add_22 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_36 = ttnn.to_memory_config(
        ttnn_add_22,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_6 = ttnn.sum(
        ttnn_to_memory_config_36,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_24 = ttnn.multiply(
        ttnn_sum_6,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_37 = ttnn.to_memory_config(
        ttnn_multiply_24,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_247 = ttnn.reshape(
        ttnn_to_memory_config_37,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_3 = ttnn.neg(
        ttnn_reshape_247,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_23 = ttnn.add(
        ttnn_to_memory_config_36,
        ttnn_neg_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_38 = ttnn.to_memory_config(
        ttnn_add_23,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_25 = ttnn.multiply(
        ttnn_to_memory_config_38,
        ttnn_to_memory_config_38,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_39 = ttnn.to_memory_config(
        ttnn_multiply_25,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_7 = ttnn.sum(
        ttnn_to_memory_config_39,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_26 = ttnn.multiply(
        ttnn_sum_7,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_24 = ttnn.add(
        ttnn_multiply_26,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_40 = ttnn.to_memory_config(
        ttnn_add_24,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_to_memory_config_40,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_248 = ttnn.reshape(
        ttnn_rsqrt_3,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_22, False)
    # ttnn.deallocate(ttnn_to_memory_config_36, False)
    ttnn.deallocate(ttnn_sum_6, False)
    ttnn.deallocate(ttnn_multiply_24, False)
    ttnn.deallocate(ttnn_to_memory_config_37, False)
    ttnn.deallocate(ttnn_reshape_247, False)
    ttnn.deallocate(ttnn_neg_3, False)
    ttnn.deallocate(ttnn_add_23, False)
    # ttnn.deallocate(ttnn_to_memory_config_38, False)
    ttnn.deallocate(ttnn_multiply_25, False)
    ttnn.deallocate(ttnn_to_memory_config_39, False)
    ttnn.deallocate(ttnn_sum_7, False)
    ttnn.deallocate(ttnn_multiply_26, False)
    ttnn.deallocate(ttnn_add_24, False)
    ttnn.deallocate(ttnn_to_memory_config_40, False)
    ttnn.deallocate(ttnn_rsqrt_3, False)
    # ttnn.deallocate(ttnn_reshape_248, False)
    return ttnn_to_memory_config_36, ttnn_reshape_248, ttnn_to_memory_config_38


def LayerNorm_118_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_27 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_28 = ttnn.multiply(
        ttnn_multiply_27,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_25 = ttnn.add(
        ttnn_multiply_28,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_27, False)
    ttnn.deallocate(ttnn_multiply_28, False)
    return ttnn_add_25


def CLIPEncoderLayer_14_0(input_0, input_1, input_2, input_3):
    ttnn_add_26 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_41 = ttnn.to_memory_config(
        ttnn_add_26,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_8 = ttnn.sum(
        ttnn_to_memory_config_41,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_29 = ttnn.multiply(
        ttnn_sum_8,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_42 = ttnn.to_memory_config(
        ttnn_multiply_29,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_249 = ttnn.reshape(
        ttnn_to_memory_config_42,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_4 = ttnn.neg(
        ttnn_reshape_249,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_27 = ttnn.add(
        ttnn_to_memory_config_41,
        ttnn_neg_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_43 = ttnn.to_memory_config(
        ttnn_add_27,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_to_memory_config_43,
        ttnn_to_memory_config_43,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_44 = ttnn.to_memory_config(
        ttnn_multiply_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_9 = ttnn.sum(
        ttnn_to_memory_config_44,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_31 = ttnn.multiply(
        ttnn_sum_9,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_28 = ttnn.add(
        ttnn_multiply_31,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_45 = ttnn.to_memory_config(
        ttnn_add_28,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_to_memory_config_45,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_250 = ttnn.reshape(
        ttnn_rsqrt_4,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_26, False)
    # ttnn.deallocate(ttnn_to_memory_config_41, False)
    ttnn.deallocate(ttnn_sum_8, False)
    ttnn.deallocate(ttnn_multiply_29, False)
    ttnn.deallocate(ttnn_to_memory_config_42, False)
    ttnn.deallocate(ttnn_reshape_249, False)
    ttnn.deallocate(ttnn_neg_4, False)
    ttnn.deallocate(ttnn_add_27, False)
    # ttnn.deallocate(ttnn_to_memory_config_43, False)
    ttnn.deallocate(ttnn_multiply_30, False)
    ttnn.deallocate(ttnn_to_memory_config_44, False)
    ttnn.deallocate(ttnn_sum_9, False)
    ttnn.deallocate(ttnn_multiply_31, False)
    ttnn.deallocate(ttnn_add_28, False)
    ttnn.deallocate(ttnn_to_memory_config_45, False)
    ttnn.deallocate(ttnn_rsqrt_4, False)
    # ttnn.deallocate(ttnn_reshape_250, False)
    return ttnn_to_memory_config_43, ttnn_to_memory_config_41, ttnn_reshape_250


def Linear_107_0(input_0, input_1, input_2):
    ttnn_matmul_11 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_29 = ttnn.add(
        ttnn_matmul_11,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_46 = ttnn.to_memory_config(
        ttnn_add_29,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn.deallocate(ttnn_add_29, False)
    return ttnn_to_memory_config_46


def Linear_83_0(input_0, input_1, input_2):
    ttnn_matmul_12 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_30 = ttnn.add(
        ttnn_matmul_12,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_47 = ttnn.to_memory_config(
        ttnn_add_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn.deallocate(ttnn_add_30, False)
    return ttnn_to_memory_config_47


def Linear_17_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_13 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_48 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_31 = ttnn.add(
        ttnn_matmul_13,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_49 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_to_memory_config_48,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_50 = ttnn.to_memory_config(
        ttnn_add_31,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_to_memory_config_49,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_32 = ttnn.add(
        ttnn_matmul_14,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_33 = ttnn.add(
        ttnn_matmul_15,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_251 = ttnn.reshape(
        ttnn_to_memory_config_50,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_51 = ttnn.to_memory_config(
        ttnn_add_32,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_52 = ttnn.to_memory_config(
        ttnn_add_33,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_56 = ttnn.permute(
        ttnn_reshape_251,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_252 = ttnn.reshape(
        ttnn_to_memory_config_51,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_253 = ttnn.reshape(
        ttnn_to_memory_config_52,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_254 = ttnn.reshape(
        ttnn_permute_56,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_57 = ttnn.permute(
        ttnn_reshape_252,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_58 = ttnn.permute(
        ttnn_reshape_253,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_255 = ttnn.reshape(
        ttnn_permute_57,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_256 = ttnn.reshape(
        ttnn_permute_58,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn.deallocate(ttnn_add_31, False)
    ttnn.deallocate(ttnn_to_memory_config_48, False)
    ttnn.deallocate(ttnn_to_memory_config_49, False)
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn.deallocate(ttnn_to_memory_config_50, False)
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn.deallocate(ttnn_reshape_251, False)
    ttnn.deallocate(ttnn_add_33, False)
    ttnn.deallocate(ttnn_add_32, False)
    ttnn.deallocate(ttnn_to_memory_config_51, False)
    ttnn.deallocate(ttnn_permute_56, False)
    ttnn.deallocate(ttnn_to_memory_config_52, False)
    ttnn.deallocate(ttnn_reshape_252, False)
    ttnn.deallocate(ttnn_reshape_253, False)
    ttnn.deallocate(ttnn_permute_57, False)
    ttnn.deallocate(ttnn_permute_58, False)
    return ttnn_reshape_255, ttnn_reshape_254, ttnn_reshape_256


def Linear_101_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_16 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_53 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_34 = ttnn.add(
        ttnn_matmul_16,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_54 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_to_memory_config_53,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_55 = ttnn.to_memory_config(
        ttnn_add_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_18 = ttnn.matmul(
        ttnn_to_memory_config_54,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_35 = ttnn.add(
        ttnn_matmul_17,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_36 = ttnn.add(
        ttnn_matmul_18,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_257 = ttnn.reshape(
        ttnn_to_memory_config_55,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_56 = ttnn.to_memory_config(
        ttnn_add_35,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_57 = ttnn.to_memory_config(
        ttnn_add_36,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_59 = ttnn.permute(
        ttnn_reshape_257,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_258 = ttnn.reshape(
        ttnn_to_memory_config_56,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_259 = ttnn.reshape(
        ttnn_to_memory_config_57,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_260 = ttnn.reshape(
        ttnn_permute_59,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_60 = ttnn.permute(
        ttnn_reshape_258,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_61 = ttnn.permute(
        ttnn_reshape_259,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_261 = ttnn.reshape(
        ttnn_permute_60,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_262 = ttnn.reshape(
        ttnn_permute_61,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_16, False)
    ttnn.deallocate(ttnn_add_34, False)
    ttnn.deallocate(ttnn_to_memory_config_53, False)
    ttnn.deallocate(ttnn_to_memory_config_54, False)
    ttnn.deallocate(ttnn_matmul_18, False)
    ttnn.deallocate(ttnn_to_memory_config_55, False)
    ttnn.deallocate(ttnn_matmul_17, False)
    ttnn.deallocate(ttnn_reshape_257, False)
    ttnn.deallocate(ttnn_add_36, False)
    ttnn.deallocate(ttnn_add_35, False)
    ttnn.deallocate(ttnn_to_memory_config_56, False)
    ttnn.deallocate(ttnn_permute_59, False)
    ttnn.deallocate(ttnn_to_memory_config_57, False)
    ttnn.deallocate(ttnn_reshape_258, False)
    ttnn.deallocate(ttnn_reshape_259, False)
    ttnn.deallocate(ttnn_permute_60, False)
    ttnn.deallocate(ttnn_permute_61, False)
    return ttnn_reshape_261, ttnn_reshape_260, ttnn_reshape_262


def CLIPVisionEmbeddings_0_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_permute_62 = ttnn.permute(
        input_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_263 = ttnn.reshape(
        ttnn_permute_62,
        [1, 1, 100352, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_reshape_263,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_0,
        weight_tensor=input_4,
        device=input_2,
        in_channels=3,
        out_channels=768,
        batch_size=2,
        input_height=224,
        input_width=224,
        kernel_size=[32, 32],
        stride=[32, 32],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
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
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_264 = ttnn.reshape(
        ttnn_conv2d_0,
        [2, 7, 7, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_63 = ttnn.permute(
        ttnn_reshape_264,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_265 = ttnn.reshape(
        ttnn_permute_63,
        [2, 768, 49],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_261 = [input_1, ttnn_reshape_265]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_261,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_37 = ttnn.add(
        ttnn_concat_0,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_58 = ttnn.to_memory_config(
        ttnn_add_37,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_64 = ttnn.permute(
        ttnn_to_memory_config_58,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_permute_62, False)
    ttnn.deallocate(ttnn_reshape_263, False)
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn.deallocate(ttnn_reshape_264, False)
    ttnn.deallocate(ttnn_permute_63, False)
    ttnn.deallocate(ttnn_reshape_265, False)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn.deallocate(ttnn_add_37, False)
    ttnn.deallocate(ttnn_to_memory_config_58, False)
    return ttnn_permute_64


def LayerNorm_106_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_32 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_33 = ttnn.multiply(
        ttnn_multiply_32,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_38 = ttnn.add(
        ttnn_multiply_33,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_32, False)
    ttnn.deallocate(ttnn_multiply_33, False)
    return ttnn_add_38


def Linear_73_0(input_0, input_1, input_2):
    ttnn_matmul_19 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_39 = ttnn.add(
        ttnn_matmul_19,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_59 = ttnn.to_memory_config(
        ttnn_add_39,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_266 = ttnn.reshape(
        ttnn_to_memory_config_59,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_19, False)
    ttnn.deallocate(ttnn_add_39, False)
    ttnn.deallocate(ttnn_to_memory_config_59, False)
    return ttnn_reshape_266


def Linear_121_0(input_0, input_1, input_2):
    ttnn_matmul_20 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_40 = ttnn.add(
        ttnn_matmul_20,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_60 = ttnn.to_memory_config(
        ttnn_add_40,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_267 = ttnn.reshape(
        ttnn_to_memory_config_60,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_20, False)
    ttnn.deallocate(ttnn_add_40, False)
    ttnn.deallocate(ttnn_to_memory_config_60, False)
    return ttnn_reshape_267


def Linear_105_0(input):
    ttnn_reshape_268 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_268


def Linear_133_0(input_0, input_1, input_2):
    ttnn_matmul_21 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_41 = ttnn.add(
        ttnn_matmul_21,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_61 = ttnn.to_memory_config(
        ttnn_add_41,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_269 = ttnn.reshape(
        ttnn_to_memory_config_61,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_21, False)
    ttnn.deallocate(ttnn_add_41, False)
    ttnn.deallocate(ttnn_to_memory_config_61, False)
    return ttnn_reshape_269


def Linear_131_0(input_0, input_1, input_2):
    ttnn_matmul_22 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_42 = ttnn.add(
        ttnn_matmul_22,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_62 = ttnn.to_memory_config(
        ttnn_add_42,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_22, False)
    ttnn.deallocate(ttnn_add_42, False)
    return ttnn_to_memory_config_62


def Linear_7_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_25 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_270 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_25,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_63 = ttnn.to_memory_config(
        ttnn_reshape_270,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_23 = ttnn.matmul(
        ttnn_to_memory_config_63,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_43 = ttnn.add(
        ttnn_matmul_23,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_64 = ttnn.to_memory_config(
        ttnn_add_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_271 = ttnn.reshape(
        ttnn_to_memory_config_64,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_25, False)
    ttnn.deallocate(ttnn_reshape_270, False)
    ttnn.deallocate(ttnn_to_memory_config_63, False)
    ttnn.deallocate(ttnn_matmul_23, False)
    ttnn.deallocate(ttnn_add_43, False)
    ttnn.deallocate(ttnn_to_memory_config_64, False)
    return ttnn_reshape_271


def QuickGELUActivation_96_0(input_0, input_1):
    ttnn_multiply_34 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_65 = ttnn.to_memory_config(
        ttnn_multiply_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_2 = ttnn.sigmoid(
        ttnn_to_memory_config_65,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_35 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_34, False)
    ttnn.deallocate(ttnn_to_memory_config_65, False)
    ttnn.deallocate(ttnn_sigmoid_2, False)
    return ttnn_multiply_35


def QuickGELUActivation_132_0(input_0, input_1):
    ttnn_multiply_36 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_66 = ttnn.to_memory_config(
        ttnn_multiply_36,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_3 = ttnn.sigmoid(
        ttnn_to_memory_config_66,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_37 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_36, False)
    ttnn.deallocate(ttnn_to_memory_config_66, False)
    ttnn.deallocate(ttnn_sigmoid_3, False)
    return ttnn_multiply_37


def QuickGELUActivation_48_0(input_0, input_1):
    ttnn_multiply_38 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_67 = ttnn.to_memory_config(
        ttnn_multiply_38,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_4 = ttnn.sigmoid(
        ttnn_to_memory_config_67,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_39 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_38, False)
    ttnn.deallocate(ttnn_to_memory_config_67, False)
    ttnn.deallocate(ttnn_sigmoid_4, False)
    return ttnn_multiply_39


def CLIPEncoderLayer_80_0(input_0, input_1, input_2, input_3):
    ttnn_add_44 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_68 = ttnn.to_memory_config(
        ttnn_add_44,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_10 = ttnn.sum(
        ttnn_to_memory_config_68,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_40 = ttnn.multiply(
        ttnn_sum_10,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_69 = ttnn.to_memory_config(
        ttnn_multiply_40,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_272 = ttnn.reshape(
        ttnn_to_memory_config_69,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_5 = ttnn.neg(
        ttnn_reshape_272,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_45 = ttnn.add(
        ttnn_to_memory_config_68,
        ttnn_neg_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_70 = ttnn.to_memory_config(
        ttnn_add_45,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_41 = ttnn.multiply(
        ttnn_to_memory_config_70,
        ttnn_to_memory_config_70,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_71 = ttnn.to_memory_config(
        ttnn_multiply_41,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_11 = ttnn.sum(
        ttnn_to_memory_config_71,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_42 = ttnn.multiply(
        ttnn_sum_11,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_46 = ttnn.add(
        ttnn_multiply_42,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_72 = ttnn.to_memory_config(
        ttnn_add_46,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_5 = ttnn.rsqrt(
        ttnn_to_memory_config_72,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_273 = ttnn.reshape(
        ttnn_rsqrt_5,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_44, False)
    # ttnn.deallocate(ttnn_to_memory_config_68, False)
    ttnn.deallocate(ttnn_sum_10, False)
    ttnn.deallocate(ttnn_multiply_40, False)
    ttnn.deallocate(ttnn_to_memory_config_69, False)
    ttnn.deallocate(ttnn_reshape_272, False)
    ttnn.deallocate(ttnn_neg_5, False)
    ttnn.deallocate(ttnn_add_45, False)
    # ttnn.deallocate(ttnn_to_memory_config_70, False)
    ttnn.deallocate(ttnn_multiply_41, False)
    ttnn.deallocate(ttnn_to_memory_config_71, False)
    ttnn.deallocate(ttnn_sum_11, False)
    ttnn.deallocate(ttnn_multiply_42, False)
    ttnn.deallocate(ttnn_add_46, False)
    ttnn.deallocate(ttnn_to_memory_config_72, False)
    ttnn.deallocate(ttnn_rsqrt_5, False)
    # ttnn.deallocate(ttnn_reshape_273, False)
    return ttnn_to_memory_config_68, ttnn_to_memory_config_70, ttnn_reshape_273


def QuickGELUActivation_84_0(input_0, input_1):
    ttnn_multiply_43 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_73 = ttnn.to_memory_config(
        ttnn_multiply_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_5 = ttnn.sigmoid(
        ttnn_to_memory_config_73,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_44 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_43, False)
    ttnn.deallocate(ttnn_to_memory_config_73, False)
    ttnn.deallocate(ttnn_sigmoid_5, False)
    return ttnn_multiply_44


def LayerNorm_58_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_45 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_46 = ttnn.multiply(
        ttnn_multiply_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_47 = ttnn.add(
        ttnn_multiply_46,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_45, False)
    ttnn.deallocate(ttnn_multiply_46, False)
    return ttnn_add_47


def CLIPEncoderLayer_26_0(input_0, input_1, input_2, input_3):
    ttnn_add_48 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_74 = ttnn.to_memory_config(
        ttnn_add_48,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_12 = ttnn.sum(
        ttnn_to_memory_config_74,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_47 = ttnn.multiply(
        ttnn_sum_12,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_75 = ttnn.to_memory_config(
        ttnn_multiply_47,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_274 = ttnn.reshape(
        ttnn_to_memory_config_75,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_6 = ttnn.neg(
        ttnn_reshape_274,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_49 = ttnn.add(
        ttnn_to_memory_config_74,
        ttnn_neg_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_76 = ttnn.to_memory_config(
        ttnn_add_49,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_48 = ttnn.multiply(
        ttnn_to_memory_config_76,
        ttnn_to_memory_config_76,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_77 = ttnn.to_memory_config(
        ttnn_multiply_48,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_13 = ttnn.sum(
        ttnn_to_memory_config_77,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_49 = ttnn.multiply(
        ttnn_sum_13,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_50 = ttnn.add(
        ttnn_multiply_49,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_78 = ttnn.to_memory_config(
        ttnn_add_50,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_6 = ttnn.rsqrt(
        ttnn_to_memory_config_78,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_275 = ttnn.reshape(
        ttnn_rsqrt_6,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_48, False)
    # ttnn.deallocate(ttnn_to_memory_config_74, False)
    ttnn.deallocate(ttnn_sum_12, False)
    ttnn.deallocate(ttnn_multiply_47, False)
    ttnn.deallocate(ttnn_to_memory_config_75, False)
    ttnn.deallocate(ttnn_reshape_274, False)
    ttnn.deallocate(ttnn_neg_6, False)
    ttnn.deallocate(ttnn_add_49, False)
    # ttnn.deallocate(ttnn_to_memory_config_76, False)
    ttnn.deallocate(ttnn_multiply_48, False)
    ttnn.deallocate(ttnn_to_memory_config_77, False)
    ttnn.deallocate(ttnn_sum_13, False)
    ttnn.deallocate(ttnn_multiply_49, False)
    ttnn.deallocate(ttnn_add_50, False)
    ttnn.deallocate(ttnn_to_memory_config_78, False)
    ttnn.deallocate(ttnn_rsqrt_6, False)
    # ttnn.deallocate(ttnn_reshape_275, False)
    return ttnn_to_memory_config_76, ttnn_to_memory_config_74, ttnn_reshape_275


def Linear_67_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_26 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_276 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_26,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_79 = ttnn.to_memory_config(
        ttnn_reshape_276,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_24 = ttnn.matmul(
        ttnn_to_memory_config_79,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_51 = ttnn.add(
        ttnn_matmul_24,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_80 = ttnn.to_memory_config(
        ttnn_add_51,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_277 = ttnn.reshape(
        ttnn_to_memory_config_80,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_26, False)
    ttnn.deallocate(ttnn_reshape_276, False)
    ttnn.deallocate(ttnn_to_memory_config_79, False)
    ttnn.deallocate(ttnn_matmul_24, False)
    ttnn.deallocate(ttnn_add_51, False)
    ttnn.deallocate(ttnn_to_memory_config_80, False)
    return ttnn_reshape_277


def LayerNorm_40_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_50 = ttnn.multiply(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_51 = ttnn.multiply(
        ttnn_multiply_50,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_52 = ttnn.add(
        ttnn_multiply_51,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_81 = ttnn.to_memory_config(
        ttnn_add_52,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_50, False)
    ttnn.deallocate(ttnn_multiply_51, False)
    return ttnn_add_52, ttnn_to_memory_config_81


def CLIPEncoderLayer_32_0(input_0, input_1, input_2, input_3):
    ttnn_add_53 = ttnn.add(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_82 = ttnn.to_memory_config(
        ttnn_add_53,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_14 = ttnn.sum(
        ttnn_to_memory_config_82,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_52 = ttnn.multiply(
        ttnn_sum_14,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_83 = ttnn.to_memory_config(
        ttnn_multiply_52,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_278 = ttnn.reshape(
        ttnn_to_memory_config_83,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_7 = ttnn.neg(
        ttnn_reshape_278,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_54 = ttnn.add(
        ttnn_to_memory_config_82,
        ttnn_neg_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_84 = ttnn.to_memory_config(
        ttnn_add_54,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_53 = ttnn.multiply(
        ttnn_to_memory_config_84,
        ttnn_to_memory_config_84,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_85 = ttnn.to_memory_config(
        ttnn_multiply_53,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_15 = ttnn.sum(
        ttnn_to_memory_config_85,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_sum_15,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_55 = ttnn.add(
        ttnn_multiply_54,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_86 = ttnn.to_memory_config(
        ttnn_add_55,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_7 = ttnn.rsqrt(
        ttnn_to_memory_config_86,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_279 = ttnn.reshape(
        ttnn_rsqrt_7,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_53, False)
    # ttnn.deallocate(ttnn_to_memory_config_82, False)
    ttnn.deallocate(ttnn_sum_14, False)
    ttnn.deallocate(ttnn_multiply_52, False)
    ttnn.deallocate(ttnn_to_memory_config_83, False)
    ttnn.deallocate(ttnn_reshape_278, False)
    ttnn.deallocate(ttnn_neg_7, False)
    ttnn.deallocate(ttnn_add_54, False)
    # ttnn.deallocate(ttnn_to_memory_config_84, False)
    ttnn.deallocate(ttnn_multiply_53, False)
    ttnn.deallocate(ttnn_to_memory_config_85, False)
    ttnn.deallocate(ttnn_sum_15, False)
    ttnn.deallocate(ttnn_multiply_54, False)
    ttnn.deallocate(ttnn_add_55, False)
    ttnn.deallocate(ttnn_to_memory_config_86, False)
    ttnn.deallocate(ttnn_rsqrt_7, False)
    # ttnn.deallocate(ttnn_reshape_279, False)
    return ttnn_reshape_279, ttnn_to_memory_config_82, ttnn_to_memory_config_84


def Linear_19_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_27 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_280 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_27,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_87 = ttnn.to_memory_config(
        ttnn_reshape_280,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_to_memory_config_87,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_56 = ttnn.add(
        ttnn_matmul_25,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_88 = ttnn.to_memory_config(
        ttnn_add_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_281 = ttnn.reshape(
        ttnn_to_memory_config_88,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_27, False)
    ttnn.deallocate(ttnn_reshape_280, False)
    ttnn.deallocate(ttnn_to_memory_config_87, False)
    ttnn.deallocate(ttnn_matmul_25, False)
    ttnn.deallocate(ttnn_add_56, False)
    ttnn.deallocate(ttnn_to_memory_config_88, False)
    return ttnn_reshape_281


def CLIPEncoderLayer_50_0(input_0, input_1, input_2, input_3):
    ttnn_add_57 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_89 = ttnn.to_memory_config(
        ttnn_add_57,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_16 = ttnn.sum(
        ttnn_to_memory_config_89,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_sum_16,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_90 = ttnn.to_memory_config(
        ttnn_multiply_55,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_282 = ttnn.reshape(
        ttnn_to_memory_config_90,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_8 = ttnn.neg(
        ttnn_reshape_282,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_58 = ttnn.add(
        ttnn_to_memory_config_89,
        ttnn_neg_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_91 = ttnn.to_memory_config(
        ttnn_add_58,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_56 = ttnn.multiply(
        ttnn_to_memory_config_91,
        ttnn_to_memory_config_91,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_92 = ttnn.to_memory_config(
        ttnn_multiply_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_17 = ttnn.sum(
        ttnn_to_memory_config_92,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_57 = ttnn.multiply(
        ttnn_sum_17,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_59 = ttnn.add(
        ttnn_multiply_57,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_93 = ttnn.to_memory_config(
        ttnn_add_59,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_8 = ttnn.rsqrt(
        ttnn_to_memory_config_93,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_283 = ttnn.reshape(
        ttnn_rsqrt_8,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_57, False)
    # ttnn.deallocate(ttnn_to_memory_config_89, False)
    ttnn.deallocate(ttnn_sum_16, False)
    ttnn.deallocate(ttnn_multiply_55, False)
    ttnn.deallocate(ttnn_to_memory_config_90, False)
    ttnn.deallocate(ttnn_reshape_282, False)
    ttnn.deallocate(ttnn_neg_8, False)
    ttnn.deallocate(ttnn_add_58, False)
    # ttnn.deallocate(ttnn_to_memory_config_91, False)
    ttnn.deallocate(ttnn_multiply_56, False)
    ttnn.deallocate(ttnn_to_memory_config_92, False)
    ttnn.deallocate(ttnn_sum_17, False)
    ttnn.deallocate(ttnn_multiply_57, False)
    ttnn.deallocate(ttnn_add_59, False)
    ttnn.deallocate(ttnn_to_memory_config_93, False)
    ttnn.deallocate(ttnn_rsqrt_8, False)
    # ttnn.deallocate(ttnn_reshape_283, False)
    return ttnn_to_memory_config_91, ttnn_reshape_283, ttnn_to_memory_config_89


def Linear_29_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_26 = ttnn.matmul(
        input_7,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_94 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_60 = ttnn.add(
        ttnn_matmul_26,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_95 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_to_memory_config_94,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_96 = ttnn.to_memory_config(
        ttnn_add_60,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_28 = ttnn.matmul(
        ttnn_to_memory_config_95,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_61 = ttnn.add(
        ttnn_matmul_27,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_62 = ttnn.add(
        ttnn_matmul_28,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_284 = ttnn.reshape(
        ttnn_to_memory_config_96,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_97 = ttnn.to_memory_config(
        ttnn_add_61,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_98 = ttnn.to_memory_config(
        ttnn_add_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_65 = ttnn.permute(
        ttnn_reshape_284,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_285 = ttnn.reshape(
        ttnn_to_memory_config_97,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_286 = ttnn.reshape(
        ttnn_to_memory_config_98,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_287 = ttnn.reshape(
        ttnn_permute_65,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_66 = ttnn.permute(
        ttnn_reshape_285,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_67 = ttnn.permute(
        ttnn_reshape_286,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_288 = ttnn.reshape(
        ttnn_permute_66,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_289 = ttnn.reshape(
        ttnn_permute_67,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_7, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_26, False)
    ttnn.deallocate(ttnn_add_60, False)
    ttnn.deallocate(ttnn_to_memory_config_94, False)
    ttnn.deallocate(ttnn_to_memory_config_95, False)
    ttnn.deallocate(ttnn_matmul_28, False)
    ttnn.deallocate(ttnn_to_memory_config_96, False)
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn.deallocate(ttnn_reshape_284, False)
    ttnn.deallocate(ttnn_add_62, False)
    ttnn.deallocate(ttnn_add_61, False)
    ttnn.deallocate(ttnn_to_memory_config_97, False)
    ttnn.deallocate(ttnn_permute_65, False)
    ttnn.deallocate(ttnn_to_memory_config_98, False)
    ttnn.deallocate(ttnn_reshape_285, False)
    ttnn.deallocate(ttnn_reshape_286, False)
    ttnn.deallocate(ttnn_permute_66, False)
    ttnn.deallocate(ttnn_permute_67, False)
    return ttnn_reshape_288, ttnn_reshape_289, ttnn_reshape_287


def Linear_61_0(input_0, input_1, input_2):
    ttnn_matmul_29 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_63 = ttnn.add(
        ttnn_matmul_29,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_99 = ttnn.to_memory_config(
        ttnn_add_63,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_290 = ttnn.reshape(
        ttnn_to_memory_config_99,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_29, False)
    ttnn.deallocate(ttnn_add_63, False)
    ttnn.deallocate(ttnn_to_memory_config_99, False)
    return ttnn_reshape_290


def CLIPAttention_30_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_100 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_to_memory_config_100,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_101 = ttnn.to_memory_config(
        ttnn_matmul_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_291 = ttnn.reshape(
        ttnn_to_memory_config_101,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_58 = ttnn.multiply(
        ttnn_reshape_291,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_multiply_58,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_1 = ttnn.softmax(
        ttnn_typecast_3,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_softmax_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_292 = ttnn.reshape(
        ttnn_typecast_4,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_102 = ttnn.to_memory_config(
        ttnn_reshape_292,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_to_memory_config_102,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_103 = ttnn.to_memory_config(
        ttnn_matmul_31,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_293 = ttnn.reshape(
        ttnn_to_memory_config_103,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_100, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_101, False)
    ttnn.deallocate(ttnn_reshape_291, False)
    ttnn.deallocate(ttnn_multiply_58, False)
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn.deallocate(ttnn_softmax_1, False)
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn.deallocate(ttnn_reshape_292, False)
    ttnn.deallocate(ttnn_to_memory_config_102, False)
    ttnn.deallocate(ttnn_matmul_31, False)
    ttnn.deallocate(ttnn_to_memory_config_103, False)
    return ttnn_reshape_293


def Linear_41_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_32 = ttnn.matmul(
        input_0,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_104 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_64 = ttnn.add(
        ttnn_matmul_32,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_105 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_to_memory_config_104,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_106 = ttnn.to_memory_config(
        ttnn_add_64,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_34 = ttnn.matmul(
        ttnn_to_memory_config_105,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_65 = ttnn.add(
        ttnn_matmul_33,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_66 = ttnn.add(
        ttnn_matmul_34,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_294 = ttnn.reshape(
        ttnn_to_memory_config_106,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_107 = ttnn.to_memory_config(
        ttnn_add_65,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_108 = ttnn.to_memory_config(
        ttnn_add_66,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_68 = ttnn.permute(
        ttnn_reshape_294,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_295 = ttnn.reshape(
        ttnn_to_memory_config_107,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_296 = ttnn.reshape(
        ttnn_to_memory_config_108,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_297 = ttnn.reshape(
        ttnn_permute_68,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_69 = ttnn.permute(
        ttnn_reshape_295,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_70 = ttnn.permute(
        ttnn_reshape_296,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_298 = ttnn.reshape(
        ttnn_permute_69,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_299 = ttnn.reshape(
        ttnn_permute_70,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn.deallocate(ttnn_add_64, False)
    ttnn.deallocate(ttnn_to_memory_config_104, False)
    ttnn.deallocate(ttnn_to_memory_config_105, False)
    ttnn.deallocate(ttnn_matmul_34, False)
    ttnn.deallocate(ttnn_to_memory_config_106, False)
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn.deallocate(ttnn_reshape_294, False)
    ttnn.deallocate(ttnn_add_66, False)
    ttnn.deallocate(ttnn_add_65, False)
    ttnn.deallocate(ttnn_to_memory_config_107, False)
    ttnn.deallocate(ttnn_permute_68, False)
    ttnn.deallocate(ttnn_to_memory_config_108, False)
    ttnn.deallocate(ttnn_reshape_295, False)
    ttnn.deallocate(ttnn_reshape_296, False)
    ttnn.deallocate(ttnn_permute_69, False)
    ttnn.deallocate(ttnn_permute_70, False)
    return ttnn_reshape_297, ttnn_reshape_298, ttnn_reshape_299


def Linear_97_0(input_0, input_1, input_2):
    ttnn_matmul_35 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_67 = ttnn.add(
        ttnn_matmul_35,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_109 = ttnn.to_memory_config(
        ttnn_add_67,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_300 = ttnn.reshape(
        ttnn_to_memory_config_109,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_35, False)
    ttnn.deallocate(ttnn_add_67, False)
    ttnn.deallocate(ttnn_to_memory_config_109, False)
    return ttnn_reshape_300


def CLIPEncoderLayer_44_0(input_0, input_1, input_2, input_3):
    ttnn_add_68 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_110 = ttnn.to_memory_config(
        ttnn_add_68,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_18 = ttnn.sum(
        ttnn_to_memory_config_110,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_sum_18,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_111 = ttnn.to_memory_config(
        ttnn_multiply_59,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_301 = ttnn.reshape(
        ttnn_to_memory_config_111,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_9 = ttnn.neg(
        ttnn_reshape_301,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_69 = ttnn.add(
        ttnn_to_memory_config_110,
        ttnn_neg_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_112 = ttnn.to_memory_config(
        ttnn_add_69,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_60 = ttnn.multiply(
        ttnn_to_memory_config_112,
        ttnn_to_memory_config_112,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_113 = ttnn.to_memory_config(
        ttnn_multiply_60,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_19 = ttnn.sum(
        ttnn_to_memory_config_113,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_61 = ttnn.multiply(
        ttnn_sum_19,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_70 = ttnn.add(
        ttnn_multiply_61,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_114 = ttnn.to_memory_config(
        ttnn_add_70,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_9 = ttnn.rsqrt(
        ttnn_to_memory_config_114,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_302 = ttnn.reshape(
        ttnn_rsqrt_9,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_68, False)
    # ttnn.deallocate(ttnn_to_memory_config_110, False)
    ttnn.deallocate(ttnn_sum_18, False)
    ttnn.deallocate(ttnn_multiply_59, False)
    ttnn.deallocate(ttnn_to_memory_config_111, False)
    ttnn.deallocate(ttnn_reshape_301, False)
    ttnn.deallocate(ttnn_neg_9, False)
    ttnn.deallocate(ttnn_add_69, False)
    # ttnn.deallocate(ttnn_to_memory_config_112, False)
    ttnn.deallocate(ttnn_multiply_60, False)
    ttnn.deallocate(ttnn_to_memory_config_113, False)
    ttnn.deallocate(ttnn_sum_19, False)
    ttnn.deallocate(ttnn_multiply_61, False)
    ttnn.deallocate(ttnn_add_70, False)
    ttnn.deallocate(ttnn_to_memory_config_114, False)
    ttnn.deallocate(ttnn_rsqrt_9, False)
    # ttnn.deallocate(ttnn_reshape_302, False)
    return ttnn_reshape_302, ttnn_to_memory_config_112, ttnn_to_memory_config_110


def CLIPEncoderLayer_68_0(input_0, input_1, input_2, input_3):
    ttnn_add_71 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_115 = ttnn.to_memory_config(
        ttnn_add_71,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_20 = ttnn.sum(
        ttnn_to_memory_config_115,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_62 = ttnn.multiply(
        ttnn_sum_20,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_116 = ttnn.to_memory_config(
        ttnn_multiply_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_303 = ttnn.reshape(
        ttnn_to_memory_config_116,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_10 = ttnn.neg(
        ttnn_reshape_303,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_72 = ttnn.add(
        ttnn_to_memory_config_115,
        ttnn_neg_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_117 = ttnn.to_memory_config(
        ttnn_add_72,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_63 = ttnn.multiply(
        ttnn_to_memory_config_117,
        ttnn_to_memory_config_117,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_118 = ttnn.to_memory_config(
        ttnn_multiply_63,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_21 = ttnn.sum(
        ttnn_to_memory_config_118,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_64 = ttnn.multiply(
        ttnn_sum_21,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_73 = ttnn.add(
        ttnn_multiply_64,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_119 = ttnn.to_memory_config(
        ttnn_add_73,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_10 = ttnn.rsqrt(
        ttnn_to_memory_config_119,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_304 = ttnn.reshape(
        ttnn_rsqrt_10,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_71, False)
    # ttnn.deallocate(ttnn_to_memory_config_115, False)
    ttnn.deallocate(ttnn_sum_20, False)
    ttnn.deallocate(ttnn_multiply_62, False)
    ttnn.deallocate(ttnn_to_memory_config_116, False)
    ttnn.deallocate(ttnn_reshape_303, False)
    ttnn.deallocate(ttnn_neg_10, False)
    ttnn.deallocate(ttnn_add_72, False)
    # ttnn.deallocate(ttnn_to_memory_config_117, False)
    ttnn.deallocate(ttnn_multiply_63, False)
    ttnn.deallocate(ttnn_to_memory_config_118, False)
    ttnn.deallocate(ttnn_sum_21, False)
    ttnn.deallocate(ttnn_multiply_64, False)
    ttnn.deallocate(ttnn_add_73, False)
    ttnn.deallocate(ttnn_to_memory_config_119, False)
    ttnn.deallocate(ttnn_rsqrt_10, False)
    # ttnn.deallocate(ttnn_reshape_304, False)
    return ttnn_reshape_304, ttnn_to_memory_config_115, ttnn_to_memory_config_117


def CLIPEncoderLayer_116_0(input_0, input_1, input_2, input_3):
    ttnn_add_74 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_120 = ttnn.to_memory_config(
        ttnn_add_74,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_22 = ttnn.sum(
        ttnn_to_memory_config_120,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_65 = ttnn.multiply(
        ttnn_sum_22,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_121 = ttnn.to_memory_config(
        ttnn_multiply_65,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_305 = ttnn.reshape(
        ttnn_to_memory_config_121,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_11 = ttnn.neg(
        ttnn_reshape_305,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_75 = ttnn.add(
        ttnn_to_memory_config_120,
        ttnn_neg_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_122 = ttnn.to_memory_config(
        ttnn_add_75,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_66 = ttnn.multiply(
        ttnn_to_memory_config_122,
        ttnn_to_memory_config_122,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_123 = ttnn.to_memory_config(
        ttnn_multiply_66,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_23 = ttnn.sum(
        ttnn_to_memory_config_123,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_67 = ttnn.multiply(
        ttnn_sum_23,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_76 = ttnn.add(
        ttnn_multiply_67,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_124 = ttnn.to_memory_config(
        ttnn_add_76,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_11 = ttnn.rsqrt(
        ttnn_to_memory_config_124,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_306 = ttnn.reshape(
        ttnn_rsqrt_11,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_74, False)
    # ttnn.deallocate(ttnn_to_memory_config_120, False)
    ttnn.deallocate(ttnn_sum_22, False)
    ttnn.deallocate(ttnn_multiply_65, False)
    ttnn.deallocate(ttnn_to_memory_config_121, False)
    ttnn.deallocate(ttnn_reshape_305, False)
    ttnn.deallocate(ttnn_neg_11, False)
    ttnn.deallocate(ttnn_add_75, False)
    # ttnn.deallocate(ttnn_to_memory_config_122, False)
    ttnn.deallocate(ttnn_multiply_66, False)
    ttnn.deallocate(ttnn_to_memory_config_123, False)
    ttnn.deallocate(ttnn_sum_23, False)
    ttnn.deallocate(ttnn_multiply_67, False)
    ttnn.deallocate(ttnn_add_76, False)
    ttnn.deallocate(ttnn_to_memory_config_124, False)
    ttnn.deallocate(ttnn_rsqrt_11, False)
    # ttnn.deallocate(ttnn_reshape_306, False)
    return ttnn_to_memory_config_120, ttnn_to_memory_config_122, ttnn_reshape_306


def Linear_5_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_36 = ttnn.matmul(
        input_3,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_125 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_77 = ttnn.add(
        ttnn_matmul_36,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_126 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_to_memory_config_125,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_127 = ttnn.to_memory_config(
        ttnn_add_77,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_38 = ttnn.matmul(
        ttnn_to_memory_config_126,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_78 = ttnn.add(
        ttnn_matmul_37,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_79 = ttnn.add(
        ttnn_matmul_38,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_307 = ttnn.reshape(
        ttnn_to_memory_config_127,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_128 = ttnn.to_memory_config(
        ttnn_add_78,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_129 = ttnn.to_memory_config(
        ttnn_add_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_71 = ttnn.permute(
        ttnn_reshape_307,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_308 = ttnn.reshape(
        ttnn_to_memory_config_128,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_309 = ttnn.reshape(
        ttnn_to_memory_config_129,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_310 = ttnn.reshape(
        ttnn_permute_71,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_72 = ttnn.permute(
        ttnn_reshape_308,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_73 = ttnn.permute(
        ttnn_reshape_309,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_311 = ttnn.reshape(
        ttnn_permute_72,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_312 = ttnn.reshape(
        ttnn_permute_73,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_36, False)
    ttnn.deallocate(ttnn_add_77, False)
    ttnn.deallocate(ttnn_to_memory_config_125, False)
    ttnn.deallocate(ttnn_to_memory_config_126, False)
    ttnn.deallocate(ttnn_matmul_38, False)
    ttnn.deallocate(ttnn_to_memory_config_127, False)
    ttnn.deallocate(ttnn_matmul_37, False)
    ttnn.deallocate(ttnn_reshape_307, False)
    ttnn.deallocate(ttnn_add_79, False)
    ttnn.deallocate(ttnn_add_78, False)
    ttnn.deallocate(ttnn_to_memory_config_128, False)
    ttnn.deallocate(ttnn_permute_71, False)
    ttnn.deallocate(ttnn_to_memory_config_129, False)
    ttnn.deallocate(ttnn_reshape_308, False)
    ttnn.deallocate(ttnn_reshape_309, False)
    ttnn.deallocate(ttnn_permute_72, False)
    ttnn.deallocate(ttnn_permute_73, False)
    return ttnn_reshape_310, ttnn_reshape_312, ttnn_reshape_311


def LayerNorm_10_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_68 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_69 = ttnn.multiply(
        ttnn_multiply_68,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_80 = ttnn.add(
        ttnn_multiply_69,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_68, False)
    ttnn.deallocate(ttnn_multiply_69, False)
    return ttnn_add_80


def Linear_93_0(input):
    ttnn_reshape_313 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_313


def Linear_115_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_28 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_314 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_28,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_130 = ttnn.to_memory_config(
        ttnn_reshape_314,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_39 = ttnn.matmul(
        ttnn_to_memory_config_130,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_81 = ttnn.add(
        ttnn_matmul_39,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_131 = ttnn.to_memory_config(
        ttnn_add_81,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_315 = ttnn.reshape(
        ttnn_to_memory_config_131,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_28, False)
    ttnn.deallocate(ttnn_reshape_314, False)
    ttnn.deallocate(ttnn_to_memory_config_130, False)
    ttnn.deallocate(ttnn_matmul_39, False)
    ttnn.deallocate(ttnn_add_81, False)
    ttnn.deallocate(ttnn_to_memory_config_131, False)
    return ttnn_reshape_315


def CLIPEncoderLayer_110_0(input_0, input_1, input_2, input_3):
    ttnn_add_82 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_132 = ttnn.to_memory_config(
        ttnn_add_82,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_24 = ttnn.sum(
        ttnn_to_memory_config_132,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_70 = ttnn.multiply(
        ttnn_sum_24,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_133 = ttnn.to_memory_config(
        ttnn_multiply_70,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_316 = ttnn.reshape(
        ttnn_to_memory_config_133,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_12 = ttnn.neg(
        ttnn_reshape_316,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_83 = ttnn.add(
        ttnn_to_memory_config_132,
        ttnn_neg_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_134 = ttnn.to_memory_config(
        ttnn_add_83,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_71 = ttnn.multiply(
        ttnn_to_memory_config_134,
        ttnn_to_memory_config_134,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_135 = ttnn.to_memory_config(
        ttnn_multiply_71,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_25 = ttnn.sum(
        ttnn_to_memory_config_135,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_72 = ttnn.multiply(
        ttnn_sum_25,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_84 = ttnn.add(
        ttnn_multiply_72,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_136 = ttnn.to_memory_config(
        ttnn_add_84,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_12 = ttnn.rsqrt(
        ttnn_to_memory_config_136,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_317 = ttnn.reshape(
        ttnn_rsqrt_12,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_82, False)
    # ttnn.deallocate(ttnn_to_memory_config_132, False)
    ttnn.deallocate(ttnn_sum_24, False)
    ttnn.deallocate(ttnn_multiply_70, False)
    ttnn.deallocate(ttnn_to_memory_config_133, False)
    ttnn.deallocate(ttnn_reshape_316, False)
    ttnn.deallocate(ttnn_neg_12, False)
    ttnn.deallocate(ttnn_add_83, False)
    # ttnn.deallocate(ttnn_to_memory_config_134, False)
    ttnn.deallocate(ttnn_multiply_71, False)
    ttnn.deallocate(ttnn_to_memory_config_135, False)
    ttnn.deallocate(ttnn_sum_25, False)
    ttnn.deallocate(ttnn_multiply_72, False)
    ttnn.deallocate(ttnn_add_84, False)
    ttnn.deallocate(ttnn_to_memory_config_136, False)
    ttnn.deallocate(ttnn_rsqrt_12, False)
    # ttnn.deallocate(ttnn_reshape_317, False)
    return ttnn_to_memory_config_132, ttnn_reshape_317, ttnn_to_memory_config_134


def Linear_71_0(input_0, input_1, input_2):
    ttnn_matmul_40 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_85 = ttnn.add(
        ttnn_matmul_40,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_137 = ttnn.to_memory_config(
        ttnn_add_85,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_40, False)
    ttnn.deallocate(ttnn_add_85, False)
    return ttnn_to_memory_config_137


def QuickGELUActivation_120_0(input_0, input_1):
    ttnn_multiply_73 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_138 = ttnn.to_memory_config(
        ttnn_multiply_73,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_6 = ttnn.sigmoid(
        ttnn_to_memory_config_138,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_74 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_73, False)
    ttnn.deallocate(ttnn_to_memory_config_138, False)
    ttnn.deallocate(ttnn_sigmoid_6, False)
    return ttnn_multiply_74


def Linear_59_0(input_0, input_1, input_2):
    ttnn_matmul_41 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_86 = ttnn.add(
        ttnn_matmul_41,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_139 = ttnn.to_memory_config(
        ttnn_add_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_41, False)
    ttnn.deallocate(ttnn_add_86, False)
    return ttnn_to_memory_config_139


def Linear_129_0(input):
    ttnn_reshape_318 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_318


def CLIPAttention_102_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_140 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_42 = ttnn.matmul(
        ttnn_to_memory_config_140,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_141 = ttnn.to_memory_config(
        ttnn_matmul_42,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_319 = ttnn.reshape(
        ttnn_to_memory_config_141,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_75 = ttnn.multiply(
        ttnn_reshape_319,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_multiply_75,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_2 = ttnn.softmax(
        ttnn_typecast_5,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_softmax_2,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_320 = ttnn.reshape(
        ttnn_typecast_6,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_142 = ttnn.to_memory_config(
        ttnn_reshape_320,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_43 = ttnn.matmul(
        ttnn_to_memory_config_142,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_143 = ttnn.to_memory_config(
        ttnn_matmul_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_321 = ttnn.reshape(
        ttnn_to_memory_config_143,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_140, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_42, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_141, False)
    ttnn.deallocate(ttnn_reshape_319, False)
    ttnn.deallocate(ttnn_multiply_75, False)
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn.deallocate(ttnn_softmax_2, False)
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn.deallocate(ttnn_reshape_320, False)
    ttnn.deallocate(ttnn_to_memory_config_142, False)
    ttnn.deallocate(ttnn_matmul_43, False)
    ttnn.deallocate(ttnn_to_memory_config_143, False)
    return ttnn_reshape_321


def Linear_13_0(input_0, input_1, input_2):
    ttnn_matmul_44 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_87 = ttnn.add(
        ttnn_matmul_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_144 = ttnn.to_memory_config(
        ttnn_add_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_322 = ttnn.reshape(
        ttnn_to_memory_config_144,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_44, False)
    ttnn.deallocate(ttnn_add_87, False)
    ttnn.deallocate(ttnn_to_memory_config_144, False)
    return ttnn_reshape_322


def Linear_11_0(input_0, input_1, input_2):
    ttnn_matmul_45 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_88 = ttnn.add(
        ttnn_matmul_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_145 = ttnn.to_memory_config(
        ttnn_add_88,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_45, False)
    ttnn.deallocate(ttnn_add_88, False)
    return ttnn_to_memory_config_145


def Linear_125_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_46 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_146 = ttnn.to_memory_config(
        input_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_89 = ttnn.add(
        ttnn_matmul_46,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_147 = ttnn.to_memory_config(
        input_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_47 = ttnn.matmul(
        ttnn_to_memory_config_146,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_148 = ttnn.to_memory_config(
        ttnn_add_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_48 = ttnn.matmul(
        ttnn_to_memory_config_147,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_90 = ttnn.add(
        ttnn_matmul_47,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_91 = ttnn.add(
        ttnn_matmul_48,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_323 = ttnn.reshape(
        ttnn_to_memory_config_148,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_149 = ttnn.to_memory_config(
        ttnn_add_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_150 = ttnn.to_memory_config(
        ttnn_add_91,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_74 = ttnn.permute(
        ttnn_reshape_323,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_324 = ttnn.reshape(
        ttnn_to_memory_config_149,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_325 = ttnn.reshape(
        ttnn_to_memory_config_150,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_326 = ttnn.reshape(
        ttnn_permute_74,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_75 = ttnn.permute(
        ttnn_reshape_324,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_76 = ttnn.permute(
        ttnn_reshape_325,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_327 = ttnn.reshape(
        ttnn_permute_75,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_328 = ttnn.reshape(
        ttnn_permute_76,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_4, False)
    ttnn.deallocate(ttnn_matmul_46, False)
    ttnn.deallocate(ttnn_add_89, False)
    ttnn.deallocate(ttnn_to_memory_config_146, False)
    ttnn.deallocate(ttnn_to_memory_config_147, False)
    ttnn.deallocate(ttnn_matmul_48, False)
    ttnn.deallocate(ttnn_to_memory_config_148, False)
    ttnn.deallocate(ttnn_matmul_47, False)
    ttnn.deallocate(ttnn_reshape_323, False)
    ttnn.deallocate(ttnn_add_91, False)
    ttnn.deallocate(ttnn_add_90, False)
    ttnn.deallocate(ttnn_to_memory_config_149, False)
    ttnn.deallocate(ttnn_permute_74, False)
    ttnn.deallocate(ttnn_to_memory_config_150, False)
    ttnn.deallocate(ttnn_reshape_324, False)
    ttnn.deallocate(ttnn_reshape_325, False)
    ttnn.deallocate(ttnn_permute_75, False)
    ttnn.deallocate(ttnn_permute_76, False)
    return ttnn_reshape_328, ttnn_reshape_327, ttnn_reshape_326


def Linear_87_0(input):
    ttnn_reshape_329 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_329


def CLIPAttention_78_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_151 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_49 = ttnn.matmul(
        ttnn_to_memory_config_151,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_152 = ttnn.to_memory_config(
        ttnn_matmul_49,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_330 = ttnn.reshape(
        ttnn_to_memory_config_152,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_76 = ttnn.multiply(
        ttnn_reshape_330,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_multiply_76,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_3 = ttnn.softmax(
        ttnn_typecast_7,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_softmax_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_331 = ttnn.reshape(
        ttnn_typecast_8,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_153 = ttnn.to_memory_config(
        ttnn_reshape_331,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_50 = ttnn.matmul(
        ttnn_to_memory_config_153,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_154 = ttnn.to_memory_config(
        ttnn_matmul_50,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_332 = ttnn.reshape(
        ttnn_to_memory_config_154,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_151, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_49, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_152, False)
    ttnn.deallocate(ttnn_reshape_330, False)
    ttnn.deallocate(ttnn_multiply_76, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn.deallocate(ttnn_softmax_3, False)
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn.deallocate(ttnn_reshape_331, False)
    ttnn.deallocate(ttnn_to_memory_config_153, False)
    ttnn.deallocate(ttnn_matmul_50, False)
    ttnn.deallocate(ttnn_to_memory_config_154, False)
    return ttnn_reshape_332


def Linear_63_0(input):
    ttnn_reshape_333 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_333


def Linear_75_0(input):
    ttnn_reshape_334 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_334


def LayerNorm_88_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_77 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_78 = ttnn.multiply(
        ttnn_multiply_77,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_92 = ttnn.add(
        ttnn_multiply_78,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_155 = ttnn.to_memory_config(
        ttnn_add_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_77, False)
    ttnn.deallocate(ttnn_multiply_78, False)
    return ttnn_to_memory_config_155, ttnn_add_92


def LayerNorm_1_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_sum_26 = ttnn.sum(
        input_0,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_79 = ttnn.multiply(
        ttnn_sum_26,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_156 = ttnn.to_memory_config(
        ttnn_multiply_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_335 = ttnn.reshape(
        ttnn_to_memory_config_156,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_13 = ttnn.neg(
        ttnn_reshape_335,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_93 = ttnn.add(
        input_0,
        ttnn_neg_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_157 = ttnn.to_memory_config(
        ttnn_add_93,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_80 = ttnn.multiply(
        ttnn_to_memory_config_157,
        ttnn_to_memory_config_157,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_158 = ttnn.to_memory_config(
        ttnn_multiply_80,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_27 = ttnn.sum(
        ttnn_to_memory_config_158,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_81 = ttnn.multiply(
        ttnn_sum_27,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_159 = ttnn.to_memory_config(
        ttnn_multiply_81,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_336 = ttnn.reshape(
        ttnn_to_memory_config_159,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_94 = ttnn.add(
        ttnn_reshape_336,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_160 = ttnn.to_memory_config(
        ttnn_add_94,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_13 = ttnn.rsqrt(
        ttnn_to_memory_config_160,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_82 = ttnn.multiply(
        ttnn_to_memory_config_157,
        ttnn_rsqrt_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_83 = ttnn.multiply(
        ttnn_multiply_82,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_95 = ttnn.add(
        ttnn_multiply_83,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_161 = ttnn.to_memory_config(
        ttnn_add_95,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_sum_26, False)
    ttnn.deallocate(ttnn_multiply_79, False)
    ttnn.deallocate(ttnn_to_memory_config_156, False)
    ttnn.deallocate(ttnn_reshape_335, False)
    ttnn.deallocate(ttnn_neg_13, False)
    ttnn.deallocate(ttnn_add_93, False)
    ttnn.deallocate(ttnn_to_memory_config_157, False)
    ttnn.deallocate(ttnn_multiply_80, False)
    ttnn.deallocate(ttnn_to_memory_config_158, False)
    ttnn.deallocate(ttnn_sum_27, False)
    ttnn.deallocate(ttnn_multiply_81, False)
    ttnn.deallocate(ttnn_to_memory_config_159, False)
    ttnn.deallocate(ttnn_reshape_336, False)
    ttnn.deallocate(ttnn_add_94, False)
    ttnn.deallocate(ttnn_to_memory_config_160, False)
    ttnn.deallocate(ttnn_rsqrt_13, False)
    ttnn.deallocate(ttnn_multiply_82, False)
    ttnn.deallocate(ttnn_multiply_83, False)
    ttnn.deallocate(ttnn_add_95, False)
    return ttnn_to_memory_config_161


def Linear_3_0(input):
    ttnn_reshape_337 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_337


def Linear_117_0(input):
    ttnn_reshape_338 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_338


def Linear_103_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_29 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_339 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_29,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_162 = ttnn.to_memory_config(
        ttnn_reshape_339,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_51 = ttnn.matmul(
        ttnn_to_memory_config_162,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_96 = ttnn.add(
        ttnn_matmul_51,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_163 = ttnn.to_memory_config(
        ttnn_add_96,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_340 = ttnn.reshape(
        ttnn_to_memory_config_163,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_29, False)
    ttnn.deallocate(ttnn_reshape_339, False)
    ttnn.deallocate(ttnn_to_memory_config_162, False)
    ttnn.deallocate(ttnn_matmul_51, False)
    ttnn.deallocate(ttnn_add_96, False)
    ttnn.deallocate(ttnn_to_memory_config_163, False)
    return ttnn_reshape_340


def Linear_123_0(input):
    ttnn_reshape_341 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_341


def LayerNorm_64_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_84 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_85 = ttnn.multiply(
        ttnn_multiply_84,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_97 = ttnn.add(
        ttnn_multiply_85,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_164 = ttnn.to_memory_config(
        ttnn_add_97,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_84, False)
    ttnn.deallocate(ttnn_multiply_85, False)
    return ttnn_add_97, ttnn_to_memory_config_164


def CLIPEncoderLayer_62_0(input_0, input_1, input_2, input_3):
    ttnn_add_98 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_165 = ttnn.to_memory_config(
        ttnn_add_98,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_28 = ttnn.sum(
        ttnn_to_memory_config_165,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_86 = ttnn.multiply(
        ttnn_sum_28,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_166 = ttnn.to_memory_config(
        ttnn_multiply_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_342 = ttnn.reshape(
        ttnn_to_memory_config_166,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_14 = ttnn.neg(
        ttnn_reshape_342,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_99 = ttnn.add(
        ttnn_to_memory_config_165,
        ttnn_neg_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_167 = ttnn.to_memory_config(
        ttnn_add_99,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_87 = ttnn.multiply(
        ttnn_to_memory_config_167,
        ttnn_to_memory_config_167,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_168 = ttnn.to_memory_config(
        ttnn_multiply_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_29 = ttnn.sum(
        ttnn_to_memory_config_168,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_88 = ttnn.multiply(
        ttnn_sum_29,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_100 = ttnn.add(
        ttnn_multiply_88,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_169 = ttnn.to_memory_config(
        ttnn_add_100,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_14 = ttnn.rsqrt(
        ttnn_to_memory_config_169,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_343 = ttnn.reshape(
        ttnn_rsqrt_14,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_98, False)
    # ttnn.deallocate(ttnn_to_memory_config_165, False)
    ttnn.deallocate(ttnn_sum_28, False)
    ttnn.deallocate(ttnn_multiply_86, False)
    ttnn.deallocate(ttnn_to_memory_config_166, False)
    ttnn.deallocate(ttnn_reshape_342, False)
    ttnn.deallocate(ttnn_neg_14, False)
    ttnn.deallocate(ttnn_add_99, False)
    # ttnn.deallocate(ttnn_to_memory_config_167, False)
    ttnn.deallocate(ttnn_multiply_87, False)
    ttnn.deallocate(ttnn_to_memory_config_168, False)
    ttnn.deallocate(ttnn_sum_29, False)
    ttnn.deallocate(ttnn_multiply_88, False)
    ttnn.deallocate(ttnn_add_100, False)
    ttnn.deallocate(ttnn_to_memory_config_169, False)
    ttnn.deallocate(ttnn_rsqrt_14, False)
    # ttnn.deallocate(ttnn_reshape_343, False)
    return ttnn_to_memory_config_167, ttnn_reshape_343, ttnn_to_memory_config_165


def CLIPEncoderLayer_86_0(input_0, input_1, input_2, input_3):
    ttnn_add_101 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_170 = ttnn.to_memory_config(
        ttnn_add_101,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_30 = ttnn.sum(
        ttnn_to_memory_config_170,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_89 = ttnn.multiply(
        ttnn_sum_30,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_171 = ttnn.to_memory_config(
        ttnn_multiply_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_344 = ttnn.reshape(
        ttnn_to_memory_config_171,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_15 = ttnn.neg(
        ttnn_reshape_344,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_102 = ttnn.add(
        ttnn_to_memory_config_170,
        ttnn_neg_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_172 = ttnn.to_memory_config(
        ttnn_add_102,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_90 = ttnn.multiply(
        ttnn_to_memory_config_172,
        ttnn_to_memory_config_172,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_173 = ttnn.to_memory_config(
        ttnn_multiply_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_31 = ttnn.sum(
        ttnn_to_memory_config_173,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_91 = ttnn.multiply(
        ttnn_sum_31,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_103 = ttnn.add(
        ttnn_multiply_91,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_174 = ttnn.to_memory_config(
        ttnn_add_103,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_15 = ttnn.rsqrt(
        ttnn_to_memory_config_174,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_345 = ttnn.reshape(
        ttnn_rsqrt_15,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_101, False)
    # ttnn.deallocate(ttnn_to_memory_config_170, False)
    ttnn.deallocate(ttnn_sum_30, False)
    ttnn.deallocate(ttnn_multiply_89, False)
    ttnn.deallocate(ttnn_to_memory_config_171, False)
    ttnn.deallocate(ttnn_reshape_344, False)
    ttnn.deallocate(ttnn_neg_15, False)
    ttnn.deallocate(ttnn_add_102, False)
    # ttnn.deallocate(ttnn_to_memory_config_172, False)
    ttnn.deallocate(ttnn_multiply_90, False)
    ttnn.deallocate(ttnn_to_memory_config_173, False)
    ttnn.deallocate(ttnn_sum_31, False)
    ttnn.deallocate(ttnn_multiply_91, False)
    ttnn.deallocate(ttnn_add_103, False)
    ttnn.deallocate(ttnn_to_memory_config_174, False)
    ttnn.deallocate(ttnn_rsqrt_15, False)
    # ttnn.deallocate(ttnn_reshape_345, False)
    return ttnn_to_memory_config_172, ttnn_to_memory_config_170, ttnn_reshape_345


def QuickGELUActivation_108_0(input_0, input_1):
    ttnn_multiply_92 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_175 = ttnn.to_memory_config(
        ttnn_multiply_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_7 = ttnn.sigmoid(
        ttnn_to_memory_config_175,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_93 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_92, False)
    ttnn.deallocate(ttnn_to_memory_config_175, False)
    ttnn.deallocate(ttnn_sigmoid_7, False)
    return ttnn_multiply_93


def Linear_21_0(input):
    ttnn_reshape_346 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_346


def QuickGELUActivation_144_0(input_0, input_1):
    ttnn_multiply_94 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_176 = ttnn.to_memory_config(
        ttnn_multiply_94,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_8 = ttnn.sigmoid(
        ttnn_to_memory_config_176,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_95 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_94, False)
    ttnn.deallocate(ttnn_to_memory_config_176, False)
    ttnn.deallocate(ttnn_sigmoid_8, False)
    return ttnn_multiply_95


def CLIPEncoderLayer_104_0(input_0, input_1, input_2, input_3):
    ttnn_add_104 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_177 = ttnn.to_memory_config(
        ttnn_add_104,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_32 = ttnn.sum(
        ttnn_to_memory_config_177,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_96 = ttnn.multiply(
        ttnn_sum_32,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_178 = ttnn.to_memory_config(
        ttnn_multiply_96,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_347 = ttnn.reshape(
        ttnn_to_memory_config_178,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_16 = ttnn.neg(
        ttnn_reshape_347,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_105 = ttnn.add(
        ttnn_to_memory_config_177,
        ttnn_neg_16,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_179 = ttnn.to_memory_config(
        ttnn_add_105,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_97 = ttnn.multiply(
        ttnn_to_memory_config_179,
        ttnn_to_memory_config_179,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_180 = ttnn.to_memory_config(
        ttnn_multiply_97,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_33 = ttnn.sum(
        ttnn_to_memory_config_180,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_98 = ttnn.multiply(
        ttnn_sum_33,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_106 = ttnn.add(
        ttnn_multiply_98,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_181 = ttnn.to_memory_config(
        ttnn_add_106,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_16 = ttnn.rsqrt(
        ttnn_to_memory_config_181,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_348 = ttnn.reshape(
        ttnn_rsqrt_16,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_104, False)
    # ttnn.deallocate(ttnn_to_memory_config_177, False)
    ttnn.deallocate(ttnn_sum_32, False)
    ttnn.deallocate(ttnn_multiply_96, False)
    ttnn.deallocate(ttnn_to_memory_config_178, False)
    ttnn.deallocate(ttnn_reshape_347, False)
    ttnn.deallocate(ttnn_neg_16, False)
    ttnn.deallocate(ttnn_add_105, False)
    # ttnn.deallocate(ttnn_to_memory_config_179, False)
    ttnn.deallocate(ttnn_multiply_97, False)
    ttnn.deallocate(ttnn_to_memory_config_180, False)
    ttnn.deallocate(ttnn_sum_33, False)
    ttnn.deallocate(ttnn_multiply_98, False)
    ttnn.deallocate(ttnn_add_106, False)
    ttnn.deallocate(ttnn_to_memory_config_181, False)
    ttnn.deallocate(ttnn_rsqrt_16, False)
    # ttnn.deallocate(ttnn_reshape_348, False)
    return ttnn_reshape_348, ttnn_to_memory_config_177, ttnn_to_memory_config_179


def Linear_25_0(input_0, input_1, input_2):
    ttnn_matmul_52 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_107 = ttnn.add(
        ttnn_matmul_52,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_182 = ttnn.to_memory_config(
        ttnn_add_107,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_349 = ttnn.reshape(
        ttnn_to_memory_config_182,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_52, False)
    ttnn.deallocate(ttnn_add_107, False)
    ttnn.deallocate(ttnn_to_memory_config_182, False)
    return ttnn_reshape_349


def LayerNorm_94_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_99 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_100 = ttnn.multiply(
        ttnn_multiply_99,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_108 = ttnn.add(
        ttnn_multiply_100,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_99, False)
    ttnn.deallocate(ttnn_multiply_100, False)
    return ttnn_add_108


def Linear_57_0(input):
    ttnn_reshape_350 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_350


def Linear_35_0(input_0, input_1, input_2):
    ttnn_matmul_53 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_109 = ttnn.add(
        ttnn_matmul_53,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_183 = ttnn.to_memory_config(
        ttnn_add_109,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_53, False)
    ttnn.deallocate(ttnn_add_109, False)
    return ttnn_to_memory_config_183


def Linear_148_0(input_0, input_1):
    ttnn_matmul_54 = ttnn.matmul(
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
    ttnn.deallocate(input_0, False)
    return ttnn_matmul_54


def Linear_45_0(input):
    ttnn_reshape_351 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_351


def LayerNorm_112_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_101 = ttnn.multiply(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_102 = ttnn.multiply(
        ttnn_multiply_101,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_110 = ttnn.add(
        ttnn_multiply_102,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_184 = ttnn.to_memory_config(
        ttnn_add_110,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_101, False)
    ttnn.deallocate(ttnn_multiply_102, False)
    return ttnn_to_memory_config_184, ttnn_add_110


def CLIPEncoderLayer_92_0(input_0, input_1, input_2, input_3):
    ttnn_add_111 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_185 = ttnn.to_memory_config(
        ttnn_add_111,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_34 = ttnn.sum(
        ttnn_to_memory_config_185,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_103 = ttnn.multiply(
        ttnn_sum_34,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_186 = ttnn.to_memory_config(
        ttnn_multiply_103,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_352 = ttnn.reshape(
        ttnn_to_memory_config_186,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_17 = ttnn.neg(
        ttnn_reshape_352,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_112 = ttnn.add(
        ttnn_to_memory_config_185,
        ttnn_neg_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_187 = ttnn.to_memory_config(
        ttnn_add_112,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_104 = ttnn.multiply(
        ttnn_to_memory_config_187,
        ttnn_to_memory_config_187,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_188 = ttnn.to_memory_config(
        ttnn_multiply_104,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_35 = ttnn.sum(
        ttnn_to_memory_config_188,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_105 = ttnn.multiply(
        ttnn_sum_35,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_113 = ttnn.add(
        ttnn_multiply_105,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_189 = ttnn.to_memory_config(
        ttnn_add_113,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_17 = ttnn.rsqrt(
        ttnn_to_memory_config_189,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_353 = ttnn.reshape(
        ttnn_rsqrt_17,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_111, False)
    # ttnn.deallocate(ttnn_to_memory_config_185, False)
    ttnn.deallocate(ttnn_sum_34, False)
    ttnn.deallocate(ttnn_multiply_103, False)
    ttnn.deallocate(ttnn_to_memory_config_186, False)
    ttnn.deallocate(ttnn_reshape_352, False)
    ttnn.deallocate(ttnn_neg_17, False)
    ttnn.deallocate(ttnn_add_112, False)
    # ttnn.deallocate(ttnn_to_memory_config_187, False)
    ttnn.deallocate(ttnn_multiply_104, False)
    ttnn.deallocate(ttnn_to_memory_config_188, False)
    ttnn.deallocate(ttnn_sum_35, False)
    ttnn.deallocate(ttnn_multiply_105, False)
    ttnn.deallocate(ttnn_add_113, False)
    ttnn.deallocate(ttnn_to_memory_config_189, False)
    ttnn.deallocate(ttnn_rsqrt_17, False)
    # ttnn.deallocate(ttnn_reshape_353, False)
    return ttnn_to_memory_config_185, ttnn_reshape_353, ttnn_to_memory_config_187


def Linear_27_0(input):
    ttnn_reshape_354 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_354


def LayerNorm_124_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_106 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_107 = ttnn.multiply(
        ttnn_multiply_106,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_114 = ttnn.add(
        ttnn_multiply_107,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_190 = ttnn.to_memory_config(
        ttnn_add_114,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_106, False)
    ttnn.deallocate(ttnn_multiply_107, False)
    return ttnn_add_114, ttnn_to_memory_config_190


def Linear_143_0(input_0, input_1, input_2):
    ttnn_matmul_55 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_115 = ttnn.add(
        ttnn_matmul_55,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_191 = ttnn.to_memory_config(
        ttnn_add_115,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_55, False)
    ttnn.deallocate(ttnn_add_115, False)
    return ttnn_to_memory_config_191


def CLIPAttention_18_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_192 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_56 = ttnn.matmul(
        ttnn_to_memory_config_192,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_193 = ttnn.to_memory_config(
        ttnn_matmul_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_355 = ttnn.reshape(
        ttnn_to_memory_config_193,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_108 = ttnn.multiply(
        ttnn_reshape_355,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_multiply_108,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_4 = ttnn.softmax(
        ttnn_typecast_9,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_softmax_4,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_356 = ttnn.reshape(
        ttnn_typecast_10,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_194 = ttnn.to_memory_config(
        ttnn_reshape_356,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_57 = ttnn.matmul(
        ttnn_to_memory_config_194,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_195 = ttnn.to_memory_config(
        ttnn_matmul_57,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_357 = ttnn.reshape(
        ttnn_to_memory_config_195,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_192, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_56, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_193, False)
    ttnn.deallocate(ttnn_reshape_355, False)
    ttnn.deallocate(ttnn_multiply_108, False)
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_softmax_4, False)
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn.deallocate(ttnn_reshape_356, False)
    ttnn.deallocate(ttnn_to_memory_config_194, False)
    ttnn.deallocate(ttnn_matmul_57, False)
    ttnn.deallocate(ttnn_to_memory_config_195, False)
    return ttnn_reshape_357


def Linear_145_0(input_0, input_1, input_2):
    ttnn_matmul_58 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_116 = ttnn.add(
        ttnn_matmul_58,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_196 = ttnn.to_memory_config(
        ttnn_add_116,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_358 = ttnn.reshape(
        ttnn_to_memory_config_196,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_58, False)
    ttnn.deallocate(ttnn_add_116, False)
    ttnn.deallocate(ttnn_to_memory_config_196, False)
    return ttnn_reshape_358


def Linear_49_0(input_0, input_1, input_2):
    ttnn_matmul_59 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_117 = ttnn.add(
        ttnn_matmul_59,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_197 = ttnn.to_memory_config(
        ttnn_add_117,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_359 = ttnn.reshape(
        ttnn_to_memory_config_197,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_59, False)
    ttnn.deallocate(ttnn_add_117, False)
    ttnn.deallocate(ttnn_to_memory_config_197, False)
    return ttnn_reshape_359


def LayerNorm_100_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_109 = ttnn.multiply(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_110 = ttnn.multiply(
        ttnn_multiply_109,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_118 = ttnn.add(
        ttnn_multiply_110,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_198 = ttnn.to_memory_config(
        ttnn_add_118,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_109, False)
    ttnn.deallocate(ttnn_multiply_110, False)
    return ttnn_to_memory_config_198, ttnn_add_118


def Linear_91_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_30 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_360 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_30,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_199 = ttnn.to_memory_config(
        ttnn_reshape_360,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_60 = ttnn.matmul(
        ttnn_to_memory_config_199,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_119 = ttnn.add(
        ttnn_matmul_60,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_200 = ttnn.to_memory_config(
        ttnn_add_119,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_361 = ttnn.reshape(
        ttnn_to_memory_config_200,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_30, False)
    ttnn.deallocate(ttnn_reshape_360, False)
    ttnn.deallocate(ttnn_to_memory_config_199, False)
    ttnn.deallocate(ttnn_matmul_60, False)
    ttnn.deallocate(ttnn_add_119, False)
    ttnn.deallocate(ttnn_to_memory_config_200, False)
    return ttnn_reshape_361


def Linear_39_0(input):
    ttnn_reshape_362 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_362


def CLIPAttention_54_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_201 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_61 = ttnn.matmul(
        ttnn_to_memory_config_201,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_202 = ttnn.to_memory_config(
        ttnn_matmul_61,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_363 = ttnn.reshape(
        ttnn_to_memory_config_202,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_111 = ttnn.multiply(
        ttnn_reshape_363,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_multiply_111,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_5 = ttnn.softmax(
        ttnn_typecast_11,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_softmax_5,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_364 = ttnn.reshape(
        ttnn_typecast_12,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_203 = ttnn.to_memory_config(
        ttnn_reshape_364,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_62 = ttnn.matmul(
        ttnn_to_memory_config_203,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_204 = ttnn.to_memory_config(
        ttnn_matmul_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_365 = ttnn.reshape(
        ttnn_to_memory_config_204,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_201, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_61, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_202, False)
    ttnn.deallocate(ttnn_reshape_363, False)
    ttnn.deallocate(ttnn_multiply_111, False)
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn.deallocate(ttnn_softmax_5, False)
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn.deallocate(ttnn_reshape_364, False)
    ttnn.deallocate(ttnn_to_memory_config_203, False)
    ttnn.deallocate(ttnn_matmul_62, False)
    ttnn.deallocate(ttnn_to_memory_config_204, False)
    return ttnn_reshape_365


def QuickGELUActivation_72_0(input_0, input_1):
    ttnn_multiply_112 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_205 = ttnn.to_memory_config(
        ttnn_multiply_112,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_9 = ttnn.sigmoid(
        ttnn_to_memory_config_205,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_113 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_112, False)
    ttnn.deallocate(ttnn_to_memory_config_205, False)
    ttnn.deallocate(ttnn_sigmoid_9, False)
    return ttnn_multiply_113


def Linear_77_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_63 = ttnn.matmul(
        input_6,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_206 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_120 = ttnn.add(
        ttnn_matmul_63,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_207 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_64 = ttnn.matmul(
        ttnn_to_memory_config_206,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_208 = ttnn.to_memory_config(
        ttnn_add_120,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_65 = ttnn.matmul(
        ttnn_to_memory_config_207,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_121 = ttnn.add(
        ttnn_matmul_64,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_122 = ttnn.add(
        ttnn_matmul_65,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_366 = ttnn.reshape(
        ttnn_to_memory_config_208,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_209 = ttnn.to_memory_config(
        ttnn_add_121,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_210 = ttnn.to_memory_config(
        ttnn_add_122,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_77 = ttnn.permute(
        ttnn_reshape_366,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_367 = ttnn.reshape(
        ttnn_to_memory_config_209,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_368 = ttnn.reshape(
        ttnn_to_memory_config_210,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_369 = ttnn.reshape(
        ttnn_permute_77,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_78 = ttnn.permute(
        ttnn_reshape_367,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_79 = ttnn.permute(
        ttnn_reshape_368,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_370 = ttnn.reshape(
        ttnn_permute_78,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_371 = ttnn.reshape(
        ttnn_permute_79,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_6, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_63, False)
    ttnn.deallocate(ttnn_add_120, False)
    ttnn.deallocate(ttnn_to_memory_config_206, False)
    ttnn.deallocate(ttnn_to_memory_config_207, False)
    ttnn.deallocate(ttnn_matmul_65, False)
    ttnn.deallocate(ttnn_to_memory_config_208, False)
    ttnn.deallocate(ttnn_matmul_64, False)
    ttnn.deallocate(ttnn_reshape_366, False)
    ttnn.deallocate(ttnn_add_122, False)
    ttnn.deallocate(ttnn_add_121, False)
    ttnn.deallocate(ttnn_to_memory_config_209, False)
    ttnn.deallocate(ttnn_permute_77, False)
    ttnn.deallocate(ttnn_to_memory_config_210, False)
    ttnn.deallocate(ttnn_reshape_367, False)
    ttnn.deallocate(ttnn_reshape_368, False)
    ttnn.deallocate(ttnn_permute_78, False)
    ttnn.deallocate(ttnn_permute_79, False)
    return ttnn_reshape_371, ttnn_reshape_369, ttnn_reshape_370


def Linear_47_0(input_0, input_1, input_2):
    ttnn_matmul_66 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_123 = ttnn.add(
        ttnn_matmul_66,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_211 = ttnn.to_memory_config(
        ttnn_add_123,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_66, False)
    ttnn.deallocate(ttnn_add_123, False)
    return ttnn_to_memory_config_211


def CLIPAttention_90_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_212 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_67 = ttnn.matmul(
        ttnn_to_memory_config_212,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_213 = ttnn.to_memory_config(
        ttnn_matmul_67,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_372 = ttnn.reshape(
        ttnn_to_memory_config_213,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_114 = ttnn.multiply(
        ttnn_reshape_372,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_multiply_114,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_6 = ttnn.softmax(
        ttnn_typecast_13,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_softmax_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_373 = ttnn.reshape(
        ttnn_typecast_14,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_214 = ttnn.to_memory_config(
        ttnn_reshape_373,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_68 = ttnn.matmul(
        ttnn_to_memory_config_214,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_215 = ttnn.to_memory_config(
        ttnn_matmul_68,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_374 = ttnn.reshape(
        ttnn_to_memory_config_215,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_212, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_67, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_213, False)
    ttnn.deallocate(ttnn_reshape_372, False)
    ttnn.deallocate(ttnn_multiply_114, False)
    ttnn.deallocate(ttnn_typecast_13, False)
    ttnn.deallocate(ttnn_softmax_6, False)
    ttnn.deallocate(ttnn_typecast_14, False)
    ttnn.deallocate(ttnn_reshape_373, False)
    ttnn.deallocate(ttnn_to_memory_config_214, False)
    ttnn.deallocate(ttnn_matmul_68, False)
    ttnn.deallocate(ttnn_to_memory_config_215, False)
    return ttnn_reshape_374


def LayerNorm_82_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_115 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_116 = ttnn.multiply(
        ttnn_multiply_115,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_124 = ttnn.add(
        ttnn_multiply_116,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_115, False)
    ttnn.deallocate(ttnn_multiply_116, False)
    return ttnn_add_124


def LayerNorm_136_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_117 = ttnn.multiply(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_118 = ttnn.multiply(
        ttnn_multiply_117,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_125 = ttnn.add(
        ttnn_multiply_118,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_216 = ttnn.to_memory_config(
        ttnn_add_125,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_117, False)
    ttnn.deallocate(ttnn_multiply_118, False)
    return ttnn_add_125, ttnn_to_memory_config_216


def LayerNorm_76_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_119 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_120 = ttnn.multiply(
        ttnn_multiply_119,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_126 = ttnn.add(
        ttnn_multiply_120,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_217 = ttnn.to_memory_config(
        ttnn_add_126,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_119, False)
    ttnn.deallocate(ttnn_multiply_120, False)
    return ttnn_to_memory_config_217, ttnn_add_126


def Linear_89_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_69 = ttnn.matmul(
        input_3,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_218 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_127 = ttnn.add(
        ttnn_matmul_69,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_219 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_70 = ttnn.matmul(
        ttnn_to_memory_config_218,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_220 = ttnn.to_memory_config(
        ttnn_add_127,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_71 = ttnn.matmul(
        ttnn_to_memory_config_219,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_128 = ttnn.add(
        ttnn_matmul_70,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_129 = ttnn.add(
        ttnn_matmul_71,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_375 = ttnn.reshape(
        ttnn_to_memory_config_220,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_221 = ttnn.to_memory_config(
        ttnn_add_128,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_222 = ttnn.to_memory_config(
        ttnn_add_129,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_80 = ttnn.permute(
        ttnn_reshape_375,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_376 = ttnn.reshape(
        ttnn_to_memory_config_221,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_377 = ttnn.reshape(
        ttnn_to_memory_config_222,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_378 = ttnn.reshape(
        ttnn_permute_80,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_81 = ttnn.permute(
        ttnn_reshape_376,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_82 = ttnn.permute(
        ttnn_reshape_377,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_379 = ttnn.reshape(
        ttnn_permute_81,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_380 = ttnn.reshape(
        ttnn_permute_82,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_69, False)
    ttnn.deallocate(ttnn_add_127, False)
    ttnn.deallocate(ttnn_to_memory_config_218, False)
    ttnn.deallocate(ttnn_to_memory_config_219, False)
    ttnn.deallocate(ttnn_matmul_71, False)
    ttnn.deallocate(ttnn_to_memory_config_220, False)
    ttnn.deallocate(ttnn_matmul_70, False)
    ttnn.deallocate(ttnn_reshape_375, False)
    ttnn.deallocate(ttnn_add_129, False)
    ttnn.deallocate(ttnn_add_128, False)
    ttnn.deallocate(ttnn_to_memory_config_221, False)
    ttnn.deallocate(ttnn_permute_80, False)
    ttnn.deallocate(ttnn_to_memory_config_222, False)
    ttnn.deallocate(ttnn_reshape_376, False)
    ttnn.deallocate(ttnn_reshape_377, False)
    ttnn.deallocate(ttnn_permute_81, False)
    ttnn.deallocate(ttnn_permute_82, False)
    return ttnn_reshape_379, ttnn_reshape_380, ttnn_reshape_378


def LayerNorm_52_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_121 = ttnn.multiply(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_122 = ttnn.multiply(
        ttnn_multiply_121,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_130 = ttnn.add(
        ttnn_multiply_122,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_223 = ttnn.to_memory_config(
        ttnn_add_130,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_121, False)
    ttnn.deallocate(ttnn_multiply_122, False)
    return ttnn_add_130, ttnn_to_memory_config_223


def Linear_69_0(input):
    ttnn_reshape_381 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_381


def CLIPEncoderLayer_140_0(input_0, input_1, input_2, input_3):
    ttnn_add_131 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_224 = ttnn.to_memory_config(
        ttnn_add_131,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_36 = ttnn.sum(
        ttnn_to_memory_config_224,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_123 = ttnn.multiply(
        ttnn_sum_36,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_225 = ttnn.to_memory_config(
        ttnn_multiply_123,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_382 = ttnn.reshape(
        ttnn_to_memory_config_225,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_18 = ttnn.neg(
        ttnn_reshape_382,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_132 = ttnn.add(
        ttnn_to_memory_config_224,
        ttnn_neg_18,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_226 = ttnn.to_memory_config(
        ttnn_add_132,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_124 = ttnn.multiply(
        ttnn_to_memory_config_226,
        ttnn_to_memory_config_226,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_227 = ttnn.to_memory_config(
        ttnn_multiply_124,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_37 = ttnn.sum(
        ttnn_to_memory_config_227,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_125 = ttnn.multiply(
        ttnn_sum_37,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_133 = ttnn.add(
        ttnn_multiply_125,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_228 = ttnn.to_memory_config(
        ttnn_add_133,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_18 = ttnn.rsqrt(
        ttnn_to_memory_config_228,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_383 = ttnn.reshape(
        ttnn_rsqrt_18,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_131, False)
    # ttnn.deallocate(ttnn_to_memory_config_224, False)
    ttnn.deallocate(ttnn_sum_36, False)
    ttnn.deallocate(ttnn_multiply_123, False)
    ttnn.deallocate(ttnn_to_memory_config_225, False)
    ttnn.deallocate(ttnn_reshape_382, False)
    ttnn.deallocate(ttnn_neg_18, False)
    ttnn.deallocate(ttnn_add_132, False)
    # ttnn.deallocate(ttnn_to_memory_config_226, False)
    ttnn.deallocate(ttnn_multiply_124, False)
    ttnn.deallocate(ttnn_to_memory_config_227, False)
    ttnn.deallocate(ttnn_sum_37, False)
    ttnn.deallocate(ttnn_multiply_125, False)
    ttnn.deallocate(ttnn_add_133, False)
    ttnn.deallocate(ttnn_to_memory_config_228, False)
    ttnn.deallocate(ttnn_rsqrt_18, False)
    # ttnn.deallocate(ttnn_reshape_383, False)
    return ttnn_reshape_383, ttnn_to_memory_config_226, ttnn_to_memory_config_224


def CLIPEncoderLayer_146_0(input_0, input_1):
    ttnn_add_134 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_229 = ttnn.to_memory_config(
        ttnn_add_134,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_134, False)
    return ttnn_to_memory_config_229


def Linear_55_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_31 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_384 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_31,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_230 = ttnn.to_memory_config(
        ttnn_reshape_384,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_72 = ttnn.matmul(
        ttnn_to_memory_config_230,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_135 = ttnn.add(
        ttnn_matmul_72,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_231 = ttnn.to_memory_config(
        ttnn_add_135,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_385 = ttnn.reshape(
        ttnn_to_memory_config_231,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_31, False)
    ttnn.deallocate(ttnn_reshape_384, False)
    ttnn.deallocate(ttnn_to_memory_config_230, False)
    ttnn.deallocate(ttnn_matmul_72, False)
    ttnn.deallocate(ttnn_add_135, False)
    ttnn.deallocate(ttnn_to_memory_config_231, False)
    return ttnn_reshape_385


def Linear_127_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_32 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_386 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_32,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_232 = ttnn.to_memory_config(
        ttnn_reshape_386,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_73 = ttnn.matmul(
        ttnn_to_memory_config_232,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_136 = ttnn.add(
        ttnn_matmul_73,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_233 = ttnn.to_memory_config(
        ttnn_add_136,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_387 = ttnn.reshape(
        ttnn_to_memory_config_233,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_32, False)
    ttnn.deallocate(ttnn_reshape_386, False)
    ttnn.deallocate(ttnn_to_memory_config_232, False)
    ttnn.deallocate(ttnn_matmul_73, False)
    ttnn.deallocate(ttnn_add_136, False)
    ttnn.deallocate(ttnn_to_memory_config_233, False)
    return ttnn_reshape_387


def Linear_95_0(input_0, input_1, input_2):
    ttnn_matmul_74 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_137 = ttnn.add(
        ttnn_matmul_74,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_234 = ttnn.to_memory_config(
        ttnn_add_137,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_74, False)
    ttnn.deallocate(ttnn_add_137, False)
    return ttnn_to_memory_config_234


def QuickGELUActivation_24_0(input_0, input_1):
    ttnn_multiply_126 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_235 = ttnn.to_memory_config(
        ttnn_multiply_126,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_10 = ttnn.sigmoid(
        ttnn_to_memory_config_235,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_127 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_126, False)
    ttnn.deallocate(ttnn_to_memory_config_235, False)
    ttnn.deallocate(ttnn_sigmoid_10, False)
    return ttnn_multiply_127


def Linear_113_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_75 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_236 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_138 = ttnn.add(
        ttnn_matmul_75,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_237 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_76 = ttnn.matmul(
        ttnn_to_memory_config_236,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_238 = ttnn.to_memory_config(
        ttnn_add_138,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_77 = ttnn.matmul(
        ttnn_to_memory_config_237,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_139 = ttnn.add(
        ttnn_matmul_76,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_140 = ttnn.add(
        ttnn_matmul_77,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_388 = ttnn.reshape(
        ttnn_to_memory_config_238,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_239 = ttnn.to_memory_config(
        ttnn_add_139,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_240 = ttnn.to_memory_config(
        ttnn_add_140,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_83 = ttnn.permute(
        ttnn_reshape_388,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_389 = ttnn.reshape(
        ttnn_to_memory_config_239,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_390 = ttnn.reshape(
        ttnn_to_memory_config_240,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_391 = ttnn.reshape(
        ttnn_permute_83,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_84 = ttnn.permute(
        ttnn_reshape_389,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_85 = ttnn.permute(
        ttnn_reshape_390,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_392 = ttnn.reshape(
        ttnn_permute_84,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_393 = ttnn.reshape(
        ttnn_permute_85,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_75, False)
    ttnn.deallocate(ttnn_add_138, False)
    ttnn.deallocate(ttnn_to_memory_config_236, False)
    ttnn.deallocate(ttnn_to_memory_config_237, False)
    ttnn.deallocate(ttnn_matmul_77, False)
    ttnn.deallocate(ttnn_to_memory_config_238, False)
    ttnn.deallocate(ttnn_matmul_76, False)
    ttnn.deallocate(ttnn_reshape_388, False)
    ttnn.deallocate(ttnn_add_140, False)
    ttnn.deallocate(ttnn_add_139, False)
    ttnn.deallocate(ttnn_to_memory_config_239, False)
    ttnn.deallocate(ttnn_permute_83, False)
    ttnn.deallocate(ttnn_to_memory_config_240, False)
    ttnn.deallocate(ttnn_reshape_389, False)
    ttnn.deallocate(ttnn_reshape_390, False)
    ttnn.deallocate(ttnn_permute_84, False)
    ttnn.deallocate(ttnn_permute_85, False)
    return ttnn_reshape_393, ttnn_reshape_392, ttnn_reshape_391


def LayerNorm_130_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_128 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_129 = ttnn.multiply(
        ttnn_multiply_128,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_141 = ttnn.add(
        ttnn_multiply_129,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_128, False)
    ttnn.deallocate(ttnn_multiply_129, False)
    return ttnn_add_141


def CLIPAttention_42_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_241 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_78 = ttnn.matmul(
        ttnn_to_memory_config_241,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_242 = ttnn.to_memory_config(
        ttnn_matmul_78,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_394 = ttnn.reshape(
        ttnn_to_memory_config_242,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_130 = ttnn.multiply(
        ttnn_reshape_394,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_multiply_130,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_7 = ttnn.softmax(
        ttnn_typecast_15,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_softmax_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_395 = ttnn.reshape(
        ttnn_typecast_16,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_243 = ttnn.to_memory_config(
        ttnn_reshape_395,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_79 = ttnn.matmul(
        ttnn_to_memory_config_243,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_244 = ttnn.to_memory_config(
        ttnn_matmul_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_396 = ttnn.reshape(
        ttnn_to_memory_config_244,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_241, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_78, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_242, False)
    ttnn.deallocate(ttnn_reshape_394, False)
    ttnn.deallocate(ttnn_multiply_130, False)
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn.deallocate(ttnn_softmax_7, False)
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn.deallocate(ttnn_reshape_395, False)
    ttnn.deallocate(ttnn_to_memory_config_243, False)
    ttnn.deallocate(ttnn_matmul_79, False)
    ttnn.deallocate(ttnn_to_memory_config_244, False)
    return ttnn_reshape_396


def CLIPEncoderLayer_8_0(input_0, input_1, input_2, input_3):
    ttnn_add_142 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_245 = ttnn.to_memory_config(
        ttnn_add_142,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_38 = ttnn.sum(
        ttnn_to_memory_config_245,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_131 = ttnn.multiply(
        ttnn_sum_38,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_246 = ttnn.to_memory_config(
        ttnn_multiply_131,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_397 = ttnn.reshape(
        ttnn_to_memory_config_246,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_19 = ttnn.neg(
        ttnn_reshape_397,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_143 = ttnn.add(
        ttnn_to_memory_config_245,
        ttnn_neg_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_247 = ttnn.to_memory_config(
        ttnn_add_143,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_132 = ttnn.multiply(
        ttnn_to_memory_config_247,
        ttnn_to_memory_config_247,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_248 = ttnn.to_memory_config(
        ttnn_multiply_132,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_39 = ttnn.sum(
        ttnn_to_memory_config_248,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_133 = ttnn.multiply(
        ttnn_sum_39,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_144 = ttnn.add(
        ttnn_multiply_133,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_249 = ttnn.to_memory_config(
        ttnn_add_144,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_19 = ttnn.rsqrt(
        ttnn_to_memory_config_249,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_398 = ttnn.reshape(
        ttnn_rsqrt_19,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_142, False)
    # ttnn.deallocate(ttnn_to_memory_config_245, False)
    ttnn.deallocate(ttnn_sum_38, False)
    ttnn.deallocate(ttnn_multiply_131, False)
    ttnn.deallocate(ttnn_to_memory_config_246, False)
    ttnn.deallocate(ttnn_reshape_397, False)
    ttnn.deallocate(ttnn_neg_19, False)
    ttnn.deallocate(ttnn_add_143, False)
    # ttnn.deallocate(ttnn_to_memory_config_247, False)
    ttnn.deallocate(ttnn_multiply_132, False)
    ttnn.deallocate(ttnn_to_memory_config_248, False)
    ttnn.deallocate(ttnn_sum_39, False)
    ttnn.deallocate(ttnn_multiply_133, False)
    ttnn.deallocate(ttnn_add_144, False)
    ttnn.deallocate(ttnn_to_memory_config_249, False)
    ttnn.deallocate(ttnn_rsqrt_19, False)
    # ttnn.deallocate(ttnn_reshape_398, False)
    return ttnn_to_memory_config_247, ttnn_to_memory_config_245, ttnn_reshape_398


def Linear_23_0(input_0, input_1, input_2):
    ttnn_matmul_80 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_145 = ttnn.add(
        ttnn_matmul_80,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_250 = ttnn.to_memory_config(
        ttnn_add_145,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_80, False)
    ttnn.deallocate(ttnn_add_145, False)
    return ttnn_to_memory_config_250


def Linear_99_0(input):
    ttnn_reshape_399 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_399


def Linear_79_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_33 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_400 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_33,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_251 = ttnn.to_memory_config(
        ttnn_reshape_400,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_81 = ttnn.matmul(
        ttnn_to_memory_config_251,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_146 = ttnn.add(
        ttnn_matmul_81,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_252 = ttnn.to_memory_config(
        ttnn_add_146,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_401 = ttnn.reshape(
        ttnn_to_memory_config_252,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_33, False)
    ttnn.deallocate(ttnn_reshape_400, False)
    ttnn.deallocate(ttnn_to_memory_config_251, False)
    ttnn.deallocate(ttnn_matmul_81, False)
    ttnn.deallocate(ttnn_add_146, False)
    ttnn.deallocate(ttnn_to_memory_config_252, False)
    return ttnn_reshape_401


def CLIPAttention_114_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_253 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_82 = ttnn.matmul(
        ttnn_to_memory_config_253,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_254 = ttnn.to_memory_config(
        ttnn_matmul_82,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_402 = ttnn.reshape(
        ttnn_to_memory_config_254,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_134 = ttnn.multiply(
        ttnn_reshape_402,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_multiply_134,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_8 = ttnn.softmax(
        ttnn_typecast_17,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_softmax_8,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_403 = ttnn.reshape(
        ttnn_typecast_18,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_255 = ttnn.to_memory_config(
        ttnn_reshape_403,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_83 = ttnn.matmul(
        ttnn_to_memory_config_255,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_256 = ttnn.to_memory_config(
        ttnn_matmul_83,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_404 = ttnn.reshape(
        ttnn_to_memory_config_256,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_253, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_82, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_254, False)
    ttnn.deallocate(ttnn_reshape_402, False)
    ttnn.deallocate(ttnn_multiply_134, False)
    ttnn.deallocate(ttnn_typecast_17, False)
    ttnn.deallocate(ttnn_softmax_8, False)
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn.deallocate(ttnn_reshape_403, False)
    ttnn.deallocate(ttnn_to_memory_config_255, False)
    ttnn.deallocate(ttnn_matmul_83, False)
    ttnn.deallocate(ttnn_to_memory_config_256, False)
    return ttnn_reshape_404


def LayerNorm_16_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_135 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_136 = ttnn.multiply(
        ttnn_multiply_135,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_147 = ttnn.add(
        ttnn_multiply_136,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_257 = ttnn.to_memory_config(
        ttnn_add_147,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_135, False)
    ttnn.deallocate(ttnn_multiply_136, False)
    return ttnn_to_memory_config_257, ttnn_add_147


def Linear_9_0(input):
    ttnn_reshape_405 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_405


def Linear_81_0(input):
    ttnn_reshape_406 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_406


def LayerNorm_46_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_137 = ttnn.multiply(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_138 = ttnn.multiply(
        ttnn_multiply_137,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_148 = ttnn.add(
        ttnn_multiply_138,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_137, False)
    ttnn.deallocate(ttnn_multiply_138, False)
    return ttnn_add_148


def CLIPEncoderLayer_98_0(input_0, input_1, input_2, input_3):
    ttnn_add_149 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_258 = ttnn.to_memory_config(
        ttnn_add_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_40 = ttnn.sum(
        ttnn_to_memory_config_258,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_139 = ttnn.multiply(
        ttnn_sum_40,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_259 = ttnn.to_memory_config(
        ttnn_multiply_139,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_407 = ttnn.reshape(
        ttnn_to_memory_config_259,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_20 = ttnn.neg(
        ttnn_reshape_407,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_150 = ttnn.add(
        ttnn_to_memory_config_258,
        ttnn_neg_20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_260 = ttnn.to_memory_config(
        ttnn_add_150,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_140 = ttnn.multiply(
        ttnn_to_memory_config_260,
        ttnn_to_memory_config_260,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_261 = ttnn.to_memory_config(
        ttnn_multiply_140,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_41 = ttnn.sum(
        ttnn_to_memory_config_261,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_141 = ttnn.multiply(
        ttnn_sum_41,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_151 = ttnn.add(
        ttnn_multiply_141,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_262 = ttnn.to_memory_config(
        ttnn_add_151,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_20 = ttnn.rsqrt(
        ttnn_to_memory_config_262,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_408 = ttnn.reshape(
        ttnn_rsqrt_20,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_149, False)
    # ttnn.deallocate(ttnn_to_memory_config_258, False)
    ttnn.deallocate(ttnn_sum_40, False)
    ttnn.deallocate(ttnn_multiply_139, False)
    ttnn.deallocate(ttnn_to_memory_config_259, False)
    ttnn.deallocate(ttnn_reshape_407, False)
    ttnn.deallocate(ttnn_neg_20, False)
    ttnn.deallocate(ttnn_add_150, False)
    # ttnn.deallocate(ttnn_to_memory_config_260, False)
    ttnn.deallocate(ttnn_multiply_140, False)
    ttnn.deallocate(ttnn_to_memory_config_261, False)
    ttnn.deallocate(ttnn_sum_41, False)
    ttnn.deallocate(ttnn_multiply_141, False)
    ttnn.deallocate(ttnn_add_151, False)
    ttnn.deallocate(ttnn_to_memory_config_262, False)
    ttnn.deallocate(ttnn_rsqrt_20, False)
    # ttnn.deallocate(ttnn_reshape_408, False)
    return ttnn_to_memory_config_260, ttnn_to_memory_config_258, ttnn_reshape_408


def Linear_109_0(input_0, input_1, input_2):
    ttnn_matmul_84 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_152 = ttnn.add(
        ttnn_matmul_84,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_263 = ttnn.to_memory_config(
        ttnn_add_152,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_409 = ttnn.reshape(
        ttnn_to_memory_config_263,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_84, False)
    ttnn.deallocate(ttnn_add_152, False)
    ttnn.deallocate(ttnn_to_memory_config_263, False)
    return ttnn_reshape_409


def CLIPEncoderLayer_74_0(input_0, input_1, input_2, input_3):
    ttnn_add_153 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_264 = ttnn.to_memory_config(
        ttnn_add_153,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_42 = ttnn.sum(
        ttnn_to_memory_config_264,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_142 = ttnn.multiply(
        ttnn_sum_42,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_265 = ttnn.to_memory_config(
        ttnn_multiply_142,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_410 = ttnn.reshape(
        ttnn_to_memory_config_265,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_21 = ttnn.neg(
        ttnn_reshape_410,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_154 = ttnn.add(
        ttnn_to_memory_config_264,
        ttnn_neg_21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_266 = ttnn.to_memory_config(
        ttnn_add_154,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_143 = ttnn.multiply(
        ttnn_to_memory_config_266,
        ttnn_to_memory_config_266,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_267 = ttnn.to_memory_config(
        ttnn_multiply_143,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_43 = ttnn.sum(
        ttnn_to_memory_config_267,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_144 = ttnn.multiply(
        ttnn_sum_43,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_155 = ttnn.add(
        ttnn_multiply_144,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_268 = ttnn.to_memory_config(
        ttnn_add_155,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_21 = ttnn.rsqrt(
        ttnn_to_memory_config_268,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_411 = ttnn.reshape(
        ttnn_rsqrt_21,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_153, False)
    # ttnn.deallocate(ttnn_to_memory_config_264, False)
    ttnn.deallocate(ttnn_sum_42, False)
    ttnn.deallocate(ttnn_multiply_142, False)
    ttnn.deallocate(ttnn_to_memory_config_265, False)
    ttnn.deallocate(ttnn_reshape_410, False)
    ttnn.deallocate(ttnn_neg_21, False)
    ttnn.deallocate(ttnn_add_154, False)
    # ttnn.deallocate(ttnn_to_memory_config_266, False)
    ttnn.deallocate(ttnn_multiply_143, False)
    ttnn.deallocate(ttnn_to_memory_config_267, False)
    ttnn.deallocate(ttnn_sum_43, False)
    ttnn.deallocate(ttnn_multiply_144, False)
    ttnn.deallocate(ttnn_add_155, False)
    ttnn.deallocate(ttnn_to_memory_config_268, False)
    ttnn.deallocate(ttnn_rsqrt_21, False)
    # ttnn.deallocate(ttnn_reshape_411, False)
    return ttnn_to_memory_config_264, ttnn_reshape_411, ttnn_to_memory_config_266


def CLIPEncoderLayer_128_0(input_0, input_1, input_2, input_3):
    ttnn_add_156 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_269 = ttnn.to_memory_config(
        ttnn_add_156,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_44 = ttnn.sum(
        ttnn_to_memory_config_269,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_145 = ttnn.multiply(
        ttnn_sum_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_270 = ttnn.to_memory_config(
        ttnn_multiply_145,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_412 = ttnn.reshape(
        ttnn_to_memory_config_270,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_22 = ttnn.neg(
        ttnn_reshape_412,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_157 = ttnn.add(
        ttnn_to_memory_config_269,
        ttnn_neg_22,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_271 = ttnn.to_memory_config(
        ttnn_add_157,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_146 = ttnn.multiply(
        ttnn_to_memory_config_271,
        ttnn_to_memory_config_271,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_272 = ttnn.to_memory_config(
        ttnn_multiply_146,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_45 = ttnn.sum(
        ttnn_to_memory_config_272,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_147 = ttnn.multiply(
        ttnn_sum_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_158 = ttnn.add(
        ttnn_multiply_147,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_273 = ttnn.to_memory_config(
        ttnn_add_158,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_22 = ttnn.rsqrt(
        ttnn_to_memory_config_273,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_413 = ttnn.reshape(
        ttnn_rsqrt_22,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_156, False)
    # ttnn.deallocate(ttnn_to_memory_config_269, False)
    ttnn.deallocate(ttnn_sum_44, False)
    ttnn.deallocate(ttnn_multiply_145, False)
    ttnn.deallocate(ttnn_to_memory_config_270, False)
    ttnn.deallocate(ttnn_reshape_412, False)
    ttnn.deallocate(ttnn_neg_22, False)
    ttnn.deallocate(ttnn_add_157, False)
    # ttnn.deallocate(ttnn_to_memory_config_271, False)
    ttnn.deallocate(ttnn_multiply_146, False)
    ttnn.deallocate(ttnn_to_memory_config_272, False)
    ttnn.deallocate(ttnn_sum_45, False)
    ttnn.deallocate(ttnn_multiply_147, False)
    ttnn.deallocate(ttnn_add_158, False)
    ttnn.deallocate(ttnn_to_memory_config_273, False)
    ttnn.deallocate(ttnn_rsqrt_22, False)
    # ttnn.deallocate(ttnn_reshape_413, False)
    return ttnn_reshape_413, ttnn_to_memory_config_271, ttnn_to_memory_config_269


def Linear_43_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_34 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_414 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_34,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_274 = ttnn.to_memory_config(
        ttnn_reshape_414,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_85 = ttnn.matmul(
        ttnn_to_memory_config_274,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_159 = ttnn.add(
        ttnn_matmul_85,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_275 = ttnn.to_memory_config(
        ttnn_add_159,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_415 = ttnn.reshape(
        ttnn_to_memory_config_275,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_34, False)
    ttnn.deallocate(ttnn_reshape_414, False)
    ttnn.deallocate(ttnn_to_memory_config_274, False)
    ttnn.deallocate(ttnn_matmul_85, False)
    ttnn.deallocate(ttnn_add_159, False)
    ttnn.deallocate(ttnn_to_memory_config_275, False)
    return ttnn_reshape_415


def CLIPAttention_138_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_276 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_86 = ttnn.matmul(
        ttnn_to_memory_config_276,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_277 = ttnn.to_memory_config(
        ttnn_matmul_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_416 = ttnn.reshape(
        ttnn_to_memory_config_277,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_148 = ttnn.multiply(
        ttnn_reshape_416,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_multiply_148,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_9 = ttnn.softmax(
        ttnn_typecast_19,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_softmax_9,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_417 = ttnn.reshape(
        ttnn_typecast_20,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_278 = ttnn.to_memory_config(
        ttnn_reshape_417,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_87 = ttnn.matmul(
        ttnn_to_memory_config_278,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_279 = ttnn.to_memory_config(
        ttnn_matmul_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_418 = ttnn.reshape(
        ttnn_to_memory_config_279,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_276, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_86, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_277, False)
    ttnn.deallocate(ttnn_reshape_416, False)
    ttnn.deallocate(ttnn_multiply_148, False)
    ttnn.deallocate(ttnn_typecast_19, False)
    ttnn.deallocate(ttnn_softmax_9, False)
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn.deallocate(ttnn_reshape_417, False)
    ttnn.deallocate(ttnn_to_memory_config_278, False)
    ttnn.deallocate(ttnn_matmul_87, False)
    ttnn.deallocate(ttnn_to_memory_config_279, False)
    return ttnn_reshape_418


def Linear_85_0(input_0, input_1, input_2):
    ttnn_matmul_88 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_160 = ttnn.add(
        ttnn_matmul_88,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_280 = ttnn.to_memory_config(
        ttnn_add_160,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_419 = ttnn.reshape(
        ttnn_to_memory_config_280,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_88, False)
    ttnn.deallocate(ttnn_add_160, False)
    ttnn.deallocate(ttnn_to_memory_config_280, False)
    return ttnn_reshape_419


def CLIPAttention_6_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_281 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_89 = ttnn.matmul(
        ttnn_to_memory_config_281,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_282 = ttnn.to_memory_config(
        ttnn_matmul_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_420 = ttnn.reshape(
        ttnn_to_memory_config_282,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_149 = ttnn.multiply(
        ttnn_reshape_420,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_multiply_149,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_10 = ttnn.softmax(
        ttnn_typecast_21,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_softmax_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_421 = ttnn.reshape(
        ttnn_typecast_22,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_283 = ttnn.to_memory_config(
        ttnn_reshape_421,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_90 = ttnn.matmul(
        ttnn_to_memory_config_283,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_284 = ttnn.to_memory_config(
        ttnn_matmul_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_422 = ttnn.reshape(
        ttnn_to_memory_config_284,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_281, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_89, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_282, False)
    ttnn.deallocate(ttnn_reshape_420, False)
    ttnn.deallocate(ttnn_multiply_149, False)
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn.deallocate(ttnn_softmax_10, False)
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn.deallocate(ttnn_reshape_421, False)
    ttnn.deallocate(ttnn_to_memory_config_283, False)
    ttnn.deallocate(ttnn_matmul_90, False)
    ttnn.deallocate(ttnn_to_memory_config_284, False)
    return ttnn_reshape_422


def Linear_51_0(input):
    ttnn_reshape_423 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_423


def Linear_31_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_35 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_424 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_35,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_285 = ttnn.to_memory_config(
        ttnn_reshape_424,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_91 = ttnn.matmul(
        ttnn_to_memory_config_285,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_161 = ttnn.add(
        ttnn_matmul_91,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_286 = ttnn.to_memory_config(
        ttnn_add_161,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_425 = ttnn.reshape(
        ttnn_to_memory_config_286,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_35, False)
    ttnn.deallocate(ttnn_reshape_424, False)
    ttnn.deallocate(ttnn_to_memory_config_285, False)
    ttnn.deallocate(ttnn_matmul_91, False)
    ttnn.deallocate(ttnn_add_161, False)
    ttnn.deallocate(ttnn_to_memory_config_286, False)
    return ttnn_reshape_425


def Linear_15_0(input):
    ttnn_reshape_426 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_426


def CLIPAttention_66_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_287 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_92 = ttnn.matmul(
        ttnn_to_memory_config_287,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_288 = ttnn.to_memory_config(
        ttnn_matmul_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_427 = ttnn.reshape(
        ttnn_to_memory_config_288,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_150 = ttnn.multiply(
        ttnn_reshape_427,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_multiply_150,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_11 = ttnn.softmax(
        ttnn_typecast_23,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_softmax_11,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_428 = ttnn.reshape(
        ttnn_typecast_24,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_289 = ttnn.to_memory_config(
        ttnn_reshape_428,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_93 = ttnn.matmul(
        ttnn_to_memory_config_289,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_290 = ttnn.to_memory_config(
        ttnn_matmul_93,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_429 = ttnn.reshape(
        ttnn_to_memory_config_290,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_287, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_92, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_288, False)
    ttnn.deallocate(ttnn_reshape_427, False)
    ttnn.deallocate(ttnn_multiply_150, False)
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn.deallocate(ttnn_softmax_11, False)
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn.deallocate(ttnn_reshape_428, False)
    ttnn.deallocate(ttnn_to_memory_config_289, False)
    ttnn.deallocate(ttnn_matmul_93, False)
    ttnn.deallocate(ttnn_to_memory_config_290, False)
    return ttnn_reshape_429


def Linear_111_0(input):
    ttnn_reshape_430 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_430


def LayerNorm_142_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_151 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_152 = ttnn.multiply(
        ttnn_multiply_151,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_162 = ttnn.add(
        ttnn_multiply_152,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_151, False)
    ttnn.deallocate(ttnn_multiply_152, False)
    return ttnn_add_162


def CLIPEncoderLayer_134_0(input_0, input_1, input_2, input_3):
    ttnn_add_163 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_291 = ttnn.to_memory_config(
        ttnn_add_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_46 = ttnn.sum(
        ttnn_to_memory_config_291,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_153 = ttnn.multiply(
        ttnn_sum_46,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_292 = ttnn.to_memory_config(
        ttnn_multiply_153,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_431 = ttnn.reshape(
        ttnn_to_memory_config_292,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_23 = ttnn.neg(
        ttnn_reshape_431,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_164 = ttnn.add(
        ttnn_to_memory_config_291,
        ttnn_neg_23,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_293 = ttnn.to_memory_config(
        ttnn_add_164,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_154 = ttnn.multiply(
        ttnn_to_memory_config_293,
        ttnn_to_memory_config_293,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_294 = ttnn.to_memory_config(
        ttnn_multiply_154,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_47 = ttnn.sum(
        ttnn_to_memory_config_294,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_155 = ttnn.multiply(
        ttnn_sum_47,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_165 = ttnn.add(
        ttnn_multiply_155,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_295 = ttnn.to_memory_config(
        ttnn_add_165,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_23 = ttnn.rsqrt(
        ttnn_to_memory_config_295,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_432 = ttnn.reshape(
        ttnn_rsqrt_23,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_163, False)
    # ttnn.deallocate(ttnn_to_memory_config_291, False)
    ttnn.deallocate(ttnn_sum_46, False)
    ttnn.deallocate(ttnn_multiply_153, False)
    ttnn.deallocate(ttnn_to_memory_config_292, False)
    ttnn.deallocate(ttnn_reshape_431, False)
    ttnn.deallocate(ttnn_neg_23, False)
    ttnn.deallocate(ttnn_add_164, False)
    # ttnn.deallocate(ttnn_to_memory_config_293, False)
    ttnn.deallocate(ttnn_multiply_154, False)
    ttnn.deallocate(ttnn_to_memory_config_294, False)
    ttnn.deallocate(ttnn_sum_47, False)
    ttnn.deallocate(ttnn_multiply_155, False)
    ttnn.deallocate(ttnn_add_165, False)
    ttnn.deallocate(ttnn_to_memory_config_295, False)
    ttnn.deallocate(ttnn_rsqrt_23, False)
    # ttnn.deallocate(ttnn_reshape_432, False)
    return ttnn_to_memory_config_293, ttnn_reshape_432, ttnn_to_memory_config_291


def CLIPEncoderLayer_20_0(input_0, input_1, input_2, input_3):
    ttnn_add_166 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_296 = ttnn.to_memory_config(
        ttnn_add_166,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_48 = ttnn.sum(
        ttnn_to_memory_config_296,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_156 = ttnn.multiply(
        ttnn_sum_48,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_297 = ttnn.to_memory_config(
        ttnn_multiply_156,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_433 = ttnn.reshape(
        ttnn_to_memory_config_297,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_24 = ttnn.neg(
        ttnn_reshape_433,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_167 = ttnn.add(
        ttnn_to_memory_config_296,
        ttnn_neg_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_298 = ttnn.to_memory_config(
        ttnn_add_167,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_157 = ttnn.multiply(
        ttnn_to_memory_config_298,
        ttnn_to_memory_config_298,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_299 = ttnn.to_memory_config(
        ttnn_multiply_157,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_49 = ttnn.sum(
        ttnn_to_memory_config_299,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_158 = ttnn.multiply(
        ttnn_sum_49,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_168 = ttnn.add(
        ttnn_multiply_158,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_300 = ttnn.to_memory_config(
        ttnn_add_168,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_24 = ttnn.rsqrt(
        ttnn_to_memory_config_300,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_434 = ttnn.reshape(
        ttnn_rsqrt_24,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_166, False)
    # ttnn.deallocate(ttnn_to_memory_config_296, False)
    ttnn.deallocate(ttnn_sum_48, False)
    ttnn.deallocate(ttnn_multiply_156, False)
    ttnn.deallocate(ttnn_to_memory_config_297, False)
    ttnn.deallocate(ttnn_reshape_433, False)
    ttnn.deallocate(ttnn_neg_24, False)
    ttnn.deallocate(ttnn_add_167, False)
    # ttnn.deallocate(ttnn_to_memory_config_298, False)
    ttnn.deallocate(ttnn_multiply_157, False)
    ttnn.deallocate(ttnn_to_memory_config_299, False)
    ttnn.deallocate(ttnn_sum_49, False)
    ttnn.deallocate(ttnn_multiply_158, False)
    ttnn.deallocate(ttnn_add_168, False)
    ttnn.deallocate(ttnn_to_memory_config_300, False)
    ttnn.deallocate(ttnn_rsqrt_24, False)
    # ttnn.deallocate(ttnn_reshape_434, False)
    return ttnn_to_memory_config_296, ttnn_to_memory_config_298, ttnn_reshape_434


def Linear_53_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_94 = ttnn.matmul(
        input_3,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_301 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_169 = ttnn.add(
        ttnn_matmul_94,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_302 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_95 = ttnn.matmul(
        ttnn_to_memory_config_301,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_303 = ttnn.to_memory_config(
        ttnn_add_169,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_96 = ttnn.matmul(
        ttnn_to_memory_config_302,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_170 = ttnn.add(
        ttnn_matmul_95,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_171 = ttnn.add(
        ttnn_matmul_96,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_435 = ttnn.reshape(
        ttnn_to_memory_config_303,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_304 = ttnn.to_memory_config(
        ttnn_add_170,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_305 = ttnn.to_memory_config(
        ttnn_add_171,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_86 = ttnn.permute(
        ttnn_reshape_435,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_436 = ttnn.reshape(
        ttnn_to_memory_config_304,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_437 = ttnn.reshape(
        ttnn_to_memory_config_305,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_438 = ttnn.reshape(
        ttnn_permute_86,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_87 = ttnn.permute(
        ttnn_reshape_436,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_88 = ttnn.permute(
        ttnn_reshape_437,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_439 = ttnn.reshape(
        ttnn_permute_87,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_440 = ttnn.reshape(
        ttnn_permute_88,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_94, False)
    ttnn.deallocate(ttnn_add_169, False)
    ttnn.deallocate(ttnn_to_memory_config_301, False)
    ttnn.deallocate(ttnn_to_memory_config_302, False)
    ttnn.deallocate(ttnn_matmul_96, False)
    ttnn.deallocate(ttnn_to_memory_config_303, False)
    ttnn.deallocate(ttnn_matmul_95, False)
    ttnn.deallocate(ttnn_reshape_435, False)
    ttnn.deallocate(ttnn_add_171, False)
    ttnn.deallocate(ttnn_add_170, False)
    ttnn.deallocate(ttnn_to_memory_config_304, False)
    ttnn.deallocate(ttnn_permute_86, False)
    ttnn.deallocate(ttnn_to_memory_config_305, False)
    ttnn.deallocate(ttnn_reshape_436, False)
    ttnn.deallocate(ttnn_reshape_437, False)
    ttnn.deallocate(ttnn_permute_87, False)
    ttnn.deallocate(ttnn_permute_88, False)
    return ttnn_reshape_440, ttnn_reshape_439, ttnn_reshape_438


def QuickGELUActivation_36_0(input_0, input_1):
    ttnn_multiply_159 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_306 = ttnn.to_memory_config(
        ttnn_multiply_159,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_11 = ttnn.sigmoid(
        ttnn_to_memory_config_306,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_160 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_159, False)
    ttnn.deallocate(ttnn_to_memory_config_306, False)
    ttnn.deallocate(ttnn_sigmoid_11, False)
    return ttnn_multiply_160


def CLIPVisionTransformer_147_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_slice_0 = ttnn.slice(
        input_2,
        [0, 0, 0],
        [2, 1, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_441 = ttnn.reshape(
        ttnn_slice_0,
        [2, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_50 = ttnn.sum(
        ttnn_reshape_441,
        [1],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_161 = ttnn.multiply(
        ttnn_sum_50,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_307 = ttnn.to_memory_config(
        ttnn_multiply_161,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_442 = ttnn.reshape(
        ttnn_to_memory_config_307,
        [2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_25 = ttnn.neg(
        ttnn_reshape_442,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_172 = ttnn.add(
        ttnn_reshape_441,
        ttnn_neg_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_308 = ttnn.to_memory_config(
        ttnn_add_172,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_162 = ttnn.multiply(
        ttnn_to_memory_config_308,
        ttnn_to_memory_config_308,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_309 = ttnn.to_memory_config(
        ttnn_multiply_162,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_51 = ttnn.sum(
        ttnn_to_memory_config_309,
        [1],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_163 = ttnn.multiply(
        ttnn_sum_51,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_310 = ttnn.to_memory_config(
        ttnn_multiply_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_443 = ttnn.reshape(
        ttnn_to_memory_config_310,
        [2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_173 = ttnn.add(
        ttnn_reshape_443,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_311 = ttnn.to_memory_config(
        ttnn_add_173,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_25 = ttnn.rsqrt(
        ttnn_to_memory_config_311,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_164 = ttnn.multiply(
        ttnn_to_memory_config_308,
        ttnn_rsqrt_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_165 = ttnn.multiply(
        ttnn_multiply_164,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_174 = ttnn.add(
        ttnn_multiply_165,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_312 = ttnn.to_memory_config(
        ttnn_add_174,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn.deallocate(ttnn_reshape_441, False)
    ttnn.deallocate(ttnn_sum_50, False)
    ttnn.deallocate(ttnn_multiply_161, False)
    ttnn.deallocate(ttnn_to_memory_config_307, False)
    ttnn.deallocate(ttnn_reshape_442, False)
    ttnn.deallocate(ttnn_neg_25, False)
    ttnn.deallocate(ttnn_add_172, False)
    ttnn.deallocate(ttnn_to_memory_config_308, False)
    ttnn.deallocate(ttnn_multiply_162, False)
    ttnn.deallocate(ttnn_to_memory_config_309, False)
    ttnn.deallocate(ttnn_sum_51, False)
    ttnn.deallocate(ttnn_multiply_163, False)
    ttnn.deallocate(ttnn_to_memory_config_310, False)
    ttnn.deallocate(ttnn_reshape_443, False)
    ttnn.deallocate(ttnn_add_173, False)
    ttnn.deallocate(ttnn_to_memory_config_311, False)
    ttnn.deallocate(ttnn_rsqrt_25, False)
    ttnn.deallocate(ttnn_multiply_164, False)
    ttnn.deallocate(ttnn_multiply_165, False)
    ttnn.deallocate(ttnn_add_174, False)
    return ttnn_to_memory_config_312


def Linear_135_0(input):
    ttnn_reshape_444 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_444


