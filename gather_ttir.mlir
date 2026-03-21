module @SyncTensorsGraph.36 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.36 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<32x1x16xi64> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1x16xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "L__self___topk_indices"}, %arg1: tensor<32x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}) -> (tensor<32x16x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x16x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.constant"() <{value = dense<128> : tensor<32x16xi64>}> : () -> tensor<32x16xi64>
        %1 = "ttir.constant"() <{value = dense<32> : tensor<32x16xi64>}> : () -> tensor<32x16xi64>
        %2 = "ttir.constant"() <{value = dense<0> : tensor<32x16xi64>}> : () -> tensor<32x16xi64>
        %3 = "ttir.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>}> : () -> tensor<32xi64>
        %4 = "ttir.reshape"(%3) <{shape = [32 : i32, 1 : i32]}> : (tensor<32xi64>) -> tensor<32x1xi64>
        %5 = "ttir.broadcast"(%4) <{broadcast_dimensions = array<i64: 1, 16>}> : (tensor<32x1xi64>) -> tensor<32x16xi64>
        %6 = "ttir.lt"(%5, %2) : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi1>
        %7 = "ttir.add"(%5, %1) : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi64>
        %8 = "ttir.where"(%6, %7, %5) : (tensor<32x16xi1>, tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi64>
        %9 = "ttir.reshape"(%8) <{shape = [32 : i32, 16 : i32, 1 : i32]}> : (tensor<32x16xi64>) -> tensor<32x16x1xi64>
        %10 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 16 : i32]}> : (tensor<32x1x16xi64>) -> tensor<32x16xi64>
        %11 = "ttir.lt"(%10, %2) : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi1>
        %12 = "ttir.add"(%10, %0) : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi64>
        %13 = "ttir.where"(%11, %12, %10) : (tensor<32x16xi1>, tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi64>
        %14 = "ttir.reshape"(%13) <{shape = [32 : i32, 16 : i32, 1 : i32]}> : (tensor<32x16xi64>) -> tensor<32x16x1xi64>
        %15 = "ttir.concat"(%9, %14) <{dim = 2 : si32}> : (tensor<32x16x1xi64>, tensor<32x16x1xi64>) -> tensor<32x16x2xi64>
        %16 = "ttir.gather"(%arg1, %15) <{collapsed_slice_dims = array<i64: 0, 1>, index_vector_dim = 2 : si64, indices_are_sorted = false, offset_dims = array<i64: 2>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 1, 512>, start_index_map = array<i64: 0, 1>, start_indices_batching_dims = array<i64>}> : (tensor<32x128x512xbf16>, tensor<32x16x2xi64>) -> tensor<32x16x512xbf16>
        return %16 : tensor<32x16x512xbf16>
      }
    }
  }
}
