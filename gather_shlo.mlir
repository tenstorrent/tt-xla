module @SyncTensorsGraph.36 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @main(%arg0: tensor<32x1x16xi64> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1x16xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "L__self___topk_indices"}, %arg1: tensor<32x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}) -> (tensor<32x16x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x16x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %c = stablehlo.constant dense<128> : tensor<32x16xi64>
    %c_0 = stablehlo.constant dense<32> : tensor<32x16xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<32x16xi64>
    %c_2 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>
    %0 = stablehlo.broadcast_in_dim %c_2, dims = [0] : (tensor<32xi64>) -> tensor<32x16xi64>
    %1 = stablehlo.compare  LT, %0, %c_1 : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi1>
    %2 = stablehlo.add %0, %c_0 : tensor<32x16xi64>
    %3 = stablehlo.select %1, %2, %0 : tensor<32x16xi1>, tensor<32x16xi64>
    %4 = stablehlo.reshape %3 : (tensor<32x16xi64>) -> tensor<32x16x1xi64>
    %5 = stablehlo.reshape %arg0 : (tensor<32x1x16xi64>) -> tensor<32x16xi64>
    %6 = stablehlo.compare  LT, %5, %c_1 : (tensor<32x16xi64>, tensor<32x16xi64>) -> tensor<32x16xi1>
    %7 = stablehlo.add %5, %c : tensor<32x16xi64>
    %8 = stablehlo.select %6, %7, %5 : tensor<32x16xi1>, tensor<32x16xi64>
    %9 = stablehlo.reshape %8 : (tensor<32x16xi64>) -> tensor<32x16x1xi64>
    %10 = stablehlo.concatenate %4, %9, dim = 2 : (tensor<32x16x1xi64>, tensor<32x16x1xi64>) -> tensor<32x16x2xi64>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<32x128x512xbf16>, tensor<32x16x2xi64>) -> tensor<32x16x512xbf16>
    return %11 : tensor<32x16x512xbf16>
  }
}
