module @SyncTensorsGraph.739 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<1x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<1x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<1x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<1x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<1x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:4 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) in_shardings=[<@mesh, [{}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {"_axis_1"}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{"_axis_1"}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}, {}]>] manual_axes={"_axis_0", "_axis_1"} (%arg19: tensor<1x64x512xbf16>, %arg20: tensor<576x2048xbf16>, %arg21: tensor<1x1x2048xbf16>, %arg22: tensor<512xbf16>, %arg23: tensor<i1>, %arg24: tensor<1x64x64xbf16>, %arg25: tensor<1x32x2xbf16>, %arg26: tensor<1x64x128xbf16>, %arg27: tensor<128x128xbf16>, %arg28: tensor<128xbf16>, %arg29: tensor<128xbf16>, %arg30: tensor<128x2048xbf16>, %arg31: tensor<2048x512xbf16>, %arg32: tensor<1024x512xbf16>, %arg33: tensor<16x2048xbf16>, %arg34: tensor<2048x3072xbf16>, %arg35: tensor<3072x2048xbf16>, %arg36: tensor<3072xbf16>, %arg37: tensor<768x3072xbf16>) {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %c = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
      %c_1 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %cst_4 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
      %c_5 = stablehlo.constant dense<1> : tensor<i64>
      %cst_6 = stablehlo.constant dense<3.25520843E-4> : tensor<1x1xf32>
      %cst_7 = stablehlo.constant dense<1.250000e-01> : tensor<bf16>
      %cst_8 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
      %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %cst_10 = stablehlo.constant dense<0.001953125> : tensor<1x1xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %c_12 = stablehlo.constant dense<0> : tensor<ui32>
      %cst_13 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<bf16>) -> tensor<1x1x4x33xbf16>
      %2 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1x33x64x1xui32>
      %3 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<1x33x512x1xui32>
      %4 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x64xbf16>
      %5 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x512xf32>
      %6 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x512xbf16>
      %7 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<1x1x16x1xbf16>
      %8 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<1x1x16xbf16>
      %9 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x33x16xbf16>
      %10 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %11 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %12 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %13 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x128xbf16>
      %14 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<i1>) -> tensor<64xi1>
      %15 = stablehlo.and %14, %c : tensor<64xi1>
      %16 = stablehlo.and %15, %c_1 : tensor<64xi1>
      %17 = stablehlo.reshape %16 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      %18 = stablehlo.reshape %16 : (tensor<64xi1>) -> tensor<1x64xi1>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x128xi1>
      %20 = stablehlo.not %17 : tensor<1x64x1xi1>
      %21 = stablehlo.reshape %20 : (tensor<1x64x1xi1>) -> tensor<1x64xi1>
      %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x128xi1>
      %23 = stablehlo.reshape %arg21 : (tensor<1x1x2048xbf16>) -> tensor<1x2048xbf16>
      %24 = stablehlo.reshape %arg30 : (tensor<128x2048xbf16>) -> tensor<1x128x2048xbf16>
      %25 = stablehlo.reshape %24 : (tensor<1x128x2048xbf16>) -> tensor<128x2048xbf16>
      %26 = stablehlo.transpose %25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x2048xbf16>) -> tensor<2048x128xbf16>
      %27 = stablehlo.dot_general %23, %26, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x128xbf16>) -> tensor<1x128xbf16>
      %28 = stablehlo.reshape %27 : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
      %29 = stablehlo.reshape %arg29 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %30 = stablehlo.reshape %29 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %31 = stablehlo.reshape %arg28 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %32 = stablehlo.reshape %31 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %33 = stablehlo.composite "tenstorrent.layer_norm" %28, %30, %32 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @outlined_composite_tenstorrent.layer_norm.impl} : (tensor<1x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %34 = stablehlo.slice %33 [0:1, 0:1, 0:64] : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
      %35 = stablehlo.reshape %34 : (tensor<1x1x64xbf16>) -> tensor<1x1x1x2x32xbf16>
      %36 = stablehlo.transpose %35, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,1,1,32,2]{3,4,2,1,0}"} : (tensor<1x1x1x2x32xbf16>) -> tensor<1x1x1x32x2xbf16>
      %37 = stablehlo.convert %36 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,1,1,32,2]{3,4,2,1,0}"} : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x2xf32>
      %38 = stablehlo.slice %37 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %39 = stablehlo.reshape %38 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %40 = stablehlo.reshape %arg25 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %41 = stablehlo.slice %40 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %42 = stablehlo.reshape %41 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %43 = stablehlo.convert %42 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %44 = stablehlo.multiply %39, %43 : tensor<1x1x1x32xf32>
      %45 = stablehlo.slice %37 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %46 = stablehlo.reshape %45 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %47 = stablehlo.slice %40 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %48 = stablehlo.reshape %47 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %49 = stablehlo.convert %48 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %50 = stablehlo.multiply %46, %49 : tensor<1x1x1x32xf32>
      %51 = stablehlo.subtract %44, %50 : tensor<1x1x1x32xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %53 = stablehlo.multiply %39, %49 : tensor<1x1x1x32xf32>
      %54 = stablehlo.multiply %46, %43 : tensor<1x1x1x32xf32>
      %55 = stablehlo.add %53, %54 : tensor<1x1x1x32xf32>
      %56 = stablehlo.reshape %55 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %57 = stablehlo.concatenate %52, %56, dim = 4 : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
      %58 = stablehlo.reshape %57 : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
      %59 = stablehlo.slice %58 [0:1, 0:1, 0:1, 0:64:2] : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
      %60 = stablehlo.slice %58 [0:1, 0:1, 0:1, 1:64:2] : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
      %61 = stablehlo.concatenate %59, %60, dim = 3 : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x64xf32>
      %62 = stablehlo.convert %61 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
      %63 = stablehlo.reshape %62 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
      %64 = stablehlo.slice %33 [0:1, 0:1, 64:128] : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
      %65 = stablehlo.concatenate %63, %64, dim = 2 : (tensor<1x1x64xbf16>, tensor<1x1x64xbf16>) -> tensor<1x1x128xbf16>
      %66 = stablehlo.reshape %65 : (tensor<1x1x128xbf16>) -> tensor<1x128xbf16>
      %67 = stablehlo.reshape %arg27 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %68 = stablehlo.reshape %67 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
      %69 = stablehlo.transpose %68, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %70 = stablehlo.dot_general %66, %69, contracting_dims = [1] x [0] : (tensor<1x128xbf16>, tensor<128x128xbf16>) -> tensor<1x128xbf16>
      %71 = stablehlo.reshape %70 : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
      %72 = stablehlo.floor %cst_4 : tensor<64xf32>
      %73 = stablehlo.convert %72 : (tensor<64xf32>) -> tensor<64xi64>
      %74 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %75 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %76 = stablehlo.clamp %75, %73, %74 : tensor<64xi64>
      %77 = stablehlo.compare  LT, %76, %12 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %78 = stablehlo.add %76, %11 : tensor<64xi64>
      %79 = stablehlo.select %77, %78, %76 : tensor<64xi1>, tensor<64xi64>
      %80 = stablehlo.reshape %79 : (tensor<64xi64>) -> tensor<64x1xi64>
      %81 = "stablehlo.gather"(%71, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 128>}> : (tensor<1x1x128xbf16>, tensor<64x1xi64>) -> tensor<1x64x128xbf16>
      %82 = stablehlo.select %22, %13, %81 : tensor<1x64x128xi1>, tensor<1x64x128xbf16>
      %83 = stablehlo.select %19, %82, %arg26 : tensor<1x64x128xi1>, tensor<1x64x128xbf16>
      %84 = stablehlo.slice %83 [0:1, 0:33, 0:128] : (tensor<1x64x128xbf16>) -> tensor<1x33x128xbf16>
      %85 = stablehlo.reshape %arg36 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %86 = stablehlo.reshape %85 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %87 = stablehlo.convert %86 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %88 = stablehlo.reshape %87 : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %89 = stablehlo.reshape %arg35 : (tensor<3072x2048xbf16>) -> tensor<1x3072x2048xbf16>
      %90 = stablehlo.reshape %89 : (tensor<1x3072x2048xbf16>) -> tensor<3072x2048xbf16>
      %91 = stablehlo.transpose %90, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x2048xbf16>) -> tensor<2048x3072xbf16>
      %92 = stablehlo.dot_general %23, %91, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x3072xbf16>) -> tensor<1x3072xbf16>
      %93 = stablehlo.reshape %92 : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
      %94 = stablehlo.convert %93 : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
      %95 = stablehlo.power %94, %10 : tensor<1x1x3072xf32>
      %96 = stablehlo.reduce(%95 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x3072xf32>, tensor<f32>) -> tensor<1x1xf32>
      %97 = stablehlo.multiply %96, %cst_6 : tensor<1x1xf32>
      %98 = stablehlo.reshape %97 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %99 = stablehlo.add %98, %cst_2 : tensor<1x1x1xf32>
      %100 = stablehlo.rsqrt %99 : tensor<1x1x1xf32>
      %101 = stablehlo.reshape %100 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x3072xf32>
      %103 = stablehlo.multiply %94, %102 : tensor<1x1x3072xf32>
      %104 = stablehlo.multiply %88, %103 : tensor<1x1x3072xf32>
      %105 = stablehlo.convert %104 : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
      %106 = stablehlo.reshape %105 : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
      %107 = stablehlo.reshape %arg34 : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
      %108 = stablehlo.reshape %107 : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
      %109 = stablehlo.transpose %108, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
      %110 = stablehlo.dot_general %106, %109, contracting_dims = [1] x [0] : (tensor<1x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<1x2048xbf16>
      %111 = stablehlo.reshape %110 : (tensor<1x2048xbf16>) -> tensor<1x1x16x128xbf16>
      %112 = stablehlo.slice %111 [0:1, 0:1, 0:16, 0:64] : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
      %113 = stablehlo.reshape %112 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x2x32xbf16>
      %114 = stablehlo.transpose %113, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,1,64,32,2]{3,4,2,1,0}"} : (tensor<1x1x16x2x32xbf16>) -> tensor<1x1x16x32x2xbf16>
      %115 = stablehlo.convert %114 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,1,64,32,2]{3,4,2,1,0}"} : (tensor<1x1x16x32x2xbf16>) -> tensor<1x1x16x32x2xf32>
      %116 = stablehlo.slice %115 [0:1, 0:1, 0:16, 0:32, 0:1] : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
      %117 = stablehlo.reshape %116 : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
      %118 = stablehlo.reshape %43 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x16x32xf32>
      %120 = stablehlo.multiply %117, %119 : tensor<1x1x16x32xf32>
      %121 = stablehlo.slice %115 [0:1, 0:1, 0:16, 0:32, 1:2] : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
      %122 = stablehlo.reshape %121 : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
      %123 = stablehlo.reshape %49 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %124 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x16x32xf32>
      %125 = stablehlo.multiply %122, %124 : tensor<1x1x16x32xf32>
      %126 = stablehlo.subtract %120, %125 : tensor<1x1x16x32xf32>
      %127 = stablehlo.reshape %126 : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
      %128 = stablehlo.multiply %117, %124 : tensor<1x1x16x32xf32>
      %129 = stablehlo.multiply %122, %119 : tensor<1x1x16x32xf32>
      %130 = stablehlo.add %128, %129 : tensor<1x1x16x32xf32>
      %131 = stablehlo.reshape %130 : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
      %132 = stablehlo.concatenate %127, %131, dim = 4 : (tensor<1x1x16x32x1xf32>, tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32x2xf32>
      %133 = stablehlo.reshape %132 : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x64xf32>
      %134 = stablehlo.slice %133 [0:1, 0:1, 0:16, 0:64:2] : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
      %135 = stablehlo.slice %133 [0:1, 0:1, 0:16, 1:64:2] : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
      %136 = stablehlo.concatenate %134, %135, dim = 3 : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x64xf32>
      %137 = stablehlo.convert %136 : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x64xbf16>
      %138 = stablehlo.slice %111 [0:1, 0:1, 0:16, 64:128] : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
      %139 = stablehlo.concatenate %137, %138, dim = 3 : (tensor<1x1x16x64xbf16>, tensor<1x1x16x64xbf16>) -> tensor<1x1x16x128xbf16>
      %140 = stablehlo.dot_general %139, %69, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<1x1x16x128xbf16>
      %141 = stablehlo.reshape %140 : (tensor<1x1x16x128xbf16>) -> tensor<1x16x128xbf16>
      %142 = stablehlo.transpose %141, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,128,64]{1,2,0}"} : (tensor<1x16x128xbf16>) -> tensor<1x128x16xbf16>
      %143 = stablehlo.dot_general %84, %142, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x33x128xbf16>, tensor<1x128x16xbf16>) -> tensor<1x33x16xbf16>
      %144 = stablehlo.maximum %143, %9 : tensor<1x33x16xbf16>
      %145 = stablehlo.reshape %arg33 : (tensor<16x2048xbf16>) -> tensor<1x16x2048xbf16>
      %146 = stablehlo.reshape %145 : (tensor<1x16x2048xbf16>) -> tensor<16x2048xbf16>
      %147 = stablehlo.transpose %146, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<16x2048xbf16>) -> tensor<2048x16xbf16>
      %148 = stablehlo.dot_general %23, %147, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x16xbf16>) -> tensor<1x16xbf16>
      %149 = stablehlo.reshape %148 : (tensor<1x16xbf16>) -> tensor<1x1x16xbf16>
      %150 = stablehlo.multiply %149, %8 : tensor<1x1x16xbf16>
      %151 = stablehlo.reshape %150 : (tensor<1x1x16xbf16>) -> tensor<1x1x16x1xbf16>
      %152 = stablehlo.multiply %151, %7 : tensor<1x1x16x1xbf16>
      %153 = stablehlo.reshape %152 : (tensor<1x1x16x1xbf16>) -> tensor<1x16xbf16>
      %154 = stablehlo.broadcast_in_dim %153, dims = [0, 2] : (tensor<1x16xbf16>) -> tensor<1x33x16xbf16>
      %155 = stablehlo.multiply %144, %154 : tensor<1x33x16xbf16>
      %156 = stablehlo.reduce(%155 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x33x16xbf16>, tensor<bf16>) -> tensor<1x33xbf16>
      %157 = "stablehlo.all_reduce"(%156) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %281 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %281 : tensor<bf16>
      }) : (tensor<1x33xbf16>) -> tensor<1x33xbf16>
      %158 = stablehlo.reshape %157 : (tensor<1x33xbf16>) -> tensor<1x1x33xbf16>
      %159 = stablehlo.composite "tenstorrent.topk_indices" %158 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @outlined_composite_tenstorrent.topk_indices.impl} : (tensor<1x1x33xbf16>) -> tensor<1x1x33xi64>
      %160 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x512xi1>
      %161 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x512xi1>
      %162 = stablehlo.reshape %arg22 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %163 = stablehlo.reshape %162 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %164 = stablehlo.convert %163 : (tensor<512xbf16>) -> tensor<512xf32>
      %165 = stablehlo.reshape %164 : (tensor<512xf32>) -> tensor<1x1x512xf32>
      %166 = stablehlo.reshape %arg20 : (tensor<576x2048xbf16>) -> tensor<1x576x2048xbf16>
      %167 = stablehlo.reshape %166 : (tensor<1x576x2048xbf16>) -> tensor<576x2048xbf16>
      %168 = stablehlo.transpose %167, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x2048xbf16>) -> tensor<2048x576xbf16>
      %169 = stablehlo.dot_general %23, %168, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x576xbf16>) -> tensor<1x576xbf16>
      %170 = stablehlo.reshape %169 : (tensor<1x576xbf16>) -> tensor<1x1x576xbf16>
      %171 = stablehlo.slice %170 [0:1, 0:1, 0:512] : (tensor<1x1x576xbf16>) -> tensor<1x1x512xbf16>
      %172 = stablehlo.convert %171 : (tensor<1x1x512xbf16>) -> tensor<1x1x512xf32>
      %173 = stablehlo.power %172, %5 : tensor<1x1x512xf32>
      %174 = stablehlo.reduce(%173 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1xf32>
      %175 = stablehlo.multiply %174, %cst_10 : tensor<1x1xf32>
      %176 = stablehlo.reshape %175 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %177 = stablehlo.add %176, %cst_2 : tensor<1x1x1xf32>
      %178 = stablehlo.rsqrt %177 : tensor<1x1x1xf32>
      %179 = stablehlo.reshape %178 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x512xf32>
      %181 = stablehlo.multiply %172, %180 : tensor<1x1x512xf32>
      %182 = stablehlo.multiply %165, %181 : tensor<1x1x512xf32>
      %183 = stablehlo.convert %182 : (tensor<1x1x512xf32>) -> tensor<1x1x512xbf16>
      %184 = "stablehlo.gather"(%183, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<1x1x512xbf16>, tensor<64x1xi64>) -> tensor<1x64x512xbf16>
      %185 = stablehlo.select %161, %6, %184 : tensor<1x64x512xi1>, tensor<1x64x512xbf16>
      %186 = stablehlo.select %160, %185, %arg19 : tensor<1x64x512xi1>, tensor<1x64x512xbf16>
      %187 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x64xi1>
      %188 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x64xi1>
      %189 = stablehlo.slice %170 [0:1, 0:1, 512:576] : (tensor<1x1x576xbf16>) -> tensor<1x1x64xbf16>
      %190 = stablehlo.reshape %189 : (tensor<1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
      %191 = stablehlo.convert %190 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xf32>
      %192 = stablehlo.reshape %191 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32x2xf32>
      %193 = stablehlo.slice %192 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %194 = stablehlo.reshape %193 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %195 = stablehlo.multiply %194, %43 : tensor<1x1x1x32xf32>
      %196 = stablehlo.slice %192 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %197 = stablehlo.reshape %196 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %198 = stablehlo.multiply %197, %49 : tensor<1x1x1x32xf32>
      %199 = stablehlo.subtract %195, %198 : tensor<1x1x1x32xf32>
      %200 = stablehlo.reshape %199 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %201 = stablehlo.multiply %194, %49 : tensor<1x1x1x32xf32>
      %202 = stablehlo.multiply %197, %43 : tensor<1x1x1x32xf32>
      %203 = stablehlo.add %201, %202 : tensor<1x1x1x32xf32>
      %204 = stablehlo.reshape %203 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %205 = stablehlo.concatenate %200, %204, dim = 4 : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
      %206 = stablehlo.reshape %205 : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
      %207 = stablehlo.convert %206 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
      %208 = stablehlo.reshape %207 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
      %209 = "stablehlo.gather"(%208, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 64>}> : (tensor<1x1x64xbf16>, tensor<64x1xi64>) -> tensor<1x64x64xbf16>
      %210 = stablehlo.select %188, %4, %209 : tensor<1x64x64xi1>, tensor<1x64x64xbf16>
      %211 = stablehlo.select %187, %210, %arg24 : tensor<1x64x64xi1>, tensor<1x64x64xbf16>
      %212 = stablehlo.reshape %arg37 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %213 = stablehlo.reshape %212 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %214 = stablehlo.transpose %213, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %215 = stablehlo.dot_general %106, %214, contracting_dims = [1] x [0] : (tensor<1x3072xbf16>, tensor<3072x768xbf16>) -> tensor<1x768xbf16>
      %216 = stablehlo.reshape %215 : (tensor<1x768xbf16>) -> tensor<1x1x4x192xbf16>
      %217 = stablehlo.slice %216 [0:1, 0:1, 0:4, 0:128] : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x128xbf16>
      %218 = stablehlo.reshape %arg32 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %219 = stablehlo.reshape %218 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %220 = stablehlo.slice %219 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %221 = stablehlo.dot_general %217, %220, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x512xbf16>
      %222 = stablehlo.reshape %221 : (tensor<4x1x1x512xbf16>) -> tensor<1x1x4x512xbf16>
      %223 = stablehlo.slice %186 [0:1, 0:33, 0:512] : (tensor<1x64x512xbf16>) -> tensor<1x33x512xbf16>
      %224 = stablehlo.reshape %159 : (tensor<1x1x33xi64>) -> tensor<1x33xi64>
      %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1] : (tensor<1x33xi64>) -> tensor<1x33x512xi64>
      %226 = stablehlo.convert %225 : (tensor<1x33x512xi64>) -> tensor<1x33x512xui32>
      %227 = stablehlo.reshape %226 : (tensor<1x33x512xui32>) -> tensor<1x33x512x1xui32>
      %228 = stablehlo.iota dim = 0 : tensor<512xui32>
      %229 = stablehlo.broadcast_in_dim %228, dims = [2] : (tensor<512xui32>) -> tensor<1x33x512x1xui32>
      %230 = stablehlo.concatenate %3, %227, %229, dim = 3 : (tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>) -> tensor<1x33x512x3xui32>
      %231 = "stablehlo.gather"(%223, %230) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x33x512xbf16>, tensor<1x33x512x3xui32>) -> tensor<1x33x512xbf16>
      %232 = stablehlo.dot_general %222, %231, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x512xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x33xbf16>
      %233 = stablehlo.slice %216 [0:1, 0:1, 0:4, 128:192] : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x64xbf16>
      %234 = stablehlo.convert %233 : (tensor<1x1x4x64xbf16>) -> tensor<1x1x4x64xf32>
      %235 = stablehlo.reshape %234 : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x32x2xf32>
      %236 = stablehlo.slice %235 [0:1, 0:1, 0:4, 0:32, 0:1] : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
      %237 = stablehlo.reshape %236 : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
      %238 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
      %239 = stablehlo.multiply %237, %238 : tensor<1x1x4x32xf32>
      %240 = stablehlo.slice %235 [0:1, 0:1, 0:4, 0:32, 1:2] : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
      %241 = stablehlo.reshape %240 : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
      %242 = stablehlo.broadcast_in_dim %123, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
      %243 = stablehlo.multiply %241, %242 : tensor<1x1x4x32xf32>
      %244 = stablehlo.subtract %239, %243 : tensor<1x1x4x32xf32>
      %245 = stablehlo.reshape %244 : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
      %246 = stablehlo.multiply %237, %242 : tensor<1x1x4x32xf32>
      %247 = stablehlo.multiply %241, %238 : tensor<1x1x4x32xf32>
      %248 = stablehlo.add %246, %247 : tensor<1x1x4x32xf32>
      %249 = stablehlo.reshape %248 : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
      %250 = stablehlo.concatenate %245, %249, dim = 4 : (tensor<1x1x4x32x1xf32>, tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32x2xf32>
      %251 = stablehlo.reshape %250 : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x64xf32>
      %252 = stablehlo.convert %251 : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x64xbf16>
      %253 = stablehlo.slice %211 [0:1, 0:33, 0:64] : (tensor<1x64x64xbf16>) -> tensor<1x33x64xbf16>
      %254 = stablehlo.broadcast_in_dim %224, dims = [0, 1] : (tensor<1x33xi64>) -> tensor<1x33x64xi64>
      %255 = stablehlo.convert %254 : (tensor<1x33x64xi64>) -> tensor<1x33x64xui32>
      %256 = stablehlo.reshape %255 : (tensor<1x33x64xui32>) -> tensor<1x33x64x1xui32>
      %257 = stablehlo.iota dim = 0 : tensor<64xui32>
      %258 = stablehlo.broadcast_in_dim %257, dims = [2] : (tensor<64xui32>) -> tensor<1x33x64x1xui32>
      %259 = stablehlo.concatenate %2, %256, %258, dim = 3 : (tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>) -> tensor<1x33x64x3xui32>
      %260 = "stablehlo.gather"(%253, %259) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x33x64xbf16>, tensor<1x33x64x3xui32>) -> tensor<1x33x64xbf16>
      %261 = stablehlo.dot_general %252, %260, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x64xbf16>, tensor<1x33x64xbf16>) -> tensor<1x1x4x33xbf16>
      %262 = stablehlo.add %232, %261 : tensor<1x1x4x33xbf16>
      %263 = stablehlo.multiply %262, %1 : tensor<1x1x4x33xbf16>
      %264 = stablehlo.reduce(%263 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x33xbf16>, tensor<bf16>) -> tensor<1x1x4xbf16>
      %265 = stablehlo.broadcast_in_dim %264, dims = [0, 1, 2] : (tensor<1x1x4xbf16>) -> tensor<1x1x4x33xbf16>
      %266 = stablehlo.subtract %263, %265 : tensor<1x1x4x33xbf16>
      %267 = stablehlo.exponential %266 : tensor<1x1x4x33xbf16>
      %268 = stablehlo.reduce(%267 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x33xbf16>, tensor<bf16>) -> tensor<1x1x4xbf16>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x1x4xbf16>) -> tensor<1x1x4x33xbf16>
      %270 = stablehlo.divide %267, %269 : tensor<1x1x4x33xbf16>
      %271 = stablehlo.dot_general %270, %231, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x33xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x512xbf16>
      %272 = stablehlo.slice %219 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %273 = stablehlo.dot_general %271, %272, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x128xbf16>
      %274 = stablehlo.reshape %273 : (tensor<4x1x1x128xbf16>) -> tensor<1x512xbf16>
      %275 = stablehlo.reshape %arg31 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
      %276 = stablehlo.reshape %275 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
      %277 = stablehlo.transpose %276, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
      %278 = stablehlo.dot_general %274, %277, contracting_dims = [1] x [0] : (tensor<1x512xbf16>, tensor<512x2048xbf16>) -> tensor<1x2048xbf16>
      %279 = "stablehlo.all_reduce"(%278) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %281 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %281 : tensor<bf16>
      }) : (tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
      %280 = stablehlo.reshape %279 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
      sdy.return %186, %211, %83, %280 : tensor<1x64x512xbf16>, tensor<1x64x64xbf16>, tensor<1x64x128xbf16>, tensor<1x1x2048xbf16>
    } : (tensor<1x64x512xbf16>, tensor<576x2048xbf16>, tensor<1x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<1x64x64xbf16>, tensor<1x32x2xbf16>, tensor<1x64x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128x2048xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<64x2048xbf16>, tensor<8192x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>, tensor<3072x3072xbf16>) -> (tensor<1x64x512xbf16>, tensor<1x64x64xbf16>, tensor<1x64x128xbf16>, tensor<1x1x2048xbf16>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<1x64x512xbf16>, tensor<1x64x64xbf16>, tensor<1x64x128xbf16>, tensor<1x1x2048xbf16>
  }
  func.func private @outlined_composite_tenstorrent.topk_indices.impl(%arg0: tensor<1x1x33xbf16>) -> tensor<1x1x33xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<33xi32>
    %1 = stablehlo.reshape %0 : (tensor<33xi32>) -> tensor<1x1x33xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 2 : i64}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %4 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) : (tensor<1x1x33xbf16>, tensor<1x1x33xi32>) -> (tensor<1x1x33xbf16>, tensor<1x1x33xi32>)
    %3 = stablehlo.convert %2#1 : (tensor<1x1x33xi32>) -> tensor<1x1x33xi64>
    return %3 : tensor<1x1x33xi64>
  }
  func.func private @outlined_composite_tenstorrent.layer_norm.impl(%arg0: tensor<1x1x128xbf16>, %arg1: tensor<128xbf16>, %arg2: tensor<128xbf16>) -> tensor<1x1x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<1x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x1x128xbf16>) -> tensor<1x1x128xf32>
    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<1x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x128xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<1x1x128xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x1x128xf32>
    %6 = stablehlo.reduce(%5 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x1x128xf32>, tensor<f32>) -> tensor<1x1xf32>
    %7 = stablehlo.multiply %6, %cst_0 : tensor<1x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %9 = stablehlo.add %8, %cst : tensor<1x1x1xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x1x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x128xf32>
    %13 = stablehlo.multiply %4, %12 : tensor<1x1x128xf32>
    %14 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %15 = stablehlo.reshape %14 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %16 = stablehlo.multiply %13, %15 : tensor<1x1x128xf32>
    %17 = stablehlo.convert %arg2 : (tensor<128xbf16>) -> tensor<128xf32>
    %18 = stablehlo.reshape %17 : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %19 = stablehlo.add %16, %18 : tensor<1x1x128xf32>
    %20 = stablehlo.convert %19 : (tensor<1x1x128xf32>) -> tensor<1x1x128xbf16>
    return %20 : tensor<1x1x128xbf16>
  }
}
