module @SyncTensorsGraph.762 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x64x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<4x64x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {"_axis_1"}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {"_axis_0"}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>, ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<4x64x512xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x128xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %c = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
    %c_1 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<4x1x1xf32>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %cst_4 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %cst_6 = stablehlo.constant dense<3.25520843E-4> : tensor<4x1xf32>
    %cst_7 = stablehlo.constant dense<1.250000e-01> : tensor<bf16>
    %cst_8 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
    %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_10 = stablehlo.constant dense<0.001953125> : tensor<4x1xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_12 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<bf16>) -> tensor<4x1x16x33xbf16>
    %1 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<4x64x64xbf16>
    %2 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<4x1x512xf32>
    %3 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<4x64x512xbf16>
    %4 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<4x1x64x1xbf16>
    %5 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<4x1x64xbf16>
    %6 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<4x33x64xbf16>
    %7 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<4x1x3072xf32>
    %8 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %10 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<4x64x128xbf16>
    %11 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<i1>) -> tensor<64xi1>
    %12 = stablehlo.and %11, %c : tensor<64xi1>
    %13 = stablehlo.and %12, %c_1 : tensor<64xi1>
    %14 = stablehlo.reshape %13 : (tensor<64xi1>) -> tensor<1x64x1xi1>
    %15 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<64xi1>) -> tensor<4x64x128xi1>
    %16 = stablehlo.not %14 : tensor<1x64x1xi1>
    %17 = stablehlo.reshape %16 : (tensor<1x64x1xi1>) -> tensor<64xi1>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<4x64x128xi1>
    %19 = stablehlo.reshape %arg2 : (tensor<4x1x2048xbf16>) -> tensor<4x2048xbf16>
    %20 = stablehlo.reshape %arg11 : (tensor<128x2048xbf16>) -> tensor<1x128x2048xbf16>
    %21 = stablehlo.reshape %20 : (tensor<1x128x2048xbf16>) -> tensor<128x2048xbf16>
    %22 = stablehlo.transpose %21, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x2048xbf16>) -> tensor<2048x128xbf16>
    %23 = stablehlo.dot_general %19, %22, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x128xbf16>) -> tensor<4x128xbf16>
    %24 = stablehlo.reshape %23 : (tensor<4x128xbf16>) -> tensor<4x1x128xbf16>
    %25 = stablehlo.reshape %arg10 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
    %26 = stablehlo.reshape %25 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
    %27 = stablehlo.reshape %arg9 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
    %28 = stablehlo.reshape %27 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
    %29 = stablehlo.composite "tenstorrent.layer_norm" %24, %26, %28 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @tenstorrent.layer_norm.impl} : (tensor<4x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<4x1x128xbf16>
    %30 = stablehlo.slice %29 [0:4, 0:1, 0:64] : (tensor<4x1x128xbf16>) -> tensor<4x1x64xbf16>
    %31 = stablehlo.reshape %30 : (tensor<4x1x64xbf16>) -> tensor<4x1x1x2x32xbf16>
    %32 = stablehlo.transpose %31, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<4x1x1x2x32xbf16>) -> tensor<4x1x1x32x2xbf16>
    %33 = stablehlo.convert %32 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<4x1x1x32x2xbf16>) -> tensor<4x1x1x32x2xf32>
    %34 = stablehlo.slice %33 [0:4, 0:1, 0:1, 0:32, 0:1] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %35 = stablehlo.reshape %34 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %36 = stablehlo.reshape %arg6 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
    %37 = stablehlo.slice %36 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
    %38 = stablehlo.reshape %37 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
    %39 = stablehlo.convert %38 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<4x1x1x32xf32>
    %42 = stablehlo.multiply %35, %41 : tensor<4x1x1x32xf32>
    %43 = stablehlo.slice %33 [0:4, 0:1, 0:1, 0:32, 1:2] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %44 = stablehlo.reshape %43 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %45 = stablehlo.slice %36 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
    %46 = stablehlo.reshape %45 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
    %47 = stablehlo.convert %46 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<4x1x1x32xf32>
    %50 = stablehlo.multiply %44, %49 : tensor<4x1x1x32xf32>
    %51 = stablehlo.subtract %42, %50 : tensor<4x1x1x32xf32>
    %52 = stablehlo.reshape %51 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %53 = stablehlo.multiply %35, %49 : tensor<4x1x1x32xf32>
    %54 = stablehlo.multiply %44, %41 : tensor<4x1x1x32xf32>
    %55 = stablehlo.add %53, %54 : tensor<4x1x1x32xf32>
    %56 = stablehlo.reshape %55 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %57 = stablehlo.concatenate %52, %56, dim = 4 : (tensor<4x1x1x32x1xf32>, tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32x2xf32>
    %58 = stablehlo.reshape %57 : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x64xf32>
    %59 = stablehlo.slice %58 [0:4, 0:1, 0:1, 0:64:2] : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32xf32>
    %60 = stablehlo.slice %58 [0:4, 0:1, 0:1, 1:64:2] : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32xf32>
    %61 = stablehlo.concatenate %59, %60, dim = 3 : (tensor<4x1x1x32xf32>, tensor<4x1x1x32xf32>) -> tensor<4x1x1x64xf32>
    %62 = stablehlo.convert %61 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x64xbf16>
    %63 = stablehlo.reshape %62 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x64xbf16>
    %64 = stablehlo.slice %29 [0:4, 0:1, 64:128] : (tensor<4x1x128xbf16>) -> tensor<4x1x64xbf16>
    %65 = stablehlo.concatenate %63, %64, dim = 2 : (tensor<4x1x64xbf16>, tensor<4x1x64xbf16>) -> tensor<4x1x128xbf16>
    %66 = stablehlo.reshape %65 : (tensor<4x1x128xbf16>) -> tensor<4x128xbf16>
    %67 = stablehlo.reshape %arg8 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
    %68 = stablehlo.reshape %67 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
    %69 = stablehlo.transpose %68, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %70 = stablehlo.dot_general %66, %69, contracting_dims = [1] x [0] : (tensor<4x128xbf16>, tensor<128x128xbf16>) -> tensor<4x128xbf16>
    %71 = stablehlo.reshape %70 : (tensor<4x128xbf16>) -> tensor<4x1x128xbf16>
    %72 = stablehlo.floor %cst_4 : tensor<64xf32>
    %73 = stablehlo.convert %72 : (tensor<64xf32>) -> tensor<64xi64>
    %74 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %75 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %76 = stablehlo.clamp %75, %73, %74 : tensor<64xi64>
    %77 = stablehlo.compare  LT, %76, %9 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %78 = stablehlo.add %76, %8 : tensor<64xi64>
    %79 = stablehlo.select %77, %78, %76 : tensor<64xi1>, tensor<64xi64>
    %80 = stablehlo.reshape %79 : (tensor<64xi64>) -> tensor<64x1xi64>
    %81 = "stablehlo.gather"(%71, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 128>}> : (tensor<4x1x128xbf16>, tensor<64x1xi64>) -> tensor<4x64x128xbf16>
    %82 = stablehlo.select %18, %10, %81 : tensor<4x64x128xi1>, tensor<4x64x128xbf16>
    %83 = stablehlo.select %15, %82, %arg7 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x128xi1>, tensor<4x64x128xbf16>
    %84 = stablehlo.slice %83 [0:4, 0:33, 0:128] : (tensor<4x64x128xbf16>) -> tensor<4x33x128xbf16>
    %85 = stablehlo.reshape %arg17 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
    %86 = stablehlo.reshape %85 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
    %87 = stablehlo.convert %86 : (tensor<3072xbf16>) -> tensor<3072xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [2] : (tensor<3072xf32>) -> tensor<4x1x3072xf32>
    %89 = stablehlo.reshape %arg16 : (tensor<3072x2048xbf16>) -> tensor<1x3072x2048xbf16>
    %90 = stablehlo.reshape %89 : (tensor<1x3072x2048xbf16>) -> tensor<3072x2048xbf16>
    %91 = stablehlo.transpose %90, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x2048xbf16>) -> tensor<2048x3072xbf16>
    %92 = stablehlo.dot_general %19, %91, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x3072xbf16>) -> tensor<4x3072xbf16>
    %93 = stablehlo.reshape %92 : (tensor<4x3072xbf16>) -> tensor<4x1x3072xbf16>
    %94 = stablehlo.convert %93 : (tensor<4x1x3072xbf16>) -> tensor<4x1x3072xf32>
    %95 = stablehlo.power %94, %7 : tensor<4x1x3072xf32>
    %96 = stablehlo.reduce(%95 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4x1x3072xf32>, tensor<f32>) -> tensor<4x1xf32>
    %97 = stablehlo.multiply %96, %cst_6 : tensor<4x1xf32>
    %98 = stablehlo.reshape %97 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %99 = stablehlo.add %98, %cst_2 : tensor<4x1x1xf32>
    %100 = stablehlo.rsqrt %99 : tensor<4x1x1xf32>
    %101 = stablehlo.reshape %100 : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x3072xf32>
    %103 = stablehlo.multiply %94, %102 : tensor<4x1x3072xf32>
    %104 = stablehlo.multiply %88, %103 : tensor<4x1x3072xf32>
    %105 = stablehlo.convert %104 : (tensor<4x1x3072xf32>) -> tensor<4x1x3072xbf16>
    %106 = stablehlo.reshape %105 : (tensor<4x1x3072xbf16>) -> tensor<4x3072xbf16>
    %107 = stablehlo.reshape %arg15 : (tensor<8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %108 = stablehlo.reshape %107 : (tensor<1x8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %109 = stablehlo.transpose %108, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<8192x3072xbf16>) -> tensor<3072x8192xbf16>
    %110 = stablehlo.dot_general %106, %109, contracting_dims = [1] x [0] : (tensor<4x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<4x8192xbf16>
    %111 = stablehlo.reshape %110 : (tensor<4x8192xbf16>) -> tensor<4x1x64x128xbf16>
    %112 = stablehlo.slice %111 [0:4, 0:1, 0:64, 0:64] : (tensor<4x1x64x128xbf16>) -> tensor<4x1x64x64xbf16>
    %113 = stablehlo.reshape %112 : (tensor<4x1x64x64xbf16>) -> tensor<4x1x64x2x32xbf16>
    %114 = stablehlo.transpose %113, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<4x1x64x2x32xbf16>) -> tensor<4x1x64x32x2xbf16>
    %115 = stablehlo.convert %114 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<4x1x64x32x2xbf16>) -> tensor<4x1x64x32x2xf32>
    %116 = stablehlo.slice %115 [0:4, 0:1, 0:64, 0:32, 0:1] : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x32x1xf32>
    %117 = stablehlo.reshape %116 : (tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32xf32>
    %118 = stablehlo.reshape %39 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x64x32xf32>
    %120 = stablehlo.multiply %117, %119 : tensor<4x1x64x32xf32>
    %121 = stablehlo.slice %115 [0:4, 0:1, 0:64, 0:32, 1:2] : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x32x1xf32>
    %122 = stablehlo.reshape %121 : (tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32xf32>
    %123 = stablehlo.reshape %47 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x64x32xf32>
    %125 = stablehlo.multiply %122, %124 : tensor<4x1x64x32xf32>
    %126 = stablehlo.subtract %120, %125 : tensor<4x1x64x32xf32>
    %127 = stablehlo.reshape %126 : (tensor<4x1x64x32xf32>) -> tensor<4x1x64x32x1xf32>
    %128 = stablehlo.multiply %117, %124 : tensor<4x1x64x32xf32>
    %129 = stablehlo.multiply %122, %119 : tensor<4x1x64x32xf32>
    %130 = stablehlo.add %128, %129 : tensor<4x1x64x32xf32>
    %131 = stablehlo.reshape %130 : (tensor<4x1x64x32xf32>) -> tensor<4x1x64x32x1xf32>
    %132 = stablehlo.concatenate %127, %131, dim = 4 : (tensor<4x1x64x32x1xf32>, tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32x2xf32>
    %133 = stablehlo.reshape %132 : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x64xf32>
    %134 = stablehlo.slice %133 [0:4, 0:1, 0:64, 0:64:2] : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x32xf32>
    %135 = stablehlo.slice %133 [0:4, 0:1, 0:64, 1:64:2] : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x32xf32>
    %136 = stablehlo.concatenate %134, %135, dim = 3 : (tensor<4x1x64x32xf32>, tensor<4x1x64x32xf32>) -> tensor<4x1x64x64xf32>
    %137 = stablehlo.convert %136 : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x64xbf16>
    %138 = stablehlo.slice %111 [0:4, 0:1, 0:64, 64:128] : (tensor<4x1x64x128xbf16>) -> tensor<4x1x64x64xbf16>
    %139 = stablehlo.concatenate %137, %138, dim = 3 : (tensor<4x1x64x64xbf16>, tensor<4x1x64x64xbf16>) -> tensor<4x1x64x128xbf16>
    %140 = stablehlo.dot_general %139, %69, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x64x128xbf16>, tensor<128x128xbf16>) -> tensor<4x1x64x128xbf16>
    %141 = stablehlo.reshape %140 : (tensor<4x1x64x128xbf16>) -> tensor<4x64x128xbf16>
    %142 = stablehlo.transpose %141, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[4,128,64]{1,2,0}"} : (tensor<4x64x128xbf16>) -> tensor<4x128x64xbf16>
    %143 = stablehlo.dot_general %84, %142, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x33x128xbf16>, tensor<4x128x64xbf16>) -> tensor<4x33x64xbf16>
    %144 = stablehlo.maximum %143, %6 : tensor<4x33x64xbf16>
    %145 = stablehlo.reshape %arg14 : (tensor<64x2048xbf16>) -> tensor<1x64x2048xbf16>
    %146 = stablehlo.reshape %145 : (tensor<1x64x2048xbf16>) -> tensor<64x2048xbf16>
    %147 = stablehlo.transpose %146, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<64x2048xbf16>) -> tensor<2048x64xbf16>
    %148 = stablehlo.dot_general %19, %147, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x64xbf16>) -> tensor<4x64xbf16>
    %149 = stablehlo.reshape %148 : (tensor<4x64xbf16>) -> tensor<4x1x64xbf16>
    %150 = stablehlo.multiply %149, %5 : tensor<4x1x64xbf16>
    %151 = stablehlo.reshape %150 : (tensor<4x1x64xbf16>) -> tensor<4x1x64x1xbf16>
    %152 = stablehlo.multiply %151, %4 : tensor<4x1x64x1xbf16>
    %153 = stablehlo.reshape %152 : (tensor<4x1x64x1xbf16>) -> tensor<4x64xbf16>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 2] : (tensor<4x64xbf16>) -> tensor<4x33x64xbf16>
    %155 = stablehlo.multiply %144, %154 : tensor<4x33x64xbf16>
    %156 = stablehlo.reduce(%155 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<4x33x64xbf16>, tensor<bf16>) -> tensor<4x33xbf16>
    %157 = stablehlo.reshape %156 : (tensor<4x33xbf16>) -> tensor<4x1x33xbf16>
    %158 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<64xi1>) -> tensor<4x64x512xi1>
    %159 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<4x64x512xi1>
    %160 = stablehlo.reshape %arg3 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
    %161 = stablehlo.reshape %160 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
    %162 = stablehlo.convert %161 : (tensor<512xbf16>) -> tensor<512xf32>
    %163 = stablehlo.broadcast_in_dim %162, dims = [2] : (tensor<512xf32>) -> tensor<4x1x512xf32>
    %164 = stablehlo.reshape %arg1 : (tensor<576x2048xbf16>) -> tensor<1x576x2048xbf16>
    %165 = stablehlo.reshape %164 : (tensor<1x576x2048xbf16>) -> tensor<576x2048xbf16>
    %166 = stablehlo.transpose %165, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x2048xbf16>) -> tensor<2048x576xbf16>
    %167 = stablehlo.dot_general %19, %166, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x576xbf16>) -> tensor<4x576xbf16>
    %168 = stablehlo.reshape %167 : (tensor<4x576xbf16>) -> tensor<4x1x576xbf16>
    %169 = stablehlo.slice %168 [0:4, 0:1, 0:512] : (tensor<4x1x576xbf16>) -> tensor<4x1x512xbf16>
    %170 = stablehlo.convert %169 : (tensor<4x1x512xbf16>) -> tensor<4x1x512xf32>
    %171 = stablehlo.power %170, %2 : tensor<4x1x512xf32>
    %172 = stablehlo.reduce(%171 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4x1x512xf32>, tensor<f32>) -> tensor<4x1xf32>
    %173 = stablehlo.multiply %172, %cst_10 : tensor<4x1xf32>
    %174 = stablehlo.reshape %173 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %175 = stablehlo.add %174, %cst_2 : tensor<4x1x1xf32>
    %176 = stablehlo.rsqrt %175 : tensor<4x1x1xf32>
    %177 = stablehlo.reshape %176 : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x512xf32>
    %179 = stablehlo.multiply %170, %178 : tensor<4x1x512xf32>
    %180 = stablehlo.multiply %163, %179 : tensor<4x1x512xf32>
    %181 = stablehlo.convert %180 : (tensor<4x1x512xf32>) -> tensor<4x1x512xbf16>
    %182 = "stablehlo.gather"(%181, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 512>}> : (tensor<4x1x512xbf16>, tensor<64x1xi64>) -> tensor<4x64x512xbf16>
    %183 = stablehlo.select %159, %3, %182 : tensor<4x64x512xi1>, tensor<4x64x512xbf16>
    %184 = stablehlo.select %158, %183, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x512xi1>, tensor<4x64x512xbf16>
    %185 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<64xi1>) -> tensor<4x64x64xi1>
    %186 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<4x64x64xi1>
    %187 = stablehlo.slice %168 [0:4, 0:1, 512:576] : (tensor<4x1x576xbf16>) -> tensor<4x1x64xbf16>
    %188 = stablehlo.reshape %187 : (tensor<4x1x64xbf16>) -> tensor<4x1x1x64xbf16>
    %189 = stablehlo.convert %188 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x1x64xf32>
    %190 = stablehlo.reshape %189 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32x2xf32>
    %191 = stablehlo.slice %190 [0:4, 0:1, 0:1, 0:32, 0:1] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %192 = stablehlo.reshape %191 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %193 = stablehlo.multiply %192, %41 : tensor<4x1x1x32xf32>
    %194 = stablehlo.slice %190 [0:4, 0:1, 0:1, 0:32, 1:2] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %195 = stablehlo.reshape %194 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %196 = stablehlo.multiply %195, %49 : tensor<4x1x1x32xf32>
    %197 = stablehlo.subtract %193, %196 : tensor<4x1x1x32xf32>
    %198 = stablehlo.reshape %197 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %199 = stablehlo.multiply %192, %49 : tensor<4x1x1x32xf32>
    %200 = stablehlo.multiply %195, %41 : tensor<4x1x1x32xf32>
    %201 = stablehlo.add %199, %200 : tensor<4x1x1x32xf32>
    %202 = stablehlo.reshape %201 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %203 = stablehlo.concatenate %198, %202, dim = 4 : (tensor<4x1x1x32x1xf32>, tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32x2xf32>
    %204 = stablehlo.reshape %203 : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x64xf32>
    %205 = stablehlo.convert %204 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x64xbf16>
    %206 = stablehlo.reshape %205 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x64xbf16>
    %207 = "stablehlo.gather"(%206, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 64>}> : (tensor<4x1x64xbf16>, tensor<64x1xi64>) -> tensor<4x64x64xbf16>
    %208 = stablehlo.select %186, %1, %207 : tensor<4x64x64xi1>, tensor<4x64x64xbf16>
    %209 = stablehlo.select %185, %208, %arg5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x64xi1>, tensor<4x64x64xbf16>
    %210 = stablehlo.reshape %arg18 : (tensor<3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %211 = stablehlo.reshape %210 : (tensor<1x3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %212 = stablehlo.transpose %211, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %213 = stablehlo.dot_general %106, %212, contracting_dims = [1] x [0] : (tensor<4x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<4x3072xbf16>
    %214 = stablehlo.reshape %213 : (tensor<4x3072xbf16>) -> tensor<4x1x16x192xbf16>
    %215 = stablehlo.slice %214 [0:4, 0:1, 0:16, 0:128] : (tensor<4x1x16x192xbf16>) -> tensor<4x1x16x128xbf16>
    %216 = stablehlo.reshape %arg13 : (tensor<4096x512xbf16>) -> tensor<1x4096x512xbf16>
    %217 = stablehlo.reshape %216 : (tensor<1x4096x512xbf16>) -> tensor<16x256x512xbf16>
    %218 = stablehlo.slice %217 [0:16, 0:128, 0:512] : (tensor<16x256x512xbf16>) -> tensor<16x128x512xbf16>
    %219 = stablehlo.dot_general %215, %218, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x128xbf16>, tensor<16x128x512xbf16>) -> tensor<16x4x1x512xbf16>
    %220 = stablehlo.transpose %219, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<16x4x1x512xbf16>) -> tensor<4x1x16x512xbf16>
    %221 = stablehlo.slice %184 [0:4, 0:33, 0:512] : (tensor<4x64x512xbf16>) -> tensor<4x33x512xbf16>
    %222 = stablehlo.composite "tenstorrent.topk_indices" %157 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk_indices.impl} : (tensor<4x1x33xbf16>) -> tensor<4x1x33xi64>
    %223 = stablehlo.reshape %222 : (tensor<4x1x33xi64>) -> tensor<4x33xi64>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 1] : (tensor<4x33xi64>) -> tensor<4x33x512xi64>
    %225 = stablehlo.composite "tenstorrent.gather" %221, %224 {composite_attributes = {dim = 1 : i64, sparse_grad = false}, decomposition = @tenstorrent.gather.impl_0} : (tensor<4x33x512xbf16>, tensor<4x33x512xi64>) -> tensor<4x33x512xbf16>
    %226 = stablehlo.dot_general %220, %225, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x512xbf16>, tensor<4x33x512xbf16>) -> tensor<4x1x16x33xbf16>
    %227 = stablehlo.slice %214 [0:4, 0:1, 0:16, 128:192] : (tensor<4x1x16x192xbf16>) -> tensor<4x1x16x64xbf16>
    %228 = stablehlo.convert %227 : (tensor<4x1x16x64xbf16>) -> tensor<4x1x16x64xf32>
    %229 = stablehlo.reshape %228 : (tensor<4x1x16x64xf32>) -> tensor<4x1x16x32x2xf32>
    %230 = stablehlo.slice %229 [0:4, 0:1, 0:16, 0:32, 0:1] : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x32x1xf32>
    %231 = stablehlo.reshape %230 : (tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32xf32>
    %232 = stablehlo.broadcast_in_dim %118, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x16x32xf32>
    %233 = stablehlo.multiply %231, %232 : tensor<4x1x16x32xf32>
    %234 = stablehlo.slice %229 [0:4, 0:1, 0:16, 0:32, 1:2] : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x32x1xf32>
    %235 = stablehlo.reshape %234 : (tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32xf32>
    %236 = stablehlo.broadcast_in_dim %123, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x16x32xf32>
    %237 = stablehlo.multiply %235, %236 : tensor<4x1x16x32xf32>
    %238 = stablehlo.subtract %233, %237 : tensor<4x1x16x32xf32>
    %239 = stablehlo.reshape %238 : (tensor<4x1x16x32xf32>) -> tensor<4x1x16x32x1xf32>
    %240 = stablehlo.multiply %231, %236 : tensor<4x1x16x32xf32>
    %241 = stablehlo.multiply %235, %232 : tensor<4x1x16x32xf32>
    %242 = stablehlo.add %240, %241 : tensor<4x1x16x32xf32>
    %243 = stablehlo.reshape %242 : (tensor<4x1x16x32xf32>) -> tensor<4x1x16x32x1xf32>
    %244 = stablehlo.concatenate %239, %243, dim = 4 : (tensor<4x1x16x32x1xf32>, tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32x2xf32>
    %245 = stablehlo.reshape %244 : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x64xf32>
    %246 = stablehlo.convert %245 : (tensor<4x1x16x64xf32>) -> tensor<4x1x16x64xbf16>
    %247 = stablehlo.slice %209 [0:4, 0:33, 0:64] : (tensor<4x64x64xbf16>) -> tensor<4x33x64xbf16>
    %248 = stablehlo.broadcast_in_dim %223, dims = [0, 1] : (tensor<4x33xi64>) -> tensor<4x33x64xi64>
    %249 = stablehlo.composite "tenstorrent.gather" %247, %248 {composite_attributes = {dim = 1 : i64, sparse_grad = false}, decomposition = @tenstorrent.gather.impl} : (tensor<4x33x64xbf16>, tensor<4x33x64xi64>) -> tensor<4x33x64xbf16>
    %250 = stablehlo.dot_general %246, %249, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x64xbf16>, tensor<4x33x64xbf16>) -> tensor<4x1x16x33xbf16>
    %251 = stablehlo.add %226, %250 : tensor<4x1x16x33xbf16>
    %252 = stablehlo.multiply %251, %0 : tensor<4x1x16x33xbf16>
    %253 = stablehlo.reduce(%252 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<4x1x16x33xbf16>, tensor<bf16>) -> tensor<4x1x16xbf16>
    %254 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2] : (tensor<4x1x16xbf16>) -> tensor<4x1x16x33xbf16>
    %255 = stablehlo.subtract %252, %254 : tensor<4x1x16x33xbf16>
    %256 = stablehlo.exponential %255 : tensor<4x1x16x33xbf16>
    %257 = stablehlo.reduce(%256 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<4x1x16x33xbf16>, tensor<bf16>) -> tensor<4x1x16xbf16>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2] : (tensor<4x1x16xbf16>) -> tensor<4x1x16x33xbf16>
    %259 = stablehlo.divide %256, %258 : tensor<4x1x16x33xbf16>
    %260 = stablehlo.dot_general %259, %225, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x33xbf16>, tensor<4x33x512xbf16>) -> tensor<4x1x16x512xbf16>
    %261 = stablehlo.slice %217 [0:16, 128:256, 0:512] : (tensor<16x256x512xbf16>) -> tensor<16x128x512xbf16>
    %262 = stablehlo.dot_general %260, %261, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x512xbf16>, tensor<16x128x512xbf16>) -> tensor<16x4x1x128xbf16>
    %263 = stablehlo.transpose %262, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<16x4x1x128xbf16>) -> tensor<4x1x16x128xbf16>
    %264 = stablehlo.reshape %263 : (tensor<4x1x16x128xbf16>) -> tensor<4x2048xbf16>
    %265 = stablehlo.reshape %arg12 : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %266 = stablehlo.reshape %265 : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %267 = stablehlo.transpose %266, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %268 = stablehlo.dot_general %264, %267, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<4x2048xbf16>
    %269 = stablehlo.reshape %268 : (tensor<4x2048xbf16>) -> tensor<4x1x2048xbf16>
    %270 = sdy.sharding_constraint %269 <@mesh, [{}, {}, {}]> : tensor<4x1x2048xbf16>
    return %184, %209, %83, %270 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
  }
  func.func private @tenstorrent.gather.impl(%arg0: tensor<4x33x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<4x33x64xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<4x33x64xbf16> {
    %0 = stablehlo.iota dim = 0 : tensor<4xui32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<4xui32>) -> tensor<4x33x64x1xui32>
    %2 = stablehlo.convert %arg1 : (tensor<4x33x64xi64>) -> tensor<4x33x64xui32>
    %3 = stablehlo.reshape %2 : (tensor<4x33x64xui32>) -> tensor<4x33x64x1xui32>
    %4 = stablehlo.iota dim = 0 : tensor<64xui32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<64xui32>) -> tensor<4x33x64x1xui32>
    %6 = stablehlo.concatenate %1, %3, %5, dim = 3 : (tensor<4x33x64x1xui32>, tensor<4x33x64x1xui32>, tensor<4x33x64x1xui32>) -> tensor<4x33x64x3xui32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<4x33x64xbf16>, tensor<4x33x64x3xui32>) -> tensor<4x33x64xbf16>
    return %7 : tensor<4x33x64xbf16>
  }
  func.func private @tenstorrent.gather.impl_0(%arg0: tensor<4x33x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<4x33x512xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<4x33x512xbf16> {
    %0 = stablehlo.iota dim = 0 : tensor<4xui32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<4xui32>) -> tensor<4x33x512x1xui32>
    %2 = stablehlo.convert %arg1 : (tensor<4x33x512xi64>) -> tensor<4x33x512xui32>
    %3 = stablehlo.reshape %2 : (tensor<4x33x512xui32>) -> tensor<4x33x512x1xui32>
    %4 = stablehlo.iota dim = 0 : tensor<512xui32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<512xui32>) -> tensor<4x33x512x1xui32>
    %6 = stablehlo.concatenate %1, %3, %5, dim = 3 : (tensor<4x33x512x1xui32>, tensor<4x33x512x1xui32>, tensor<4x33x512x1xui32>) -> tensor<4x33x512x3xui32>
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<4x33x512xbf16>, tensor<4x33x512x3xui32>) -> tensor<4x33x512xbf16>
    return %7 : tensor<4x33x512xbf16>
  }
  func.func private @tenstorrent.topk_indices.impl(%arg0: tensor<4x1x33xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<4x1x33xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<33xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<33xi32>) -> tensor<4x1x33xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 2 : i64}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %4 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) : (tensor<4x1x33xbf16>, tensor<4x1x33xi32>) -> (tensor<4x1x33xbf16>, tensor<4x1x33xi32>)
    %3 = stablehlo.convert %2#1 : (tensor<4x1x33xi32>) -> tensor<4x1x33xi64>
    return %3 : tensor<4x1x33xi64>
  }
  func.func private @tenstorrent.layer_norm.impl(%arg0: tensor<4x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> tensor<4x1x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<4x1x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<4x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<4x1x128xbf16>) -> tensor<4x1x128xf32>
    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<4x1x128xf32>, tensor<f32>) -> tensor<4x1xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<4x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x128xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<4x1x128xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<4x1x128xf32>
    %6 = stablehlo.reduce(%5 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<4x1x128xf32>, tensor<f32>) -> tensor<4x1xf32>
    %7 = stablehlo.multiply %6, %cst_0 : tensor<4x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %9 = stablehlo.add %8, %cst : tensor<4x1x1xf32>
    %10 = stablehlo.rsqrt %9 : tensor<4x1x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x128xf32>
    %13 = stablehlo.multiply %4, %12 : tensor<4x1x128xf32>
    %14 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2] : (tensor<128xf32>) -> tensor<4x1x128xf32>
    %16 = stablehlo.multiply %13, %15 : tensor<4x1x128xf32>
    %17 = stablehlo.convert %arg2 : (tensor<128xbf16>) -> tensor<128xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [2] : (tensor<128xf32>) -> tensor<4x1x128xf32>
    %19 = stablehlo.add %16, %18 : tensor<4x1x128xf32>
    %20 = stablehlo.convert %19 : (tensor<4x1x128xf32>) -> tensor<4x1x128xbf16>
    return %20 : tensor<4x1x128xbf16>
  }
}
