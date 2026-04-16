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
    %cst_13 = stablehlo.constant {reoutline.comp_attrs = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, reoutline.group = "composite_tenstorrent.layer_norm.impl", reoutline.orig_name = "tenstorrent.layer_norm", reoutline.seed} dense<9.99999997E-7> : tensor<4x1x1xf32>
    %cst_14 = stablehlo.constant {reoutline.group = "composite_tenstorrent.layer_norm.impl"} dense<7.812500e-03> : tensor<4x1xf32>
    %cst_15 = stablehlo.constant {reoutline.group = "composite_tenstorrent.layer_norm.impl"} dense<0.000000e+00> : tensor<f32>
    %29 = stablehlo.convert %24 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1x128xbf16>) -> tensor<4x1x128xf32>
    %30 = stablehlo.reduce(%29 init: %cst_15) applies stablehlo.add across dimensions = [2] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1x128xf32>, tensor<f32>) -> tensor<4x1xf32>
    %31 = stablehlo.multiply %30, %cst_14 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1xf32>) -> tensor<4x1x128xf32>
    %33 = stablehlo.subtract %29, %32 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x128xf32>
    %34 = stablehlo.multiply %33, %33 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x128xf32>
    %35 = stablehlo.reduce(%34 init: %cst_15) applies stablehlo.add across dimensions = [2] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1x128xf32>, tensor<f32>) -> tensor<4x1xf32>
    %36 = stablehlo.multiply %35, %cst_14 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1xf32>
    %37 = stablehlo.reshape %36 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %38 = stablehlo.add %37, %cst_13 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x1xf32>
    %39 = stablehlo.rsqrt %38 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x1xf32>
    %40 = stablehlo.reshape %39 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<4x1xf32>) -> tensor<4x1x128xf32>
    %42 = stablehlo.multiply %33, %41 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x128xf32>
    %43 = stablehlo.convert %26 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<128xbf16>) -> tensor<128xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [2] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<128xf32>) -> tensor<4x1x128xf32>
    %45 = stablehlo.multiply %42, %44 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x128xf32>
    %46 = stablehlo.convert %28 {reoutline.arg_operand_indices = array<i64: 2>, reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<128xbf16>) -> tensor<128xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [2] {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : (tensor<128xf32>) -> tensor<4x1x128xf32>
    %48 = stablehlo.add %45, %47 {reoutline.group = "composite_tenstorrent.layer_norm.impl"} : tensor<4x1x128xf32>
    %49 = stablehlo.convert %48 {reoutline.group = "composite_tenstorrent.layer_norm.impl", reoutline.result_pos = array<i64: 0>} : (tensor<4x1x128xf32>) -> tensor<4x1x128xbf16>
    %50 = stablehlo.slice %49 [0:4, 0:1, 0:64] : (tensor<4x1x128xbf16>) -> tensor<4x1x64xbf16>
    %51 = stablehlo.reshape %50 : (tensor<4x1x64xbf16>) -> tensor<4x1x1x2x32xbf16>
    %52 = stablehlo.transpose %51, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<4x1x1x2x32xbf16>) -> tensor<4x1x1x32x2xbf16>
    %53 = stablehlo.convert %52 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<4x1x1x32x2xbf16>) -> tensor<4x1x1x32x2xf32>
    %54 = stablehlo.slice %53 [0:4, 0:1, 0:1, 0:32, 0:1] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %55 = stablehlo.reshape %54 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %56 = stablehlo.reshape %arg6 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
    %57 = stablehlo.slice %56 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
    %58 = stablehlo.reshape %57 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
    %59 = stablehlo.convert %58 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<4x1x1x32xf32>
    %62 = stablehlo.multiply %55, %61 : tensor<4x1x1x32xf32>
    %63 = stablehlo.slice %53 [0:4, 0:1, 0:1, 0:32, 1:2] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %64 = stablehlo.reshape %63 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %65 = stablehlo.slice %56 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
    %66 = stablehlo.reshape %65 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
    %67 = stablehlo.convert %66 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<4x1x1x32xf32>
    %70 = stablehlo.multiply %64, %69 : tensor<4x1x1x32xf32>
    %71 = stablehlo.subtract %62, %70 : tensor<4x1x1x32xf32>
    %72 = stablehlo.reshape %71 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %73 = stablehlo.multiply %55, %69 : tensor<4x1x1x32xf32>
    %74 = stablehlo.multiply %64, %61 : tensor<4x1x1x32xf32>
    %75 = stablehlo.add %73, %74 : tensor<4x1x1x32xf32>
    %76 = stablehlo.reshape %75 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %77 = stablehlo.concatenate %72, %76, dim = 4 : (tensor<4x1x1x32x1xf32>, tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32x2xf32>
    %78 = stablehlo.reshape %77 : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x64xf32>
    %79 = stablehlo.slice %78 [0:4, 0:1, 0:1, 0:64:2] : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32xf32>
    %80 = stablehlo.slice %78 [0:4, 0:1, 0:1, 1:64:2] : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32xf32>
    %81 = stablehlo.concatenate %79, %80, dim = 3 : (tensor<4x1x1x32xf32>, tensor<4x1x1x32xf32>) -> tensor<4x1x1x64xf32>
    %82 = stablehlo.convert %81 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x64xbf16>
    %83 = stablehlo.reshape %82 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x64xbf16>
    %84 = stablehlo.slice %49 [0:4, 0:1, 64:128] : (tensor<4x1x128xbf16>) -> tensor<4x1x64xbf16>
    %85 = stablehlo.concatenate %83, %84, dim = 2 : (tensor<4x1x64xbf16>, tensor<4x1x64xbf16>) -> tensor<4x1x128xbf16>
    %86 = stablehlo.reshape %85 : (tensor<4x1x128xbf16>) -> tensor<4x128xbf16>
    %87 = stablehlo.reshape %arg8 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
    %88 = stablehlo.reshape %87 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
    %89 = stablehlo.transpose %88, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %90 = stablehlo.dot_general %86, %89, contracting_dims = [1] x [0] : (tensor<4x128xbf16>, tensor<128x128xbf16>) -> tensor<4x128xbf16>
    %91 = stablehlo.reshape %90 : (tensor<4x128xbf16>) -> tensor<4x1x128xbf16>
    %92 = stablehlo.floor %cst_4 : tensor<64xf32>
    %93 = stablehlo.convert %92 : (tensor<64xf32>) -> tensor<64xi64>
    %94 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %95 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
    %96 = stablehlo.clamp %95, %93, %94 : tensor<64xi64>
    %97 = stablehlo.compare  LT, %96, %9 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %98 = stablehlo.add %96, %8 : tensor<64xi64>
    %99 = stablehlo.select %97, %98, %96 : tensor<64xi1>, tensor<64xi64>
    %100 = stablehlo.reshape %99 : (tensor<64xi64>) -> tensor<64x1xi64>
    %101 = "stablehlo.gather"(%91, %100) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 128>}> : (tensor<4x1x128xbf16>, tensor<64x1xi64>) -> tensor<4x64x128xbf16>
    %102 = stablehlo.select %18, %10, %101 : tensor<4x64x128xi1>, tensor<4x64x128xbf16>
    %103 = stablehlo.select %15, %102, %arg7 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x128xi1>, tensor<4x64x128xbf16>
    %104 = stablehlo.slice %103 [0:4, 0:33, 0:128] : (tensor<4x64x128xbf16>) -> tensor<4x33x128xbf16>
    %105 = stablehlo.reshape %arg17 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
    %106 = stablehlo.reshape %105 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
    %107 = stablehlo.convert %106 : (tensor<3072xbf16>) -> tensor<3072xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [2] : (tensor<3072xf32>) -> tensor<4x1x3072xf32>
    %109 = stablehlo.reshape %arg16 : (tensor<3072x2048xbf16>) -> tensor<1x3072x2048xbf16>
    %110 = stablehlo.reshape %109 : (tensor<1x3072x2048xbf16>) -> tensor<3072x2048xbf16>
    %111 = stablehlo.transpose %110, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x2048xbf16>) -> tensor<2048x3072xbf16>
    %112 = stablehlo.dot_general %19, %111, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x3072xbf16>) -> tensor<4x3072xbf16>
    %113 = stablehlo.reshape %112 : (tensor<4x3072xbf16>) -> tensor<4x1x3072xbf16>
    %114 = stablehlo.convert %113 : (tensor<4x1x3072xbf16>) -> tensor<4x1x3072xf32>
    %115 = stablehlo.power %114, %7 : tensor<4x1x3072xf32>
    %116 = stablehlo.reduce(%115 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4x1x3072xf32>, tensor<f32>) -> tensor<4x1xf32>
    %117 = stablehlo.multiply %116, %cst_6 : tensor<4x1xf32>
    %118 = stablehlo.reshape %117 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %119 = stablehlo.add %118, %cst_2 : tensor<4x1x1xf32>
    %120 = stablehlo.rsqrt %119 : tensor<4x1x1xf32>
    %121 = stablehlo.reshape %120 : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x3072xf32>
    %123 = stablehlo.multiply %114, %122 : tensor<4x1x3072xf32>
    %124 = stablehlo.multiply %108, %123 : tensor<4x1x3072xf32>
    %125 = stablehlo.convert %124 : (tensor<4x1x3072xf32>) -> tensor<4x1x3072xbf16>
    %126 = stablehlo.reshape %125 : (tensor<4x1x3072xbf16>) -> tensor<4x3072xbf16>
    %127 = stablehlo.reshape %arg15 : (tensor<8192x3072xbf16>) -> tensor<1x8192x3072xbf16>
    %128 = stablehlo.reshape %127 : (tensor<1x8192x3072xbf16>) -> tensor<8192x3072xbf16>
    %129 = stablehlo.transpose %128, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<8192x3072xbf16>) -> tensor<3072x8192xbf16>
    %130 = stablehlo.dot_general %126, %129, contracting_dims = [1] x [0] : (tensor<4x3072xbf16>, tensor<3072x8192xbf16>) -> tensor<4x8192xbf16>
    %131 = stablehlo.reshape %130 : (tensor<4x8192xbf16>) -> tensor<4x1x64x128xbf16>
    %132 = stablehlo.slice %131 [0:4, 0:1, 0:64, 0:64] : (tensor<4x1x64x128xbf16>) -> tensor<4x1x64x64xbf16>
    %133 = stablehlo.reshape %132 : (tensor<4x1x64x64xbf16>) -> tensor<4x1x64x2x32xbf16>
    %134 = stablehlo.transpose %133, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<4x1x64x2x32xbf16>) -> tensor<4x1x64x32x2xbf16>
    %135 = stablehlo.convert %134 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<4x1x64x32x2xbf16>) -> tensor<4x1x64x32x2xf32>
    %136 = stablehlo.slice %135 [0:4, 0:1, 0:64, 0:32, 0:1] : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x32x1xf32>
    %137 = stablehlo.reshape %136 : (tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32xf32>
    %138 = stablehlo.reshape %59 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x64x32xf32>
    %140 = stablehlo.multiply %137, %139 : tensor<4x1x64x32xf32>
    %141 = stablehlo.slice %135 [0:4, 0:1, 0:64, 0:32, 1:2] : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x32x1xf32>
    %142 = stablehlo.reshape %141 : (tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32xf32>
    %143 = stablehlo.reshape %67 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x64x32xf32>
    %145 = stablehlo.multiply %142, %144 : tensor<4x1x64x32xf32>
    %146 = stablehlo.subtract %140, %145 : tensor<4x1x64x32xf32>
    %147 = stablehlo.reshape %146 : (tensor<4x1x64x32xf32>) -> tensor<4x1x64x32x1xf32>
    %148 = stablehlo.multiply %137, %144 : tensor<4x1x64x32xf32>
    %149 = stablehlo.multiply %142, %139 : tensor<4x1x64x32xf32>
    %150 = stablehlo.add %148, %149 : tensor<4x1x64x32xf32>
    %151 = stablehlo.reshape %150 : (tensor<4x1x64x32xf32>) -> tensor<4x1x64x32x1xf32>
    %152 = stablehlo.concatenate %147, %151, dim = 4 : (tensor<4x1x64x32x1xf32>, tensor<4x1x64x32x1xf32>) -> tensor<4x1x64x32x2xf32>
    %153 = stablehlo.reshape %152 : (tensor<4x1x64x32x2xf32>) -> tensor<4x1x64x64xf32>
    %154 = stablehlo.slice %153 [0:4, 0:1, 0:64, 0:64:2] : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x32xf32>
    %155 = stablehlo.slice %153 [0:4, 0:1, 0:64, 1:64:2] : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x32xf32>
    %156 = stablehlo.concatenate %154, %155, dim = 3 : (tensor<4x1x64x32xf32>, tensor<4x1x64x32xf32>) -> tensor<4x1x64x64xf32>
    %157 = stablehlo.convert %156 : (tensor<4x1x64x64xf32>) -> tensor<4x1x64x64xbf16>
    %158 = stablehlo.slice %131 [0:4, 0:1, 0:64, 64:128] : (tensor<4x1x64x128xbf16>) -> tensor<4x1x64x64xbf16>
    %159 = stablehlo.concatenate %157, %158, dim = 3 : (tensor<4x1x64x64xbf16>, tensor<4x1x64x64xbf16>) -> tensor<4x1x64x128xbf16>
    %160 = stablehlo.dot_general %159, %89, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x64x128xbf16>, tensor<128x128xbf16>) -> tensor<4x1x64x128xbf16>
    %161 = stablehlo.reshape %160 : (tensor<4x1x64x128xbf16>) -> tensor<4x64x128xbf16>
    %162 = stablehlo.transpose %161, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[4,128,64]{1,2,0}"} : (tensor<4x64x128xbf16>) -> tensor<4x128x64xbf16>
    %163 = stablehlo.dot_general %104, %162, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x33x128xbf16>, tensor<4x128x64xbf16>) -> tensor<4x33x64xbf16>
    %164 = stablehlo.maximum %163, %6 : tensor<4x33x64xbf16>
    %165 = stablehlo.reshape %arg14 : (tensor<64x2048xbf16>) -> tensor<1x64x2048xbf16>
    %166 = stablehlo.reshape %165 : (tensor<1x64x2048xbf16>) -> tensor<64x2048xbf16>
    %167 = stablehlo.transpose %166, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<64x2048xbf16>) -> tensor<2048x64xbf16>
    %168 = stablehlo.dot_general %19, %167, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x64xbf16>) -> tensor<4x64xbf16>
    %169 = stablehlo.reshape %168 : (tensor<4x64xbf16>) -> tensor<4x1x64xbf16>
    %170 = stablehlo.multiply %169, %5 : tensor<4x1x64xbf16>
    %171 = stablehlo.reshape %170 : (tensor<4x1x64xbf16>) -> tensor<4x1x64x1xbf16>
    %172 = stablehlo.multiply %171, %4 : tensor<4x1x64x1xbf16>
    %173 = stablehlo.reshape %172 : (tensor<4x1x64x1xbf16>) -> tensor<4x64xbf16>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 2] : (tensor<4x64xbf16>) -> tensor<4x33x64xbf16>
    %175 = stablehlo.multiply %164, %174 : tensor<4x33x64xbf16>
    %176 = stablehlo.reduce(%175 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<4x33x64xbf16>, tensor<bf16>) -> tensor<4x33xbf16>
    %177 = stablehlo.reshape %176 : (tensor<4x33xbf16>) -> tensor<4x1x33xbf16>
    %178 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<64xi1>) -> tensor<4x64x512xi1>
    %179 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<4x64x512xi1>
    %180 = stablehlo.reshape %arg3 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
    %181 = stablehlo.reshape %180 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
    %182 = stablehlo.convert %181 : (tensor<512xbf16>) -> tensor<512xf32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [2] : (tensor<512xf32>) -> tensor<4x1x512xf32>
    %184 = stablehlo.reshape %arg1 : (tensor<576x2048xbf16>) -> tensor<1x576x2048xbf16>
    %185 = stablehlo.reshape %184 : (tensor<1x576x2048xbf16>) -> tensor<576x2048xbf16>
    %186 = stablehlo.transpose %185, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x2048xbf16>) -> tensor<2048x576xbf16>
    %187 = stablehlo.dot_general %19, %186, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x576xbf16>) -> tensor<4x576xbf16>
    %188 = stablehlo.reshape %187 : (tensor<4x576xbf16>) -> tensor<4x1x576xbf16>
    %189 = stablehlo.slice %188 [0:4, 0:1, 0:512] : (tensor<4x1x576xbf16>) -> tensor<4x1x512xbf16>
    %190 = stablehlo.convert %189 : (tensor<4x1x512xbf16>) -> tensor<4x1x512xf32>
    %191 = stablehlo.power %190, %2 : tensor<4x1x512xf32>
    %192 = stablehlo.reduce(%191 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<4x1x512xf32>, tensor<f32>) -> tensor<4x1xf32>
    %193 = stablehlo.multiply %192, %cst_10 : tensor<4x1xf32>
    %194 = stablehlo.reshape %193 : (tensor<4x1xf32>) -> tensor<4x1x1xf32>
    %195 = stablehlo.add %194, %cst_2 : tensor<4x1x1xf32>
    %196 = stablehlo.rsqrt %195 : tensor<4x1x1xf32>
    %197 = stablehlo.reshape %196 : (tensor<4x1x1xf32>) -> tensor<4x1xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1] : (tensor<4x1xf32>) -> tensor<4x1x512xf32>
    %199 = stablehlo.multiply %190, %198 : tensor<4x1x512xf32>
    %200 = stablehlo.multiply %183, %199 : tensor<4x1x512xf32>
    %201 = stablehlo.convert %200 : (tensor<4x1x512xf32>) -> tensor<4x1x512xbf16>
    %202 = "stablehlo.gather"(%201, %100) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 512>}> : (tensor<4x1x512xbf16>, tensor<64x1xi64>) -> tensor<4x64x512xbf16>
    %203 = stablehlo.select %179, %3, %202 : tensor<4x64x512xi1>, tensor<4x64x512xbf16>
    %204 = stablehlo.select %178, %203, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x512xi1>, tensor<4x64x512xbf16>
    %205 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<64xi1>) -> tensor<4x64x64xi1>
    %206 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<4x64x64xi1>
    %207 = stablehlo.slice %188 [0:4, 0:1, 512:576] : (tensor<4x1x576xbf16>) -> tensor<4x1x64xbf16>
    %208 = stablehlo.reshape %207 : (tensor<4x1x64xbf16>) -> tensor<4x1x1x64xbf16>
    %209 = stablehlo.convert %208 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x1x64xf32>
    %210 = stablehlo.reshape %209 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x32x2xf32>
    %211 = stablehlo.slice %210 [0:4, 0:1, 0:1, 0:32, 0:1] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %212 = stablehlo.reshape %211 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %213 = stablehlo.multiply %212, %61 : tensor<4x1x1x32xf32>
    %214 = stablehlo.slice %210 [0:4, 0:1, 0:1, 0:32, 1:2] : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x32x1xf32>
    %215 = stablehlo.reshape %214 : (tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32xf32>
    %216 = stablehlo.multiply %215, %69 : tensor<4x1x1x32xf32>
    %217 = stablehlo.subtract %213, %216 : tensor<4x1x1x32xf32>
    %218 = stablehlo.reshape %217 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %219 = stablehlo.multiply %212, %69 : tensor<4x1x1x32xf32>
    %220 = stablehlo.multiply %215, %61 : tensor<4x1x1x32xf32>
    %221 = stablehlo.add %219, %220 : tensor<4x1x1x32xf32>
    %222 = stablehlo.reshape %221 : (tensor<4x1x1x32xf32>) -> tensor<4x1x1x32x1xf32>
    %223 = stablehlo.concatenate %218, %222, dim = 4 : (tensor<4x1x1x32x1xf32>, tensor<4x1x1x32x1xf32>) -> tensor<4x1x1x32x2xf32>
    %224 = stablehlo.reshape %223 : (tensor<4x1x1x32x2xf32>) -> tensor<4x1x1x64xf32>
    %225 = stablehlo.convert %224 : (tensor<4x1x1x64xf32>) -> tensor<4x1x1x64xbf16>
    %226 = stablehlo.reshape %225 : (tensor<4x1x1x64xbf16>) -> tensor<4x1x64xbf16>
    %227 = "stablehlo.gather"(%226, %100) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 4, 1, 64>}> : (tensor<4x1x64xbf16>, tensor<64x1xi64>) -> tensor<4x64x64xbf16>
    %228 = stablehlo.select %206, %1, %227 : tensor<4x64x64xi1>, tensor<4x64x64xbf16>
    %229 = stablehlo.select %205, %228, %arg5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} : tensor<4x64x64xi1>, tensor<4x64x64xbf16>
    %230 = stablehlo.reshape %arg18 : (tensor<3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %231 = stablehlo.reshape %230 : (tensor<1x3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %232 = stablehlo.transpose %231, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %233 = stablehlo.dot_general %126, %232, contracting_dims = [1] x [0] : (tensor<4x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<4x3072xbf16>
    %234 = stablehlo.reshape %233 : (tensor<4x3072xbf16>) -> tensor<4x1x16x192xbf16>
    %235 = stablehlo.slice %234 [0:4, 0:1, 0:16, 0:128] : (tensor<4x1x16x192xbf16>) -> tensor<4x1x16x128xbf16>
    %236 = stablehlo.reshape %arg13 : (tensor<4096x512xbf16>) -> tensor<1x4096x512xbf16>
    %237 = stablehlo.reshape %236 : (tensor<1x4096x512xbf16>) -> tensor<16x256x512xbf16>
    %238 = stablehlo.slice %237 [0:16, 0:128, 0:512] : (tensor<16x256x512xbf16>) -> tensor<16x128x512xbf16>
    %239 = stablehlo.dot_general %235, %238, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x128xbf16>, tensor<16x128x512xbf16>) -> tensor<16x4x1x512xbf16>
    %240 = stablehlo.transpose %239, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<16x4x1x512xbf16>) -> tensor<4x1x16x512xbf16>
    %241 = stablehlo.slice %204 [0:4, 0:33, 0:512] : (tensor<4x64x512xbf16>) -> tensor<4x33x512xbf16>
    %242 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, reoutline.group = "composite_tenstorrent.topk_indices.impl", reoutline.orig_name = "tenstorrent.topk_indices", reoutline.seed} : tensor<33xi32>
    %243 = stablehlo.broadcast_in_dim %242, dims = [2] {reoutline.group = "composite_tenstorrent.topk_indices.impl"} : (tensor<33xi32>) -> tensor<4x1x33xi32>
    %244:2 = "stablehlo.sort"(%177, %243) <{dimension = 2 : i64}> ({
    ^bb0(%arg19: tensor<bf16>, %arg20: tensor<bf16>, %arg21: tensor<i32>, %arg22: tensor<i32>):
      %308 = stablehlo.compare  GT, %arg19, %arg20,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %308 : tensor<i1>
    }) {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.topk_indices.impl"} : (tensor<4x1x33xbf16>, tensor<4x1x33xi32>) -> (tensor<4x1x33xbf16>, tensor<4x1x33xi32>)
    %245 = stablehlo.convert %244#1 {reoutline.group = "composite_tenstorrent.topk_indices.impl", reoutline.result_pos = array<i64: 0>} : (tensor<4x1x33xi32>) -> tensor<4x1x33xi64>
    %246 = stablehlo.reshape %245 : (tensor<4x1x33xi64>) -> tensor<4x33xi64>
    %247 = stablehlo.broadcast_in_dim %246, dims = [0, 1] : (tensor<4x33xi64>) -> tensor<4x33x512xi64>
    %248 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<4xui32>
    %249 = stablehlo.broadcast_in_dim %248, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<4xui32>) -> tensor<4x33x512x1xui32>
    %250 = stablehlo.convert %247 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<4x33x512xi64>) -> tensor<4x33x512xui32>
    %251 = stablehlo.reshape %250 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<4x33x512xui32>) -> tensor<4x33x512x1xui32>
    %252 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : tensor<512xui32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<512xui32>) -> tensor<4x33x512x1xui32>
    %254 = stablehlo.concatenate %249, %251, %253, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<4x33x512x1xui32>, tensor<4x33x512x1xui32>, tensor<4x33x512x1xui32>) -> tensor<4x33x512x3xui32>
    %255 = "stablehlo.gather"(%241, %254) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x512xbf16>, tensor<4x33x512x3xui32>) -> tensor<4x33x512xbf16>
    %256 = stablehlo.dot_general %240, %255, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x512xbf16>, tensor<4x33x512xbf16>) -> tensor<4x1x16x33xbf16>
    %257 = stablehlo.slice %234 [0:4, 0:1, 0:16, 128:192] : (tensor<4x1x16x192xbf16>) -> tensor<4x1x16x64xbf16>
    %258 = stablehlo.convert %257 : (tensor<4x1x16x64xbf16>) -> tensor<4x1x16x64xf32>
    %259 = stablehlo.reshape %258 : (tensor<4x1x16x64xf32>) -> tensor<4x1x16x32x2xf32>
    %260 = stablehlo.slice %259 [0:4, 0:1, 0:16, 0:32, 0:1] : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x32x1xf32>
    %261 = stablehlo.reshape %260 : (tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32xf32>
    %262 = stablehlo.broadcast_in_dim %138, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x16x32xf32>
    %263 = stablehlo.multiply %261, %262 : tensor<4x1x16x32xf32>
    %264 = stablehlo.slice %259 [0:4, 0:1, 0:16, 0:32, 1:2] : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x32x1xf32>
    %265 = stablehlo.reshape %264 : (tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32xf32>
    %266 = stablehlo.broadcast_in_dim %143, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<4x1x16x32xf32>
    %267 = stablehlo.multiply %265, %266 : tensor<4x1x16x32xf32>
    %268 = stablehlo.subtract %263, %267 : tensor<4x1x16x32xf32>
    %269 = stablehlo.reshape %268 : (tensor<4x1x16x32xf32>) -> tensor<4x1x16x32x1xf32>
    %270 = stablehlo.multiply %261, %266 : tensor<4x1x16x32xf32>
    %271 = stablehlo.multiply %265, %262 : tensor<4x1x16x32xf32>
    %272 = stablehlo.add %270, %271 : tensor<4x1x16x32xf32>
    %273 = stablehlo.reshape %272 : (tensor<4x1x16x32xf32>) -> tensor<4x1x16x32x1xf32>
    %274 = stablehlo.concatenate %269, %273, dim = 4 : (tensor<4x1x16x32x1xf32>, tensor<4x1x16x32x1xf32>) -> tensor<4x1x16x32x2xf32>
    %275 = stablehlo.reshape %274 : (tensor<4x1x16x32x2xf32>) -> tensor<4x1x16x64xf32>
    %276 = stablehlo.convert %275 : (tensor<4x1x16x64xf32>) -> tensor<4x1x16x64xbf16>
    %277 = stablehlo.slice %229 [0:4, 0:33, 0:64] : (tensor<4x64x64xbf16>) -> tensor<4x33x64xbf16>
    %278 = stablehlo.broadcast_in_dim %246, dims = [0, 1] : (tensor<4x33xi64>) -> tensor<4x33x64xi64>
    %279 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<4xui32>
    %280 = stablehlo.broadcast_in_dim %279, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4xui32>) -> tensor<4x33x64x1xui32>
    %281 = stablehlo.convert %278 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4x33x64xi64>) -> tensor<4x33x64xui32>
    %282 = stablehlo.reshape %281 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4x33x64xui32>) -> tensor<4x33x64x1xui32>
    %283 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<64xui32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<64xui32>) -> tensor<4x33x64x1xui32>
    %285 = stablehlo.concatenate %280, %282, %284, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4x33x64x1xui32>, tensor<4x33x64x1xui32>, tensor<4x33x64x1xui32>) -> tensor<4x33x64x3xui32>
    %286 = "stablehlo.gather"(%277, %285) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x64xbf16>, tensor<4x33x64x3xui32>) -> tensor<4x33x64xbf16>
    %287 = stablehlo.dot_general %276, %286, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x64xbf16>, tensor<4x33x64xbf16>) -> tensor<4x1x16x33xbf16>
    %288 = stablehlo.add %256, %287 : tensor<4x1x16x33xbf16>
    %289 = stablehlo.multiply %288, %0 : tensor<4x1x16x33xbf16>
    %290 = stablehlo.reduce(%289 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<4x1x16x33xbf16>, tensor<bf16>) -> tensor<4x1x16xbf16>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<4x1x16xbf16>) -> tensor<4x1x16x33xbf16>
    %292 = stablehlo.subtract %289, %291 : tensor<4x1x16x33xbf16>
    %293 = stablehlo.exponential %292 : tensor<4x1x16x33xbf16>
    %294 = stablehlo.reduce(%293 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<4x1x16x33xbf16>, tensor<bf16>) -> tensor<4x1x16xbf16>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1, 2] : (tensor<4x1x16xbf16>) -> tensor<4x1x16x33xbf16>
    %296 = stablehlo.divide %293, %295 : tensor<4x1x16x33xbf16>
    %297 = stablehlo.dot_general %296, %255, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x33xbf16>, tensor<4x33x512xbf16>) -> tensor<4x1x16x512xbf16>
    %298 = stablehlo.slice %237 [0:16, 128:256, 0:512] : (tensor<16x256x512xbf16>) -> tensor<16x128x512xbf16>
    %299 = stablehlo.dot_general %297, %298, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<4x1x16x512xbf16>, tensor<16x128x512xbf16>) -> tensor<16x4x1x128xbf16>
    %300 = stablehlo.transpose %299, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<16x4x1x128xbf16>) -> tensor<4x1x16x128xbf16>
    %301 = stablehlo.reshape %300 : (tensor<4x1x16x128xbf16>) -> tensor<4x2048xbf16>
    %302 = stablehlo.reshape %arg12 : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %303 = stablehlo.reshape %302 : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %304 = stablehlo.transpose %303, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %305 = stablehlo.dot_general %301, %304, contracting_dims = [1] x [0] : (tensor<4x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<4x2048xbf16>
    %306 = stablehlo.reshape %305 : (tensor<4x2048xbf16>) -> tensor<4x1x2048xbf16>
    %307 = sdy.sharding_constraint %306 <@mesh, [{}, {}, {}]> : tensor<4x1x2048xbf16>
    return %204, %229, %103, %307 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
  }
}
