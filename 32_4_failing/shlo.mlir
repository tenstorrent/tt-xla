module @SyncTensorsGraph.761 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<4x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<4x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:4 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18) in_shardings=[<@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {}, {"_axis_0"}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{"_axis_0"}, {"_axis_1"}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{"_axis_1"}, {"_axis_0"}]>, <@mesh, [{"_axis_1"}, {}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}]>, <@mesh, [{"_axis_1"}, {}]>] out_shardings=[<@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{"_axis_0"}, {}, {}]>, <@mesh, [{}, {}, {"_axis_0"}]>] manual_axes={"_axis_0", "_axis_1"} (%arg19: tensor<2x64x512xbf16>, %arg20: tensor<576x1024xbf16>, %arg21: tensor<4x1x1024xbf16>, %arg22: tensor<512xbf16>, %arg23: tensor<i1>, %arg24: tensor<2x64x64xbf16>, %arg25: tensor<1x32x2xbf16>, %arg26: tensor<2x64x128xbf16>, %arg27: tensor<128x128xbf16>, %arg28: tensor<128xbf16>, %arg29: tensor<128xbf16>, %arg30: tensor<128x1024xbf16>, %arg31: tensor<1024x512xbf16>, %arg32: tensor<1024x512xbf16>, %arg33: tensor<16x1024xbf16>, %arg34: tensor<2048x3072xbf16>, %arg35: tensor<3072x1024xbf16>, %arg36: tensor<3072xbf16>, %arg37: tensor<768x3072xbf16>) {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %c = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
      %c_1 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<2x1x1xf32>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %cst_4 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
      %c_5 = stablehlo.constant dense<1> : tensor<i64>
      %cst_6 = stablehlo.constant dense<3.25520843E-4> : tensor<2x1xf32>
      %cst_7 = stablehlo.constant dense<1.250000e-01> : tensor<bf16>
      %cst_8 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
      %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %cst_10 = stablehlo.constant dense<0.001953125> : tensor<2x1xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %cst_12 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<bf16>) -> tensor<2x1x4x33xbf16>
      %2 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x64xbf16>
      %3 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<2x1x512xf32>
      %4 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x512xbf16>
      %5 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<2x1x16x1xbf16>
      %6 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x1x16xbf16>
      %7 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x33x16xbf16>
      %8 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %9 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %10 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %11 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x128xbf16>
      %12 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<i1>) -> tensor<64xi1>
      %13 = stablehlo.and %12, %c : tensor<64xi1>
      %14 = stablehlo.and %13, %c_1 : tensor<64xi1>
      %15 = stablehlo.reshape %14 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      %16 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x128xi1>
      %17 = stablehlo.not %15 : tensor<1x64x1xi1>
      %18 = stablehlo.reshape %17 : (tensor<1x64x1xi1>) -> tensor<64xi1>
      %19 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<64xi1>) -> tensor<2x64x128xi1>
      %20 = stablehlo.reshape %arg21 : (tensor<4x1x1024xbf16>) -> tensor<4x1024xbf16>
      %21 = stablehlo.reshape %arg30 : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
      %22 = stablehlo.reshape %21 : (tensor<1x128x1024xbf16>) -> tensor<128x1024xbf16>
      %23 = stablehlo.transpose %22, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x1024xbf16>) -> tensor<1024x128xbf16>
      %24 = stablehlo.dot_general %20, %23, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x128xbf16>) -> tensor<4x128xbf16>
      %25 = "stablehlo.reduce_scatter"(%24) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<4x128xbf16>) -> tensor<2x128xbf16>
      %26 = stablehlo.reshape %25 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %27 = stablehlo.reshape %arg29 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %28 = stablehlo.reshape %27 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %29 = stablehlo.reshape %arg28 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %30 = stablehlo.reshape %29 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %31 = stablehlo.composite "tenstorrent.layer_norm" %26, %28, %30 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @outlined_composite_tenstorrent.layer_norm.impl} : (tensor<2x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<2x1x128xbf16>
      %32 = stablehlo.slice %31 [0:2, 0:1, 0:64] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %33 = stablehlo.reshape %32 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x2x32xbf16>
      %34 = stablehlo.transpose %33, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x2x32xbf16>) -> tensor<2x1x1x32x2xbf16>
      %35 = stablehlo.convert %34 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x32x2xbf16>) -> tensor<2x1x1x32x2xf32>
      %36 = stablehlo.slice %35 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %37 = stablehlo.reshape %36 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %38 = stablehlo.reshape %arg25 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %39 = stablehlo.slice %38 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %40 = stablehlo.reshape %39 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %41 = stablehlo.convert %40 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %42 = stablehlo.reshape %41 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %43 = stablehlo.broadcast_in_dim %42, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %44 = stablehlo.multiply %37, %43 : tensor<2x1x1x32xf32>
      %45 = stablehlo.slice %35 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %46 = stablehlo.reshape %45 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %47 = stablehlo.slice %38 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %48 = stablehlo.reshape %47 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %49 = stablehlo.convert %48 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %51 = stablehlo.broadcast_in_dim %50, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %52 = stablehlo.multiply %46, %51 : tensor<2x1x1x32xf32>
      %53 = stablehlo.subtract %44, %52 : tensor<2x1x1x32xf32>
      %54 = stablehlo.reshape %53 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %55 = stablehlo.multiply %37, %51 : tensor<2x1x1x32xf32>
      %56 = stablehlo.multiply %46, %43 : tensor<2x1x1x32xf32>
      %57 = stablehlo.add %55, %56 : tensor<2x1x1x32xf32>
      %58 = stablehlo.reshape %57 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %59 = stablehlo.concatenate %54, %58, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %60 = stablehlo.reshape %59 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %61 = stablehlo.slice %60 [0:2, 0:1, 0:1, 0:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %62 = stablehlo.slice %60 [0:2, 0:1, 0:1, 1:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %63 = stablehlo.concatenate %61, %62, dim = 3 : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x64xf32>
      %64 = stablehlo.convert %63 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %65 = stablehlo.reshape %64 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %66 = stablehlo.slice %31 [0:2, 0:1, 64:128] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %67 = stablehlo.concatenate %65, %66, dim = 2 : (tensor<2x1x64xbf16>, tensor<2x1x64xbf16>) -> tensor<2x1x128xbf16>
      %68 = stablehlo.reshape %67 : (tensor<2x1x128xbf16>) -> tensor<2x128xbf16>
      %69 = stablehlo.reshape %arg27 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %70 = stablehlo.reshape %69 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
      %71 = stablehlo.transpose %70, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %72 = stablehlo.dot_general %68, %71, contracting_dims = [1] x [0] : (tensor<2x128xbf16>, tensor<128x128xbf16>) -> tensor<2x128xbf16>
      %73 = stablehlo.reshape %72 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %74 = stablehlo.floor %cst_4 : tensor<64xf32>
      %75 = stablehlo.convert %74 : (tensor<64xf32>) -> tensor<64xi64>
      %76 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %77 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %78 = stablehlo.clamp %77, %75, %76 : tensor<64xi64>
      %79 = stablehlo.compare  LT, %78, %10 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %80 = stablehlo.add %78, %9 : tensor<64xi64>
      %81 = stablehlo.select %79, %80, %78 : tensor<64xi1>, tensor<64xi64>
      %82 = stablehlo.reshape %81 : (tensor<64xi64>) -> tensor<64x1xi64>
      %83 = "stablehlo.gather"(%73, %82) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 128>}> : (tensor<2x1x128xbf16>, tensor<64x1xi64>) -> tensor<2x64x128xbf16>
      %84 = stablehlo.select %19, %11, %83 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %85 = stablehlo.select %16, %84, %arg26 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %86 = stablehlo.slice %85 [0:2, 0:33, 0:128] : (tensor<2x64x128xbf16>) -> tensor<2x33x128xbf16>
      %87 = stablehlo.reshape %arg36 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %88 = stablehlo.reshape %87 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %89 = stablehlo.convert %88 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %90 = stablehlo.broadcast_in_dim %89, dims = [2] : (tensor<3072xf32>) -> tensor<2x1x3072xf32>
      %91 = stablehlo.reshape %arg35 : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
      %92 = stablehlo.reshape %91 : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
      %93 = stablehlo.transpose %92, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
      %94 = stablehlo.dot_general %20, %93, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
      %95 = "stablehlo.reduce_scatter"(%94) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
      %96 = stablehlo.reshape %95 : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
      %97 = stablehlo.convert %96 : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
      %98 = stablehlo.power %97, %8 : tensor<2x1x3072xf32>
      %99 = stablehlo.reduce(%98 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x3072xf32>, tensor<f32>) -> tensor<2x1xf32>
      %100 = stablehlo.multiply %99, %cst_6 : tensor<2x1xf32>
      %101 = stablehlo.reshape %100 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %102 = stablehlo.add %101, %cst_2 : tensor<2x1x1xf32>
      %103 = stablehlo.rsqrt %102 : tensor<2x1x1xf32>
      %104 = stablehlo.reshape %103 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x3072xf32>
      %106 = stablehlo.multiply %97, %105 : tensor<2x1x3072xf32>
      %107 = stablehlo.multiply %90, %106 : tensor<2x1x3072xf32>
      %108 = stablehlo.convert %107 : (tensor<2x1x3072xf32>) -> tensor<2x1x3072xbf16>
      %109 = stablehlo.reshape %108 : (tensor<2x1x3072xbf16>) -> tensor<2x3072xbf16>
      %110 = stablehlo.reshape %arg34 : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
      %111 = stablehlo.reshape %110 : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
      %112 = stablehlo.transpose %111, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
      %113 = stablehlo.dot_general %109, %112, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<2x2048xbf16>
      %114 = stablehlo.reshape %113 : (tensor<2x2048xbf16>) -> tensor<2x1x16x128xbf16>
      %115 = stablehlo.slice %114 [0:2, 0:1, 0:16, 0:64] : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
      %116 = stablehlo.reshape %115 : (tensor<2x1x16x64xbf16>) -> tensor<2x1x16x2x32xbf16>
      %117 = stablehlo.transpose %116, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<2x1x16x2x32xbf16>) -> tensor<2x1x16x32x2xbf16>
      %118 = stablehlo.convert %117 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<2x1x16x32x2xbf16>) -> tensor<2x1x16x32x2xf32>
      %119 = stablehlo.slice %118 [0:2, 0:1, 0:16, 0:32, 0:1] : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
      %120 = stablehlo.reshape %119 : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
      %121 = stablehlo.reshape %41 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %122 = stablehlo.broadcast_in_dim %121, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x16x32xf32>
      %123 = stablehlo.multiply %120, %122 : tensor<2x1x16x32xf32>
      %124 = stablehlo.slice %118 [0:2, 0:1, 0:16, 0:32, 1:2] : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
      %125 = stablehlo.reshape %124 : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
      %126 = stablehlo.reshape %49 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %127 = stablehlo.broadcast_in_dim %126, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x16x32xf32>
      %128 = stablehlo.multiply %125, %127 : tensor<2x1x16x32xf32>
      %129 = stablehlo.subtract %123, %128 : tensor<2x1x16x32xf32>
      %130 = stablehlo.reshape %129 : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
      %131 = stablehlo.multiply %120, %127 : tensor<2x1x16x32xf32>
      %132 = stablehlo.multiply %125, %122 : tensor<2x1x16x32xf32>
      %133 = stablehlo.add %131, %132 : tensor<2x1x16x32xf32>
      %134 = stablehlo.reshape %133 : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
      %135 = stablehlo.concatenate %130, %134, dim = 4 : (tensor<2x1x16x32x1xf32>, tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32x2xf32>
      %136 = stablehlo.reshape %135 : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x64xf32>
      %137 = stablehlo.slice %136 [0:2, 0:1, 0:16, 0:64:2] : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
      %138 = stablehlo.slice %136 [0:2, 0:1, 0:16, 1:64:2] : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
      %139 = stablehlo.concatenate %137, %138, dim = 3 : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x64xf32>
      %140 = stablehlo.convert %139 : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x64xbf16>
      %141 = stablehlo.slice %114 [0:2, 0:1, 0:16, 64:128] : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
      %142 = stablehlo.concatenate %140, %141, dim = 3 : (tensor<2x1x16x64xbf16>, tensor<2x1x16x64xbf16>) -> tensor<2x1x16x128xbf16>
      %143 = stablehlo.dot_general %142, %71, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<2x1x16x128xbf16>
      %144 = stablehlo.reshape %143 : (tensor<2x1x16x128xbf16>) -> tensor<2x16x128xbf16>
      %145 = stablehlo.transpose %144, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[4,128,64]{1,2,0}"} : (tensor<2x16x128xbf16>) -> tensor<2x128x16xbf16>
      %146 = stablehlo.dot_general %86, %145, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x33x128xbf16>, tensor<2x128x16xbf16>) -> tensor<2x33x16xbf16>
      %147 = stablehlo.maximum %146, %7 : tensor<2x33x16xbf16>
      %148 = stablehlo.reshape %arg33 : (tensor<16x1024xbf16>) -> tensor<1x16x1024xbf16>
      %149 = stablehlo.reshape %148 : (tensor<1x16x1024xbf16>) -> tensor<16x1024xbf16>
      %150 = stablehlo.transpose %149, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<16x1024xbf16>) -> tensor<1024x16xbf16>
      %151 = stablehlo.dot_general %20, %150, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x16xbf16>) -> tensor<4x16xbf16>
      %152 = "stablehlo.reduce_scatter"(%151) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<4x16xbf16>) -> tensor<2x16xbf16>
      %153 = stablehlo.reshape %152 : (tensor<2x16xbf16>) -> tensor<2x1x16xbf16>
      %154 = stablehlo.multiply %153, %6 : tensor<2x1x16xbf16>
      %155 = stablehlo.reshape %154 : (tensor<2x1x16xbf16>) -> tensor<2x1x16x1xbf16>
      %156 = stablehlo.multiply %155, %5 : tensor<2x1x16x1xbf16>
      %157 = stablehlo.reshape %156 : (tensor<2x1x16x1xbf16>) -> tensor<2x16xbf16>
      %158 = stablehlo.broadcast_in_dim %157, dims = [0, 2] : (tensor<2x16xbf16>) -> tensor<2x33x16xbf16>
      %159 = stablehlo.multiply %147, %158 : tensor<2x33x16xbf16>
      %160 = stablehlo.reduce(%159 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<2x33x16xbf16>, tensor<bf16>) -> tensor<2x33xbf16>
      %161 = "stablehlo.all_reduce"(%160) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<2x33xbf16>) -> tensor<2x33xbf16>
      %162 = stablehlo.reshape %161 : (tensor<2x33xbf16>) -> tensor<2x1x33xbf16>
      %163 = stablehlo.composite "tenstorrent.topk_indices" %162 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @outlined_composite_tenstorrent.topk_indices.impl} : (tensor<2x1x33xbf16>) -> tensor<2x1x33xi64>
      %164 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %165 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %166 = stablehlo.reshape %arg22 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %167 = stablehlo.reshape %166 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %168 = stablehlo.convert %167 : (tensor<512xbf16>) -> tensor<512xf32>
      %169 = stablehlo.broadcast_in_dim %168, dims = [2] : (tensor<512xf32>) -> tensor<2x1x512xf32>
      %170 = stablehlo.reshape %arg20 : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
      %171 = stablehlo.reshape %170 : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
      %172 = stablehlo.transpose %171, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
      %173 = stablehlo.dot_general %20, %172, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
      %174 = "stablehlo.reduce_scatter"(%173) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
      %175 = stablehlo.reshape %174 : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
      %176 = stablehlo.slice %175 [0:2, 0:1, 0:512] : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
      %177 = stablehlo.convert %176 : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
      %178 = stablehlo.power %177, %3 : tensor<2x1x512xf32>
      %179 = stablehlo.reduce(%178 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x512xf32>, tensor<f32>) -> tensor<2x1xf32>
      %180 = stablehlo.multiply %179, %cst_10 : tensor<2x1xf32>
      %181 = stablehlo.reshape %180 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %182 = stablehlo.add %181, %cst_2 : tensor<2x1x1xf32>
      %183 = stablehlo.rsqrt %182 : tensor<2x1x1xf32>
      %184 = stablehlo.reshape %183 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x512xf32>
      %186 = stablehlo.multiply %177, %185 : tensor<2x1x512xf32>
      %187 = stablehlo.multiply %169, %186 : tensor<2x1x512xf32>
      %188 = stablehlo.convert %187 : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
      %189 = "stablehlo.gather"(%188, %82) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 512>}> : (tensor<2x1x512xbf16>, tensor<64x1xi64>) -> tensor<2x64x512xbf16>
      %190 = stablehlo.select %165, %4, %189 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %191 = stablehlo.select %164, %190, %arg19 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %192 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %193 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %194 = stablehlo.slice %175 [0:2, 0:1, 512:576] : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
      %195 = stablehlo.reshape %194 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
      %196 = stablehlo.convert %195 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
      %197 = stablehlo.reshape %196 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
      %198 = stablehlo.slice %197 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %199 = stablehlo.reshape %198 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %200 = stablehlo.multiply %199, %43 : tensor<2x1x1x32xf32>
      %201 = stablehlo.slice %197 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %202 = stablehlo.reshape %201 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %203 = stablehlo.multiply %202, %51 : tensor<2x1x1x32xf32>
      %204 = stablehlo.subtract %200, %203 : tensor<2x1x1x32xf32>
      %205 = stablehlo.reshape %204 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %206 = stablehlo.multiply %199, %51 : tensor<2x1x1x32xf32>
      %207 = stablehlo.multiply %202, %43 : tensor<2x1x1x32xf32>
      %208 = stablehlo.add %206, %207 : tensor<2x1x1x32xf32>
      %209 = stablehlo.reshape %208 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %210 = stablehlo.concatenate %205, %209, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %211 = stablehlo.reshape %210 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %212 = stablehlo.convert %211 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %213 = stablehlo.reshape %212 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %214 = "stablehlo.gather"(%213, %82) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 64>}> : (tensor<2x1x64xbf16>, tensor<64x1xi64>) -> tensor<2x64x64xbf16>
      %215 = stablehlo.select %193, %2, %214 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %216 = stablehlo.select %192, %215, %arg24 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %217 = stablehlo.reshape %arg37 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %218 = stablehlo.reshape %217 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %219 = stablehlo.transpose %218, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %220 = stablehlo.dot_general %109, %219, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
      %221 = stablehlo.reshape %220 : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
      %222 = stablehlo.slice %221 [0:2, 0:1, 0:4, 0:128] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
      %223 = stablehlo.reshape %arg32 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %224 = stablehlo.reshape %223 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %225 = stablehlo.slice %224 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %226 = stablehlo.dot_general %222, %225, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
      %227 = stablehlo.transpose %226, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
      %228 = stablehlo.slice %191 [0:2, 0:33, 0:512] : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
      %229 = stablehlo.reshape %163 : (tensor<2x1x33xi64>) -> tensor<2x33xi64>
      %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<2x33xi64>) -> tensor<2x33x512xi64>
      %231 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<2xui32>
      %232 = stablehlo.broadcast_in_dim %231, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2xui32>) -> tensor<2x33x512x1xui32>
      %233 = stablehlo.convert %230 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512xi64>) -> tensor<2x33x512xui32>
      %234 = stablehlo.reshape %233 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512xui32>) -> tensor<2x33x512x1xui32>
      %235 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : tensor<512xui32>
      %236 = stablehlo.broadcast_in_dim %235, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<512xui32>) -> tensor<2x33x512x1xui32>
      %237 = stablehlo.concatenate %232, %234, %236, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>) -> tensor<2x33x512x3xui32>
      %238 = "stablehlo.all_gather"(%228) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x512xbf16>) -> tensor<4x33x512xbf16>
      %239 = "stablehlo.gather"(%238, %237) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x512xbf16>, tensor<2x33x512x3xui32>) -> tensor<2x33x512xbf16>
      %240 = stablehlo.dot_general %227, %239, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
      %241 = stablehlo.slice %221 [0:2, 0:1, 0:4, 128:192] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
      %242 = stablehlo.convert %241 : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
      %243 = stablehlo.reshape %242 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
      %244 = stablehlo.slice %243 [0:2, 0:1, 0:4, 0:32, 0:1] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %245 = stablehlo.reshape %244 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %246 = stablehlo.broadcast_in_dim %121, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %247 = stablehlo.multiply %245, %246 : tensor<2x1x4x32xf32>
      %248 = stablehlo.slice %243 [0:2, 0:1, 0:4, 0:32, 1:2] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %249 = stablehlo.reshape %248 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %250 = stablehlo.broadcast_in_dim %126, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %251 = stablehlo.multiply %249, %250 : tensor<2x1x4x32xf32>
      %252 = stablehlo.subtract %247, %251 : tensor<2x1x4x32xf32>
      %253 = stablehlo.reshape %252 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %254 = stablehlo.multiply %245, %250 : tensor<2x1x4x32xf32>
      %255 = stablehlo.multiply %249, %246 : tensor<2x1x4x32xf32>
      %256 = stablehlo.add %254, %255 : tensor<2x1x4x32xf32>
      %257 = stablehlo.reshape %256 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %258 = stablehlo.concatenate %253, %257, dim = 4 : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
      %259 = stablehlo.reshape %258 : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
      %260 = stablehlo.convert %259 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
      %261 = stablehlo.slice %216 [0:2, 0:33, 0:64] : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
      %262 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<2x33xi64>) -> tensor<2x33x64xi64>
      %263 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<2xui32>
      %264 = stablehlo.broadcast_in_dim %263, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2xui32>) -> tensor<2x33x64x1xui32>
      %265 = stablehlo.convert %262 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64xi64>) -> tensor<2x33x64xui32>
      %266 = stablehlo.reshape %265 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64xui32>) -> tensor<2x33x64x1xui32>
      %267 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<64xui32>
      %268 = stablehlo.broadcast_in_dim %267, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<64xui32>) -> tensor<2x33x64x1xui32>
      %269 = stablehlo.concatenate %264, %266, %268, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>) -> tensor<2x33x64x3xui32>
      %270 = "stablehlo.all_gather"(%261) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x64xbf16>) -> tensor<4x33x64xbf16>
      %271 = "stablehlo.gather"(%270, %269) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x64xbf16>, tensor<2x33x64x3xui32>) -> tensor<2x33x64xbf16>
      %272 = stablehlo.dot_general %260, %271, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
      %273 = stablehlo.add %240, %272 : tensor<2x1x4x33xbf16>
      %274 = stablehlo.multiply %273, %1 : tensor<2x1x4x33xbf16>
      %275 = stablehlo.reduce(%274 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %276 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %277 = stablehlo.subtract %274, %276 : tensor<2x1x4x33xbf16>
      %278 = stablehlo.exponential %277 : tensor<2x1x4x33xbf16>
      %279 = stablehlo.reduce(%278 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %280 = stablehlo.broadcast_in_dim %279, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %281 = stablehlo.divide %278, %280 : tensor<2x1x4x33xbf16>
      %282 = stablehlo.dot_general %281, %239, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
      %283 = stablehlo.slice %224 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %284 = stablehlo.dot_general %282, %283, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
      %285 = stablehlo.transpose %284, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
      %286 = stablehlo.reshape %285 : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
      %287 = stablehlo.reshape %arg31 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %288 = stablehlo.reshape %287 : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
      %289 = stablehlo.transpose %288, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
      %290 = "stablehlo.all_gather"(%286) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
      %291 = stablehlo.dot_general %290, %289, contracting_dims = [1] x [0] : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
      %292 = "stablehlo.all_reduce"(%291) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %294 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %294 : tensor<bf16>
      }) : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
      %293 = stablehlo.reshape %292 : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
      sdy.return %191, %216, %85, %293 : tensor<2x64x512xbf16>, tensor<2x64x64xbf16>, tensor<2x64x128xbf16>, tensor<4x1x1024xbf16>
    } : (tensor<4x64x512xbf16>, tensor<576x2048xbf16>, tensor<4x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<4x64x64xbf16>, tensor<1x32x2xbf16>, tensor<4x64x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128x2048xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<64x2048xbf16>, tensor<8192x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>, tensor<3072x3072xbf16>) -> (tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
  }
  func.func private @outlined_composite_tenstorrent.layer_norm.impl(%arg0: tensor<2x1x128xbf16>, %arg1: tensor<128xbf16>, %arg2: tensor<128xbf16>) -> tensor<2x1x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<2x1x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<2x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<2x1x128xbf16>) -> tensor<2x1x128xf32>
    %1 = stablehlo.reduce(%0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<2x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<2x1x128xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<2x1x128xf32>
    %6 = stablehlo.reduce(%5 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
    %7 = stablehlo.multiply %6, %cst_0 : tensor<2x1xf32>
    %8 = stablehlo.reshape %7 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %9 = stablehlo.add %8, %cst : tensor<2x1x1xf32>
    %10 = stablehlo.rsqrt %9 : tensor<2x1x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %13 = stablehlo.multiply %4, %12 : tensor<2x1x128xf32>
    %14 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %16 = stablehlo.multiply %13, %15 : tensor<2x1x128xf32>
    %17 = stablehlo.convert %arg2 : (tensor<128xbf16>) -> tensor<128xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x1x128xf32>
    %20 = stablehlo.convert %19 : (tensor<2x1x128xf32>) -> tensor<2x1x128xbf16>
    return %20 : tensor<2x1x128xbf16>
  }
  func.func private @outlined_composite_tenstorrent.topk_indices.impl(%arg0: tensor<2x1x33xbf16>) -> tensor<2x1x33xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<33xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<33xi32>) -> tensor<2x1x33xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 2 : i64}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %4 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    }) : (tensor<2x1x33xbf16>, tensor<2x1x33xi32>) -> (tensor<2x1x33xbf16>, tensor<2x1x33xi32>)
    %3 = stablehlo.convert %2#1 : (tensor<2x1x33xi32>) -> tensor<2x1x33xi64>
    return %3 : tensor<2x1x33xi64>
  }
}



module @ReplicateShardedData.6 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {"_axis_0"}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"_axis_0", "_axis_1"} (%arg1: tensor<4x1x1024xbf16>) {
      %1 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<4x1x1024xbf16>) -> tensor<4x1x2048xbf16>
      sdy.return %1 : tensor<4x1x2048xbf16>
    } : (tensor<4x1x2048xbf16>) -> tensor<4x1x2048xbf16>
    return %0 : tensor<4x1x2048xbf16>
  }
}
