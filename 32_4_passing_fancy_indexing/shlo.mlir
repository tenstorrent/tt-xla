module @SyncTensorsGraph.791 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<4x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<4x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:4 = sdy.manual_computation(
        %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
        %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16,
        %arg17, %arg18)
        in_shardings=[
          <@mesh, [{"_axis_0"}, {}, {}]>,        // %arg0:  kv_cache
          <@mesh, [{}, {"_axis_0"}]>,             // %arg1:  wkv_a.weight
          <@mesh, [{}, {}, {"_axis_0"}]>,         // %arg2:  args_0
          <@mesh, [{}]>,                          // %arg3:  kv_norm.weight
          <@mesh, []>,                            // %arg4:  (scalar i1)
          <@mesh, [{"_axis_0"}, {}, {}]>,         // %arg5:  pe_cache
          <@mesh, [{}, {}, {}]>,                  // %arg6:  args_1
          <@mesh, [{"_axis_0"}, {}, {}]>,         // %arg7:  indexer.k_cache
          <@mesh, [{}, {}]>,                      // %arg8:  indexer.haddamard
          <@mesh, [{}]>,                          // %arg9:  indexer.k_norm.bias
          <@mesh, [{}]>,                          // %arg10: indexer.k_norm.weight
          <@mesh, [{}, {"_axis_0"}]>,             // %arg11: indexer.wk.weight
          <@mesh, [{"_axis_0"}, {"_axis_1"}]>,    // %arg12: wo.weight
          <@mesh, [{"_axis_1"}, {}]>,             // %arg13: wkv_b.weight
          <@mesh, [{"_axis_1"}, {"_axis_0"}]>,    // %arg14: indexer.weights_proj.weight
          <@mesh, [{"_axis_1"}, {}]>,             // %arg15: indexer.wq_b.weight
          <@mesh, [{}, {"_axis_0"}]>,             // %arg16: wq_a.weight
          <@mesh, [{}]>,                          // %arg17: q_norm.weight
          <@mesh, [{"_axis_1"}, {}]>]             // %arg18: wq_b.weight
        out_shardings=[
          <@mesh, [{"_axis_0"}, {}, {}]>,         // %0#0: kv_cache out
          <@mesh, [{"_axis_0"}, {}, {}]>,         // %0#1: pe_cache out
          <@mesh, [{"_axis_0"}, {}, {}]>,         // %0#2: indexer.k_cache out
          <@mesh, [{}, {}, {"_axis_0"}]>]         // %0#3: args_0 out
        manual_axes={"_axis_0", "_axis_1"}
        (%arg19: tensor<2x64x512xbf16>,
         %arg20: tensor<576x1024xbf16>,
         %arg21: tensor<4x1x1024xbf16>,
         %arg22: tensor<512xbf16>,
         %arg23: tensor<i1>,
         %arg24: tensor<2x64x64xbf16>,
         %arg25: tensor<1x32x2xbf16>,
         %arg26: tensor<2x64x128xbf16>,
         %arg27: tensor<128x128xbf16>,
         %arg28: tensor<128xbf16>,
         %arg29: tensor<128xbf16>,
         %arg30: tensor<128x1024xbf16>,
         %arg31: tensor<1024x512xbf16>,
         %arg32: tensor<1024x512xbf16>,
         %arg33: tensor<16x1024xbf16>,
         %arg34: tensor<2048x3072xbf16>,
         %arg35: tensor<3072x1024xbf16>,
         %arg36: tensor<3072xbf16>,
         %arg37: tensor<768x3072xbf16>) {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %c_1 = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
      %c_2 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
      %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<2x1x1xf32>
      %cst_4 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
      %c_5 = stablehlo.constant dense<1> : tensor<i64>
      %cst_6 = stablehlo.constant dense<3.25520843E-4> : tensor<2x1xf32>
      %cst_7 = stablehlo.constant dense<1.250000e-01> : tensor<bf16>
      %cst_8 = stablehlo.constant dense<8.837890e-02> : tensor<bf16>
      %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %cst_10 = stablehlo.constant dense<0.001953125> : tensor<2x1xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %c_12 = stablehlo.constant dense<0> : tensor<i64>
      %c_13 = stablehlo.constant dense<4> : tensor<i64>
      %c_14 = stablehlo.constant dense<33> : tensor<i64>
      %cst_15 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<bf16>) -> tensor<2x1x4x33xbf16>
      %2 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i64>) -> tensor<2x33xi64>
      %3 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<2x33xi64>
      %4 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<2x33xi64>
      %5 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x64xbf16>
      %6 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<2x1x512xf32>
      %7 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x512xbf16>
      %8 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<2x1x16x1xbf16>
      %9 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x1x16xbf16>
      %10 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x33x16xbf16>
      %11 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %12 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %13 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %14 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<2x64x128xbf16>
      %15 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<i1>) -> tensor<64xi1>
      %16 = stablehlo.and %15, %c_1 : tensor<64xi1>
      %17 = stablehlo.and %16, %c_2 : tensor<64xi1>
      %18 = stablehlo.reshape %17 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      %19 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<2x64x128xi1>
      %20 = stablehlo.not %18 : tensor<1x64x1xi1>
      %21 = stablehlo.reshape %20 : (tensor<1x64x1xi1>) -> tensor<64xi1>
      %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<64xi1>) -> tensor<2x64x128xi1>
      %23 = stablehlo.reshape %arg21 : (tensor<4x1x1024xbf16>) -> tensor<4x1024xbf16>
      %24 = stablehlo.reshape %arg30 : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
      %25 = stablehlo.reshape %24 : (tensor<1x128x1024xbf16>) -> tensor<128x1024xbf16>
      %26 = stablehlo.transpose %25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x1024xbf16>) -> tensor<1024x128xbf16>
      %27 = stablehlo.dot_general %23, %26, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x128xbf16>) -> tensor<4x128xbf16>
      %28 = "stablehlo.reduce_scatter"(%27) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<4x128xbf16>) -> tensor<2x128xbf16>
      %29 = stablehlo.reshape %28 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %30 = stablehlo.reshape %arg29 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %31 = stablehlo.reshape %30 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %32 = stablehlo.reshape %arg28 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %33 = stablehlo.reshape %32 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %34 = stablehlo.composite "tenstorrent.layer_norm" %29, %31, %33 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @outlined_composite_tenstorrent.layer_norm.impl} : (tensor<2x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<2x1x128xbf16>
      %35 = stablehlo.slice %34 [0:2, 0:1, 0:64] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %36 = stablehlo.reshape %35 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x2x32xbf16>
      %37 = stablehlo.transpose %36, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x2x32xbf16>) -> tensor<2x1x1x32x2xbf16>
      %38 = stablehlo.convert %37 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x32x2xbf16>) -> tensor<2x1x1x32x2xf32>
      %39 = stablehlo.slice %38 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %40 = stablehlo.reshape %39 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %41 = stablehlo.reshape %arg25 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %42 = stablehlo.slice %41 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %43 = stablehlo.reshape %42 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %44 = stablehlo.convert %43 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %45 = stablehlo.reshape %44 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %46 = stablehlo.broadcast_in_dim %45, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %47 = stablehlo.multiply %40, %46 : tensor<2x1x1x32xf32>
      %48 = stablehlo.slice %38 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %49 = stablehlo.reshape %48 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %50 = stablehlo.slice %41 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %51 = stablehlo.reshape %50 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %52 = stablehlo.convert %51 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %53 = stablehlo.reshape %52 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %54 = stablehlo.broadcast_in_dim %53, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %55 = stablehlo.multiply %49, %54 : tensor<2x1x1x32xf32>
      %56 = stablehlo.subtract %47, %55 : tensor<2x1x1x32xf32>
      %57 = stablehlo.reshape %56 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %58 = stablehlo.multiply %40, %54 : tensor<2x1x1x32xf32>
      %59 = stablehlo.multiply %49, %46 : tensor<2x1x1x32xf32>
      %60 = stablehlo.add %58, %59 : tensor<2x1x1x32xf32>
      %61 = stablehlo.reshape %60 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %62 = stablehlo.concatenate %57, %61, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %63 = stablehlo.reshape %62 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %64 = stablehlo.slice %63 [0:2, 0:1, 0:1, 0:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %65 = stablehlo.slice %63 [0:2, 0:1, 0:1, 1:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %66 = stablehlo.concatenate %64, %65, dim = 3 : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x64xf32>
      %67 = stablehlo.convert %66 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %68 = stablehlo.reshape %67 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %69 = stablehlo.slice %34 [0:2, 0:1, 64:128] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %70 = stablehlo.concatenate %68, %69, dim = 2 : (tensor<2x1x64xbf16>, tensor<2x1x64xbf16>) -> tensor<2x1x128xbf16>
      %71 = stablehlo.reshape %70 : (tensor<2x1x128xbf16>) -> tensor<2x128xbf16>
      %72 = stablehlo.reshape %arg27 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %73 = stablehlo.reshape %72 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
      %74 = stablehlo.transpose %73, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %75 = stablehlo.dot_general %71, %74, contracting_dims = [1] x [0] : (tensor<2x128xbf16>, tensor<128x128xbf16>) -> tensor<2x128xbf16>
      %76 = stablehlo.reshape %75 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %77 = stablehlo.floor %cst_4 : tensor<64xf32>
      %78 = stablehlo.convert %77 : (tensor<64xf32>) -> tensor<64xi64>
      %79 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %80 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %81 = stablehlo.clamp %80, %78, %79 : tensor<64xi64>
      %82 = stablehlo.compare  LT, %81, %13 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %83 = stablehlo.add %81, %12 : tensor<64xi64>
      %84 = stablehlo.select %82, %83, %81 : tensor<64xi1>, tensor<64xi64>
      %85 = stablehlo.reshape %84 : (tensor<64xi64>) -> tensor<64x1xi64>
      %86 = "stablehlo.gather"(%76, %85) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 128>}> : (tensor<2x1x128xbf16>, tensor<64x1xi64>) -> tensor<2x64x128xbf16>
      %87 = stablehlo.select %22, %14, %86 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %88 = stablehlo.select %19, %87, %arg26 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %89 = stablehlo.slice %88 [0:2, 0:33, 0:128] : (tensor<2x64x128xbf16>) -> tensor<2x33x128xbf16>
      %90 = stablehlo.reshape %arg36 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %91 = stablehlo.reshape %90 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %92 = stablehlo.convert %91 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %93 = stablehlo.broadcast_in_dim %92, dims = [2] : (tensor<3072xf32>) -> tensor<2x1x3072xf32>
      %94 = stablehlo.reshape %arg35 : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
      %95 = stablehlo.reshape %94 : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
      %96 = stablehlo.transpose %95, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
      %97 = stablehlo.dot_general %23, %96, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
      %98 = "stablehlo.reduce_scatter"(%97) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
      %99 = stablehlo.reshape %98 : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
      %100 = stablehlo.convert %99 : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
      %101 = stablehlo.power %100, %11 : tensor<2x1x3072xf32>
      %102 = stablehlo.reduce(%101 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x3072xf32>, tensor<f32>) -> tensor<2x1xf32>
      %103 = stablehlo.multiply %102, %cst_6 : tensor<2x1xf32>
      %104 = stablehlo.reshape %103 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %105 = stablehlo.add %104, %cst_3 : tensor<2x1x1xf32>
      %106 = stablehlo.rsqrt %105 : tensor<2x1x1xf32>
      %107 = stablehlo.reshape %106 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x3072xf32>
      %109 = stablehlo.multiply %100, %108 : tensor<2x1x3072xf32>
      %110 = stablehlo.multiply %93, %109 : tensor<2x1x3072xf32>
      %111 = stablehlo.convert %110 : (tensor<2x1x3072xf32>) -> tensor<2x1x3072xbf16>
      %112 = stablehlo.reshape %111 : (tensor<2x1x3072xbf16>) -> tensor<2x3072xbf16>
      %113 = stablehlo.reshape %arg34 : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
      %114 = stablehlo.reshape %113 : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
      %115 = stablehlo.transpose %114, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
      %116 = stablehlo.dot_general %112, %115, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<2x2048xbf16>
      %117 = stablehlo.reshape %116 : (tensor<2x2048xbf16>) -> tensor<2x1x16x128xbf16>
      %118 = stablehlo.slice %117 [0:2, 0:1, 0:16, 0:64] : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
      %119 = stablehlo.reshape %118 : (tensor<2x1x16x64xbf16>) -> tensor<2x1x16x2x32xbf16>
      %120 = stablehlo.transpose %119, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<2x1x16x2x32xbf16>) -> tensor<2x1x16x32x2xbf16>
      %121 = stablehlo.convert %120 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,64,32,2]{3,4,2,1,0}"} : (tensor<2x1x16x32x2xbf16>) -> tensor<2x1x16x32x2xf32>
      %122 = stablehlo.slice %121 [0:2, 0:1, 0:16, 0:32, 0:1] : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
      %123 = stablehlo.reshape %122 : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
      %124 = stablehlo.reshape %44 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %125 = stablehlo.broadcast_in_dim %124, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x16x32xf32>
      %126 = stablehlo.multiply %123, %125 : tensor<2x1x16x32xf32>
      %127 = stablehlo.slice %121 [0:2, 0:1, 0:16, 0:32, 1:2] : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
      %128 = stablehlo.reshape %127 : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
      %129 = stablehlo.reshape %52 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %130 = stablehlo.broadcast_in_dim %129, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x16x32xf32>
      %131 = stablehlo.multiply %128, %130 : tensor<2x1x16x32xf32>
      %132 = stablehlo.subtract %126, %131 : tensor<2x1x16x32xf32>
      %133 = stablehlo.reshape %132 : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
      %134 = stablehlo.multiply %123, %130 : tensor<2x1x16x32xf32>
      %135 = stablehlo.multiply %128, %125 : tensor<2x1x16x32xf32>
      %136 = stablehlo.add %134, %135 : tensor<2x1x16x32xf32>
      %137 = stablehlo.reshape %136 : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
      %138 = stablehlo.concatenate %133, %137, dim = 4 : (tensor<2x1x16x32x1xf32>, tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32x2xf32>
      %139 = stablehlo.reshape %138 : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x64xf32>
      %140 = stablehlo.slice %139 [0:2, 0:1, 0:16, 0:64:2] : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
      %141 = stablehlo.slice %139 [0:2, 0:1, 0:16, 1:64:2] : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
      %142 = stablehlo.concatenate %140, %141, dim = 3 : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x64xf32>
      %143 = stablehlo.convert %142 : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x64xbf16>
      %144 = stablehlo.slice %117 [0:2, 0:1, 0:16, 64:128] : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
      %145 = stablehlo.concatenate %143, %144, dim = 3 : (tensor<2x1x16x64xbf16>, tensor<2x1x16x64xbf16>) -> tensor<2x1x16x128xbf16>
      %146 = stablehlo.dot_general %145, %74, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<2x1x16x128xbf16>
      %147 = stablehlo.reshape %146 : (tensor<2x1x16x128xbf16>) -> tensor<2x16x128xbf16>
      %148 = stablehlo.transpose %147, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[4,128,64]{1,2,0}"} : (tensor<2x16x128xbf16>) -> tensor<2x128x16xbf16>
      %149 = stablehlo.dot_general %89, %148, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x33x128xbf16>, tensor<2x128x16xbf16>) -> tensor<2x33x16xbf16>
      %150 = stablehlo.maximum %149, %10 : tensor<2x33x16xbf16>
      %151 = stablehlo.reshape %arg33 : (tensor<16x1024xbf16>) -> tensor<1x16x1024xbf16>
      %152 = stablehlo.reshape %151 : (tensor<1x16x1024xbf16>) -> tensor<16x1024xbf16>
      %153 = stablehlo.transpose %152, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<16x1024xbf16>) -> tensor<1024x16xbf16>
      %154 = stablehlo.dot_general %23, %153, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x16xbf16>) -> tensor<4x16xbf16>
      %155 = "stablehlo.reduce_scatter"(%154) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<4x16xbf16>) -> tensor<2x16xbf16>
      %156 = stablehlo.reshape %155 : (tensor<2x16xbf16>) -> tensor<2x1x16xbf16>
      %157 = stablehlo.multiply %156, %9 : tensor<2x1x16xbf16>
      %158 = stablehlo.reshape %157 : (tensor<2x1x16xbf16>) -> tensor<2x1x16x1xbf16>
      %159 = stablehlo.multiply %158, %8 : tensor<2x1x16x1xbf16>
      %160 = stablehlo.reshape %159 : (tensor<2x1x16x1xbf16>) -> tensor<2x16xbf16>
      %161 = stablehlo.broadcast_in_dim %160, dims = [0, 2] : (tensor<2x16xbf16>) -> tensor<2x33x16xbf16>
      %162 = stablehlo.multiply %150, %161 : tensor<2x33x16xbf16>
      %163 = stablehlo.reduce(%162 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<2x33x16xbf16>, tensor<bf16>) -> tensor<2x33xbf16>
      %164 = "stablehlo.all_reduce"(%163) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<2x33xbf16>) -> tensor<2x33xbf16>
      %165 = stablehlo.reshape %164 : (tensor<2x33xbf16>) -> tensor<2x1x33xbf16>
      %166 = stablehlo.composite "tenstorrent.topk_indices" %165 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @outlined_composite_tenstorrent.topk_indices.impl} : (tensor<2x1x33xbf16>) -> tensor<2x1x33xi64>
      %167 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %168 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %169 = stablehlo.reshape %arg22 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %170 = stablehlo.reshape %169 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %171 = stablehlo.convert %170 : (tensor<512xbf16>) -> tensor<512xf32>
      %172 = stablehlo.broadcast_in_dim %171, dims = [2] : (tensor<512xf32>) -> tensor<2x1x512xf32>
      %173 = stablehlo.reshape %arg20 : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
      %174 = stablehlo.reshape %173 : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
      %175 = stablehlo.transpose %174, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
      %176 = stablehlo.dot_general %23, %175, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
      %177 = "stablehlo.reduce_scatter"(%176) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
      %178 = stablehlo.reshape %177 : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
      %179 = stablehlo.slice %178 [0:2, 0:1, 0:512] : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
      %180 = stablehlo.convert %179 : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
      %181 = stablehlo.power %180, %6 : tensor<2x1x512xf32>
      %182 = stablehlo.reduce(%181 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x512xf32>, tensor<f32>) -> tensor<2x1xf32>
      %183 = stablehlo.multiply %182, %cst_10 : tensor<2x1xf32>
      %184 = stablehlo.reshape %183 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %185 = stablehlo.add %184, %cst_3 : tensor<2x1x1xf32>
      %186 = stablehlo.rsqrt %185 : tensor<2x1x1xf32>
      %187 = stablehlo.reshape %186 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x512xf32>
      %189 = stablehlo.multiply %180, %188 : tensor<2x1x512xf32>
      %190 = stablehlo.multiply %172, %189 : tensor<2x1x512xf32>
      %191 = stablehlo.convert %190 : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
      %192 = "stablehlo.gather"(%191, %85) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 512>}> : (tensor<2x1x512xbf16>, tensor<64x1xi64>) -> tensor<2x64x512xbf16>
      %193 = stablehlo.select %168, %7, %192 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %194 = stablehlo.select %167, %193, %arg19 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %195 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %196 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %197 = stablehlo.slice %178 [0:2, 0:1, 512:576] : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
      %198 = stablehlo.reshape %197 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
      %199 = stablehlo.convert %198 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
      %200 = stablehlo.reshape %199 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
      %201 = stablehlo.slice %200 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %202 = stablehlo.reshape %201 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %203 = stablehlo.multiply %202, %46 : tensor<2x1x1x32xf32>
      %204 = stablehlo.slice %200 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %205 = stablehlo.reshape %204 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %206 = stablehlo.multiply %205, %54 : tensor<2x1x1x32xf32>
      %207 = stablehlo.subtract %203, %206 : tensor<2x1x1x32xf32>
      %208 = stablehlo.reshape %207 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %209 = stablehlo.multiply %202, %54 : tensor<2x1x1x32xf32>
      %210 = stablehlo.multiply %205, %46 : tensor<2x1x1x32xf32>
      %211 = stablehlo.add %209, %210 : tensor<2x1x1x32xf32>
      %212 = stablehlo.reshape %211 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %213 = stablehlo.concatenate %208, %212, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %214 = stablehlo.reshape %213 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %215 = stablehlo.convert %214 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %216 = stablehlo.reshape %215 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %217 = "stablehlo.gather"(%216, %85) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 64>}> : (tensor<2x1x64xbf16>, tensor<64x1xi64>) -> tensor<2x64x64xbf16>
      %218 = stablehlo.select %196, %5, %217 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %219 = stablehlo.select %195, %218, %arg24 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %220 = stablehlo.reshape %arg37 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %221 = stablehlo.reshape %220 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %222 = stablehlo.transpose %221, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %223 = stablehlo.dot_general %112, %222, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
      %224 = stablehlo.reshape %223 : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
      %225 = stablehlo.slice %224 [0:2, 0:1, 0:4, 0:128] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
      %226 = stablehlo.reshape %arg32 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %227 = stablehlo.reshape %226 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %228 = stablehlo.slice %227 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %229 = stablehlo.dot_general %225, %228, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
      %230 = stablehlo.transpose %229, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
      %231 = stablehlo.slice %194 [0:2, 0:33, 0:512] : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
      %232 = stablehlo.composite "sdy.all_slice" %c {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}]>}, decomposition = @sdy.all_slice1} : (tensor<4xi64>) -> tensor<2xi64>
      %233 = stablehlo.broadcast_in_dim %232, dims = [0] : (tensor<2xi64>) -> tensor<2x33xi64>
      %234 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<2x33xi64>
      %235 = stablehlo.composite "sdy.all_slice" %c {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}]>}, decomposition = @sdy.all_slice2} : (tensor<4xi64>) -> tensor<2xi64>
      %236 = stablehlo.broadcast_in_dim %235, dims = [0] : (tensor<2xi64>) -> tensor<2x33xi64>
      %237 = stablehlo.compare  LT, %236, %234 : (tensor<2x33xi64>, tensor<2x33xi64>) -> tensor<2x33xi1>
      %238 = stablehlo.composite "sdy.all_slice" %c {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}]>}, decomposition = @sdy.all_slice3} : (tensor<4xi64>) -> tensor<2xi64>
      %239 = stablehlo.broadcast_in_dim %238, dims = [0] : (tensor<2xi64>) -> tensor<2x33xi64>
      %240 = stablehlo.add %239, %3 : tensor<2x33xi64>
      %241 = stablehlo.select %237, %240, %233 : tensor<2x33xi1>, tensor<2x33xi64>
      %242 = stablehlo.reshape %241 : (tensor<2x33xi64>) -> tensor<2x33x1xi64>
      %243 = stablehlo.reshape %166 : (tensor<2x1x33xi64>) -> tensor<2x33xi64>
      %244 = stablehlo.compare  LT, %243, %4 : (tensor<2x33xi64>, tensor<2x33xi64>) -> tensor<2x33xi1>
      %245 = stablehlo.add %243, %2 : tensor<2x33xi64>
      %246 = stablehlo.select %244, %245, %243 : tensor<2x33xi1>, tensor<2x33xi64>
      %247 = stablehlo.reshape %246 : (tensor<2x33xi64>) -> tensor<2x33x1xi64>
      %248 = stablehlo.concatenate %242, %247, dim = 2 : (tensor<2x33x1xi64>, tensor<2x33x1xi64>) -> tensor<2x33x2xi64>
      %249 = "stablehlo.all_gather"(%231) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x512xbf16>) -> tensor<4x33x512xbf16>
      %250 = "stablehlo.gather"(%249, %248) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<4x33x512xbf16>, tensor<2x33x2xi64>) -> tensor<2x33x512xbf16>
      %251 = stablehlo.dot_general %230, %250, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
      %252 = stablehlo.slice %224 [0:2, 0:1, 0:4, 128:192] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
      %253 = stablehlo.convert %252 : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
      %254 = stablehlo.reshape %253 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
      %255 = stablehlo.slice %254 [0:2, 0:1, 0:4, 0:32, 0:1] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %256 = stablehlo.reshape %255 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %257 = stablehlo.broadcast_in_dim %124, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %258 = stablehlo.multiply %256, %257 : tensor<2x1x4x32xf32>
      %259 = stablehlo.slice %254 [0:2, 0:1, 0:4, 0:32, 1:2] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %260 = stablehlo.reshape %259 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %261 = stablehlo.broadcast_in_dim %129, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %262 = stablehlo.multiply %260, %261 : tensor<2x1x4x32xf32>
      %263 = stablehlo.subtract %258, %262 : tensor<2x1x4x32xf32>
      %264 = stablehlo.reshape %263 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %265 = stablehlo.multiply %256, %261 : tensor<2x1x4x32xf32>
      %266 = stablehlo.multiply %260, %257 : tensor<2x1x4x32xf32>
      %267 = stablehlo.add %265, %266 : tensor<2x1x4x32xf32>
      %268 = stablehlo.reshape %267 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %269 = stablehlo.concatenate %264, %268, dim = 4 : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
      %270 = stablehlo.reshape %269 : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
      %271 = stablehlo.convert %270 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
      %272 = stablehlo.slice %219 [0:2, 0:33, 0:64] : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
      %273 = "stablehlo.all_gather"(%272) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x64xbf16>) -> tensor<4x33x64xbf16>
      %274 = "stablehlo.gather"(%273, %248) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 64>}> : (tensor<4x33x64xbf16>, tensor<2x33x2xi64>) -> tensor<2x33x64xbf16>
      %275 = stablehlo.dot_general %271, %274, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
      %276 = stablehlo.add %251, %275 : tensor<2x1x4x33xbf16>
      %277 = stablehlo.multiply %276, %1 : tensor<2x1x4x33xbf16>
      %278 = stablehlo.reduce(%277 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %280 = stablehlo.subtract %277, %279 : tensor<2x1x4x33xbf16>
      %281 = stablehlo.exponential %280 : tensor<2x1x4x33xbf16>
      %282 = stablehlo.reduce(%281 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %284 = stablehlo.divide %281, %283 : tensor<2x1x4x33xbf16>
      %285 = stablehlo.dot_general %284, %250, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
      %286 = stablehlo.slice %227 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %287 = stablehlo.dot_general %285, %286, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
      %288 = stablehlo.transpose %287, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
      %289 = stablehlo.reshape %288 : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
      %290 = stablehlo.reshape %arg31 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %291 = stablehlo.reshape %290 : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
      %292 = stablehlo.transpose %291, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
      %293 = "stablehlo.all_gather"(%289) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
      %294 = stablehlo.dot_general %293, %292, contracting_dims = [1] x [0] : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
      %295 = "stablehlo.all_reduce"(%294) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %297 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %297 : tensor<bf16>
      }) : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
      %296 = stablehlo.reshape %295 : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
      sdy.return %194, %219, %88, %296 : tensor<2x64x512xbf16>, tensor<2x64x64xbf16>, tensor<2x64x128xbf16>, tensor<4x1x1024xbf16>
    } : (tensor<4x64x512xbf16>, tensor<576x2048xbf16>, tensor<4x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<4x64x64xbf16>, tensor<1x32x2xbf16>, tensor<4x64x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128x2048xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<64x2048xbf16>, tensor<8192x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>, tensor<3072x3072xbf16>) -> (tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<4xi64>) -> tensor<2xi64> {
    %0 = stablehlo.reshape %arg0 : (tensor<4xi64>) -> tensor<2x2xi64>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<2x2xi64>) -> tensor<2x2xi64>
    %2 = stablehlo.slice %1 [0:1, 0:2] : (tensor<2x2xi64>) -> tensor<1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x2xi64>) -> tensor<2xi64>
    return %3 : tensor<2xi64>
  }
  func.func private @sdy.all_slice2(%arg0: tensor<4xi64>) -> tensor<2xi64> {
    %0 = stablehlo.reshape %arg0 : (tensor<4xi64>) -> tensor<2x2xi64>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<2x2xi64>) -> tensor<2x2xi64>
    %2 = stablehlo.slice %1 [0:1, 0:2] : (tensor<2x2xi64>) -> tensor<1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x2xi64>) -> tensor<2xi64>
    return %3 : tensor<2xi64>
  }
  func.func private @sdy.all_slice3(%arg0: tensor<4xi64>) -> tensor<2xi64> {
    %0 = stablehlo.reshape %arg0 : (tensor<4xi64>) -> tensor<2x2xi64>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<2x2xi64>) -> tensor<2x2xi64>
    %2 = stablehlo.slice %1 [0:1, 0:2] : (tensor<2x2xi64>) -> tensor<1x2xi64>
    %3 = stablehlo.reshape %2 : (tensor<1x2xi64>) -> tensor<2xi64>
    return %3 : tensor<2xi64>
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
