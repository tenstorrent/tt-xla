module @SyncTensorsGraph.745 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<1x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<1x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<1x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<1x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<1x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:4 = sdy.manual_computation(
        %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
        %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16,
        %arg17, %arg18)
        in_shardings=[
          <@mesh, [{}, {}, {}]>,      // %arg0  (kv_cache)
          <@mesh, [{}, {}]>,          // %arg1  (wkv_a.weight)
          <@mesh, [{}, {}, {}]>,      // %arg2  (args_0)
          <@mesh, [{}]>,              // %arg3  (kv_norm.weight)
          <@mesh, []>,                // %arg4  (no name)
          <@mesh, [{}, {}, {}]>,      // %arg5  (pe_cache)
          <@mesh, [{}, {}, {}]>,      // %arg6  (args_1)
          <@mesh, [{}, {}, {}]>,      // %arg7  (indexer.k_cache)
          <@mesh, [{}, {}]>,          // %arg8  (indexer.haddamard)
          <@mesh, [{}]>,              // %arg9  (indexer.k_norm.bias)
          <@mesh, [{}]>,              // %arg10 (indexer.k_norm.weight)
          <@mesh, [{}, {}]>,          // %arg11 (indexer.wk.weight)
          <@mesh, [{}, {"_axis_1"}]>, // %arg12 (wo.weight)
          <@mesh, [{"_axis_1"}, {}]>, // %arg13 (wkv_b.weight)
          <@mesh, [{"_axis_1"}, {}]>, // %arg14 (indexer.weights_proj.weight)
          <@mesh, [{"_axis_1"}, {}]>, // %arg15 (indexer.wq_b.weight)
          <@mesh, [{}, {}]>,          // %arg16 (wq_a.weight)
          <@mesh, [{}]>,              // %arg17 (q_norm.weight)
          <@mesh, [{"_axis_1"}, {}]>] // %arg18 (wq_b.weight)
        out_shardings=[
          <@mesh, [{}, {}, {}]>,      // %0#0 (kv_cache)
          <@mesh, [{}, {}, {}]>,      // %0#1 (pe_cache)
          <@mesh, [{}, {}, {}]>,      // %0#2 (indexer.k_cache)
          <@mesh, [{}, {}, {}]>]      // %0#3 (args_0)
        manual_axes={"_axis_0", "_axis_1"}
        (%arg19: tensor<1x64x512xbf16>,
         %arg20: tensor<576x2048xbf16>,
         %arg21: tensor<1x1x2048xbf16>,
         %arg22: tensor<512xbf16>,
         %arg23: tensor<i1>,
         %arg24: tensor<1x64x64xbf16>,
         %arg25: tensor<1x32x2xbf16>,
         %arg26: tensor<1x64x128xbf16>,
         %arg27: tensor<128x128xbf16>,
         %arg28: tensor<128xbf16>,
         %arg29: tensor<128xbf16>,
         %arg30: tensor<128x2048xbf16>,
         %arg31: tensor<2048x512xbf16>,
         %arg32: tensor<1024x512xbf16>,
         %arg33: tensor<16x2048xbf16>,
         %arg34: tensor<2048x3072xbf16>,
         %arg35: tensor<3072x2048xbf16>,
         %arg36: tensor<3072xbf16>,
         %arg37: tensor<768x3072xbf16>) {
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
      %cst_12 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<bf16>) -> tensor<1x1x4x33xbf16>
      %2 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x64xbf16>
      %3 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x512xf32>
      %4 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x512xbf16>
      %5 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<bf16>) -> tensor<1x1x16x1xbf16>
      %6 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<1x1x16xbf16>
      %7 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x33x16xbf16>
      %8 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x3072xf32>
      %9 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %10 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %11 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<1x64x128xbf16>
      %12 = stablehlo.broadcast_in_dim %arg23, dims = [] : (tensor<i1>) -> tensor<64xi1>
      %13 = stablehlo.and %12, %c : tensor<64xi1>
      %14 = stablehlo.and %13, %c_1 : tensor<64xi1>
      %15 = stablehlo.reshape %14 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      %16 = stablehlo.reshape %14 : (tensor<64xi1>) -> tensor<1x64xi1>
      %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x128xi1>
      %18 = stablehlo.not %15 : tensor<1x64x1xi1>
      %19 = stablehlo.reshape %18 : (tensor<1x64x1xi1>) -> tensor<1x64xi1>
      %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x128xi1>
      %21 = stablehlo.reshape %arg21 : (tensor<1x1x2048xbf16>) -> tensor<1x2048xbf16>
      %22 = stablehlo.reshape %arg30 : (tensor<128x2048xbf16>) -> tensor<1x128x2048xbf16>
      %23 = stablehlo.reshape %22 : (tensor<1x128x2048xbf16>) -> tensor<128x2048xbf16>
      %24 = stablehlo.transpose %23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x2048xbf16>) -> tensor<2048x128xbf16>
      %25 = stablehlo.dot_general %21, %24, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x128xbf16>) -> tensor<1x128xbf16>
      %26 = stablehlo.reshape %25 : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
      %27 = stablehlo.reshape %arg29 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %28 = stablehlo.reshape %27 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %29 = stablehlo.reshape %arg28 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %30 = stablehlo.reshape %29 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %31 = stablehlo.composite "tenstorrent.layer_norm" %26, %28, %30 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @outlined_composite_tenstorrent.layer_norm.impl} : (tensor<1x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %32 = stablehlo.slice %31 [0:1, 0:1, 0:64] : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
      %33 = stablehlo.reshape %32 : (tensor<1x1x64xbf16>) -> tensor<1x1x1x2x32xbf16>
      %34 = stablehlo.transpose %33, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,1,1,32,2]{3,4,2,1,0}"} : (tensor<1x1x1x2x32xbf16>) -> tensor<1x1x1x32x2xbf16>
      %35 = stablehlo.convert %34 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,1,1,32,2]{3,4,2,1,0}"} : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x2xf32>
      %36 = stablehlo.slice %35 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %37 = stablehlo.reshape %36 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %38 = stablehlo.reshape %arg25 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %39 = stablehlo.slice %38 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %40 = stablehlo.reshape %39 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %41 = stablehlo.convert %40 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %42 = stablehlo.multiply %37, %41 : tensor<1x1x1x32xf32>
      %43 = stablehlo.slice %35 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %44 = stablehlo.reshape %43 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %45 = stablehlo.slice %38 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %46 = stablehlo.reshape %45 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %47 = stablehlo.convert %46 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %48 = stablehlo.multiply %44, %47 : tensor<1x1x1x32xf32>
      %49 = stablehlo.subtract %42, %48 : tensor<1x1x1x32xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %51 = stablehlo.multiply %37, %47 : tensor<1x1x1x32xf32>
      %52 = stablehlo.multiply %44, %41 : tensor<1x1x1x32xf32>
      %53 = stablehlo.add %51, %52 : tensor<1x1x1x32xf32>
      %54 = stablehlo.reshape %53 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %55 = stablehlo.concatenate %50, %54, dim = 4 : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
      %56 = stablehlo.reshape %55 : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
      %57 = stablehlo.slice %56 [0:1, 0:1, 0:1, 0:64:2] : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
      %58 = stablehlo.slice %56 [0:1, 0:1, 0:1, 1:64:2] : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
      %59 = stablehlo.concatenate %57, %58, dim = 3 : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x64xf32>
      %60 = stablehlo.convert %59 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
      %61 = stablehlo.reshape %60 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
      %62 = stablehlo.slice %31 [0:1, 0:1, 64:128] : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
      %63 = stablehlo.concatenate %61, %62, dim = 2 : (tensor<1x1x64xbf16>, tensor<1x1x64xbf16>) -> tensor<1x1x128xbf16>
      %64 = stablehlo.reshape %63 : (tensor<1x1x128xbf16>) -> tensor<1x128xbf16>
      %65 = stablehlo.reshape %arg27 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %66 = stablehlo.reshape %65 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
      %67 = stablehlo.transpose %66, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %68 = stablehlo.dot_general %64, %67, contracting_dims = [1] x [0] : (tensor<1x128xbf16>, tensor<128x128xbf16>) -> tensor<1x128xbf16>
      %69 = stablehlo.reshape %68 : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
      %70 = stablehlo.floor %cst_4 : tensor<64xf32>
      %71 = stablehlo.convert %70 : (tensor<64xf32>) -> tensor<64xi64>
      %72 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %73 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %74 = stablehlo.clamp %73, %71, %72 : tensor<64xi64>
      %75 = stablehlo.compare  LT, %74, %10 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %76 = stablehlo.add %74, %9 : tensor<64xi64>
      %77 = stablehlo.select %75, %76, %74 : tensor<64xi1>, tensor<64xi64>
      %78 = stablehlo.reshape %77 : (tensor<64xi64>) -> tensor<64x1xi64>
      %79 = "stablehlo.gather"(%69, %78) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 128>}> : (tensor<1x1x128xbf16>, tensor<64x1xi64>) -> tensor<1x64x128xbf16>
      %80 = stablehlo.select %20, %11, %79 : tensor<1x64x128xi1>, tensor<1x64x128xbf16>
      %81 = stablehlo.select %17, %80, %arg26 : tensor<1x64x128xi1>, tensor<1x64x128xbf16>
      %82 = stablehlo.slice %81 [0:1, 0:33, 0:128] : (tensor<1x64x128xbf16>) -> tensor<1x33x128xbf16>
      %83 = stablehlo.reshape %arg36 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %84 = stablehlo.reshape %83 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %85 = stablehlo.convert %84 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %86 = stablehlo.reshape %85 : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
      %87 = stablehlo.reshape %arg35 : (tensor<3072x2048xbf16>) -> tensor<1x3072x2048xbf16>
      %88 = stablehlo.reshape %87 : (tensor<1x3072x2048xbf16>) -> tensor<3072x2048xbf16>
      %89 = stablehlo.transpose %88, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x2048xbf16>) -> tensor<2048x3072xbf16>
      %90 = stablehlo.dot_general %21, %89, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x3072xbf16>) -> tensor<1x3072xbf16>
      %91 = stablehlo.reshape %90 : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
      %92 = stablehlo.convert %91 : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
      %93 = stablehlo.power %92, %8 : tensor<1x1x3072xf32>
      %94 = stablehlo.reduce(%93 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x3072xf32>, tensor<f32>) -> tensor<1x1xf32>
      %95 = stablehlo.multiply %94, %cst_6 : tensor<1x1xf32>
      %96 = stablehlo.reshape %95 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %97 = stablehlo.add %96, %cst_2 : tensor<1x1x1xf32>
      %98 = stablehlo.rsqrt %97 : tensor<1x1x1xf32>
      %99 = stablehlo.reshape %98 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x3072xf32>
      %101 = stablehlo.multiply %92, %100 : tensor<1x1x3072xf32>
      %102 = stablehlo.multiply %86, %101 : tensor<1x1x3072xf32>
      %103 = stablehlo.convert %102 : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
      %104 = stablehlo.reshape %103 : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
      %105 = stablehlo.reshape %arg34 : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
      %106 = stablehlo.reshape %105 : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
      %107 = stablehlo.transpose %106, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
      %108 = stablehlo.dot_general %104, %107, contracting_dims = [1] x [0] : (tensor<1x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<1x2048xbf16>
      %109 = stablehlo.reshape %108 : (tensor<1x2048xbf16>) -> tensor<1x1x16x128xbf16>
      %110 = stablehlo.slice %109 [0:1, 0:1, 0:16, 0:64] : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
      %111 = stablehlo.reshape %110 : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x2x32xbf16>
      %112 = stablehlo.transpose %111, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[1,1,64,32,2]{3,4,2,1,0}"} : (tensor<1x1x16x2x32xbf16>) -> tensor<1x1x16x32x2xbf16>
      %113 = stablehlo.convert %112 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,1,64,32,2]{3,4,2,1,0}"} : (tensor<1x1x16x32x2xbf16>) -> tensor<1x1x16x32x2xf32>
      %114 = stablehlo.slice %113 [0:1, 0:1, 0:16, 0:32, 0:1] : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
      %115 = stablehlo.reshape %114 : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
      %116 = stablehlo.reshape %41 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x16x32xf32>
      %118 = stablehlo.multiply %115, %117 : tensor<1x1x16x32xf32>
      %119 = stablehlo.slice %113 [0:1, 0:1, 0:16, 0:32, 1:2] : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
      %120 = stablehlo.reshape %119 : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
      %121 = stablehlo.reshape %47 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x16x32xf32>
      %123 = stablehlo.multiply %120, %122 : tensor<1x1x16x32xf32>
      %124 = stablehlo.subtract %118, %123 : tensor<1x1x16x32xf32>
      %125 = stablehlo.reshape %124 : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
      %126 = stablehlo.multiply %115, %122 : tensor<1x1x16x32xf32>
      %127 = stablehlo.multiply %120, %117 : tensor<1x1x16x32xf32>
      %128 = stablehlo.add %126, %127 : tensor<1x1x16x32xf32>
      %129 = stablehlo.reshape %128 : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
      %130 = stablehlo.concatenate %125, %129, dim = 4 : (tensor<1x1x16x32x1xf32>, tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32x2xf32>
      %131 = stablehlo.reshape %130 : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x64xf32>
      %132 = stablehlo.slice %131 [0:1, 0:1, 0:16, 0:64:2] : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
      %133 = stablehlo.slice %131 [0:1, 0:1, 0:16, 1:64:2] : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
      %134 = stablehlo.concatenate %132, %133, dim = 3 : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x64xf32>
      %135 = stablehlo.convert %134 : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x64xbf16>
      %136 = stablehlo.slice %109 [0:1, 0:1, 0:16, 64:128] : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
      %137 = stablehlo.concatenate %135, %136, dim = 3 : (tensor<1x1x16x64xbf16>, tensor<1x1x16x64xbf16>) -> tensor<1x1x16x128xbf16>
      %138 = stablehlo.dot_general %137, %67, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<1x1x16x128xbf16>
      %139 = stablehlo.reshape %138 : (tensor<1x1x16x128xbf16>) -> tensor<1x16x128xbf16>
      %140 = stablehlo.transpose %139, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,128,64]{1,2,0}"} : (tensor<1x16x128xbf16>) -> tensor<1x128x16xbf16>
      %141 = stablehlo.dot_general %82, %140, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x33x128xbf16>, tensor<1x128x16xbf16>) -> tensor<1x33x16xbf16>
      %142 = stablehlo.maximum %141, %7 : tensor<1x33x16xbf16>
      %143 = stablehlo.reshape %arg33 : (tensor<16x2048xbf16>) -> tensor<1x16x2048xbf16>
      %144 = stablehlo.reshape %143 : (tensor<1x16x2048xbf16>) -> tensor<16x2048xbf16>
      %145 = stablehlo.transpose %144, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<16x2048xbf16>) -> tensor<2048x16xbf16>
      %146 = stablehlo.dot_general %21, %145, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x16xbf16>) -> tensor<1x16xbf16>
      %147 = stablehlo.reshape %146 : (tensor<1x16xbf16>) -> tensor<1x1x16xbf16>
      %148 = stablehlo.multiply %147, %6 : tensor<1x1x16xbf16>
      %149 = stablehlo.reshape %148 : (tensor<1x1x16xbf16>) -> tensor<1x1x16x1xbf16>
      %150 = stablehlo.multiply %149, %5 : tensor<1x1x16x1xbf16>
      %151 = stablehlo.reshape %150 : (tensor<1x1x16x1xbf16>) -> tensor<1x16xbf16>
      %152 = stablehlo.broadcast_in_dim %151, dims = [0, 2] : (tensor<1x16xbf16>) -> tensor<1x33x16xbf16>
      %153 = stablehlo.multiply %142, %152 : tensor<1x33x16xbf16>
      %154 = stablehlo.reduce(%153 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<1x33x16xbf16>, tensor<bf16>) -> tensor<1x33xbf16>
      %155 = "stablehlo.all_reduce"(%154) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %269 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %269 : tensor<bf16>
      }) : (tensor<1x33xbf16>) -> tensor<1x33xbf16>
      %156 = stablehlo.reshape %155 : (tensor<1x33xbf16>) -> tensor<1x1x33xbf16>
      %157 = stablehlo.composite "tenstorrent.topk_indices" %156 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @outlined_composite_tenstorrent.topk_indices.impl} : (tensor<1x1x33xbf16>) -> tensor<1x1x33xi64>
      %158 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x512xi1>
      %159 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x512xi1>
      %160 = stablehlo.reshape %arg22 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %161 = stablehlo.reshape %160 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %162 = stablehlo.convert %161 : (tensor<512xbf16>) -> tensor<512xf32>
      %163 = stablehlo.reshape %162 : (tensor<512xf32>) -> tensor<1x1x512xf32>
      %164 = stablehlo.reshape %arg20 : (tensor<576x2048xbf16>) -> tensor<1x576x2048xbf16>
      %165 = stablehlo.reshape %164 : (tensor<1x576x2048xbf16>) -> tensor<576x2048xbf16>
      %166 = stablehlo.transpose %165, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x2048xbf16>) -> tensor<2048x576xbf16>
      %167 = stablehlo.dot_general %21, %166, contracting_dims = [1] x [0] : (tensor<1x2048xbf16>, tensor<2048x576xbf16>) -> tensor<1x576xbf16>
      %168 = stablehlo.reshape %167 : (tensor<1x576xbf16>) -> tensor<1x1x576xbf16>
      %169 = stablehlo.slice %168 [0:1, 0:1, 0:512] : (tensor<1x1x576xbf16>) -> tensor<1x1x512xbf16>
      %170 = stablehlo.convert %169 : (tensor<1x1x512xbf16>) -> tensor<1x1x512xf32>
      %171 = stablehlo.power %170, %3 : tensor<1x1x512xf32>
      %172 = stablehlo.reduce(%171 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x512xf32>, tensor<f32>) -> tensor<1x1xf32>
      %173 = stablehlo.multiply %172, %cst_10 : tensor<1x1xf32>
      %174 = stablehlo.reshape %173 : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
      %175 = stablehlo.add %174, %cst_2 : tensor<1x1x1xf32>
      %176 = stablehlo.rsqrt %175 : tensor<1x1x1xf32>
      %177 = stablehlo.reshape %176 : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
      %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x1x512xf32>
      %179 = stablehlo.multiply %170, %178 : tensor<1x1x512xf32>
      %180 = stablehlo.multiply %163, %179 : tensor<1x1x512xf32>
      %181 = stablehlo.convert %180 : (tensor<1x1x512xf32>) -> tensor<1x1x512xbf16>
      %182 = "stablehlo.gather"(%181, %78) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<1x1x512xbf16>, tensor<64x1xi64>) -> tensor<1x64x512xbf16>
      %183 = stablehlo.select %159, %4, %182 : tensor<1x64x512xi1>, tensor<1x64x512xbf16>
      %184 = stablehlo.select %158, %183, %arg19 : tensor<1x64x512xi1>, tensor<1x64x512xbf16>
      %185 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x64xi1>
      %186 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x64xi1>) -> tensor<1x64x64xi1>
      %187 = stablehlo.slice %168 [0:1, 0:1, 512:576] : (tensor<1x1x576xbf16>) -> tensor<1x1x64xbf16>
      %188 = stablehlo.reshape %187 : (tensor<1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
      %189 = stablehlo.convert %188 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xf32>
      %190 = stablehlo.reshape %189 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32x2xf32>
      %191 = stablehlo.slice %190 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %192 = stablehlo.reshape %191 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %193 = stablehlo.multiply %192, %41 : tensor<1x1x1x32xf32>
      %194 = stablehlo.slice %190 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
      %195 = stablehlo.reshape %194 : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
      %196 = stablehlo.multiply %195, %47 : tensor<1x1x1x32xf32>
      %197 = stablehlo.subtract %193, %196 : tensor<1x1x1x32xf32>
      %198 = stablehlo.reshape %197 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %199 = stablehlo.multiply %192, %47 : tensor<1x1x1x32xf32>
      %200 = stablehlo.multiply %195, %41 : tensor<1x1x1x32xf32>
      %201 = stablehlo.add %199, %200 : tensor<1x1x1x32xf32>
      %202 = stablehlo.reshape %201 : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
      %203 = stablehlo.concatenate %198, %202, dim = 4 : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
      %204 = stablehlo.reshape %203 : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
      %205 = stablehlo.convert %204 : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
      %206 = stablehlo.reshape %205 : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
      %207 = "stablehlo.gather"(%206, %78) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 64>}> : (tensor<1x1x64xbf16>, tensor<64x1xi64>) -> tensor<1x64x64xbf16>
      %208 = stablehlo.select %186, %2, %207 : tensor<1x64x64xi1>, tensor<1x64x64xbf16>
      %209 = stablehlo.select %185, %208, %arg24 : tensor<1x64x64xi1>, tensor<1x64x64xbf16>
      %210 = stablehlo.reshape %arg37 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %211 = stablehlo.reshape %210 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %212 = stablehlo.transpose %211, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %213 = stablehlo.dot_general %104, %212, contracting_dims = [1] x [0] : (tensor<1x3072xbf16>, tensor<3072x768xbf16>) -> tensor<1x768xbf16>
      %214 = stablehlo.reshape %213 : (tensor<1x768xbf16>) -> tensor<1x1x4x192xbf16>
      %215 = stablehlo.slice %214 [0:1, 0:1, 0:4, 0:128] : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x128xbf16>
      %216 = stablehlo.reshape %arg32 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %217 = stablehlo.reshape %216 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %218 = stablehlo.slice %217 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %219 = stablehlo.dot_general %215, %218, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x512xbf16>
      %220 = stablehlo.reshape %219 : (tensor<4x1x1x512xbf16>) -> tensor<1x1x4x512xbf16>
      %221 = stablehlo.slice %184 [0:1, 0:33, 0:512] : (tensor<1x64x512xbf16>) -> tensor<1x33x512xbf16>
      %222 = stablehlo.reshape %157 : (tensor<1x1x33xi64>) -> tensor<1x33xi64>
      %223 = stablehlo.broadcast_in_dim %222, dims = [0, 1] : (tensor<1x33xi64>) -> tensor<1x33x512xi64>
      %224 = stablehlo.composite "tenstorrent.gather" %221, %223 {composite_attributes = {dim = 1 : i64, sparse_grad = false}, decomposition = @outlined_composite_tenstorrent.gather.impl_0} : (tensor<1x33x512xbf16>, tensor<1x33x512xi64>) -> tensor<1x33x512xbf16>
      %225 = stablehlo.dot_general %220, %224, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x512xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x33xbf16>
      %226 = stablehlo.slice %214 [0:1, 0:1, 0:4, 128:192] : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x64xbf16>
      %227 = stablehlo.convert %226 : (tensor<1x1x4x64xbf16>) -> tensor<1x1x4x64xf32>
      %228 = stablehlo.reshape %227 : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x32x2xf32>
      %229 = stablehlo.slice %228 [0:1, 0:1, 0:4, 0:32, 0:1] : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
      %230 = stablehlo.reshape %229 : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
      %231 = stablehlo.broadcast_in_dim %116, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
      %232 = stablehlo.multiply %230, %231 : tensor<1x1x4x32xf32>
      %233 = stablehlo.slice %228 [0:1, 0:1, 0:4, 0:32, 1:2] : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
      %234 = stablehlo.reshape %233 : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
      %235 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 3] : (tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
      %236 = stablehlo.multiply %234, %235 : tensor<1x1x4x32xf32>
      %237 = stablehlo.subtract %232, %236 : tensor<1x1x4x32xf32>
      %238 = stablehlo.reshape %237 : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
      %239 = stablehlo.multiply %230, %235 : tensor<1x1x4x32xf32>
      %240 = stablehlo.multiply %234, %231 : tensor<1x1x4x32xf32>
      %241 = stablehlo.add %239, %240 : tensor<1x1x4x32xf32>
      %242 = stablehlo.reshape %241 : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
      %243 = stablehlo.concatenate %238, %242, dim = 4 : (tensor<1x1x4x32x1xf32>, tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32x2xf32>
      %244 = stablehlo.reshape %243 : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x64xf32>
      %245 = stablehlo.convert %244 : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x64xbf16>
      %246 = stablehlo.slice %209 [0:1, 0:33, 0:64] : (tensor<1x64x64xbf16>) -> tensor<1x33x64xbf16>
      %247 = stablehlo.broadcast_in_dim %222, dims = [0, 1] : (tensor<1x33xi64>) -> tensor<1x33x64xi64>
      %248 = stablehlo.composite "tenstorrent.gather" %246, %247 {composite_attributes = {dim = 1 : i64, sparse_grad = false}, decomposition = @outlined_composite_tenstorrent.gather.impl} : (tensor<1x33x64xbf16>, tensor<1x33x64xi64>) -> tensor<1x33x64xbf16>
      %249 = stablehlo.dot_general %245, %248, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x64xbf16>, tensor<1x33x64xbf16>) -> tensor<1x1x4x33xbf16>
      %250 = stablehlo.add %225, %249 : tensor<1x1x4x33xbf16>
      %251 = stablehlo.multiply %250, %1 : tensor<1x1x4x33xbf16>
      %252 = stablehlo.reduce(%251 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x33xbf16>, tensor<bf16>) -> tensor<1x1x4xbf16>
      %253 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2] : (tensor<1x1x4xbf16>) -> tensor<1x1x4x33xbf16>
      %254 = stablehlo.subtract %251, %253 : tensor<1x1x4x33xbf16>
      %255 = stablehlo.exponential %254 : tensor<1x1x4x33xbf16>
      %256 = stablehlo.reduce(%255 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x33xbf16>, tensor<bf16>) -> tensor<1x1x4xbf16>
      %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2] : (tensor<1x1x4xbf16>) -> tensor<1x1x4x33xbf16>
      %258 = stablehlo.divide %255, %257 : tensor<1x1x4x33xbf16>
      %259 = stablehlo.dot_general %258, %224, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x33xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x512xbf16>
      %260 = stablehlo.slice %217 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %261 = stablehlo.dot_general %259, %260, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x128xbf16>
      %262 = stablehlo.reshape %261 : (tensor<4x1x1x128xbf16>) -> tensor<1x512xbf16>
      %263 = stablehlo.reshape %arg31 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
      %264 = stablehlo.reshape %263 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
      %265 = stablehlo.transpose %264, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
      %266 = stablehlo.dot_general %262, %265, contracting_dims = [1] x [0] : (tensor<1x512xbf16>, tensor<512x2048xbf16>) -> tensor<1x2048xbf16>
      %267 = "stablehlo.all_reduce"(%266) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %269 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %269 : tensor<bf16>
      }) : (tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
      %268 = stablehlo.reshape %267 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
      sdy.return %184, %209, %81, %268 : tensor<1x64x512xbf16>, tensor<1x64x64xbf16>, tensor<1x64x128xbf16>, tensor<1x1x2048xbf16>
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
  func.func private @outlined_composite_tenstorrent.gather.impl(%arg0: tensor<1x33x64xbf16>, %arg1: tensor<1x33x64xi64>) -> tensor<1x33x64xbf16> {
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1x33x64x1xui32>
    %1 = stablehlo.convert %arg1 : (tensor<1x33x64xi64>) -> tensor<1x33x64xui32>
    %2 = stablehlo.reshape %1 : (tensor<1x33x64xui32>) -> tensor<1x33x64x1xui32>
    %3 = stablehlo.iota dim = 0 : tensor<64xui32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<64xui32>) -> tensor<1x33x64x1xui32>
    %5 = stablehlo.concatenate %0, %2, %4, dim = 3 : (tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>) -> tensor<1x33x64x3xui32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x33x64xbf16>, tensor<1x33x64x3xui32>) -> tensor<1x33x64xbf16>
    return %6 : tensor<1x33x64xbf16>
  }
  func.func private @outlined_composite_tenstorrent.gather.impl_0(%arg0: tensor<1x33x512xbf16>, %arg1: tensor<1x33x512xi64>) -> tensor<1x33x512xbf16> {
    %c = stablehlo.constant dense<0> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<1x33x512x1xui32>
    %1 = stablehlo.convert %arg1 : (tensor<1x33x512xi64>) -> tensor<1x33x512xui32>
    %2 = stablehlo.reshape %1 : (tensor<1x33x512xui32>) -> tensor<1x33x512x1xui32>
    %3 = stablehlo.iota dim = 0 : tensor<512xui32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<512xui32>) -> tensor<1x33x512x1xui32>
    %5 = stablehlo.concatenate %0, %2, %4, dim = 3 : (tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>) -> tensor<1x33x512x3xui32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x33x512xbf16>, tensor<1x33x512x3xui32>) -> tensor<1x33x512xbf16>
    return %6 : tensor<1x33x512xbf16>
  }
}
