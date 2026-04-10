// ============================================================
// MLIR Module: shlo_compiler (SyncTensorsGraph.761)
// Extracted from: replicated_input_32_4.log
// ============================================================

module @SyncTensorsGraph.761 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<4x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<4x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<4x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:4 = sdy.manual_computation(
      %arg0,  // kv_cache: tensor<4x64x512xbf16>
      %arg1,  // wkv_a.weight: tensor<576x2048xbf16>
      %arg2,  // args_0: tensor<4x1x2048xbf16>
      %arg3,  // kv_norm.weight: tensor<512xbf16>
      %arg4,  // (unnamed): tensor<i1>
      %arg5,  // pe_cache: tensor<4x64x64xbf16>
      %arg6,  // args_1: tensor<1x32x2xbf16>
      %arg7,  // indexer.k_cache: tensor<4x64x128xbf16>
      %arg8,  // indexer.haddamard: tensor<128x128xbf16>
      %arg9,  // indexer.k_norm.bias: tensor<128xbf16>
      %arg10,  // indexer.k_norm.weight: tensor<128xbf16>
      %arg11,  // indexer.wk.weight: tensor<128x2048xbf16>
      %arg12,  // wo.weight: tensor<2048x2048xbf16>
      %arg13,  // wkv_b.weight: tensor<4096x512xbf16>
      %arg14,  // indexer.weights_proj.weight: tensor<64x2048xbf16>
      %arg15,  // indexer.wq_b.weight: tensor<8192x3072xbf16>
      %arg16,  // wq_a.weight: tensor<3072x2048xbf16>
      %arg17,  // q_norm.weight: tensor<3072xbf16>
      %arg18  // wq_b.weight: tensor<3072x3072xbf16>
    )
    in_shardings=[
      // %arg0  kv_cache: tensor<4x64x512xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // %arg1  wkv_a.weight: tensor<576x2048xbf16>
      <@mesh, [{}, {"_axis_0"}]>,
      // %arg2  args_0: tensor<4x1x2048xbf16>
      <@mesh, [{}, {}, {}]>,
      // %arg3  kv_norm.weight: tensor<512xbf16>
      <@mesh, [{}]>,
      // %arg4  (unnamed): tensor<i1>
      <@mesh, []>,
      // %arg5  pe_cache: tensor<4x64x64xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // %arg6  args_1: tensor<1x32x2xbf16>
      <@mesh, [{}, {}, {}]>,
      // %arg7  indexer.k_cache: tensor<4x64x128xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // %arg8  indexer.haddamard: tensor<128x128xbf16>
      <@mesh, [{}, {}]>,
      // %arg9  indexer.k_norm.bias: tensor<128xbf16>
      <@mesh, [{}]>,
      // %arg10  indexer.k_norm.weight: tensor<128xbf16>
      <@mesh, [{}]>,
      // %arg11  indexer.wk.weight: tensor<128x2048xbf16>
      <@mesh, [{}, {"_axis_0"}]>,
      // %arg12  wo.weight: tensor<2048x2048xbf16>
      <@mesh, [{"_axis_0"}, {"_axis_1"}]>,
      // %arg13  wkv_b.weight: tensor<4096x512xbf16>
      <@mesh, [{"_axis_1"}, {}]>,
      // %arg14  indexer.weights_proj.weight: tensor<64x2048xbf16>
      <@mesh, [{"_axis_1"}, {"_axis_0"}]>,
      // %arg15  indexer.wq_b.weight: tensor<8192x3072xbf16>
      <@mesh, [{"_axis_1"}, {}]>,
      // %arg16  wq_a.weight: tensor<3072x2048xbf16>
      <@mesh, [{}, {"_axis_0"}]>,
      // %arg17  q_norm.weight: tensor<3072xbf16>
      <@mesh, [{}]>,
      // %arg18  wq_b.weight: tensor<3072x3072xbf16>
      <@mesh, [{"_axis_1"}, {}]>
    ]
    out_shardings=[
      // out0  kv_cache (updated): tensor<4x64x512xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // out1  pe_cache (updated): tensor<4x64x64xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // out2  indexer.k_cache (updated): tensor<4x64x128xbf16>
      <@mesh, [{"_axis_0"}, {}, {}]>,
      // out3  attention_output: tensor<4x1x2048xbf16>
      <@mesh, [{}, {}, {"_axis_0"}]>
    ]
    manual_axes={"_axis_0", "_axis_1"}
    (
      %arg19: tensor<2x64x512xbf16>,  // kv_cache (per-device)
      %arg20: tensor<576x1024xbf16>,  // wkv_a.weight (per-device)
      %arg21: tensor<4x1x2048xbf16>,  // args_0 (per-device)
      %arg22: tensor<512xbf16>,  // kv_norm.weight (per-device)
      %arg23: tensor<i1>,  // (unnamed) (per-device)
      %arg24: tensor<2x64x64xbf16>,  // pe_cache (per-device)
      %arg25: tensor<1x32x2xbf16>,  // args_1 (per-device)
      %arg26: tensor<2x64x128xbf16>,  // indexer.k_cache (per-device)
      %arg27: tensor<128x128xbf16>,  // indexer.haddamard (per-device)
      %arg28: tensor<128xbf16>,  // indexer.k_norm.bias (per-device)
      %arg29: tensor<128xbf16>,  // indexer.k_norm.weight (per-device)
      %arg30: tensor<128x1024xbf16>,  // indexer.wk.weight (per-device)
      %arg31: tensor<1024x512xbf16>,  // wo.weight (per-device)
      %arg32: tensor<1024x512xbf16>,  // wkv_b.weight (per-device)
      %arg33: tensor<16x1024xbf16>,  // indexer.weights_proj.weight (per-device)
      %arg34: tensor<2048x3072xbf16>,  // indexer.wq_b.weight (per-device)
      %arg35: tensor<3072x1024xbf16>,  // wq_a.weight (per-device)
      %arg36: tensor<3072xbf16>,  // q_norm.weight (per-device)
      %arg37: tensor<768x3072xbf16>  // wq_b.weight (per-device)
    ) {
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
      %20 = stablehlo.composite "sdy.all_slice" %arg21 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<4x1x2048xbf16>) -> tensor<2x1x2048xbf16>
      %21 = stablehlo.reshape %20 : (tensor<2x1x2048xbf16>) -> tensor<2x2048xbf16>
      %22 = stablehlo.reshape %arg30 : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
      %23 = stablehlo.reshape %22 : (tensor<1x128x1024xbf16>) -> tensor<128x1024xbf16>
      %24 = stablehlo.transpose %23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,128]{0,1}"} : (tensor<128x1024xbf16>) -> tensor<1024x128xbf16>
      %25 = "stablehlo.all_to_all"(%21) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x2048xbf16>) -> tensor<4x1024xbf16>
      %26 = stablehlo.dot_general %25, %24, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x128xbf16>) -> tensor<4x128xbf16>
      %27 = "stablehlo.reduce_scatter"(%26) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<4x128xbf16>) -> tensor<2x128xbf16>
      %28 = stablehlo.reshape %27 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %29 = stablehlo.reshape %arg29 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %30 = stablehlo.reshape %29 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %31 = stablehlo.reshape %arg28 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %32 = stablehlo.reshape %31 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %33 = stablehlo.composite "tenstorrent.layer_norm" %28, %30, %32 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @outlined_composite_tenstorrent.layer_norm.impl} : (tensor<2x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<2x1x128xbf16>
      %34 = stablehlo.slice %33 [0:2, 0:1, 0:64] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %35 = stablehlo.reshape %34 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x2x32xbf16>
      %36 = stablehlo.transpose %35, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x2x32xbf16>) -> tensor<2x1x1x32x2xbf16>
      %37 = stablehlo.convert %36 {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[4,1,1,32,2]{3,4,2,1,0}"} : (tensor<2x1x1x32x2xbf16>) -> tensor<2x1x1x32x2xf32>
      %38 = stablehlo.slice %37 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %39 = stablehlo.reshape %38 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %40 = stablehlo.reshape %arg25 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %41 = stablehlo.slice %40 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %42 = stablehlo.reshape %41 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %43 = stablehlo.convert %42 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %44 = stablehlo.reshape %43 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %45 = stablehlo.broadcast_in_dim %44, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %46 = stablehlo.multiply %39, %45 : tensor<2x1x1x32xf32>
      %47 = stablehlo.slice %37 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %48 = stablehlo.reshape %47 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %49 = stablehlo.slice %40 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %50 = stablehlo.reshape %49 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %51 = stablehlo.convert %50 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %53 = stablehlo.broadcast_in_dim %52, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %54 = stablehlo.multiply %48, %53 : tensor<2x1x1x32xf32>
      %55 = stablehlo.subtract %46, %54 : tensor<2x1x1x32xf32>
      %56 = stablehlo.reshape %55 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %57 = stablehlo.multiply %39, %53 : tensor<2x1x1x32xf32>
      %58 = stablehlo.multiply %48, %45 : tensor<2x1x1x32xf32>
      %59 = stablehlo.add %57, %58 : tensor<2x1x1x32xf32>
      %60 = stablehlo.reshape %59 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %61 = stablehlo.concatenate %56, %60, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %62 = stablehlo.reshape %61 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %63 = stablehlo.slice %62 [0:2, 0:1, 0:1, 0:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %64 = stablehlo.slice %62 [0:2, 0:1, 0:1, 1:64:2] : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
      %65 = stablehlo.concatenate %63, %64, dim = 3 : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x64xf32>
      %66 = stablehlo.convert %65 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %67 = stablehlo.reshape %66 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %68 = stablehlo.slice %33 [0:2, 0:1, 64:128] : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
      %69 = stablehlo.concatenate %67, %68, dim = 2 : (tensor<2x1x64xbf16>, tensor<2x1x64xbf16>) -> tensor<2x1x128xbf16>
      %70 = stablehlo.reshape %69 : (tensor<2x1x128xbf16>) -> tensor<2x128xbf16>
      %71 = stablehlo.reshape %arg27 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %72 = stablehlo.reshape %71 : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
      %73 = stablehlo.transpose %72, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[128,128]{0,1}"} : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %74 = stablehlo.dot_general %70, %73, contracting_dims = [1] x [0] : (tensor<2x128xbf16>, tensor<128x128xbf16>) -> tensor<2x128xbf16>
      %75 = stablehlo.reshape %74 : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
      %76 = stablehlo.floor %cst_4 : tensor<64xf32>
      %77 = stablehlo.convert %76 : (tensor<64xf32>) -> tensor<64xi64>
      %78 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %79 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %80 = stablehlo.clamp %79, %77, %78 : tensor<64xi64>
      %81 = stablehlo.compare  LT, %80, %10 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %82 = stablehlo.add %80, %9 : tensor<64xi64>
      %83 = stablehlo.select %81, %82, %80 : tensor<64xi1>, tensor<64xi64>
      %84 = stablehlo.reshape %83 : (tensor<64xi64>) -> tensor<64x1xi64>
      %85 = "stablehlo.gather"(%75, %84) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 128>}> : (tensor<2x1x128xbf16>, tensor<64x1xi64>) -> tensor<2x64x128xbf16>
      %86 = stablehlo.select %19, %11, %85 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %87 = stablehlo.select %16, %86, %arg26 : tensor<2x64x128xi1>, tensor<2x64x128xbf16>
      %88 = stablehlo.slice %87 [0:2, 0:33, 0:128] : (tensor<2x64x128xbf16>) -> tensor<2x33x128xbf16>
      %89 = stablehlo.reshape %arg36 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %90 = stablehlo.reshape %89 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %91 = stablehlo.convert %90 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %92 = stablehlo.broadcast_in_dim %91, dims = [2] : (tensor<3072xf32>) -> tensor<2x1x3072xf32>
      %93 = stablehlo.reshape %arg35 : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
      %94 = stablehlo.reshape %93 : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
      %95 = stablehlo.transpose %94, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
      %96 = "stablehlo.all_to_all"(%21) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x2048xbf16>) -> tensor<4x1024xbf16>
      %97 = stablehlo.dot_general %96, %95, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
      %98 = "stablehlo.reduce_scatter"(%97) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
      %99 = stablehlo.reshape %98 : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
      %100 = stablehlo.convert %99 : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
      %101 = stablehlo.power %100, %8 : tensor<2x1x3072xf32>
      %102 = stablehlo.reduce(%101 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x3072xf32>, tensor<f32>) -> tensor<2x1xf32>
      %103 = stablehlo.multiply %102, %cst_6 : tensor<2x1xf32>
      %104 = stablehlo.reshape %103 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %105 = stablehlo.add %104, %cst_2 : tensor<2x1x1xf32>
      %106 = stablehlo.rsqrt %105 : tensor<2x1x1xf32>
      %107 = stablehlo.reshape %106 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x3072xf32>
      %109 = stablehlo.multiply %100, %108 : tensor<2x1x3072xf32>
      %110 = stablehlo.multiply %92, %109 : tensor<2x1x3072xf32>
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
      %124 = stablehlo.reshape %43 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %125 = stablehlo.broadcast_in_dim %124, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x16x32xf32>
      %126 = stablehlo.multiply %123, %125 : tensor<2x1x16x32xf32>
      %127 = stablehlo.slice %121 [0:2, 0:1, 0:16, 0:32, 1:2] : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
      %128 = stablehlo.reshape %127 : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
      %129 = stablehlo.reshape %51 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
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
      %146 = stablehlo.dot_general %145, %73, contracting_dims = [3] x [0] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<2x1x16x128xbf16>
      %147 = stablehlo.reshape %146 : (tensor<2x1x16x128xbf16>) -> tensor<2x16x128xbf16>
      %148 = stablehlo.transpose %147, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[4,128,64]{1,2,0}"} : (tensor<2x16x128xbf16>) -> tensor<2x128x16xbf16>
      %149 = stablehlo.dot_general %88, %148, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x33x128xbf16>, tensor<2x128x16xbf16>) -> tensor<2x33x16xbf16>
      %150 = stablehlo.maximum %149, %7 : tensor<2x33x16xbf16>
      %151 = stablehlo.reshape %arg33 : (tensor<16x1024xbf16>) -> tensor<1x16x1024xbf16>
      %152 = stablehlo.reshape %151 : (tensor<1x16x1024xbf16>) -> tensor<16x1024xbf16>
      %153 = stablehlo.transpose %152, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,64]{0,1}"} : (tensor<16x1024xbf16>) -> tensor<1024x16xbf16>
      %154 = "stablehlo.all_to_all"(%21) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x2048xbf16>) -> tensor<4x1024xbf16>
      %155 = stablehlo.dot_general %154, %153, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x16xbf16>) -> tensor<4x16xbf16>
      %156 = "stablehlo.reduce_scatter"(%155) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<4x16xbf16>) -> tensor<2x16xbf16>
      %157 = stablehlo.reshape %156 : (tensor<2x16xbf16>) -> tensor<2x1x16xbf16>
      %158 = stablehlo.multiply %157, %6 : tensor<2x1x16xbf16>
      %159 = stablehlo.reshape %158 : (tensor<2x1x16xbf16>) -> tensor<2x1x16x1xbf16>
      %160 = stablehlo.multiply %159, %5 : tensor<2x1x16x1xbf16>
      %161 = stablehlo.reshape %160 : (tensor<2x1x16x1xbf16>) -> tensor<2x16xbf16>
      %162 = stablehlo.broadcast_in_dim %161, dims = [0, 2] : (tensor<2x16xbf16>) -> tensor<2x33x16xbf16>
      %163 = stablehlo.multiply %150, %162 : tensor<2x33x16xbf16>
      %164 = stablehlo.reduce(%163 init: %cst_11) applies stablehlo.add across dimensions = [2] : (tensor<2x33x16xbf16>, tensor<bf16>) -> tensor<2x33xbf16>
      %165 = "stablehlo.all_reduce"(%164) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<2x33xbf16>) -> tensor<2x33xbf16>
      %166 = stablehlo.reshape %165 : (tensor<2x33xbf16>) -> tensor<2x1x33xbf16>
      %167 = stablehlo.composite "tenstorrent.topk_indices" %166 {composite_attributes = {dim = -1 : i64, k = 33 : i64, largest = true, sorted = true}, decomposition = @outlined_composite_tenstorrent.topk_indices.impl} : (tensor<2x1x33xbf16>) -> tensor<2x1x33xi64>
      %168 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %169 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %170 = stablehlo.reshape %arg22 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %171 = stablehlo.reshape %170 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %172 = stablehlo.convert %171 : (tensor<512xbf16>) -> tensor<512xf32>
      %173 = stablehlo.broadcast_in_dim %172, dims = [2] : (tensor<512xf32>) -> tensor<2x1x512xf32>
      %174 = stablehlo.reshape %arg20 : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
      %175 = stablehlo.reshape %174 : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
      %176 = stablehlo.transpose %175, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
      %177 = "stablehlo.all_to_all"(%21) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x2048xbf16>) -> tensor<4x1024xbf16>
      %178 = stablehlo.dot_general %177, %176, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
      %179 = "stablehlo.reduce_scatter"(%178) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
      %180 = stablehlo.reshape %179 : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
      %181 = stablehlo.slice %180 [0:2, 0:1, 0:512] : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
      %182 = stablehlo.convert %181 : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
      %183 = stablehlo.power %182, %3 : tensor<2x1x512xf32>
      %184 = stablehlo.reduce(%183 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x512xf32>, tensor<f32>) -> tensor<2x1xf32>
      %185 = stablehlo.multiply %184, %cst_10 : tensor<2x1xf32>
      %186 = stablehlo.reshape %185 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %187 = stablehlo.add %186, %cst_2 : tensor<2x1x1xf32>
      %188 = stablehlo.rsqrt %187 : tensor<2x1x1xf32>
      %189 = stablehlo.reshape %188 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %190 = stablehlo.broadcast_in_dim %189, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x512xf32>
      %191 = stablehlo.multiply %182, %190 : tensor<2x1x512xf32>
      %192 = stablehlo.multiply %173, %191 : tensor<2x1x512xf32>
      %193 = stablehlo.convert %192 : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
      %194 = "stablehlo.gather"(%193, %84) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 512>}> : (tensor<2x1x512xbf16>, tensor<64x1xi64>) -> tensor<2x64x512xbf16>
      %195 = stablehlo.select %169, %4, %194 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %196 = stablehlo.select %168, %195, %arg19 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %197 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %198 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %199 = stablehlo.slice %180 [0:2, 0:1, 512:576] : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
      %200 = stablehlo.reshape %199 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
      %201 = stablehlo.convert %200 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
      %202 = stablehlo.reshape %201 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
      %203 = stablehlo.slice %202 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %204 = stablehlo.reshape %203 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %205 = stablehlo.multiply %204, %45 : tensor<2x1x1x32xf32>
      %206 = stablehlo.slice %202 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %207 = stablehlo.reshape %206 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %208 = stablehlo.multiply %207, %53 : tensor<2x1x1x32xf32>
      %209 = stablehlo.subtract %205, %208 : tensor<2x1x1x32xf32>
      %210 = stablehlo.reshape %209 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %211 = stablehlo.multiply %204, %53 : tensor<2x1x1x32xf32>
      %212 = stablehlo.multiply %207, %45 : tensor<2x1x1x32xf32>
      %213 = stablehlo.add %211, %212 : tensor<2x1x1x32xf32>
      %214 = stablehlo.reshape %213 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %215 = stablehlo.concatenate %210, %214, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %216 = stablehlo.reshape %215 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %217 = stablehlo.convert %216 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %218 = stablehlo.reshape %217 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %219 = "stablehlo.gather"(%218, %84) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 64>}> : (tensor<2x1x64xbf16>, tensor<64x1xi64>) -> tensor<2x64x64xbf16>
      %220 = stablehlo.select %198, %2, %219 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %221 = stablehlo.select %197, %220, %arg24 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %222 = stablehlo.reshape %arg37 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %223 = stablehlo.reshape %222 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %224 = stablehlo.transpose %223, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %225 = stablehlo.dot_general %112, %224, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
      %226 = stablehlo.reshape %225 : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
      %227 = stablehlo.slice %226 [0:2, 0:1, 0:4, 0:128] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
      %228 = stablehlo.reshape %arg32 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %229 = stablehlo.reshape %228 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %230 = stablehlo.slice %229 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %231 = stablehlo.dot_general %227, %230, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
      %232 = stablehlo.transpose %231, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
      %233 = stablehlo.slice %196 [0:2, 0:33, 0:512] : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
      %234 = stablehlo.reshape %167 : (tensor<2x1x33xi64>) -> tensor<2x33xi64>
      %235 = stablehlo.broadcast_in_dim %234, dims = [0, 1] : (tensor<2x33xi64>) -> tensor<2x33x512xi64>
      %236 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<2xui32>
      %237 = stablehlo.broadcast_in_dim %236, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2xui32>) -> tensor<2x33x512x1xui32>
      %238 = stablehlo.convert %235 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512xi64>) -> tensor<2x33x512xui32>
      %239 = stablehlo.reshape %238 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512xui32>) -> tensor<2x33x512x1xui32>
      %240 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : tensor<512xui32>
      %241 = stablehlo.broadcast_in_dim %240, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<512xui32>) -> tensor<2x33x512x1xui32>
      %242 = stablehlo.concatenate %237, %239, %241, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl_0"} : (tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>) -> tensor<2x33x512x3xui32>
      %243 = "stablehlo.all_gather"(%233) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x512xbf16>) -> tensor<4x33x512xbf16>
      %244 = "stablehlo.gather"(%243, %242) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl_0", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x512xbf16>, tensor<2x33x512x3xui32>) -> tensor<2x33x512xbf16>
      %245 = stablehlo.dot_general %232, %244, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
      %246 = stablehlo.slice %226 [0:2, 0:1, 0:4, 128:192] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
      %247 = stablehlo.convert %246 : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
      %248 = stablehlo.reshape %247 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
      %249 = stablehlo.slice %248 [0:2, 0:1, 0:4, 0:32, 0:1] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %250 = stablehlo.reshape %249 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %251 = stablehlo.broadcast_in_dim %124, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %252 = stablehlo.multiply %250, %251 : tensor<2x1x4x32xf32>
      %253 = stablehlo.slice %248 [0:2, 0:1, 0:4, 0:32, 1:2] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %254 = stablehlo.reshape %253 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %255 = stablehlo.broadcast_in_dim %129, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %256 = stablehlo.multiply %254, %255 : tensor<2x1x4x32xf32>
      %257 = stablehlo.subtract %252, %256 : tensor<2x1x4x32xf32>
      %258 = stablehlo.reshape %257 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %259 = stablehlo.multiply %250, %255 : tensor<2x1x4x32xf32>
      %260 = stablehlo.multiply %254, %251 : tensor<2x1x4x32xf32>
      %261 = stablehlo.add %259, %260 : tensor<2x1x4x32xf32>
      %262 = stablehlo.reshape %261 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %263 = stablehlo.concatenate %258, %262, dim = 4 : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
      %264 = stablehlo.reshape %263 : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
      %265 = stablehlo.convert %264 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
      %266 = stablehlo.slice %221 [0:2, 0:33, 0:64] : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
      %267 = stablehlo.broadcast_in_dim %234, dims = [0, 1] : (tensor<2x33xi64>) -> tensor<2x33x64xi64>
      %268 = stablehlo.iota dim = 0 {reoutline.comp_attrs = {dim = 1 : i64, sparse_grad = false}, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.orig_name = "tenstorrent.gather", reoutline.seed} : tensor<2xui32>
      %269 = stablehlo.broadcast_in_dim %268, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2xui32>) -> tensor<2x33x64x1xui32>
      %270 = stablehlo.convert %267 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64xi64>) -> tensor<2x33x64xui32>
      %271 = stablehlo.reshape %270 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64xui32>) -> tensor<2x33x64x1xui32>
      %272 = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<64xui32>
      %273 = stablehlo.broadcast_in_dim %272, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<64xui32>) -> tensor<2x33x64x1xui32>
      %274 = stablehlo.concatenate %269, %271, %273, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>) -> tensor<2x33x64x3xui32>
      %275 = "stablehlo.all_gather"(%266) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x33x64xbf16>) -> tensor<4x33x64xbf16>
      %276 = "stablehlo.gather"(%275, %274) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> {reoutline.arg_operand_indices = array<i64: 0, -1>, reoutline.group = "composite_tenstorrent.gather.impl", reoutline.result_pos = array<i64: 0>} : (tensor<4x33x64xbf16>, tensor<2x33x64x3xui32>) -> tensor<2x33x64xbf16>
      %277 = stablehlo.dot_general %265, %276, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
      %278 = stablehlo.add %245, %277 : tensor<2x1x4x33xbf16>
      %279 = stablehlo.multiply %278, %1 : tensor<2x1x4x33xbf16>
      %280 = stablehlo.reduce(%279 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %282 = stablehlo.subtract %279, %281 : tensor<2x1x4x33xbf16>
      %283 = stablehlo.exponential %282 : tensor<2x1x4x33xbf16>
      %284 = stablehlo.reduce(%283 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %285 = stablehlo.broadcast_in_dim %284, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %286 = stablehlo.divide %283, %285 : tensor<2x1x4x33xbf16>
      %287 = stablehlo.dot_general %286, %244, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
      %288 = stablehlo.slice %229 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %289 = stablehlo.dot_general %287, %288, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
      %290 = stablehlo.transpose %289, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
      %291 = stablehlo.reshape %290 : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
      %292 = stablehlo.reshape %arg31 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %293 = stablehlo.reshape %292 : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
      %294 = stablehlo.transpose %293, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
      %295 = "stablehlo.all_gather"(%291) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
      %296 = stablehlo.dot_general %295, %294, contracting_dims = [1] x [0] : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
      %297 = "stablehlo.all_reduce"(%296) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg38: tensor<bf16>, %arg39: tensor<bf16>):
        %299 = stablehlo.add %arg38, %arg39 : tensor<bf16>
        stablehlo.return %299 : tensor<bf16>
      }) : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
      %298 = stablehlo.reshape %297 : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
      sdy.return %196, %221, %87, %298 : tensor<2x64x512xbf16>, tensor<2x64x64xbf16>, tensor<2x64x128xbf16>, tensor<4x1x1024xbf16>
    } : (tensor<4x64x512xbf16>, tensor<576x2048xbf16>, tensor<4x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<4x64x64xbf16>, tensor<1x32x2xbf16>, tensor<4x64x128xbf16>, tensor<128x128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128x2048xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<64x2048xbf16>, tensor<8192x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>, tensor<3072x3072xbf16>) -> (tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<4x1x2048xbf16>) -> tensor<2x1x2048xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<4x1x2048xbf16>) -> tensor<2x2x1x2048xbf16>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<2x2x1x2048xbf16>) -> tensor<2x2x1x2048xbf16>
    %2 = stablehlo.slice %1 [0:1, 0:2, 0:1, 0:2048] : (tensor<2x2x1x2048xbf16>) -> tensor<1x2x1x2048xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x2x1x2048xbf16>) -> tensor<2x1x2048xbf16>
    return %3 : tensor<2x1x2048xbf16>
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


// ============================================================
// MLIR Module: shlo_compiler (ReplicateShardedData.6)
// Extracted from: replicated_input_32_4.log
// ============================================================

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
