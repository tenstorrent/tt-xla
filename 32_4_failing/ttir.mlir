module @SyncTensorsGraph.761 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.761 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @main(%arg0: tensor<4x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<4x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<4x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<4x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x64x512xbf16>) -> tensor<2x64x512xbf16>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<576x2048xbf16>) -> tensor<576x1024xbf16>
        %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x1x2048xbf16>) -> tensor<4x1x1024xbf16>
        %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16>) -> tensor<512xbf16>
        %4 = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<i1>) -> tensor<i1>
        %5 = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x64x64xbf16>) -> tensor<2x64x64xbf16>
        %6 = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x2xbf16>) -> tensor<1x32x2xbf16>
        %7 = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x64x128xbf16>) -> tensor<2x64x128xbf16>
        %8 = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %9 = "ttir.mesh_shard"(%arg9) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128xbf16>) -> tensor<128xbf16>
        %10 = "ttir.mesh_shard"(%arg10) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128xbf16>) -> tensor<128xbf16>
        %11 = "ttir.mesh_shard"(%arg11) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128x2048xbf16>) -> tensor<128x1024xbf16>
        %12 = "ttir.mesh_shard"(%arg12) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2048x2048xbf16>) -> tensor<1024x512xbf16>
        %13 = "ttir.mesh_shard"(%arg13) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x512xbf16>) -> tensor<1024x512xbf16>
        %14 = "ttir.mesh_shard"(%arg14) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64x2048xbf16>) -> tensor<16x1024xbf16>
        %15 = "ttir.mesh_shard"(%arg15) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<8192x3072xbf16>) -> tensor<2048x3072xbf16>
        %16 = "ttir.mesh_shard"(%arg16) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072x2048xbf16>) -> tensor<3072x1024xbf16>
        %17 = "ttir.mesh_shard"(%arg17) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072xbf16>) -> tensor<3072xbf16>
        %18 = "ttir.mesh_shard"(%arg18) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072x3072xbf16>) -> tensor<768x3072xbf16>
        %19 = "ttir.constant"() <{value = dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>}> : () -> tensor<64xi1>
        %20 = "ttir.constant"() <{value = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>}> : () -> tensor<64xi1>
        %21 = "ttir.constant"() <{value = dense<9.99999997E-7> : tensor<2x1x1xf32>}> : () -> tensor<2x1x1xf32>
        %22 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %23 = "ttir.constant"() <{value = dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>}> : () -> tensor<64xf32>
        %24 = "ttir.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
        %25 = "ttir.constant"() <{value = dense<3.25520843E-4> : tensor<2x1xf32>}> : () -> tensor<2x1xf32>
        %26 = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
        %27 = "ttir.constant"() <{value = dense<8.837890e-02> : tensor<bf16>}> : () -> tensor<bf16>
        %28 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %29 = "ttir.constant"() <{value = dense<0.001953125> : tensor<2x1xf32>}> : () -> tensor<2x1xf32>
        %30 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %31 = "ttir.constant"() <{value = dense<7.226560e-02> : tensor<bf16>}> : () -> tensor<bf16>
        %32 = "ttir.reshape"(%31) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %33 = "ttir.broadcast"(%32) <{broadcast_dimensions = array<i64: 2, 1, 4, 33>}> : (tensor<1x1x1x1xbf16>) -> tensor<2x1x4x33xbf16>
        %34 = "ttir.reshape"(%30) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1xbf16>
        %35 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 2, 64, 64>}> : (tensor<1x1x1xbf16>) -> tensor<2x64x64xbf16>
        %36 = "ttir.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %37 = "ttir.broadcast"(%36) <{broadcast_dimensions = array<i64: 2, 1, 512>}> : (tensor<1x1x1xf32>) -> tensor<2x1x512xf32>
        %38 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 2, 64, 512>}> : (tensor<1x1x1xbf16>) -> tensor<2x64x512xbf16>
        %39 = "ttir.reshape"(%27) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %40 = "ttir.broadcast"(%39) <{broadcast_dimensions = array<i64: 2, 1, 16, 1>}> : (tensor<1x1x1x1xbf16>) -> tensor<2x1x16x1xbf16>
        %41 = "ttir.reshape"(%26) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1xbf16>
        %42 = "ttir.broadcast"(%41) <{broadcast_dimensions = array<i64: 2, 1, 16>}> : (tensor<1x1x1xbf16>) -> tensor<2x1x16xbf16>
        %43 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 2, 33, 16>}> : (tensor<1x1x1xbf16>) -> tensor<2x33x16xbf16>
        %44 = "ttir.broadcast"(%36) <{broadcast_dimensions = array<i64: 2, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<2x1x3072xf32>
        %45 = "ttir.reshape"(%24) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %46 = "ttir.broadcast"(%45) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi64>) -> tensor<64xi64>
        %47 = "ttir.reshape"(%22) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %48 = "ttir.broadcast"(%47) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi64>) -> tensor<64xi64>
        %49 = "ttir.broadcast"(%34) <{broadcast_dimensions = array<i64: 2, 64, 128>}> : (tensor<1x1x1xbf16>) -> tensor<2x64x128xbf16>
        %50 = "ttir.reshape"(%4) <{shape = [1 : i32]}> : (tensor<i1>) -> tensor<1xi1>
        %51 = "ttir.broadcast"(%50) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi1>) -> tensor<64xi1>
        %52 = "ttir.logical_and"(%51, %19) : (tensor<64xi1>, tensor<64xi1>) -> tensor<64xi1>
        %53 = "ttir.logical_and"(%52, %20) : (tensor<64xi1>, tensor<64xi1>) -> tensor<64xi1>
        %54 = "ttir.reshape"(%53) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xi1>) -> tensor<1x64x1xi1>
        %55 = "ttir.broadcast"(%54) <{broadcast_dimensions = array<i64: 2, 1, 128>}> : (tensor<1x64x1xi1>) -> tensor<2x64x128xi1>
        %56 = "ttir.logical_not"(%54) : (tensor<1x64x1xi1>) -> tensor<1x64x1xi1>
        %57 = "ttir.reshape"(%56) <{shape = [64 : i32]}> : (tensor<1x64x1xi1>) -> tensor<64xi1>
        %58 = "ttir.reshape"(%57) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xi1>) -> tensor<1x64x1xi1>
        %59 = "ttir.broadcast"(%58) <{broadcast_dimensions = array<i64: 2, 1, 128>}> : (tensor<1x64x1xi1>) -> tensor<2x64x128xi1>
        %60 = "ttir.reshape"(%2) <{shape = [4 : i32, 1024 : i32]}> : (tensor<4x1x1024xbf16>) -> tensor<4x1024xbf16>
        %61 = "ttir.reshape"(%11) <{shape = [1 : i32, 128 : i32, 1024 : i32]}> : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
        %62 = "ttir.reshape"(%61) <{shape = [128 : i32, 1024 : i32]}> : (tensor<1x128x1024xbf16>) -> tensor<128x1024xbf16>
        %63 = "ttir.permute"(%62) <{permutation = array<i64: 1, 0>}> : (tensor<128x1024xbf16>) -> tensor<1024x128xbf16>
        %64 = "ttir.dot_general"(%60, %63) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x1024xbf16>, tensor<1024x128xbf16>) -> tensor<4x128xbf16>
        %65 = "ttir.reduce_scatter"(%64) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4x128xbf16>) -> tensor<2x128xbf16>
        %66 = "ttir.reshape"(%65) <{shape = [2 : i32, 1 : i32, 128 : i32]}> : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
        %67 = "ttir.reshape"(%10) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %68 = "ttir.reshape"(%67) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %69 = "ttir.reshape"(%9) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %70 = "ttir.reshape"(%69) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %71 = "ttir.layer_norm"(%66, %68, %70) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 128>, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<2x1x128xbf16>
        %72 = "ttir.slice_static"(%71) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
        %73 = "ttir.reshape"(%72) <{shape = [2 : i32, 1 : i32, 1 : i32, 2 : i32, 32 : i32]}> : (tensor<2x1x64xbf16>) -> tensor<2x1x1x2x32xbf16>
        %74 = "ttir.permute"(%73) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<2x1x1x2x32xbf16>) -> tensor<2x1x1x32x2xbf16>
        %75 = "ttir.typecast"(%74) <{conservative_folding = false}> : (tensor<2x1x1x32x2xbf16>) -> tensor<2x1x1x32x2xf32>
        %76 = "ttir.slice_static"(%75) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
        %77 = "ttir.reshape"(%76) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
        %78 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32]}> : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
        %79 = "ttir.slice_static"(%78) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
        %80 = "ttir.reshape"(%79) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
        %81 = "ttir.typecast"(%80) <{conservative_folding = false}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
        %82 = "ttir.reshape"(%81) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
        %83 = "ttir.reshape"(%82) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %84 = "ttir.broadcast"(%83) <{broadcast_dimensions = array<i64: 2, 1, 1, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %85 = "ttir.multiply"(%77, %84) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %86 = "ttir.slice_static"(%75) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
        %87 = "ttir.reshape"(%86) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
        %88 = "ttir.slice_static"(%78) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
        %89 = "ttir.reshape"(%88) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
        %90 = "ttir.typecast"(%89) <{conservative_folding = false}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
        %91 = "ttir.reshape"(%90) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
        %92 = "ttir.reshape"(%91) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %93 = "ttir.broadcast"(%92) <{broadcast_dimensions = array<i64: 2, 1, 1, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %94 = "ttir.multiply"(%87, %93) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %95 = "ttir.subtract"(%85, %94) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %96 = "ttir.reshape"(%95) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
        %97 = "ttir.multiply"(%77, %93) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %98 = "ttir.multiply"(%87, %84) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %99 = "ttir.add"(%97, %98) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %100 = "ttir.reshape"(%99) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
        %101 = "ttir.concat"(%96, %100) <{dim = 4 : si32}> : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
        %102 = "ttir.reshape"(%101) <{shape = [2 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
        %103 = "ttir.slice_static"(%102) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
        %104 = "ttir.slice_static"(%102) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32xf32>
        %105 = "ttir.concat"(%103, %104) <{dim = 3 : si32}> : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x64xf32>
        %106 = "ttir.typecast"(%105) <{conservative_folding = false}> : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
        %107 = "ttir.reshape"(%106) <{shape = [2 : i32, 1 : i32, 64 : i32]}> : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
        %108 = "ttir.slice_static"(%71) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [2 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x128xbf16>) -> tensor<2x1x64xbf16>
        %109 = "ttir.concat"(%107, %108) <{dim = 2 : si32}> : (tensor<2x1x64xbf16>, tensor<2x1x64xbf16>) -> tensor<2x1x128xbf16>
        %110 = "ttir.reshape"(%109) <{shape = [2 : i32, 128 : i32]}> : (tensor<2x1x128xbf16>) -> tensor<2x128xbf16>
        %111 = "ttir.reshape"(%8) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
        %112 = "ttir.reshape"(%111) <{shape = [128 : i32, 128 : i32]}> : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
        %113 = "ttir.permute"(%112) <{permutation = array<i64: 1, 0>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %114 = "ttir.dot_general"(%110, %113) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<2x128xbf16>, tensor<128x128xbf16>) -> tensor<2x128xbf16>
        %115 = "ttir.reshape"(%114) <{shape = [2 : i32, 1 : i32, 128 : i32]}> : (tensor<2x128xbf16>) -> tensor<2x1x128xbf16>
        %116 = "ttir.floor"(%23) : (tensor<64xf32>) -> tensor<64xf32>
        %117 = "ttir.typecast"(%116) <{conservative_folding = false}> : (tensor<64xf32>) -> tensor<64xi64>
        %118 = "ttir.clamp_tensor"(%117, %48, %48) : (tensor<64xi64>, tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %119 = "ttir.lt"(%118, %48) : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
        %120 = "ttir.add"(%118, %46) : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %121 = "ttir.where"(%119, %120, %118) : (tensor<64xi1>, tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %122 = "ttir.reshape"(%121) <{shape = [64 : i32, 1 : i32]}> : (tensor<64xi64>) -> tensor<64x1xi64>
        %123 = "ttir.permute"(%115) <{permutation = array<i64: 1, 0, 2>}> : (tensor<2x1x128xbf16>) -> tensor<1x2x128xbf16>
        %124 = "ttir.reshape"(%123) <{shape = [1 : i32, 256 : i32]}> : (tensor<1x2x128xbf16>) -> tensor<1x256xbf16>
        %125 = "ttir.embedding"(%122, %124) : (tensor<64x1xi64>, tensor<1x256xbf16>) -> tensor<64x1x256xbf16>
        %126 = "ttir.reshape"(%125) <{shape = [64 : i32, 2 : i32, 128 : i32]}> : (tensor<64x1x256xbf16>) -> tensor<64x2x128xbf16>
        %127 = "ttir.permute"(%126) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x2x128xbf16>) -> tensor<2x64x128xbf16>
        %128 = "ttir.where"(%59, %49, %127) : (tensor<2x64x128xi1>, tensor<2x64x128xbf16>, tensor<2x64x128xbf16>) -> tensor<2x64x128xbf16>
        %129 = "ttir.where"(%55, %128, %7) : (tensor<2x64x128xi1>, tensor<2x64x128xbf16>, tensor<2x64x128xbf16>) -> tensor<2x64x128xbf16>
        %130 = "ttir.slice_static"(%129) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 33 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x64x128xbf16>) -> tensor<2x33x128xbf16>
        %131 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %132 = "ttir.reshape"(%131) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
        %133 = "ttir.typecast"(%132) <{conservative_folding = false}> : (tensor<3072xbf16>) -> tensor<3072xf32>
        %134 = "ttir.reshape"(%133) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
        %135 = "ttir.broadcast"(%134) <{broadcast_dimensions = array<i64: 2, 1, 1>}> : (tensor<1x1x3072xf32>) -> tensor<2x1x3072xf32>
        %136 = "ttir.reshape"(%16) <{shape = [1 : i32, 3072 : i32, 1024 : i32]}> : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
        %137 = "ttir.reshape"(%136) <{shape = [3072 : i32, 1024 : i32]}> : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
        %138 = "ttir.permute"(%137) <{permutation = array<i64: 1, 0>}> : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
        %139 = "ttir.dot_general"(%60, %138) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
        %140 = "ttir.reduce_scatter"(%139) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
        %141 = "ttir.reshape"(%140) <{shape = [2 : i32, 1 : i32, 3072 : i32]}> : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
        %142 = "ttir.typecast"(%141) <{conservative_folding = false}> : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
        %143 = "ttir.pow"(%142, %44) : (tensor<2x1x3072xf32>, tensor<2x1x3072xf32>) -> tensor<2x1x3072xf32>
        %144 = "ttir.sum"(%143) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x1x3072xf32>) -> tensor<2x1xf32>
        %145 = "ttir.multiply"(%144, %25) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
        %146 = "ttir.reshape"(%145) <{shape = [2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
        %147 = "ttir.add"(%146, %21) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
        %148 = "ttir.rsqrt"(%147) : (tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
        %149 = "ttir.reshape"(%148) <{shape = [2 : i32, 1 : i32]}> : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
        %150 = "ttir.reshape"(%149) <{shape = [2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
        %151 = "ttir.broadcast"(%150) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<2x1x1xf32>) -> tensor<2x1x3072xf32>
        %152 = "ttir.multiply"(%142, %151) : (tensor<2x1x3072xf32>, tensor<2x1x3072xf32>) -> tensor<2x1x3072xf32>
        %153 = "ttir.multiply"(%135, %152) : (tensor<2x1x3072xf32>, tensor<2x1x3072xf32>) -> tensor<2x1x3072xf32>
        %154 = "ttir.typecast"(%153) <{conservative_folding = false}> : (tensor<2x1x3072xf32>) -> tensor<2x1x3072xbf16>
        %155 = "ttir.reshape"(%154) <{shape = [2 : i32, 3072 : i32]}> : (tensor<2x1x3072xbf16>) -> tensor<2x3072xbf16>
        %156 = "ttir.reshape"(%15) <{shape = [1 : i32, 2048 : i32, 3072 : i32]}> : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
        %157 = "ttir.reshape"(%156) <{shape = [2048 : i32, 3072 : i32]}> : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
        %158 = "ttir.permute"(%157) <{permutation = array<i64: 1, 0>}> : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
        %159 = "ttir.dot_general"(%155, %158) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<2x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<2x2048xbf16>
        %160 = "ttir.reshape"(%159) <{shape = [2 : i32, 1 : i32, 16 : i32, 128 : i32]}> : (tensor<2x2048xbf16>) -> tensor<2x1x16x128xbf16>
        %161 = "ttir.slice_static"(%160) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
        %162 = "ttir.reshape"(%161) <{shape = [2 : i32, 1 : i32, 16 : i32, 2 : i32, 32 : i32]}> : (tensor<2x1x16x64xbf16>) -> tensor<2x1x16x2x32xbf16>
        %163 = "ttir.permute"(%162) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<2x1x16x2x32xbf16>) -> tensor<2x1x16x32x2xbf16>
        %164 = "ttir.typecast"(%163) <{conservative_folding = false}> : (tensor<2x1x16x32x2xbf16>) -> tensor<2x1x16x32x2xf32>
        %165 = "ttir.slice_static"(%164) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
        %166 = "ttir.reshape"(%165) <{shape = [2 : i32, 1 : i32, 16 : i32, 32 : i32]}> : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
        %167 = "ttir.reshape"(%81) <{shape = [1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
        %168 = "ttir.reshape"(%167) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x32xf32>) -> tensor<1x1x1x32xf32>
        %169 = "ttir.broadcast"(%168) <{broadcast_dimensions = array<i64: 2, 1, 16, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x16x32xf32>
        %170 = "ttir.multiply"(%166, %169) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %171 = "ttir.slice_static"(%164) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x32x1xf32>
        %172 = "ttir.reshape"(%171) <{shape = [2 : i32, 1 : i32, 16 : i32, 32 : i32]}> : (tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32xf32>
        %173 = "ttir.reshape"(%90) <{shape = [1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
        %174 = "ttir.reshape"(%173) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x32xf32>) -> tensor<1x1x1x32xf32>
        %175 = "ttir.broadcast"(%174) <{broadcast_dimensions = array<i64: 2, 1, 16, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x16x32xf32>
        %176 = "ttir.multiply"(%172, %175) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %177 = "ttir.subtract"(%170, %176) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %178 = "ttir.reshape"(%177) <{shape = [2 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
        %179 = "ttir.multiply"(%166, %175) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %180 = "ttir.multiply"(%172, %169) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %181 = "ttir.add"(%179, %180) : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
        %182 = "ttir.reshape"(%181) <{shape = [2 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32x1xf32>
        %183 = "ttir.concat"(%178, %182) <{dim = 4 : si32}> : (tensor<2x1x16x32x1xf32>, tensor<2x1x16x32x1xf32>) -> tensor<2x1x16x32x2xf32>
        %184 = "ttir.reshape"(%183) <{shape = [2 : i32, 1 : i32, 16 : i32, 64 : i32]}> : (tensor<2x1x16x32x2xf32>) -> tensor<2x1x16x64xf32>
        %185 = "ttir.slice_static"(%184) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
        %186 = "ttir.slice_static"(%184) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x32xf32>
        %187 = "ttir.concat"(%185, %186) <{dim = 3 : si32}> : (tensor<2x1x16x32xf32>, tensor<2x1x16x32xf32>) -> tensor<2x1x16x64xf32>
        %188 = "ttir.typecast"(%187) <{conservative_folding = false}> : (tensor<2x1x16x64xf32>) -> tensor<2x1x16x64xbf16>
        %189 = "ttir.slice_static"(%160) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [2 : i32, 1 : i32, 16 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x16x128xbf16>) -> tensor<2x1x16x64xbf16>
        %190 = "ttir.concat"(%188, %189) <{dim = 3 : si32}> : (tensor<2x1x16x64xbf16>, tensor<2x1x16x64xbf16>) -> tensor<2x1x16x128xbf16>
        %191 = "ttir.dot_general"(%190, %113) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 0>}> : (tensor<2x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<2x1x16x128xbf16>
        %192 = "ttir.reshape"(%191) <{shape = [2 : i32, 16 : i32, 128 : i32]}> : (tensor<2x1x16x128xbf16>) -> tensor<2x16x128xbf16>
        %193 = "ttir.permute"(%192) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2x16x128xbf16>) -> tensor<2x128x16xbf16>
        %194 = "ttir.dot_general"(%130, %193) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<2x33x128xbf16>, tensor<2x128x16xbf16>) -> tensor<2x33x16xbf16>
        %195 = "ttir.maximum"(%194, %43) : (tensor<2x33x16xbf16>, tensor<2x33x16xbf16>) -> tensor<2x33x16xbf16>
        %196 = "ttir.reshape"(%14) <{shape = [1 : i32, 16 : i32, 1024 : i32]}> : (tensor<16x1024xbf16>) -> tensor<1x16x1024xbf16>
        %197 = "ttir.reshape"(%196) <{shape = [16 : i32, 1024 : i32]}> : (tensor<1x16x1024xbf16>) -> tensor<16x1024xbf16>
        %198 = "ttir.permute"(%197) <{permutation = array<i64: 1, 0>}> : (tensor<16x1024xbf16>) -> tensor<1024x16xbf16>
        %199 = "ttir.dot_general"(%60, %198) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x1024xbf16>, tensor<1024x16xbf16>) -> tensor<4x16xbf16>
        %200 = "ttir.reduce_scatter"(%199) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4x16xbf16>) -> tensor<2x16xbf16>
        %201 = "ttir.reshape"(%200) <{shape = [2 : i32, 1 : i32, 16 : i32]}> : (tensor<2x16xbf16>) -> tensor<2x1x16xbf16>
        %202 = "ttir.multiply"(%201, %42) : (tensor<2x1x16xbf16>, tensor<2x1x16xbf16>) -> tensor<2x1x16xbf16>
        %203 = "ttir.reshape"(%202) <{shape = [2 : i32, 1 : i32, 16 : i32, 1 : i32]}> : (tensor<2x1x16xbf16>) -> tensor<2x1x16x1xbf16>
        %204 = "ttir.multiply"(%203, %40) : (tensor<2x1x16x1xbf16>, tensor<2x1x16x1xbf16>) -> tensor<2x1x16x1xbf16>
        %205 = "ttir.reshape"(%204) <{shape = [2 : i32, 16 : i32]}> : (tensor<2x1x16x1xbf16>) -> tensor<2x16xbf16>
        %206 = "ttir.reshape"(%205) <{shape = [2 : i32, 1 : i32, 16 : i32]}> : (tensor<2x16xbf16>) -> tensor<2x1x16xbf16>
        %207 = "ttir.broadcast"(%206) <{broadcast_dimensions = array<i64: 1, 33, 1>}> : (tensor<2x1x16xbf16>) -> tensor<2x33x16xbf16>
        %208 = "ttir.multiply"(%195, %207) : (tensor<2x33x16xbf16>, tensor<2x33x16xbf16>) -> tensor<2x33x16xbf16>
        %209 = "ttir.sum"(%208) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x33x16xbf16>) -> tensor<2x33xbf16>
        %210 = "ttir.all_reduce"(%209) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<2x33xbf16>) -> tensor<2x33xbf16>
        %211 = "ttir.reshape"(%210) <{shape = [2 : i32, 1 : i32, 33 : i32]}> : (tensor<2x33xbf16>) -> tensor<2x1x33xbf16>
        %values, %indices = "ttir.topk"(%211) <{dim = -1 : i32, k = 33 : i32, largest = true, sorted = true}> : (tensor<2x1x33xbf16>) -> (tensor<2x1x33xbf16>, tensor<2x1x33xi64>)
        %212 = "ttir.broadcast"(%54) <{broadcast_dimensions = array<i64: 2, 1, 512>}> : (tensor<1x64x1xi1>) -> tensor<2x64x512xi1>
        %213 = "ttir.broadcast"(%58) <{broadcast_dimensions = array<i64: 2, 1, 512>}> : (tensor<1x64x1xi1>) -> tensor<2x64x512xi1>
        %214 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
        %215 = "ttir.reshape"(%214) <{shape = [512 : i32]}> : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
        %216 = "ttir.typecast"(%215) <{conservative_folding = false}> : (tensor<512xbf16>) -> tensor<512xf32>
        %217 = "ttir.reshape"(%216) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
        %218 = "ttir.broadcast"(%217) <{broadcast_dimensions = array<i64: 2, 1, 1>}> : (tensor<1x1x512xf32>) -> tensor<2x1x512xf32>
        %219 = "ttir.reshape"(%1) <{shape = [1 : i32, 576 : i32, 1024 : i32]}> : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
        %220 = "ttir.reshape"(%219) <{shape = [576 : i32, 1024 : i32]}> : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
        %221 = "ttir.permute"(%220) <{permutation = array<i64: 1, 0>}> : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
        %222 = "ttir.dot_general"(%60, %221) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
        %223 = "ttir.reduce_scatter"(%222) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
        %224 = "ttir.reshape"(%223) <{shape = [2 : i32, 1 : i32, 576 : i32]}> : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
        %225 = "ttir.slice_static"(%224) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
        %226 = "ttir.typecast"(%225) <{conservative_folding = false}> : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
        %227 = "ttir.pow"(%226, %37) : (tensor<2x1x512xf32>, tensor<2x1x512xf32>) -> tensor<2x1x512xf32>
        %228 = "ttir.sum"(%227) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x1x512xf32>) -> tensor<2x1xf32>
        %229 = "ttir.multiply"(%228, %29) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
        %230 = "ttir.reshape"(%229) <{shape = [2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
        %231 = "ttir.add"(%230, %21) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
        %232 = "ttir.rsqrt"(%231) : (tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
        %233 = "ttir.reshape"(%232) <{shape = [2 : i32, 1 : i32]}> : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
        %234 = "ttir.reshape"(%233) <{shape = [2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
        %235 = "ttir.broadcast"(%234) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<2x1x1xf32>) -> tensor<2x1x512xf32>
        %236 = "ttir.multiply"(%226, %235) : (tensor<2x1x512xf32>, tensor<2x1x512xf32>) -> tensor<2x1x512xf32>
        %237 = "ttir.multiply"(%218, %236) : (tensor<2x1x512xf32>, tensor<2x1x512xf32>) -> tensor<2x1x512xf32>
        %238 = "ttir.typecast"(%237) <{conservative_folding = false}> : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
        %239 = "ttir.permute"(%238) <{permutation = array<i64: 1, 0, 2>}> : (tensor<2x1x512xbf16>) -> tensor<1x2x512xbf16>
        %240 = "ttir.reshape"(%239) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1x2x512xbf16>) -> tensor<1x1024xbf16>
        %241 = "ttir.embedding"(%122, %240) : (tensor<64x1xi64>, tensor<1x1024xbf16>) -> tensor<64x1x1024xbf16>
        %242 = "ttir.reshape"(%241) <{shape = [64 : i32, 2 : i32, 512 : i32]}> : (tensor<64x1x1024xbf16>) -> tensor<64x2x512xbf16>
        %243 = "ttir.permute"(%242) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x2x512xbf16>) -> tensor<2x64x512xbf16>
        %244 = "ttir.where"(%213, %38, %243) : (tensor<2x64x512xi1>, tensor<2x64x512xbf16>, tensor<2x64x512xbf16>) -> tensor<2x64x512xbf16>
        %245 = "ttir.where"(%212, %244, %0) : (tensor<2x64x512xi1>, tensor<2x64x512xbf16>, tensor<2x64x512xbf16>) -> tensor<2x64x512xbf16>
        %246 = "ttir.broadcast"(%54) <{broadcast_dimensions = array<i64: 2, 1, 64>}> : (tensor<1x64x1xi1>) -> tensor<2x64x64xi1>
        %247 = "ttir.broadcast"(%58) <{broadcast_dimensions = array<i64: 2, 1, 64>}> : (tensor<1x64x1xi1>) -> tensor<2x64x64xi1>
        %248 = "ttir.slice_static"(%224) <{begins = [0 : i32, 0 : i32, 512 : i32], ends = [2 : i32, 1 : i32, 576 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
        %249 = "ttir.reshape"(%248) <{shape = [2 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
        %250 = "ttir.typecast"(%249) <{conservative_folding = false}> : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
        %251 = "ttir.reshape"(%250) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32]}> : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
        %252 = "ttir.slice_static"(%251) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
        %253 = "ttir.reshape"(%252) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
        %254 = "ttir.multiply"(%253, %84) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %255 = "ttir.slice_static"(%251) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
        %256 = "ttir.reshape"(%255) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
        %257 = "ttir.multiply"(%256, %93) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %258 = "ttir.subtract"(%254, %257) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %259 = "ttir.reshape"(%258) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
        %260 = "ttir.multiply"(%253, %93) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %261 = "ttir.multiply"(%256, %84) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %262 = "ttir.add"(%260, %261) : (tensor<2x1x1x32xf32>, tensor<2x1x1x32xf32>) -> tensor<2x1x1x32xf32>
        %263 = "ttir.reshape"(%262) <{shape = [2 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
        %264 = "ttir.concat"(%259, %263) <{dim = 4 : si32}> : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
        %265 = "ttir.reshape"(%264) <{shape = [2 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
        %266 = "ttir.typecast"(%265) <{conservative_folding = false}> : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
        %267 = "ttir.reshape"(%266) <{shape = [2 : i32, 1 : i32, 64 : i32]}> : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
        %268 = "ttir.permute"(%267) <{permutation = array<i64: 1, 0, 2>}> : (tensor<2x1x64xbf16>) -> tensor<1x2x64xbf16>
        %269 = "ttir.reshape"(%268) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x2x64xbf16>) -> tensor<1x128xbf16>
        %270 = "ttir.embedding"(%122, %269) : (tensor<64x1xi64>, tensor<1x128xbf16>) -> tensor<64x1x128xbf16>
        %271 = "ttir.reshape"(%270) <{shape = [64 : i32, 2 : i32, 64 : i32]}> : (tensor<64x1x128xbf16>) -> tensor<64x2x64xbf16>
        %272 = "ttir.permute"(%271) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x2x64xbf16>) -> tensor<2x64x64xbf16>
        %273 = "ttir.where"(%247, %35, %272) : (tensor<2x64x64xi1>, tensor<2x64x64xbf16>, tensor<2x64x64xbf16>) -> tensor<2x64x64xbf16>
        %274 = "ttir.where"(%246, %273, %5) : (tensor<2x64x64xi1>, tensor<2x64x64xbf16>, tensor<2x64x64xbf16>) -> tensor<2x64x64xbf16>
        %275 = "ttir.reshape"(%18) <{shape = [1 : i32, 768 : i32, 3072 : i32]}> : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
        %276 = "ttir.reshape"(%275) <{shape = [768 : i32, 3072 : i32]}> : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
        %277 = "ttir.permute"(%276) <{permutation = array<i64: 1, 0>}> : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
        %278 = "ttir.dot_general"(%155, %277) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
        %279 = "ttir.reshape"(%278) <{shape = [2 : i32, 1 : i32, 4 : i32, 192 : i32]}> : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
        %280 = "ttir.slice_static"(%279) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
        %281 = "ttir.reshape"(%13) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
        %282 = "ttir.reshape"(%281) <{shape = [4 : i32, 256 : i32, 512 : i32]}> : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
        %283 = "ttir.slice_static"(%282) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 128 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
        %284 = "ttir.dot_general"(%280, %283) <{batch_dims_lhs = array<i64: 2>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 1>}> : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
        %285 = "ttir.permute"(%284) <{permutation = array<i64: 1, 2, 0, 3>}> : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
        %286 = "ttir.slice_static"(%245) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 33 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
        %287 = "ttir.reshape"(%indices) <{shape = [2 : i32, 33 : i32]}> : (tensor<2x1x33xi64>) -> tensor<2x33xi64>
        %288 = "ttir.reshape"(%287) <{shape = [2 : i32, 33 : i32, 1 : i32]}> : (tensor<2x33xi64>) -> tensor<2x33x1xi64>
        %289 = "ttir.broadcast"(%288) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<2x33x1xi64>) -> tensor<2x33x512xi64>
        %290 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 2 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<2xui32>
        %291 = "ttir.reshape"(%290) <{shape = [2 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2xui32>) -> tensor<2x1x1x1xui32>
        %292 = "ttir.broadcast"(%291) <{broadcast_dimensions = array<i64: 1, 33, 512, 1>}> : (tensor<2x1x1x1xui32>) -> tensor<2x33x512x1xui32>
        %293 = "ttir.typecast"(%289) <{conservative_folding = false}> : (tensor<2x33x512xi64>) -> tensor<2x33x512xui32>
        %294 = "ttir.reshape"(%293) <{shape = [2 : i32, 33 : i32, 512 : i32, 1 : i32]}> : (tensor<2x33x512xui32>) -> tensor<2x33x512x1xui32>
        %295 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xui32>
        %296 = "ttir.reshape"(%295) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xui32>) -> tensor<1x1x512x1xui32>
        %297 = "ttir.broadcast"(%296) <{broadcast_dimensions = array<i64: 2, 33, 1, 1>}> : (tensor<1x1x512x1xui32>) -> tensor<2x33x512x1xui32>
        %298 = "ttir.concat"(%292, %294, %297) <{dim = 3 : si32}> : (tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>, tensor<2x33x512x1xui32>) -> tensor<2x33x512x3xui32>
        %299 = "ttir.all_gather"(%286) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<2x33x512xbf16>) -> tensor<4x33x512xbf16>
        %300 = "ttir.permute"(%299) <{permutation = array<i64: 0, 1, 2>}> : (tensor<4x33x512xbf16>) -> tensor<4x33x512xbf16>
        %301 = "ttir.reshape"(%300) <{shape = [67584 : i32, 1 : i32]}> : (tensor<4x33x512xbf16>) -> tensor<67584x1xbf16>
        %302 = "ttir.permute"(%298) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<2x33x512x3xui32>) -> tensor<2x33x512x3xui32>
        %303 = "ttir.typecast"(%302) <{conservative_folding = false}> : (tensor<2x33x512x3xui32>) -> tensor<2x33x512x3xf32>
        %304 = "ttir.constant"() <{value = dense<[[1.689600e+04], [5.120000e+02], [1.000000e+00]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
        %305 = "ttir.matmul"(%303, %304) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x512x3xf32>, tensor<3x1xf32>) -> tensor<2x33x512x1xf32>
        %306 = "ttir.reshape"(%305) <{shape = [1 : i32, 33792 : i32]}> : (tensor<2x33x512x1xf32>) -> tensor<1x33792xf32>
        %307 = "ttir.embedding"(%306, %301) : (tensor<1x33792xf32>, tensor<67584x1xbf16>) -> tensor<1x33792x1xbf16>
        %308 = "ttir.reshape"(%307) <{shape = [2 : i32, 33 : i32, 512 : i32]}> : (tensor<1x33792x1xbf16>) -> tensor<2x33x512xbf16>
        %309 = "ttir.permute"(%308) <{permutation = array<i64: 0, 1, 2>}> : (tensor<2x33x512xbf16>) -> tensor<2x33x512xbf16>
        %310 = "ttir.dot_general"(%285, %309) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
        %311 = "ttir.slice_static"(%279) <{begins = [0 : i32, 0 : i32, 0 : i32, 128 : i32], ends = [2 : i32, 1 : i32, 4 : i32, 192 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
        %312 = "ttir.typecast"(%311) <{conservative_folding = false}> : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
        %313 = "ttir.reshape"(%312) <{shape = [2 : i32, 1 : i32, 4 : i32, 32 : i32, 2 : i32]}> : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
        %314 = "ttir.slice_static"(%313) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
        %315 = "ttir.reshape"(%314) <{shape = [2 : i32, 1 : i32, 4 : i32, 32 : i32]}> : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
        %316 = "ttir.broadcast"(%168) <{broadcast_dimensions = array<i64: 2, 1, 4, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x4x32xf32>
        %317 = "ttir.multiply"(%315, %316) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %318 = "ttir.slice_static"(%313) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 1 : i32, 4 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
        %319 = "ttir.reshape"(%318) <{shape = [2 : i32, 1 : i32, 4 : i32, 32 : i32]}> : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
        %320 = "ttir.broadcast"(%174) <{broadcast_dimensions = array<i64: 2, 1, 4, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<2x1x4x32xf32>
        %321 = "ttir.multiply"(%319, %320) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %322 = "ttir.subtract"(%317, %321) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %323 = "ttir.reshape"(%322) <{shape = [2 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
        %324 = "ttir.multiply"(%315, %320) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %325 = "ttir.multiply"(%319, %316) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %326 = "ttir.add"(%324, %325) : (tensor<2x1x4x32xf32>, tensor<2x1x4x32xf32>) -> tensor<2x1x4x32xf32>
        %327 = "ttir.reshape"(%326) <{shape = [2 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32]}> : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
        %328 = "ttir.concat"(%323, %327) <{dim = 4 : si32}> : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
        %329 = "ttir.reshape"(%328) <{shape = [2 : i32, 1 : i32, 4 : i32, 64 : i32]}> : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
        %330 = "ttir.typecast"(%329) <{conservative_folding = false}> : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
        %331 = "ttir.slice_static"(%274) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 33 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
        %332 = "ttir.broadcast"(%288) <{broadcast_dimensions = array<i64: 1, 1, 64>}> : (tensor<2x33x1xi64>) -> tensor<2x33x64xi64>
        %333 = "ttir.broadcast"(%291) <{broadcast_dimensions = array<i64: 1, 33, 64, 1>}> : (tensor<2x1x1x1xui32>) -> tensor<2x33x64x1xui32>
        %334 = "ttir.typecast"(%332) <{conservative_folding = false}> : (tensor<2x33x64xi64>) -> tensor<2x33x64xui32>
        %335 = "ttir.reshape"(%334) <{shape = [2 : i32, 33 : i32, 64 : i32, 1 : i32]}> : (tensor<2x33x64xui32>) -> tensor<2x33x64x1xui32>
        %336 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 64 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<64xui32>
        %337 = "ttir.reshape"(%336) <{shape = [1 : i32, 1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xui32>) -> tensor<1x1x64x1xui32>
        %338 = "ttir.broadcast"(%337) <{broadcast_dimensions = array<i64: 2, 33, 1, 1>}> : (tensor<1x1x64x1xui32>) -> tensor<2x33x64x1xui32>
        %339 = "ttir.concat"(%333, %335, %338) <{dim = 3 : si32}> : (tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>, tensor<2x33x64x1xui32>) -> tensor<2x33x64x3xui32>
        %340 = "ttir.all_gather"(%331) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<2x33x64xbf16>) -> tensor<4x33x64xbf16>
        %341 = "ttir.permute"(%340) <{permutation = array<i64: 0, 1, 2>}> : (tensor<4x33x64xbf16>) -> tensor<4x33x64xbf16>
        %342 = "ttir.reshape"(%341) <{shape = [8448 : i32, 1 : i32]}> : (tensor<4x33x64xbf16>) -> tensor<8448x1xbf16>
        %343 = "ttir.permute"(%339) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<2x33x64x3xui32>) -> tensor<2x33x64x3xui32>
        %344 = "ttir.typecast"(%343) <{conservative_folding = false}> : (tensor<2x33x64x3xui32>) -> tensor<2x33x64x3xf32>
        %345 = "ttir.constant"() <{value = dense<[[2.112000e+03], [6.400000e+01], [1.000000e+00]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
        %346 = "ttir.matmul"(%344, %345) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x64x3xf32>, tensor<3x1xf32>) -> tensor<2x33x64x1xf32>
        %347 = "ttir.reshape"(%346) <{shape = [1 : i32, 4224 : i32]}> : (tensor<2x33x64x1xf32>) -> tensor<1x4224xf32>
        %348 = "ttir.embedding"(%347, %342) : (tensor<1x4224xf32>, tensor<8448x1xbf16>) -> tensor<1x4224x1xbf16>
        %349 = "ttir.reshape"(%348) <{shape = [2 : i32, 33 : i32, 64 : i32]}> : (tensor<1x4224x1xbf16>) -> tensor<2x33x64xbf16>
        %350 = "ttir.permute"(%349) <{permutation = array<i64: 0, 1, 2>}> : (tensor<2x33x64xbf16>) -> tensor<2x33x64xbf16>
        %351 = "ttir.dot_general"(%330, %350) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
        %352 = "ttir.add"(%310, %351) : (tensor<2x1x4x33xbf16>, tensor<2x1x4x33xbf16>) -> tensor<2x1x4x33xbf16>
        %353 = "ttir.multiply"(%352, %33) : (tensor<2x1x4x33xbf16>, tensor<2x1x4x33xbf16>) -> tensor<2x1x4x33xbf16>
        %354 = "ttir.max"(%353) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x1x4x33xbf16>) -> tensor<2x1x4xbf16>
        %355 = "ttir.reshape"(%354) <{shape = [2 : i32, 1 : i32, 4 : i32, 1 : i32]}> : (tensor<2x1x4xbf16>) -> tensor<2x1x4x1xbf16>
        %356 = "ttir.broadcast"(%355) <{broadcast_dimensions = array<i64: 1, 1, 1, 33>}> : (tensor<2x1x4x1xbf16>) -> tensor<2x1x4x33xbf16>
        %357 = "ttir.subtract"(%353, %356) : (tensor<2x1x4x33xbf16>, tensor<2x1x4x33xbf16>) -> tensor<2x1x4x33xbf16>
        %358 = "ttir.exp"(%357) : (tensor<2x1x4x33xbf16>) -> tensor<2x1x4x33xbf16>
        %359 = "ttir.sum"(%358) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x1x4x33xbf16>) -> tensor<2x1x4xbf16>
        %360 = "ttir.reshape"(%359) <{shape = [2 : i32, 1 : i32, 4 : i32, 1 : i32]}> : (tensor<2x1x4xbf16>) -> tensor<2x1x4x1xbf16>
        %361 = "ttir.broadcast"(%360) <{broadcast_dimensions = array<i64: 1, 1, 1, 33>}> : (tensor<2x1x4x1xbf16>) -> tensor<2x1x4x33xbf16>
        %362 = "ttir.div"(%358, %361) : (tensor<2x1x4x33xbf16>, tensor<2x1x4x33xbf16>) -> tensor<2x1x4x33xbf16>
        %363 = "ttir.dot_general"(%362, %309) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 1>}> : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
        %364 = "ttir.slice_static"(%282) <{begins = [0 : i32, 128 : i32, 0 : i32], ends = [4 : i32, 256 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
        %365 = "ttir.dot_general"(%363, %364) <{batch_dims_lhs = array<i64: 2>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
        %366 = "ttir.permute"(%365) <{permutation = array<i64: 1, 2, 0, 3>}> : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
        %367 = "ttir.reshape"(%366) <{shape = [2 : i32, 512 : i32]}> : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
        %368 = "ttir.reshape"(%12) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
        %369 = "ttir.reshape"(%368) <{shape = [1024 : i32, 512 : i32]}> : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
        %370 = "ttir.permute"(%369) <{permutation = array<i64: 1, 0>}> : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
        %371 = "ttir.all_gather"(%367) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
        %372 = "ttir.dot_general"(%371, %370) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
        %373 = "ttir.all_reduce"(%372) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
        %374 = "ttir.reshape"(%373) <{shape = [4 : i32, 1 : i32, 1024 : i32]}> : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
        %375 = "ttir.mesh_shard"(%245) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x64x512xbf16>) -> tensor<4x64x512xbf16>
        %376 = "ttir.mesh_shard"(%274) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x64x64xbf16>) -> tensor<4x64x64xbf16>
        %377 = "ttir.mesh_shard"(%129) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x64x128xbf16>) -> tensor<4x64x128xbf16>
        %378 = "ttir.mesh_shard"(%374) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x1x1024xbf16>) -> tensor<4x1x2048xbf16>
        return %375, %376, %377, %378 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x64x128xbf16>, tensor<4x1x2048xbf16>
      }
    }
  }
}




module @ReplicateShardedData.6 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module @ReplicateShardedData.6 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @main(%arg0: tensor<4x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<4x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x1x2048xbf16>) -> tensor<4x1x1024xbf16>
        %1 = "ttir.all_gather"(%0) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<4x1x1024xbf16>) -> tensor<4x1x2048xbf16>
        %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4x1x2048xbf16>) -> tensor<4x1x2048xbf16>
        return %2 : tensor<4x1x2048xbf16>
      }
    }
  }
}
