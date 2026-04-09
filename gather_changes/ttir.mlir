module @SyncTensorsGraph.739 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.739 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @main(%arg0: tensor<1x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_cache"}, %arg1: tensor<576x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_a.weight"}, %arg2: tensor<1x1x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg3: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "kv_norm.weight"}, %arg4: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg5: tensor<1x64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "pe_cache"}, %arg6: tensor<1x32x2xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg7: tensor<1x64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_cache"}, %arg8: tensor<128x128xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.haddamard"}, %arg9: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.bias"}, %arg10: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.k_norm.weight"}, %arg11: tensor<128x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wk.weight"}, %arg12: tensor<2048x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wo.weight"}, %arg13: tensor<4096x512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wkv_b.weight"}, %arg14: tensor<64x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<16x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.weights_proj.weight"}, %arg15: tensor<8192x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "indexer.wq_b.weight"}, %arg16: tensor<3072x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_a.weight"}, %arg17: tensor<3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "q_norm.weight"}, %arg18: tensor<3072x3072xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "wq_b.weight"}) -> (tensor<1x64x512xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x64x128xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x64x128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x1x2048xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x512xbf16>) -> tensor<1x64x512xbf16>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<576x2048xbf16>) -> tensor<576x2048xbf16>
        %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
        %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16>) -> tensor<512xbf16>
        %4 = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<i1>) -> tensor<i1>
        %5 = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
        %6 = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x2xbf16>) -> tensor<1x32x2xbf16>
        %7 = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x128xbf16>) -> tensor<1x64x128xbf16>
        %8 = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %9 = "ttir.mesh_shard"(%arg9) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128xbf16>) -> tensor<128xbf16>
        %10 = "ttir.mesh_shard"(%arg10) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128xbf16>) -> tensor<128xbf16>
        %11 = "ttir.mesh_shard"(%arg11) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128x2048xbf16>) -> tensor<128x2048xbf16>
        %12 = "ttir.mesh_shard"(%arg12) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2048x2048xbf16>) -> tensor<2048x512xbf16>
        %13 = "ttir.mesh_shard"(%arg13) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x512xbf16>) -> tensor<1024x512xbf16>
        %14 = "ttir.mesh_shard"(%arg14) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64x2048xbf16>) -> tensor<16x2048xbf16>
        %15 = "ttir.mesh_shard"(%arg15) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<8192x3072xbf16>) -> tensor<2048x3072xbf16>
        %16 = "ttir.mesh_shard"(%arg16) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072x2048xbf16>) -> tensor<3072x2048xbf16>
        %17 = "ttir.mesh_shard"(%arg17) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072xbf16>) -> tensor<3072xbf16>
        %18 = "ttir.mesh_shard"(%arg18) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<3072x3072xbf16>) -> tensor<768x3072xbf16>
        %19 = "ttir.constant"() <{value = dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>}> : () -> tensor<64xi1>
        %20 = "ttir.constant"() <{value = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>}> : () -> tensor<64xi1>
        %21 = "ttir.constant"() <{value = dense<9.99999997E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
        %22 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %23 = "ttir.constant"() <{value = dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>}> : () -> tensor<64xf32>
        %24 = "ttir.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
        %25 = "ttir.constant"() <{value = dense<3.25520843E-4> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %26 = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
        %27 = "ttir.constant"() <{value = dense<8.837890e-02> : tensor<bf16>}> : () -> tensor<bf16>
        %28 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %29 = "ttir.constant"() <{value = dense<0.001953125> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
        %30 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %31 = "ttir.constant"() <{value = dense<0> : tensor<ui32>}> : () -> tensor<ui32>
        %32 = "ttir.constant"() <{value = dense<7.226560e-02> : tensor<bf16>}> : () -> tensor<bf16>
        %33 = "ttir.reshape"(%32) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %34 = "ttir.broadcast"(%33) <{broadcast_dimensions = array<i64: 1, 1, 4, 33>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x1x4x33xbf16>
        %35 = "ttir.reshape"(%31) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<ui32>) -> tensor<1x1x1x1xui32>
        %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 33, 64, 1>}> : (tensor<1x1x1x1xui32>) -> tensor<1x33x64x1xui32>
        %37 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 33, 512, 1>}> : (tensor<1x1x1x1xui32>) -> tensor<1x33x512x1xui32>
        %38 = "ttir.reshape"(%30) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1xbf16>
        %39 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 1, 64, 64>}> : (tensor<1x1x1xbf16>) -> tensor<1x64x64xbf16>
        %40 = "ttir.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %41 = "ttir.broadcast"(%40) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x1x512xf32>
        %42 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 1, 64, 512>}> : (tensor<1x1x1xbf16>) -> tensor<1x64x512xbf16>
        %43 = "ttir.reshape"(%27) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %44 = "ttir.broadcast"(%43) <{broadcast_dimensions = array<i64: 1, 1, 16, 1>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x1x16x1xbf16>
        %45 = "ttir.reshape"(%26) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1xbf16>
        %46 = "ttir.broadcast"(%45) <{broadcast_dimensions = array<i64: 1, 1, 16>}> : (tensor<1x1x1xbf16>) -> tensor<1x1x16xbf16>
        %47 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 1, 33, 16>}> : (tensor<1x1x1xbf16>) -> tensor<1x33x16xbf16>
        %48 = "ttir.broadcast"(%40) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %49 = "ttir.reshape"(%24) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %50 = "ttir.broadcast"(%49) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi64>) -> tensor<64xi64>
        %51 = "ttir.reshape"(%22) <{shape = [1 : i32]}> : (tensor<i64>) -> tensor<1xi64>
        %52 = "ttir.broadcast"(%51) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi64>) -> tensor<64xi64>
        %53 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<1x1x1xbf16>) -> tensor<1x64x128xbf16>
        %54 = "ttir.reshape"(%4) <{shape = [1 : i32]}> : (tensor<i1>) -> tensor<1xi1>
        %55 = "ttir.broadcast"(%54) <{broadcast_dimensions = array<i64: 64>}> : (tensor<1xi1>) -> tensor<64xi1>
        %56 = "ttir.logical_and"(%55, %19) : (tensor<64xi1>, tensor<64xi1>) -> tensor<64xi1>
        %57 = "ttir.logical_and"(%56, %20) : (tensor<64xi1>, tensor<64xi1>) -> tensor<64xi1>
        %58 = "ttir.reshape"(%57) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xi1>) -> tensor<1x64x1xi1>
        %59 = "ttir.reshape"(%57) <{shape = [1 : i32, 64 : i32]}> : (tensor<64xi1>) -> tensor<1x64xi1>
        %60 = "ttir.reshape"(%59) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<1x64xi1>) -> tensor<1x64x1xi1>
        %61 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 1, 1, 128>}> : (tensor<1x64x1xi1>) -> tensor<1x64x128xi1>
        %62 = "ttir.logical_not"(%58) : (tensor<1x64x1xi1>) -> tensor<1x64x1xi1>
        %63 = "ttir.reshape"(%62) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x64x1xi1>) -> tensor<1x64xi1>
        %64 = "ttir.reshape"(%63) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<1x64xi1>) -> tensor<1x64x1xi1>
        %65 = "ttir.broadcast"(%64) <{broadcast_dimensions = array<i64: 1, 1, 128>}> : (tensor<1x64x1xi1>) -> tensor<1x64x128xi1>
        %66 = "ttir.reshape"(%2) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<1x2048xbf16>
        %67 = "ttir.reshape"(%11) <{shape = [1 : i32, 128 : i32, 2048 : i32]}> : (tensor<128x2048xbf16>) -> tensor<1x128x2048xbf16>
        %68 = "ttir.reshape"(%67) <{shape = [128 : i32, 2048 : i32]}> : (tensor<1x128x2048xbf16>) -> tensor<128x2048xbf16>
        %69 = "ttir.permute"(%68) <{permutation = array<i64: 1, 0>}> : (tensor<128x2048xbf16>) -> tensor<2048x128xbf16>
        %70 = "ttir.dot_general"(%66, %69) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x128xbf16>) -> tensor<1x128xbf16>
        %71 = "ttir.reshape"(%70) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
        %72 = "ttir.reshape"(%10) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %73 = "ttir.reshape"(%72) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %74 = "ttir.reshape"(%9) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %75 = "ttir.reshape"(%74) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %76 = "ttir.layer_norm"(%71, %73, %75) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 128>, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<1x1x128xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %77 = "ttir.slice_static"(%76) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
        %78 = "ttir.reshape"(%77) <{shape = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 32 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x1x1x2x32xbf16>
        %79 = "ttir.permute"(%78) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<1x1x1x2x32xbf16>) -> tensor<1x1x1x32x2xbf16>
        %80 = "ttir.typecast"(%79) <{conservative_folding = false}> : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x2xf32>
        %81 = "ttir.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
        %82 = "ttir.reshape"(%81) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
        %83 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32]}> : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
        %84 = "ttir.slice_static"(%83) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
        %85 = "ttir.reshape"(%84) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
        %86 = "ttir.typecast"(%85) <{conservative_folding = false}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
        %87 = "ttir.multiply"(%82, %86) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %88 = "ttir.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
        %89 = "ttir.reshape"(%88) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
        %90 = "ttir.slice_static"(%83) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
        %91 = "ttir.reshape"(%90) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
        %92 = "ttir.typecast"(%91) <{conservative_folding = false}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
        %93 = "ttir.multiply"(%89, %92) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %94 = "ttir.subtract"(%87, %93) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %95 = "ttir.reshape"(%94) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
        %96 = "ttir.multiply"(%82, %92) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %97 = "ttir.multiply"(%89, %86) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %98 = "ttir.add"(%96, %97) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %99 = "ttir.reshape"(%98) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
        %100 = "ttir.concat"(%95, %99) <{dim = 4 : si32}> : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
        %101 = "ttir.reshape"(%100) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
        %102 = "ttir.slice_static"(%101) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
        %103 = "ttir.slice_static"(%101) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32xf32>
        %104 = "ttir.concat"(%102, %103) <{dim = 3 : si32}> : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x64xf32>
        %105 = "ttir.typecast"(%104) <{conservative_folding = false}> : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
        %106 = "ttir.reshape"(%105) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
        %107 = "ttir.slice_static"(%76) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x1x64xbf16>
        %108 = "ttir.concat"(%106, %107) <{dim = 2 : si32}> : (tensor<1x1x64xbf16>, tensor<1x1x64xbf16>) -> tensor<1x1x128xbf16>
        %109 = "ttir.reshape"(%108) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x128xbf16>
        %110 = "ttir.reshape"(%8) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
        %111 = "ttir.reshape"(%110) <{shape = [128 : i32, 128 : i32]}> : (tensor<1x128x128xbf16>) -> tensor<128x128xbf16>
        %112 = "ttir.permute"(%111) <{permutation = array<i64: 1, 0>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %113 = "ttir.dot_general"(%109, %112) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x128xbf16>, tensor<128x128xbf16>) -> tensor<1x128xbf16>
        %114 = "ttir.reshape"(%113) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xbf16>) -> tensor<1x1x128xbf16>
        %115 = "ttir.floor"(%23) : (tensor<64xf32>) -> tensor<64xf32>
        %116 = "ttir.typecast"(%115) <{conservative_folding = false}> : (tensor<64xf32>) -> tensor<64xi64>
        %117 = "ttir.clamp_tensor"(%116, %52, %52) : (tensor<64xi64>, tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %118 = "ttir.lt"(%117, %52) : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
        %119 = "ttir.add"(%117, %50) : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %120 = "ttir.where"(%118, %119, %117) : (tensor<64xi1>, tensor<64xi64>, tensor<64xi64>) -> tensor<64xi64>
        %121 = "ttir.reshape"(%120) <{shape = [64 : i32, 1 : i32]}> : (tensor<64xi64>) -> tensor<64x1xi64>
        %122 = "ttir.permute"(%114) <{permutation = array<i64: 1, 0, 2>}> : (tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16>
        %123 = "ttir.reshape"(%122) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<1x128xbf16>
        %124 = "ttir.embedding"(%121, %123) : (tensor<64x1xi64>, tensor<1x128xbf16>) -> tensor<64x1x128xbf16>
        %125 = "ttir.reshape"(%124) <{shape = [64 : i32, 1 : i32, 128 : i32]}> : (tensor<64x1x128xbf16>) -> tensor<64x1x128xbf16>
        %126 = "ttir.permute"(%125) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x1x128xbf16>) -> tensor<1x64x128xbf16>
        %127 = "ttir.where"(%65, %53, %126) : (tensor<1x64x128xi1>, tensor<1x64x128xbf16>, tensor<1x64x128xbf16>) -> tensor<1x64x128xbf16>
        %128 = "ttir.where"(%61, %127, %7) : (tensor<1x64x128xi1>, tensor<1x64x128xbf16>, tensor<1x64x128xbf16>) -> tensor<1x64x128xbf16>
        %129 = "ttir.slice_static"(%128) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 33 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x128xbf16>) -> tensor<1x33x128xbf16>
        %130 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
        %131 = "ttir.reshape"(%130) <{shape = [3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
        %132 = "ttir.typecast"(%131) <{conservative_folding = false}> : (tensor<3072xbf16>) -> tensor<3072xf32>
        %133 = "ttir.reshape"(%132) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<3072xf32>) -> tensor<1x1x3072xf32>
        %134 = "ttir.reshape"(%16) <{shape = [1 : i32, 3072 : i32, 2048 : i32]}> : (tensor<3072x2048xbf16>) -> tensor<1x3072x2048xbf16>
        %135 = "ttir.reshape"(%134) <{shape = [3072 : i32, 2048 : i32]}> : (tensor<1x3072x2048xbf16>) -> tensor<3072x2048xbf16>
        %136 = "ttir.permute"(%135) <{permutation = array<i64: 1, 0>}> : (tensor<3072x2048xbf16>) -> tensor<2048x3072xbf16>
        %137 = "ttir.dot_general"(%66, %136) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x3072xbf16>) -> tensor<1x3072xbf16>
        %138 = "ttir.reshape"(%137) <{shape = [1 : i32, 1 : i32, 3072 : i32]}> : (tensor<1x3072xbf16>) -> tensor<1x1x3072xbf16>
        %139 = "ttir.typecast"(%138) <{conservative_folding = false}> : (tensor<1x1x3072xbf16>) -> tensor<1x1x3072xf32>
        %140 = "ttir.pow"(%139, %48) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %141 = "ttir.sum"(%140) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1xf32>
        %142 = "ttir.multiply"(%141, %25) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %143 = "ttir.reshape"(%142) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %144 = "ttir.add"(%143, %21) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %145 = "ttir.rsqrt"(%144) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %146 = "ttir.reshape"(%145) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %147 = "ttir.reshape"(%146) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %148 = "ttir.broadcast"(%147) <{broadcast_dimensions = array<i64: 1, 1, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x1x3072xf32>
        %149 = "ttir.multiply"(%139, %148) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %150 = "ttir.multiply"(%133, %149) : (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>
        %151 = "ttir.typecast"(%150) <{conservative_folding = false}> : (tensor<1x1x3072xf32>) -> tensor<1x1x3072xbf16>
        %152 = "ttir.reshape"(%151) <{shape = [1 : i32, 3072 : i32]}> : (tensor<1x1x3072xbf16>) -> tensor<1x3072xbf16>
        %153 = "ttir.reshape"(%15) <{shape = [1 : i32, 2048 : i32, 3072 : i32]}> : (tensor<2048x3072xbf16>) -> tensor<1x2048x3072xbf16>
        %154 = "ttir.reshape"(%153) <{shape = [2048 : i32, 3072 : i32]}> : (tensor<1x2048x3072xbf16>) -> tensor<2048x3072xbf16>
        %155 = "ttir.permute"(%154) <{permutation = array<i64: 1, 0>}> : (tensor<2048x3072xbf16>) -> tensor<3072x2048xbf16>
        %156 = "ttir.dot_general"(%152, %155) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<1x2048xbf16>
        %157 = "ttir.reshape"(%156) <{shape = [1 : i32, 1 : i32, 16 : i32, 128 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x1x16x128xbf16>
        %158 = "ttir.slice_static"(%157) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
        %159 = "ttir.reshape"(%158) <{shape = [1 : i32, 1 : i32, 16 : i32, 2 : i32, 32 : i32]}> : (tensor<1x1x16x64xbf16>) -> tensor<1x1x16x2x32xbf16>
        %160 = "ttir.permute"(%159) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<1x1x16x2x32xbf16>) -> tensor<1x1x16x32x2xbf16>
        %161 = "ttir.typecast"(%160) <{conservative_folding = false}> : (tensor<1x1x16x32x2xbf16>) -> tensor<1x1x16x32x2xf32>
        %162 = "ttir.slice_static"(%161) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
        %163 = "ttir.reshape"(%162) <{shape = [1 : i32, 1 : i32, 16 : i32, 32 : i32]}> : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
        %164 = "ttir.reshape"(%86) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
        %165 = "ttir.reshape"(%164) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %166 = "ttir.broadcast"(%165) <{broadcast_dimensions = array<i64: 1, 1, 16, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x16x32xf32>
        %167 = "ttir.multiply"(%163, %166) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %168 = "ttir.slice_static"(%161) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x32x1xf32>
        %169 = "ttir.reshape"(%168) <{shape = [1 : i32, 1 : i32, 16 : i32, 32 : i32]}> : (tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32xf32>
        %170 = "ttir.reshape"(%92) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
        %171 = "ttir.reshape"(%170) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %172 = "ttir.broadcast"(%171) <{broadcast_dimensions = array<i64: 1, 1, 16, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x16x32xf32>
        %173 = "ttir.multiply"(%169, %172) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %174 = "ttir.subtract"(%167, %173) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %175 = "ttir.reshape"(%174) <{shape = [1 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
        %176 = "ttir.multiply"(%163, %172) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %177 = "ttir.multiply"(%169, %166) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %178 = "ttir.add"(%176, %177) : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x32xf32>
        %179 = "ttir.reshape"(%178) <{shape = [1 : i32, 1 : i32, 16 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x16x32xf32>) -> tensor<1x1x16x32x1xf32>
        %180 = "ttir.concat"(%175, %179) <{dim = 4 : si32}> : (tensor<1x1x16x32x1xf32>, tensor<1x1x16x32x1xf32>) -> tensor<1x1x16x32x2xf32>
        %181 = "ttir.reshape"(%180) <{shape = [1 : i32, 1 : i32, 16 : i32, 64 : i32]}> : (tensor<1x1x16x32x2xf32>) -> tensor<1x1x16x64xf32>
        %182 = "ttir.slice_static"(%181) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
        %183 = "ttir.slice_static"(%181) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x32xf32>
        %184 = "ttir.concat"(%182, %183) <{dim = 3 : si32}> : (tensor<1x1x16x32xf32>, tensor<1x1x16x32xf32>) -> tensor<1x1x16x64xf32>
        %185 = "ttir.typecast"(%184) <{conservative_folding = false}> : (tensor<1x1x16x64xf32>) -> tensor<1x1x16x64xbf16>
        %186 = "ttir.slice_static"(%157) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 1 : i32, 16 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x16x128xbf16>) -> tensor<1x1x16x64xbf16>
        %187 = "ttir.concat"(%185, %186) <{dim = 3 : si32}> : (tensor<1x1x16x64xbf16>, tensor<1x1x16x64xbf16>) -> tensor<1x1x16x128xbf16>
        %188 = "ttir.dot_general"(%187, %112) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x1x16x128xbf16>, tensor<128x128xbf16>) -> tensor<1x1x16x128xbf16>
        %189 = "ttir.reshape"(%188) <{shape = [1 : i32, 16 : i32, 128 : i32]}> : (tensor<1x1x16x128xbf16>) -> tensor<1x16x128xbf16>
        %190 = "ttir.permute"(%189) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x16x128xbf16>) -> tensor<1x128x16xbf16>
        %191 = "ttir.dot_general"(%129, %190) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x33x128xbf16>, tensor<1x128x16xbf16>) -> tensor<1x33x16xbf16>
        %192 = "ttir.maximum"(%191, %47) : (tensor<1x33x16xbf16>, tensor<1x33x16xbf16>) -> tensor<1x33x16xbf16>
        %193 = "ttir.reshape"(%14) <{shape = [1 : i32, 16 : i32, 2048 : i32]}> : (tensor<16x2048xbf16>) -> tensor<1x16x2048xbf16>
        %194 = "ttir.reshape"(%193) <{shape = [16 : i32, 2048 : i32]}> : (tensor<1x16x2048xbf16>) -> tensor<16x2048xbf16>
        %195 = "ttir.permute"(%194) <{permutation = array<i64: 1, 0>}> : (tensor<16x2048xbf16>) -> tensor<2048x16xbf16>
        %196 = "ttir.dot_general"(%66, %195) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x16xbf16>) -> tensor<1x16xbf16>
        %197 = "ttir.reshape"(%196) <{shape = [1 : i32, 1 : i32, 16 : i32]}> : (tensor<1x16xbf16>) -> tensor<1x1x16xbf16>
        %198 = "ttir.multiply"(%197, %46) : (tensor<1x1x16xbf16>, tensor<1x1x16xbf16>) -> tensor<1x1x16xbf16>
        %199 = "ttir.reshape"(%198) <{shape = [1 : i32, 1 : i32, 16 : i32, 1 : i32]}> : (tensor<1x1x16xbf16>) -> tensor<1x1x16x1xbf16>
        %200 = "ttir.multiply"(%199, %44) : (tensor<1x1x16x1xbf16>, tensor<1x1x16x1xbf16>) -> tensor<1x1x16x1xbf16>
        %201 = "ttir.reshape"(%200) <{shape = [1 : i32, 16 : i32]}> : (tensor<1x1x16x1xbf16>) -> tensor<1x16xbf16>
        %202 = "ttir.reshape"(%201) <{shape = [1 : i32, 1 : i32, 16 : i32]}> : (tensor<1x16xbf16>) -> tensor<1x1x16xbf16>
        %203 = "ttir.broadcast"(%202) <{broadcast_dimensions = array<i64: 1, 33, 1>}> : (tensor<1x1x16xbf16>) -> tensor<1x33x16xbf16>
        %204 = "ttir.multiply"(%192, %203) : (tensor<1x33x16xbf16>, tensor<1x33x16xbf16>) -> tensor<1x33x16xbf16>
        %205 = "ttir.sum"(%204) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x33x16xbf16>) -> tensor<1x33xbf16>
        %206 = "ttir.all_reduce"(%205) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x33xbf16>) -> tensor<1x33xbf16>
        %207 = "ttir.reshape"(%206) <{shape = [1 : i32, 1 : i32, 33 : i32]}> : (tensor<1x33xbf16>) -> tensor<1x1x33xbf16>
        %values, %indices = "ttir.topk"(%207) <{dim = -1 : i32, k = 33 : i32, largest = true, sorted = true}> : (tensor<1x1x33xbf16>) -> (tensor<1x1x33xbf16>, tensor<1x1x33xi64>)
        %208 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x64x1xi1>) -> tensor<1x64x512xi1>
        %209 = "ttir.broadcast"(%64) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x64x1xi1>) -> tensor<1x64x512xi1>
        %210 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
        %211 = "ttir.reshape"(%210) <{shape = [512 : i32]}> : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
        %212 = "ttir.typecast"(%211) <{conservative_folding = false}> : (tensor<512xbf16>) -> tensor<512xf32>
        %213 = "ttir.reshape"(%212) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
        %214 = "ttir.reshape"(%1) <{shape = [1 : i32, 576 : i32, 2048 : i32]}> : (tensor<576x2048xbf16>) -> tensor<1x576x2048xbf16>
        %215 = "ttir.reshape"(%214) <{shape = [576 : i32, 2048 : i32]}> : (tensor<1x576x2048xbf16>) -> tensor<576x2048xbf16>
        %216 = "ttir.permute"(%215) <{permutation = array<i64: 1, 0>}> : (tensor<576x2048xbf16>) -> tensor<2048x576xbf16>
        %217 = "ttir.dot_general"(%66, %216) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x2048xbf16>, tensor<2048x576xbf16>) -> tensor<1x576xbf16>
        %218 = "ttir.reshape"(%217) <{shape = [1 : i32, 1 : i32, 576 : i32]}> : (tensor<1x576xbf16>) -> tensor<1x1x576xbf16>
        %219 = "ttir.slice_static"(%218) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x576xbf16>) -> tensor<1x1x512xbf16>
        %220 = "ttir.typecast"(%219) <{conservative_folding = false}> : (tensor<1x1x512xbf16>) -> tensor<1x1x512xf32>
        %221 = "ttir.pow"(%220, %41) : (tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
        %222 = "ttir.sum"(%221) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x512xf32>) -> tensor<1x1xf32>
        %223 = "ttir.multiply"(%222, %29) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
        %224 = "ttir.reshape"(%223) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %225 = "ttir.add"(%224, %21) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %226 = "ttir.rsqrt"(%225) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
        %227 = "ttir.reshape"(%226) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xf32>) -> tensor<1x1xf32>
        %228 = "ttir.reshape"(%227) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
        %229 = "ttir.broadcast"(%228) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x1x512xf32>
        %230 = "ttir.multiply"(%220, %229) : (tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
        %231 = "ttir.multiply"(%213, %230) : (tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
        %232 = "ttir.typecast"(%231) <{conservative_folding = false}> : (tensor<1x1x512xf32>) -> tensor<1x1x512xbf16>
        %233 = "ttir.permute"(%232) <{permutation = array<i64: 1, 0, 2>}> : (tensor<1x1x512xbf16>) -> tensor<1x1x512xbf16>
        %234 = "ttir.reshape"(%233) <{shape = [1 : i32, 512 : i32]}> : (tensor<1x1x512xbf16>) -> tensor<1x512xbf16>
        %235 = "ttir.embedding"(%121, %234) : (tensor<64x1xi64>, tensor<1x512xbf16>) -> tensor<64x1x512xbf16>
        %236 = "ttir.reshape"(%235) <{shape = [64 : i32, 1 : i32, 512 : i32]}> : (tensor<64x1x512xbf16>) -> tensor<64x1x512xbf16>
        %237 = "ttir.permute"(%236) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x1x512xbf16>) -> tensor<1x64x512xbf16>
        %238 = "ttir.where"(%209, %42, %237) : (tensor<1x64x512xi1>, tensor<1x64x512xbf16>, tensor<1x64x512xbf16>) -> tensor<1x64x512xbf16>
        %239 = "ttir.where"(%208, %238, %0) : (tensor<1x64x512xi1>, tensor<1x64x512xbf16>, tensor<1x64x512xbf16>) -> tensor<1x64x512xbf16>
        %240 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 1, 1, 64>}> : (tensor<1x64x1xi1>) -> tensor<1x64x64xi1>
        %241 = "ttir.broadcast"(%64) <{broadcast_dimensions = array<i64: 1, 1, 64>}> : (tensor<1x64x1xi1>) -> tensor<1x64x64xi1>
        %242 = "ttir.slice_static"(%218) <{begins = [0 : i32, 0 : i32, 512 : i32], ends = [1 : i32, 1 : i32, 576 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x576xbf16>) -> tensor<1x1x64xbf16>
        %243 = "ttir.reshape"(%242) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
        %244 = "ttir.typecast"(%243) <{conservative_folding = false}> : (tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xf32>
        %245 = "ttir.reshape"(%244) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32]}> : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x32x2xf32>
        %246 = "ttir.slice_static"(%245) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
        %247 = "ttir.reshape"(%246) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
        %248 = "ttir.multiply"(%247, %86) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %249 = "ttir.slice_static"(%245) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x32x1xf32>
        %250 = "ttir.reshape"(%249) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32xf32>
        %251 = "ttir.multiply"(%250, %92) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %252 = "ttir.subtract"(%248, %251) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %253 = "ttir.reshape"(%252) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
        %254 = "ttir.multiply"(%247, %92) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %255 = "ttir.multiply"(%250, %86) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %256 = "ttir.add"(%254, %255) : (tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>) -> tensor<1x1x1x32xf32>
        %257 = "ttir.reshape"(%256) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x1x32x1xf32>
        %258 = "ttir.concat"(%253, %257) <{dim = 4 : si32}> : (tensor<1x1x1x32x1xf32>, tensor<1x1x1x32x1xf32>) -> tensor<1x1x1x32x2xf32>
        %259 = "ttir.reshape"(%258) <{shape = [1 : i32, 1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x1x1x32x2xf32>) -> tensor<1x1x1x64xf32>
        %260 = "ttir.typecast"(%259) <{conservative_folding = false}> : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xbf16>
        %261 = "ttir.reshape"(%260) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x1x1x64xbf16>) -> tensor<1x1x64xbf16>
        %262 = "ttir.permute"(%261) <{permutation = array<i64: 1, 0, 2>}> : (tensor<1x1x64xbf16>) -> tensor<1x1x64xbf16>
        %263 = "ttir.reshape"(%262) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64xbf16>
        %264 = "ttir.embedding"(%121, %263) : (tensor<64x1xi64>, tensor<1x64xbf16>) -> tensor<64x1x64xbf16>
        %265 = "ttir.reshape"(%264) <{shape = [64 : i32, 1 : i32, 64 : i32]}> : (tensor<64x1x64xbf16>) -> tensor<64x1x64xbf16>
        %266 = "ttir.permute"(%265) <{permutation = array<i64: 1, 0, 2>}> : (tensor<64x1x64xbf16>) -> tensor<1x64x64xbf16>
        %267 = "ttir.where"(%241, %39, %266) : (tensor<1x64x64xi1>, tensor<1x64x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
        %268 = "ttir.where"(%240, %267, %5) : (tensor<1x64x64xi1>, tensor<1x64x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
        %269 = "ttir.reshape"(%18) <{shape = [1 : i32, 768 : i32, 3072 : i32]}> : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
        %270 = "ttir.reshape"(%269) <{shape = [768 : i32, 3072 : i32]}> : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
        %271 = "ttir.permute"(%270) <{permutation = array<i64: 1, 0>}> : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
        %272 = "ttir.dot_general"(%152, %271) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3072xbf16>, tensor<3072x768xbf16>) -> tensor<1x768xbf16>
        %273 = "ttir.reshape"(%272) <{shape = [1 : i32, 1 : i32, 4 : i32, 192 : i32]}> : (tensor<1x768xbf16>) -> tensor<1x1x4x192xbf16>
        %274 = "ttir.slice_static"(%273) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x128xbf16>
        %275 = "ttir.reshape"(%13) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
        %276 = "ttir.reshape"(%275) <{shape = [4 : i32, 256 : i32, 512 : i32]}> : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
        %277 = "ttir.slice_static"(%276) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 128 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
        %278 = "ttir.dot_general"(%274, %277) <{batch_dims_lhs = array<i64: 2>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x512xbf16>
        %279 = "ttir.reshape"(%278) <{shape = [1 : i32, 1 : i32, 4 : i32, 512 : i32]}> : (tensor<4x1x1x512xbf16>) -> tensor<1x1x4x512xbf16>
        %280 = "ttir.slice_static"(%239) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 33 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x512xbf16>) -> tensor<1x33x512xbf16>
        %281 = "ttir.reshape"(%indices) <{shape = [1 : i32, 33 : i32]}> : (tensor<1x1x33xi64>) -> tensor<1x33xi64>
        %282 = "ttir.reshape"(%281) <{shape = [1 : i32, 33 : i32, 1 : i32]}> : (tensor<1x33xi64>) -> tensor<1x33x1xi64>
        %283 = "ttir.broadcast"(%282) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x33x1xi64>) -> tensor<1x33x512xi64>
        %284 = "ttir.typecast"(%283) <{conservative_folding = false}> : (tensor<1x33x512xi64>) -> tensor<1x33x512xui32>
        %285 = "ttir.reshape"(%284) <{shape = [1 : i32, 33 : i32, 512 : i32, 1 : i32]}> : (tensor<1x33x512xui32>) -> tensor<1x33x512x1xui32>
        %286 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xui32>
        %287 = "ttir.reshape"(%286) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xui32>) -> tensor<1x1x512x1xui32>
        %288 = "ttir.broadcast"(%287) <{broadcast_dimensions = array<i64: 1, 33, 1, 1>}> : (tensor<1x1x512x1xui32>) -> tensor<1x33x512x1xui32>
        %289 = "ttir.concat"(%37, %285, %288) <{dim = 3 : si32}> : (tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>, tensor<1x33x512x1xui32>) -> tensor<1x33x512x3xui32>
        %290 = "ttir.permute"(%280) <{permutation = array<i64: 1, 2, 0>}> : (tensor<1x33x512xbf16>) -> tensor<33x512x1xbf16>
        %291 = "ttir.reshape"(%290) <{shape = [16896 : i32, 1 : i32]}> : (tensor<33x512x1xbf16>) -> tensor<16896x1xbf16>
        %292 = "ttir.permute"(%289) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x33x512x3xui32>) -> tensor<1x33x512x3xui32>
        %293 = "ttir.typecast"(%292) <{conservative_folding = false}> : (tensor<1x33x512x3xui32>) -> tensor<1x33x512x3xf32>
        %294 = "ttir.constant"() <{value = dense<[[5.120000e+02], [1.000000e+00], [1.000000e+00]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
        %295 = "ttir.matmul"(%293, %294) <{transpose_a = false, transpose_b = false}> : (tensor<1x33x512x3xf32>, tensor<3x1xf32>) -> tensor<1x33x512x1xf32>
        %296 = "ttir.reshape"(%295) <{shape = [1 : i32, 16896 : i32]}> : (tensor<1x33x512x1xf32>) -> tensor<1x16896xf32>
        %297 = "ttir.embedding"(%296, %291) : (tensor<1x16896xf32>, tensor<16896x1xbf16>) -> tensor<1x16896x1xbf16>
        %298 = "ttir.reshape"(%297) <{shape = [1 : i32, 33 : i32, 512 : i32]}> : (tensor<1x16896x1xbf16>) -> tensor<1x33x512xbf16>
        %299 = "ttir.permute"(%298) <{permutation = array<i64: 0, 1, 2>}> : (tensor<1x33x512xbf16>) -> tensor<1x33x512xbf16>
        %300 = "ttir.dot_general"(%279, %299) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<1x1x4x512xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x33xbf16>
        %301 = "ttir.slice_static"(%273) <{begins = [0 : i32, 0 : i32, 0 : i32, 128 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 192 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x192xbf16>) -> tensor<1x1x4x64xbf16>
        %302 = "ttir.typecast"(%301) <{conservative_folding = false}> : (tensor<1x1x4x64xbf16>) -> tensor<1x1x4x64xf32>
        %303 = "ttir.reshape"(%302) <{shape = [1 : i32, 1 : i32, 4 : i32, 32 : i32, 2 : i32]}> : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x32x2xf32>
        %304 = "ttir.slice_static"(%303) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
        %305 = "ttir.reshape"(%304) <{shape = [1 : i32, 1 : i32, 4 : i32, 32 : i32]}> : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
        %306 = "ttir.broadcast"(%165) <{broadcast_dimensions = array<i64: 1, 1, 4, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x4x32xf32>
        %307 = "ttir.multiply"(%305, %306) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %308 = "ttir.slice_static"(%303) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 32 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x32x1xf32>
        %309 = "ttir.reshape"(%308) <{shape = [1 : i32, 1 : i32, 4 : i32, 32 : i32]}> : (tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32xf32>
        %310 = "ttir.broadcast"(%171) <{broadcast_dimensions = array<i64: 1, 1, 4, 1>}> : (tensor<1x1x1x32xf32>) -> tensor<1x1x4x32xf32>
        %311 = "ttir.multiply"(%309, %310) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %312 = "ttir.subtract"(%307, %311) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %313 = "ttir.reshape"(%312) <{shape = [1 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
        %314 = "ttir.multiply"(%305, %310) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %315 = "ttir.multiply"(%309, %306) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %316 = "ttir.add"(%314, %315) : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32>
        %317 = "ttir.reshape"(%316) <{shape = [1 : i32, 1 : i32, 4 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x4x32xf32>) -> tensor<1x1x4x32x1xf32>
        %318 = "ttir.concat"(%313, %317) <{dim = 4 : si32}> : (tensor<1x1x4x32x1xf32>, tensor<1x1x4x32x1xf32>) -> tensor<1x1x4x32x2xf32>
        %319 = "ttir.reshape"(%318) <{shape = [1 : i32, 1 : i32, 4 : i32, 64 : i32]}> : (tensor<1x1x4x32x2xf32>) -> tensor<1x1x4x64xf32>
        %320 = "ttir.typecast"(%319) <{conservative_folding = false}> : (tensor<1x1x4x64xf32>) -> tensor<1x1x4x64xbf16>
        %321 = "ttir.slice_static"(%268) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 33 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x64xbf16>) -> tensor<1x33x64xbf16>
        %322 = "ttir.broadcast"(%282) <{broadcast_dimensions = array<i64: 1, 1, 64>}> : (tensor<1x33x1xi64>) -> tensor<1x33x64xi64>
        %323 = "ttir.typecast"(%322) <{conservative_folding = false}> : (tensor<1x33x64xi64>) -> tensor<1x33x64xui32>
        %324 = "ttir.reshape"(%323) <{shape = [1 : i32, 33 : i32, 64 : i32, 1 : i32]}> : (tensor<1x33x64xui32>) -> tensor<1x33x64x1xui32>
        %325 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 64 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<64xui32>
        %326 = "ttir.reshape"(%325) <{shape = [1 : i32, 1 : i32, 64 : i32, 1 : i32]}> : (tensor<64xui32>) -> tensor<1x1x64x1xui32>
        %327 = "ttir.broadcast"(%326) <{broadcast_dimensions = array<i64: 1, 33, 1, 1>}> : (tensor<1x1x64x1xui32>) -> tensor<1x33x64x1xui32>
        %328 = "ttir.concat"(%36, %324, %327) <{dim = 3 : si32}> : (tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>, tensor<1x33x64x1xui32>) -> tensor<1x33x64x3xui32>
        %329 = "ttir.permute"(%321) <{permutation = array<i64: 1, 2, 0>}> : (tensor<1x33x64xbf16>) -> tensor<33x64x1xbf16>
        %330 = "ttir.reshape"(%329) <{shape = [2112 : i32, 1 : i32]}> : (tensor<33x64x1xbf16>) -> tensor<2112x1xbf16>
        %331 = "ttir.permute"(%328) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x33x64x3xui32>) -> tensor<1x33x64x3xui32>
        %332 = "ttir.typecast"(%331) <{conservative_folding = false}> : (tensor<1x33x64x3xui32>) -> tensor<1x33x64x3xf32>
        %333 = "ttir.constant"() <{value = dense<[[6.400000e+01], [1.000000e+00], [1.000000e+00]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
        %334 = "ttir.matmul"(%332, %333) <{transpose_a = false, transpose_b = false}> : (tensor<1x33x64x3xf32>, tensor<3x1xf32>) -> tensor<1x33x64x1xf32>
        %335 = "ttir.reshape"(%334) <{shape = [1 : i32, 2112 : i32]}> : (tensor<1x33x64x1xf32>) -> tensor<1x2112xf32>
        %336 = "ttir.embedding"(%335, %330) : (tensor<1x2112xf32>, tensor<2112x1xbf16>) -> tensor<1x2112x1xbf16>
        %337 = "ttir.reshape"(%336) <{shape = [1 : i32, 33 : i32, 64 : i32]}> : (tensor<1x2112x1xbf16>) -> tensor<1x33x64xbf16>
        %338 = "ttir.permute"(%337) <{permutation = array<i64: 0, 1, 2>}> : (tensor<1x33x64xbf16>) -> tensor<1x33x64xbf16>
        %339 = "ttir.dot_general"(%320, %338) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<1x1x4x64xbf16>, tensor<1x33x64xbf16>) -> tensor<1x1x4x33xbf16>
        %340 = "ttir.add"(%300, %339) : (tensor<1x1x4x33xbf16>, tensor<1x1x4x33xbf16>) -> tensor<1x1x4x33xbf16>
        %341 = "ttir.multiply"(%340, %34) : (tensor<1x1x4x33xbf16>, tensor<1x1x4x33xbf16>) -> tensor<1x1x4x33xbf16>
        %342 = "ttir.max"(%341) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x1x4x33xbf16>) -> tensor<1x1x4xbf16>
        %343 = "ttir.reshape"(%342) <{shape = [1 : i32, 1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x1x4xbf16>) -> tensor<1x1x4x1xbf16>
        %344 = "ttir.broadcast"(%343) <{broadcast_dimensions = array<i64: 1, 1, 1, 33>}> : (tensor<1x1x4x1xbf16>) -> tensor<1x1x4x33xbf16>
        %345 = "ttir.subtract"(%341, %344) : (tensor<1x1x4x33xbf16>, tensor<1x1x4x33xbf16>) -> tensor<1x1x4x33xbf16>
        %346 = "ttir.exp"(%345) : (tensor<1x1x4x33xbf16>) -> tensor<1x1x4x33xbf16>
        %347 = "ttir.sum"(%346) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x1x4x33xbf16>) -> tensor<1x1x4xbf16>
        %348 = "ttir.reshape"(%347) <{shape = [1 : i32, 1 : i32, 4 : i32, 1 : i32]}> : (tensor<1x1x4xbf16>) -> tensor<1x1x4x1xbf16>
        %349 = "ttir.broadcast"(%348) <{broadcast_dimensions = array<i64: 1, 1, 1, 33>}> : (tensor<1x1x4x1xbf16>) -> tensor<1x1x4x33xbf16>
        %350 = "ttir.div"(%346, %349) : (tensor<1x1x4x33xbf16>, tensor<1x1x4x33xbf16>) -> tensor<1x1x4x33xbf16>
        %351 = "ttir.dot_general"(%350, %299) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x1x4x33xbf16>, tensor<1x33x512xbf16>) -> tensor<1x1x4x512xbf16>
        %352 = "ttir.slice_static"(%276) <{begins = [0 : i32, 128 : i32, 0 : i32], ends = [4 : i32, 256 : i32, 512 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
        %353 = "ttir.dot_general"(%351, %352) <{batch_dims_lhs = array<i64: 2>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<1x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x1x1x128xbf16>
        %354 = "ttir.reshape"(%353) <{shape = [1 : i32, 512 : i32]}> : (tensor<4x1x1x128xbf16>) -> tensor<1x512xbf16>
        %355 = "ttir.reshape"(%12) <{shape = [1 : i32, 2048 : i32, 512 : i32]}> : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
        %356 = "ttir.reshape"(%355) <{shape = [2048 : i32, 512 : i32]}> : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
        %357 = "ttir.permute"(%356) <{permutation = array<i64: 1, 0>}> : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
        %358 = "ttir.dot_general"(%354, %357) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512xbf16>, tensor<512x2048xbf16>) -> tensor<1x2048xbf16>
        %359 = "ttir.all_reduce"(%358) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
        %360 = "ttir.reshape"(%359) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
        %361 = "ttir.mesh_shard"(%239) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x512xbf16>) -> tensor<1x64x512xbf16>
        %362 = "ttir.mesh_shard"(%268) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
        %363 = "ttir.mesh_shard"(%128) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x64x128xbf16>) -> tensor<1x64x128xbf16>
        %364 = "ttir.mesh_shard"(%360) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
        return %361, %362, %363, %364 : tensor<1x64x512xbf16>, tensor<1x64x64xbf16>, tensor<1x64x128xbf16>, tensor<1x1x2048xbf16>
      }
    }
  }
}
