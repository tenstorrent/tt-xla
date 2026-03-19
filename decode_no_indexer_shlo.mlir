module @SyncTensorsGraph.388 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(
    %arg0: tensor<4x64x512xbf16> {
      ttcore.argument_type = #ttcore.argument_type<input>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "kv_cache"
    },
    %arg1: tensor<576x2048xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<576x1024xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "wkv_a.weight"
    },
    %arg2: tensor<4x1x2048xbf16> {
      ttcore.argument_type = #ttcore.argument_type<input>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "args_0"
    },
    %arg3: tensor<512xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "kv_norm.weight"
    },
    %arg4: tensor<i1> {
      ttcore.argument_type = #ttcore.argument_type<input>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>
    },
    %arg5: tensor<4x64x64xbf16> {
      ttcore.argument_type = #ttcore.argument_type<input>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "pe_cache"
    },
    %arg6: tensor<1x32x2xbf16> {
      ttcore.argument_type = #ttcore.argument_type<input>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x2xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "args_1"
    },
    %arg7: tensor<2048x2048xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "wo.weight"
    },
    %arg8: tensor<4096x512xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x512xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "wkv_b.weight"
    },
    %arg9: tensor<3072x3072xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<768x3072xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "wq_b.weight"
    },
    %arg10: tensor<3072x2048xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072x1024xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "wq_a.weight"
    },
    %arg11: tensor<3072xbf16> {
      ttcore.argument_type = #ttcore.argument_type<parameter>,
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<3072xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>,
      ttir.name = "q_norm.weight"
    }
  ) -> (
    tensor<4x64x512xbf16> {
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x512xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>
    },
    tensor<4x64x64xbf16> {
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2x64x64xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>
    },
    tensor<4x1x2048xbf16> {
      ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x1x1024xbf16>>,
      ttcore.shard_status = #ttcore.shard_status<presharded>
    }
  ) {
    %0:3 = sdy.manual_computation(
        %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6,
        %arg7, %arg8, %arg9, %arg10, %arg11
    )
    in_shardings = [
        <@mesh, [{"_axis_0"}, {}, {}]>,         // %arg0
        <@mesh, [{}, {"_axis_0"}]>,             // %arg1
        <@mesh, [{}, {}, {"_axis_0"}]>,         // %arg2
        <@mesh, [{}]>,                          // %arg3
        <@mesh, []>,                            // %arg4
        <@mesh, [{"_axis_0"}, {}, {}]>,         // %arg5
        <@mesh, [{}, {}, {}]>,                  // %arg6
        <@mesh, [{"_axis_0"}, {"_axis_1"}]>,    // %arg7
        <@mesh, [{"_axis_1"}, {}]>,             // %arg8
        <@mesh, [{"_axis_1"}, {}]>,             // %arg9
        <@mesh, [{}, {"_axis_0"}]>,             // %arg10
        <@mesh, [{}]>                           // %arg11
    ]
    out_shardings = [
        <@mesh, [{"_axis_0"}, {}, {}]>,
        <@mesh, [{"_axis_0"}, {}, {}]>,
        <@mesh, [{}, {}, {"_axis_0"}]>
    ]
    manual_axes = {"_axis_0", "_axis_1"}
    (
        %arg12: tensor<2x64x512xbf16>,
        %arg13: tensor<576x1024xbf16>,
        %arg14: tensor<4x1x1024xbf16>,
        %arg15: tensor<512xbf16>,
        %arg16: tensor<i1>,
        %arg17: tensor<2x64x64xbf16>,
        %arg18: tensor<1x32x2xbf16>,
        %arg19: tensor<1024x512xbf16>,
        %arg20: tensor<1024x512xbf16>,
        %arg21: tensor<768x3072xbf16>,
        %arg22: tensor<3072x1024xbf16>,
        %arg23: tensor<3072xbf16>
    ) {
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %cst_0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %c = stablehlo.constant dense<[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<64xi1>
      %c_1 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
      %cst_2 = stablehlo.constant dense<0.001953125> : tensor<2x1xf32>
      %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<2x1x1xf32>
      %c_4 = stablehlo.constant dense<0> : tensor<i64>
      %cst_5 = stablehlo.constant dense<[-3.200000e+01, -3.100000e+01, -3.000000e+01, -2.900000e+01, -2.800000e+01, -2.700000e+01, -2.600000e+01, -2.500000e+01, -2.400000e+01, -2.300000e+01, -2.200000e+01, -2.100000e+01, -2.000000e+01, -1.900000e+01, -1.800000e+01, -1.700000e+01, -1.600000e+01, -1.500000e+01, -1.400000e+01, -1.300000e+01, -1.200000e+01, -1.100000e+01, -1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00, -2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<64xf32>
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %cst_8 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %cst_9 = stablehlo.constant dense<3.25520843E-4> : tensor<2x1xf32>
      %cst_10 = stablehlo.constant dense<7.226560e-02> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<2x1x4x33xbf16>
      %2 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<2x1x3072xf32>
      %3 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x64x64xbf16>
      %4 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %5 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %6 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<2x1x512xf32>
      %7 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<2x64x512xbf16>
      %8 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<i1>) -> tensor<64xi1>
      %9 = stablehlo.and %8, %c : tensor<64xi1>
      %10 = stablehlo.and %9, %c_1 : tensor<64xi1>
      %11 = stablehlo.reshape %10 : (tensor<64xi1>) -> tensor<1x64x1xi1>
      %12 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %13 = stablehlo.not %11 : tensor<1x64x1xi1>
      %14 = stablehlo.reshape %13 : (tensor<1x64x1xi1>) -> tensor<64xi1>
      %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x512xi1>
      %16 = stablehlo.reshape %arg15 : (tensor<512xbf16>) -> tensor<1x1x512xbf16>
      %17 = stablehlo.reshape %16 : (tensor<1x1x512xbf16>) -> tensor<512xbf16>
      %18 = stablehlo.convert %17 : (tensor<512xbf16>) -> tensor<512xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [2] : (tensor<512xf32>) -> tensor<2x1x512xf32>
      %20 = stablehlo.reshape %arg14 : (tensor<4x1x1024xbf16>) -> tensor<4x1024xbf16>
      %21 = stablehlo.reshape %arg13 : (tensor<576x1024xbf16>) -> tensor<1x576x1024xbf16>
      %22 = stablehlo.reshape %21 : (tensor<1x576x1024xbf16>) -> tensor<576x1024xbf16>
      %23 = stablehlo.transpose %22, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,576]{0,1}"} : (tensor<576x1024xbf16>) -> tensor<1024x576xbf16>
      %24 = stablehlo.dot_general %20, %23, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x576xbf16>) -> tensor<4x576xbf16>
      %25 = "stablehlo.reduce_scatter"(%24) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x576xbf16>) -> tensor<2x576xbf16>
      %26 = stablehlo.reshape %25 : (tensor<2x576xbf16>) -> tensor<2x1x576xbf16>
      %27 = stablehlo.slice %26 [0:2, 0:1, 0:512] : (tensor<2x1x576xbf16>) -> tensor<2x1x512xbf16>
      %28 = stablehlo.convert %27 : (tensor<2x1x512xbf16>) -> tensor<2x1x512xf32>
      %29 = stablehlo.power %28, %6 : tensor<2x1x512xf32>
      %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x512xf32>, tensor<f32>) -> tensor<2x1xf32>
      %31 = stablehlo.multiply %30, %cst_2 : tensor<2x1xf32>
      %32 = stablehlo.reshape %31 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %33 = stablehlo.add %32, %cst_3 : tensor<2x1x1xf32>
      %34 = stablehlo.rsqrt %33 : tensor<2x1x1xf32>
      %35 = stablehlo.reshape %34 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x512xf32>
      %37 = stablehlo.multiply %28, %36 : tensor<2x1x512xf32>
      %38 = stablehlo.multiply %19, %37 : tensor<2x1x512xf32>
      %39 = stablehlo.convert %38 : (tensor<2x1x512xf32>) -> tensor<2x1x512xbf16>
      %40 = stablehlo.floor %cst_5 : tensor<64xf32>
      %41 = stablehlo.convert %40 : (tensor<64xf32>) -> tensor<64xi64>
      %42 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %43 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<64xi64>
      %44 = stablehlo.clamp %43, %41, %42 : tensor<64xi64>
      %45 = stablehlo.compare  LT, %44, %5 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
      %46 = stablehlo.add %44, %4 : tensor<64xi64>
      %47 = stablehlo.select %45, %46, %44 : tensor<64xi1>, tensor<64xi64>
      %48 = stablehlo.reshape %47 : (tensor<64xi64>) -> tensor<64x1xi64>
      %49 = "stablehlo.gather"(%39, %48) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 512>}> : (tensor<2x1x512xbf16>, tensor<64x1xi64>) -> tensor<2x64x512xbf16>
      %50 = stablehlo.select %15, %7, %49 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %51 = stablehlo.select %12, %50, %arg12 : tensor<2x64x512xi1>, tensor<2x64x512xbf16>
      %52 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %53 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<64xi1>) -> tensor<2x64x64xi1>
      %54 = stablehlo.slice %26 [0:2, 0:1, 512:576] : (tensor<2x1x576xbf16>) -> tensor<2x1x64xbf16>
      %55 = stablehlo.reshape %54 : (tensor<2x1x64xbf16>) -> tensor<2x1x1x64xbf16>
      %56 = stablehlo.convert %55 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x1x64xf32>
      %57 = stablehlo.reshape %56 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x32x2xf32>
      %58 = stablehlo.slice %57 [0:2, 0:1, 0:1, 0:32, 0:1] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %59 = stablehlo.reshape %58 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %60 = stablehlo.reshape %arg18 : (tensor<1x32x2xbf16>) -> tensor<1x1x1x32x2xbf16>
      %61 = stablehlo.slice %60 [0:1, 0:1, 0:1, 0:32, 0:1] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %62 = stablehlo.reshape %61 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %63 = stablehlo.convert %62 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %64 = stablehlo.reshape %63 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %65 = stablehlo.broadcast_in_dim %64, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %66 = stablehlo.multiply %59, %65 : tensor<2x1x1x32xf32>
      %67 = stablehlo.slice %57 [0:2, 0:1, 0:1, 0:32, 1:2] : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x32x1xf32>
      %68 = stablehlo.reshape %67 : (tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32xf32>
      %69 = stablehlo.slice %60 [0:1, 0:1, 0:1, 0:32, 1:2] : (tensor<1x1x1x32x2xbf16>) -> tensor<1x1x1x32x1xbf16>
      %70 = stablehlo.reshape %69 : (tensor<1x1x1x32x1xbf16>) -> tensor<1x1x1x32xbf16>
      %71 = stablehlo.convert %70 : (tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xf32>
      %72 = stablehlo.reshape %71 : (tensor<1x1x1x32xf32>) -> tensor<1x1x32xf32>
      %73 = stablehlo.broadcast_in_dim %72, dims = [1, 2, 3] : (tensor<1x1x32xf32>) -> tensor<2x1x1x32xf32>
      %74 = stablehlo.multiply %68, %73 : tensor<2x1x1x32xf32>
      %75 = stablehlo.subtract %66, %74 : tensor<2x1x1x32xf32>
      %76 = stablehlo.reshape %75 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %77 = stablehlo.multiply %59, %73 : tensor<2x1x1x32xf32>
      %78 = stablehlo.multiply %68, %65 : tensor<2x1x1x32xf32>
      %79 = stablehlo.add %77, %78 : tensor<2x1x1x32xf32>
      %80 = stablehlo.reshape %79 : (tensor<2x1x1x32xf32>) -> tensor<2x1x1x32x1xf32>
      %81 = stablehlo.concatenate %76, %80, dim = 4 : (tensor<2x1x1x32x1xf32>, tensor<2x1x1x32x1xf32>) -> tensor<2x1x1x32x2xf32>
      %82 = stablehlo.reshape %81 : (tensor<2x1x1x32x2xf32>) -> tensor<2x1x1x64xf32>
      %83 = stablehlo.convert %82 : (tensor<2x1x1x64xf32>) -> tensor<2x1x1x64xbf16>
      %84 = stablehlo.reshape %83 : (tensor<2x1x1x64xbf16>) -> tensor<2x1x64xbf16>
      %85 = "stablehlo.gather"(%84, %48) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = array<i64: 2, 1, 64>}> : (tensor<2x1x64xbf16>, tensor<64x1xi64>) -> tensor<2x64x64xbf16>
      %86 = stablehlo.select %53, %3, %85 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %87 = stablehlo.select %52, %86, %arg17 : tensor<2x64x64xi1>, tensor<2x64x64xbf16>
      %88 = stablehlo.reshape %arg23 : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
      %89 = stablehlo.reshape %88 : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
      %90 = stablehlo.convert %89 : (tensor<3072xbf16>) -> tensor<3072xf32>
      %91 = stablehlo.broadcast_in_dim %90, dims = [2] : (tensor<3072xf32>) -> tensor<2x1x3072xf32>
      %92 = stablehlo.reshape %arg22 : (tensor<3072x1024xbf16>) -> tensor<1x3072x1024xbf16>
      %93 = stablehlo.reshape %92 : (tensor<1x3072x1024xbf16>) -> tensor<3072x1024xbf16>
      %94 = stablehlo.transpose %93, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,3072]{0,1}"} : (tensor<3072x1024xbf16>) -> tensor<1024x3072xbf16>
      %95 = stablehlo.dot_general %20, %94, contracting_dims = [1] x [0] : (tensor<4x1024xbf16>, tensor<1024x3072xbf16>) -> tensor<4x3072xbf16>
      %96 = "stablehlo.reduce_scatter"(%95) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x3072xbf16>) -> tensor<2x3072xbf16>
      %97 = stablehlo.reshape %96 : (tensor<2x3072xbf16>) -> tensor<2x1x3072xbf16>
      %98 = stablehlo.convert %97 : (tensor<2x1x3072xbf16>) -> tensor<2x1x3072xf32>
      %99 = stablehlo.power %98, %2 : tensor<2x1x3072xf32>
      %100 = stablehlo.reduce(%99 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<2x1x3072xf32>, tensor<f32>) -> tensor<2x1xf32>
      %101 = stablehlo.multiply %100, %cst_9 : tensor<2x1xf32>
      %102 = stablehlo.reshape %101 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
      %103 = stablehlo.add %102, %cst_3 : tensor<2x1x1xf32>
      %104 = stablehlo.rsqrt %103 : tensor<2x1x1xf32>
      %105 = stablehlo.reshape %104 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
      %106 = stablehlo.broadcast_in_dim %105, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x3072xf32>
      %107 = stablehlo.multiply %98, %106 : tensor<2x1x3072xf32>
      %108 = stablehlo.multiply %91, %107 : tensor<2x1x3072xf32>
      %109 = stablehlo.convert %108 : (tensor<2x1x3072xf32>) -> tensor<2x1x3072xbf16>
      %110 = stablehlo.reshape %109 : (tensor<2x1x3072xbf16>) -> tensor<2x3072xbf16>
      %111 = stablehlo.reshape %arg21 : (tensor<768x3072xbf16>) -> tensor<1x768x3072xbf16>
      %112 = stablehlo.reshape %111 : (tensor<1x768x3072xbf16>) -> tensor<768x3072xbf16>
      %113 = stablehlo.transpose %112, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<768x3072xbf16>) -> tensor<3072x768xbf16>
      %114 = stablehlo.dot_general %110, %113, contracting_dims = [1] x [0] : (tensor<2x3072xbf16>, tensor<3072x768xbf16>) -> tensor<2x768xbf16>
      %115 = stablehlo.reshape %114 : (tensor<2x768xbf16>) -> tensor<2x1x4x192xbf16>
      %116 = stablehlo.slice %115 [0:2, 0:1, 0:4, 0:128] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x128xbf16>
      %117 = stablehlo.reshape %arg20 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %118 = stablehlo.reshape %117 : (tensor<1x1024x512xbf16>) -> tensor<4x256x512xbf16>
      %119 = stablehlo.slice %118 [0:4, 0:128, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %120 = stablehlo.dot_general %116, %119, batching_dims = [2] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x128xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x512xbf16>
      %121 = stablehlo.transpose %120, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,512]{3,1,0,2}"} : (tensor<4x2x1x512xbf16>) -> tensor<2x1x4x512xbf16>
      %122 = stablehlo.slice %51 [0:2, 0:33, 0:512] : (tensor<2x64x512xbf16>) -> tensor<2x33x512xbf16>
      %123 = stablehlo.dot_general %121, %122, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x33xbf16>
      %124 = stablehlo.slice %115 [0:2, 0:1, 0:4, 128:192] : (tensor<2x1x4x192xbf16>) -> tensor<2x1x4x64xbf16>
      %125 = stablehlo.convert %124 : (tensor<2x1x4x64xbf16>) -> tensor<2x1x4x64xf32>
      %126 = stablehlo.reshape %125 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x32x2xf32>
      %127 = stablehlo.slice %126 [0:2, 0:1, 0:4, 0:32, 0:1] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %128 = stablehlo.reshape %127 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %129 = stablehlo.reshape %63 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %130 = stablehlo.broadcast_in_dim %129, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %131 = stablehlo.multiply %128, %130 : tensor<2x1x4x32xf32>
      %132 = stablehlo.slice %126 [0:2, 0:1, 0:4, 0:32, 1:2] : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x32x1xf32>
      %133 = stablehlo.reshape %132 : (tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32xf32>
      %134 = stablehlo.reshape %71 : (tensor<1x1x1x32xf32>) -> tensor<1x32xf32>
      %135 = stablehlo.broadcast_in_dim %134, dims = [1, 3] : (tensor<1x32xf32>) -> tensor<2x1x4x32xf32>
      %136 = stablehlo.multiply %133, %135 : tensor<2x1x4x32xf32>
      %137 = stablehlo.subtract %131, %136 : tensor<2x1x4x32xf32>
      %138 = stablehlo.reshape %137 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %139 = stablehlo.multiply %128, %135 : tensor<2x1x4x32xf32>
      %140 = stablehlo.multiply %133, %130 : tensor<2x1x4x32xf32>
      %141 = stablehlo.add %139, %140 : tensor<2x1x4x32xf32>
      %142 = stablehlo.reshape %141 : (tensor<2x1x4x32xf32>) -> tensor<2x1x4x32x1xf32>
      %143 = stablehlo.concatenate %138, %142, dim = 4 : (tensor<2x1x4x32x1xf32>, tensor<2x1x4x32x1xf32>) -> tensor<2x1x4x32x2xf32>
      %144 = stablehlo.reshape %143 : (tensor<2x1x4x32x2xf32>) -> tensor<2x1x4x64xf32>
      %145 = stablehlo.convert %144 : (tensor<2x1x4x64xf32>) -> tensor<2x1x4x64xbf16>
      %146 = stablehlo.slice %87 [0:2, 0:33, 0:64] : (tensor<2x64x64xbf16>) -> tensor<2x33x64xbf16>
      %147 = stablehlo.dot_general %145, %146, batching_dims = [0] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x64xbf16>, tensor<2x33x64xbf16>) -> tensor<2x1x4x33xbf16>
      %148 = stablehlo.add %123, %147 : tensor<2x1x4x33xbf16>
      %149 = stablehlo.multiply %148, %1 : tensor<2x1x4x33xbf16>
      %150 = stablehlo.reduce(%149 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %152 = stablehlo.subtract %149, %151 : tensor<2x1x4x33xbf16>
      %153 = stablehlo.exponential %152 : tensor<2x1x4x33xbf16>
      %154 = stablehlo.reduce(%153 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<2x1x4x33xbf16>, tensor<bf16>) -> tensor<2x1x4xbf16>
      %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<2x1x4xbf16>) -> tensor<2x1x4x33xbf16>
      %156 = stablehlo.divide %153, %155 : tensor<2x1x4x33xbf16>
      %157 = stablehlo.dot_general %156, %122, batching_dims = [0] x [0], contracting_dims = [3] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x33xbf16>, tensor<2x33x512xbf16>) -> tensor<2x1x4x512xbf16>
      %158 = stablehlo.slice %118 [0:4, 128:256, 0:512] : (tensor<4x256x512xbf16>) -> tensor<4x128x512xbf16>
      %159 = stablehlo.dot_general %157, %158, batching_dims = [2] x [0], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x1x4x512xbf16>, tensor<4x128x512xbf16>) -> tensor<4x2x1x128xbf16>
      %160 = stablehlo.transpose %159, dims = [1, 2, 0, 3] {result_layout = dense<[3, 1, 0, 2]> : tensor<4xindex>, xla_shape = "bf16[4,1,16,128]{3,1,0,2}"} : (tensor<4x2x1x128xbf16>) -> tensor<2x1x4x128xbf16>
      %161 = stablehlo.reshape %160 : (tensor<2x1x4x128xbf16>) -> tensor<2x512xbf16>
      %162 = stablehlo.reshape %arg19 : (tensor<1024x512xbf16>) -> tensor<1x1024x512xbf16>
      %163 = stablehlo.reshape %162 : (tensor<1x1024x512xbf16>) -> tensor<1024x512xbf16>
      %164 = stablehlo.transpose %163, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<1024x512xbf16>) -> tensor<512x1024xbf16>
      %165 = "stablehlo.all_gather"(%161) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<2x512xbf16>) -> tensor<4x512xbf16>
      %166 = stablehlo.dot_general %165, %164, contracting_dims = [1] x [0] : (tensor<4x512xbf16>, tensor<512x1024xbf16>) -> tensor<4x1024xbf16>
      %167 = "stablehlo.all_reduce"(%166) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg24: tensor<bf16>, %arg25: tensor<bf16>):
        %169 = stablehlo.add %arg24, %arg25 : tensor<bf16>
        stablehlo.return %169 : tensor<bf16>
      }) : (tensor<4x1024xbf16>) -> tensor<4x1024xbf16>
      %168 = stablehlo.reshape %167 : (tensor<4x1024xbf16>) -> tensor<4x1x1024xbf16>
      sdy.return %51, %87, %168 : tensor<2x64x512xbf16>, tensor<2x64x64xbf16>, tensor<4x1x1024xbf16>
    } : (tensor<4x64x512xbf16>, tensor<576x2048xbf16>, tensor<4x1x2048xbf16>, tensor<512xbf16>, tensor<i1>, tensor<4x64x64xbf16>, tensor<1x32x2xbf16>, tensor<2048x2048xbf16>, tensor<4096x512xbf16>, tensor<3072x3072xbf16>, tensor<3072x2048xbf16>, tensor<3072xbf16>) -> (tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x1x2048xbf16>)
    return %0#0, %0#1, %0#2 : tensor<4x64x512xbf16>, tensor<4x64x64xbf16>, tensor<4x1x2048xbf16>
  }
}
