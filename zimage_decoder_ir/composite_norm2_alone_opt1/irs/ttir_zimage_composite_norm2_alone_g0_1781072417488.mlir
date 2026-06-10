#loc1 = loc("p0.2")
#loc2 = loc("p1.10")
#loc3 = loc("p2.21")
module @SyncTensorsGraph.108 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.108 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "norm.bias"} loc("p0.2"), %arg1: tensor<128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "norm.weight"} loc("p1.10"), %arg2: tensor<1x128x1280x720xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x1280x720xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p2.21")) -> (tensor<1x128x1280x720xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x1280x720xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.group_norm"(%arg2, %arg1, %arg0) <{channel_dim = 1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 32 : i64, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x128x1280x720xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<1x128x1280x720xbf16> loc(#loc4)
        return %0 : tensor<1x128x1280x720xbf16> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc4 = loc("custom-call.106")
