#loc1 = loc("p0.3")
module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]> loc(#loc)
  func.func @main(%arg0: tensor<8x8xi32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>} loc("p0.3")) -> (tensor<8x8xi32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"_axis_0"}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}]>] manual_axes={"_axis_0_updated", "_axis_0"} (%arg1: tensor<8x4xi32> loc("p0.3")) {
      %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
      %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x4xi32> loc(#loc2)
      %2 = stablehlo.add %arg1, %1 : tensor<8x4xi32> loc(#loc3)
      sdy.return %2 : tensor<8x4xi32> loc(#loc)
    } : (tensor<8x8xi32>) -> tensor<8x8xi32> loc(#loc)
    return %0 : tensor<8x8xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("broadcast.5")
#loc3 = loc("add.6")