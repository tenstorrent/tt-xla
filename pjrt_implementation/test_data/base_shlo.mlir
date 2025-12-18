#loc1 = loc("p0.3")
module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x8xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"} loc("p0.3")) -> tensor<8x8xi32> {
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc2)
    %1 = stablehlo.add %arg0, %0 : tensor<8x8xi32> loc(#loc3)
    return %1 : tensor<8x8xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("broadcast.5")
#loc3 = loc("add.6")