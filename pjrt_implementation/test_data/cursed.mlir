module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=8]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32x32xi32> {mhlo.frontend_attributes = {mhlo.sharding = "{devices=[1,8]<=[8]}"}}) -> tensor<32x32xi32> {
    %c = "stablehlo.constant" (dense<1>) : tensor<i32>
    %0 = "stablehlo.broadcast_in_dim" (%c, dims = []) : (tensor<i32>) -> tensor<32x32xi32>
    %1 = "stablehlo.add" (%arg0, %0) : tensor<32x32xi32>
    return %1 : tensor<32x32xi32>
  }
}