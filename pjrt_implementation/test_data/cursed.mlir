#loc1 = loc("p0.3")
module @SyncTensorsGraph.9 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, mhlo.spmd_output_sharding = "{devices=[1,2]<=[2]}", mhlo.spmd_parameters_shardings = ["{devices=[1,2]0,1}"]} {
  func.func @main(%arg0: tensor<8x8xi32> {mhlo.sharding = "{devices=[1,2]0,1}"} loc("p0.3")) -> (tensor<8x8xi32> {mhlo.sharding = "{devices=[1,2]0,1}"}) {
    %c = stablehlo.constant dense<1> : tensor<8x8xi32> loc(#loc2)
    %0 = stablehlo.add %arg0, %c : tensor<8x8xi32> loc(#loc3)
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[1,2]0,1}"} : (tensor<8x8xi32>) -> tensor<8x8xi32> loc(#loc4)
    return %1 : tensor<8x8xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("broadcast.5")
#loc3 = loc("add.6")
#loc4 = loc("custom-call.7")