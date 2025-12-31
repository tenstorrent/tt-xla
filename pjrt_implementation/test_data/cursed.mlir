module @SyncTensorsGraph.15 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, mhlo.spmd_output_sharding="{{devices=[1,2]<=[2]},{replicated}}" } {
  func.func @main(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2]0,1}"}, %arg1: tensor<8x8xi64>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<8x8xf32>
    %0 = stablehlo.add %arg0, %cst : tensor<8x8xf32>
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[1,2]0,1}"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = stablehlo.convert %arg1 : (tensor<8x8xi64>) -> tensor<8x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<8x8xf32>
    return %1, %3 : tensor<8x8xf32>, tensor<8x8xf32>
  }
}