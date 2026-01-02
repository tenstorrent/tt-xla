module @SyncTensorsGraph.516 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{{devices=[1,2,1,1]<=[2]},{devices=[1,2,1,1]<=[2]},{replicated},{replicated}}", mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32xi64>, %arg1: tensor<64xf32>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1x32xi64>, %arg4: tensor<128256x3072xbf16>, %arg5: tensor<3072xbf16>, %arg6: tensor<1x8x128x128xbf16>, %arg7: tensor<1024x3072xbf16>, %arg8: tensor<1x8x128x128xbf16>, %arg9: tensor<128256x3072xbf16>, %arg10: tensor<3072x8192xbf16>, %arg11: tensor<8192x3072xbf16>, %arg12: tensor<3072x3072xbf16>, %arg13: tensor<1x32xi64>, %arg14: tensor<i1>, %arg15: tensor<3072x3072xbf16>, %arg16: tensor<3072xbf16>, %arg17: tensor<8192x3072xbf16>, %arg18: tensor<3072xbf16>) -> (tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<32x128256xbf16>, tensor<1x32x128256xbf16>) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x8x128x128xbf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x8x128x128xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<32x128256xbf16>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x128256xbf16>
    return %0, %1, %2, %3 : tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<32x128256xbf16>, tensor<1x32x128256xbf16>
  }
}

