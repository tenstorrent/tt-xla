
module @SyncTensorsGraph.516 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, mhlo.spmd_output_sharding="{{devices=[1,2,1,1]<=[2]},{devices=[1,2,1,1]<=[2]},{replicated},{replicated}}"
} {
  func.func @main(%arg0: tensor<32xi64>, %arg1: tensor<64xf32>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1x32xi64>, %arg4: tensor<128256x3072xbf16>, %arg5: tensor<3072xbf16>, %arg6: tensor<1x8x128x128xbf16>, %arg7: tensor<1024x3072xbf16>, %arg8: tensor<1x8x128x128xbf16>, %arg9: tensor<128256x3072xbf16>, %arg10: tensor<3072x8192xbf16>, %arg11: tensor<8192x3072xbf16>, %arg12: tensor<3072x3072xbf16>, %arg13: tensor<1x32xi64>, %arg14: tensor<i1>, %arg15: tensor<3072x3072xbf16>, %arg16: tensor<3072xbf16>, %arg17: tensor<8192x3072xbf16>, %arg18: tensor<3072xbf16>) -> (tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<32x128256xbf16>, tensor<1x32x128256xbf16>) {
    %0 = mhlo.copy %arg0 {mhlo.sharding = "{replicated}"} : tensor<32xi64>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<32xi64>
    %2 = mhlo.copy %arg1 {mhlo.sharding = "{replicated}"} : tensor<64xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{manual}"} : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.copy %arg2 {mhlo.sharding = "{devices=[2,1]<=[2]}"} : tensor<1024x3072xbf16>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {mhlo.sharding = "{manual}"} : (tensor<1024x3072xbf16>) -> tensor<512x3072xbf16>
    %6 = mhlo.copy %arg3 {mhlo.sharding = "{replicated}"} : tensor<1x32xi64>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {mhlo.sharding = "{manual}"} : (tensor<1x32xi64>) -> tensor<1x32xi64>
    %8 = mhlo.copy %arg4 {mhlo.sharding = "{replicated}"} : tensor<128256x3072xbf16>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {mhlo.sharding = "{manual}"} : (tensor<128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %10 = mhlo.copy %arg5 {mhlo.sharding = "{replicated}"} : tensor<3072xbf16>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<3072xbf16>
    %12 = mhlo.copy %arg6 {mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : tensor<1x8x128x128xbf16>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {mhlo.sharding = "{manual}"} : (tensor<1x8x128x128xbf16>) -> tensor<1x4x128x128xbf16>
    %14 = mhlo.copy %arg7 {mhlo.sharding = "{devices=[2,1]<=[2]}"} : tensor<1024x3072xbf16>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {mhlo.sharding = "{manual}"} : (tensor<1024x3072xbf16>) -> tensor<512x3072xbf16>
    %16 = mhlo.copy %arg8 {mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : tensor<1x8x128x128xbf16>
    %17 = stablehlo.custom_call @SPMDFullToShardShape(%16) {mhlo.sharding = "{manual}"} : (tensor<1x8x128x128xbf16>) -> tensor<1x4x128x128xbf16>
    %18 = mhlo.copy %arg9 {mhlo.sharding = "{replicated}"} : tensor<128256x3072xbf16>
    %19 = stablehlo.custom_call @SPMDFullToShardShape(%18) {mhlo.sharding = "{manual}"} : (tensor<128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %20 = mhlo.copy %arg10 {mhlo.sharding = "{devices=[1,2]<=[2]}"} : tensor<3072x8192xbf16>
    %21 = stablehlo.custom_call @SPMDFullToShardShape(%20) {mhlo.sharding = "{manual}"} : (tensor<3072x8192xbf16>) -> tensor<3072x4096xbf16>
    %22 = mhlo.copy %arg11 {mhlo.sharding = "{devices=[2,1]<=[2]}"} : tensor<8192x3072xbf16>
    %23 = stablehlo.custom_call @SPMDFullToShardShape(%22) {mhlo.sharding = "{manual}"} : (tensor<8192x3072xbf16>) -> tensor<4096x3072xbf16>
    %24 = mhlo.copy %arg12 {mhlo.sharding = "{devices=[1,2]<=[2]}"} : tensor<3072x3072xbf16>
    %25 = stablehlo.custom_call @SPMDFullToShardShape(%24) {mhlo.sharding = "{manual}"} : (tensor<3072x3072xbf16>) -> tensor<3072x1536xbf16>
    %26 = mhlo.copy %arg13 {mhlo.sharding = "{replicated}"} : tensor<1x32xi64>
    %27 = stablehlo.custom_call @SPMDFullToShardShape(%26) {mhlo.sharding = "{manual}"} : (tensor<1x32xi64>) -> tensor<1x32xi64>
    %28 = mhlo.copy %arg14 {mhlo.sharding = "{replicated}"} : tensor<i1>
    %29 = stablehlo.custom_call @SPMDFullToShardShape(%28) {mhlo.sharding = "{manual}"} : (tensor<i1>) -> tensor<i1>
    %30 = mhlo.copy %arg15 {mhlo.sharding = "{devices=[2,1]<=[2]}"} : tensor<3072x3072xbf16>
    %31 = stablehlo.custom_call @SPMDFullToShardShape(%30) {mhlo.sharding = "{manual}"} : (tensor<3072x3072xbf16>) -> tensor<1536x3072xbf16>
    %32 = mhlo.copy %arg16 {mhlo.sharding = "{replicated}"} : tensor<3072xbf16>
    %33 = stablehlo.custom_call @SPMDFullToShardShape(%32) {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<3072xbf16>
    %34 = mhlo.copy %arg17 {mhlo.sharding = "{devices=[2,1]<=[2]}"} : tensor<8192x3072xbf16>
    %35 = stablehlo.custom_call @SPMDFullToShardShape(%34) {mhlo.sharding = "{manual}"} : (tensor<8192x3072xbf16>) -> tensor<4096x3072xbf16>
    %36 = mhlo.copy %arg18 {mhlo.sharding = "{replicated}"} : tensor<3072xbf16>
    %37 = stablehlo.custom_call @SPMDFullToShardShape(%36) {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<3072xbf16>
    %cst = stablehlo.constant {mhlo.sharding = "{manual}"} dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant {mhlo.sharding = "{manual}"} dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<128xi64>
    %c_0 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>
    %cst_1 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<0xFF800000> : tensor<f32>
    %c_2 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<128> : tensor<i64>
    %cst_3 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<2.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<3.25520843E-4> : tensor<f32>
    %cst_5 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<9.99999974E-6> : tensor<f32>
    %cst_6 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<8.837890e-02> : tensor<bf16>
    %c_7 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<"0xFFFFFFFF000000000000000000000000"> : tensor<128xi1>
    %c_8 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<1> : tensor<i64>
    %cst_9 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<0.000000e+00> : tensor<bf16>
    %cst_10 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<-3.389530e+38> : tensor<bf16>
    %c_11 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<0> : tensor<i64>
    %cst_12 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<"0x000000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C42000080420000824200008442000086420000884200008A4200008C4200008E42000090420000924200009442000096420000984200009A4200009C4200009E420000A0420000A2420000A4420000A6420000A8420000AA420000AC420000AE420000B0420000B2420000B4420000B6420000B8420000BA420000BC420000BE420000C0420000C2420000C4420000C6420000C8420000CA420000CC420000CE420000D0420000D2420000D4420000D6420000D8420000DA420000DC420000DE420000E0420000E2420000E4420000E6420000E8420000EA420000EC420000EE420000F0420000F2420000F4420000F6420000F8420000FA420000FC420000FE42"> : tensor<128xf32>
    %c_13 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<31> : tensor<i64>
    %c_14 = stablehlo.constant {mhlo.sharding = "{manual}"} dense<32> : tensor<i64>
    %38 = stablehlo.broadcast_in_dim %c_14, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<128xi64>
    %39 = stablehlo.broadcast_in_dim %c_13, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<128xi64>
    %40 = stablehlo.broadcast_in_dim %c_11, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<128xi64>
    %41 = stablehlo.broadcast_in_dim %cst_10, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<1x1x32x32xbf16>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<1x1x32x32xbf16>
    %43 = stablehlo.broadcast_in_dim %cst_9, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<32x128xbf16>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<32x128xbf16>
    %45 = stablehlo.broadcast_in_dim %c_8, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<32x128xi64>
    %46 = stablehlo.broadcast_in_dim %cst_9, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<1x1x32x128xbf16>
    %47 = stablehlo.broadcast_in_dim %cst_6, dims = [] {mhlo.sharding = "{manual}"} : (tensor<bf16>) -> tensor<1x12x32x128xbf16>
    %48 = stablehlo.broadcast_in_dim %cst_5, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x1xf32>
    %49 = stablehlo.broadcast_in_dim %cst_4, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32xf32>
    %50 = stablehlo.broadcast_in_dim %cst_3, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x3072xf32>
    %51 = stablehlo.broadcast_in_dim %c_2, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<32xi64>
    %52 = stablehlo.broadcast_in_dim %c_11, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<32xi64>
    %53 = stablehlo.reshape %1 {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<1x1x32xi64>
    %54 = stablehlo.reshape %53 {mhlo.sharding = "{manual}"} : (tensor<1x1x32xi64>) -> tensor<32xi64>
    %55 = stablehlo.compare  LT, %54, %52 {mhlo.sharding = "{manual}"} : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
    %56 = stablehlo.add %54, %51 {mhlo.sharding = "{manual}"} : tensor<32xi64>
    %57 = stablehlo.select %55, %56, %54 {mhlo.sharding = "{manual}"} : tensor<32xi1>, tensor<32xi64>
    %58 = stablehlo.reshape %57 {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<32x1xi64>
    %59 = stablehlo.reshape %11 {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
    %60 = stablehlo.reshape %59 {mhlo.sharding = "{manual}"} : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
    %61 = stablehlo.broadcast_in_dim %60, dims = [2] {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x32x3072xbf16>
    %62 = stablehlo.reshape %9 {mhlo.sharding = "{manual}"} : (tensor<128256x3072xbf16>) -> tensor<1x128256x3072xbf16>
    %63 = stablehlo.reshape %62 {mhlo.sharding = "{manual}"} : (tensor<1x128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %64 = stablehlo.reshape %7 {mhlo.sharding = "{manual}"} : (tensor<1x32xi64>) -> tensor<1x1x32xi64>
    %65 = stablehlo.reshape %64 {mhlo.sharding = "{manual}"} : (tensor<1x1x32xi64>) -> tensor<32xi64>
    %66 = stablehlo.convert %65 {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<32xui32>
    %67 = "stablehlo.gather"(%63, %66) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3072>}> {mhlo.sharding = "{manual}"} : (tensor<128256x3072xbf16>, tensor<32xui32>) -> tensor<32x3072xbf16>
    %68 = stablehlo.reshape %67 {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>) -> tensor<1x32x3072xbf16>
    %69 = stablehlo.convert %68 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<1x32x3072xf32>
    %70 = stablehlo.broadcast_in_dim %cst_3, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x3072xf32>
    %71 = stablehlo.power %69, %70 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %72 = stablehlo.reduce(%71 init: %cst) applies stablehlo.add across dimensions = [2] {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>, tensor<f32>) -> tensor<1x32xf32>
    %73 = stablehlo.broadcast_in_dim %cst_4, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32xf32>
    %74 = stablehlo.multiply %72, %73 {mhlo.sharding = "{manual}"} : tensor<1x32xf32>
    %75 = stablehlo.reshape %74 {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %76 = stablehlo.broadcast_in_dim %cst_5, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x1xf32>
    %77 = stablehlo.add %75, %76 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %78 = stablehlo.rsqrt %77 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %79 = stablehlo.reshape %78 {mhlo.sharding = "{manual}"} : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1] {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x3072xf32>
    %81 = stablehlo.multiply %69, %80 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %82 = stablehlo.convert %81 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>) -> tensor<1x32x3072xbf16>
    %83 = stablehlo.multiply %61, %82 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xbf16>
    %84 = stablehlo.reshape %83 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<32x3072xbf16>
    %85 = stablehlo.reshape %5 {mhlo.sharding = "{manual}"} : (tensor<512x3072xbf16>) -> tensor<1x512x3072xbf16>
    %86 = stablehlo.reshape %85 {mhlo.sharding = "{manual}"} : (tensor<1x512x3072xbf16>) -> tensor<512x3072xbf16>
    %87 = stablehlo.transpose %86, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,1024]{0,1}"} : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
    %88 = stablehlo.dot_general %84, %87, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x512xbf16>) -> tensor<32x512xbf16>
    %89 = stablehlo.reshape %88 {mhlo.sharding = "{manual}"} : (tensor<32x512xbf16>) -> tensor<1x32x4x128xbf16>
    %90 = stablehlo.transpose %89, dims = [0, 2, 1, 3] {mhlo.sharding = "{manual}", result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,8,32,128]{3,1,2,0}"} : (tensor<1x32x4x128xbf16>) -> tensor<1x4x32x128xbf16>
    %91 = stablehlo.reshape %3 {mhlo.sharding = "{manual}"} : (tensor<64xf32>) -> tensor<1x1x64xf32>
    %92 = stablehlo.reshape %91 {mhlo.sharding = "{manual}"} : (tensor<1x1x64xf32>) -> tensor<1x64x1xf32>
    %93 = stablehlo.convert %53 {mhlo.sharding = "{manual}"} : (tensor<1x1x32xi64>) -> tensor<1x1x32xf32>
    %94 = stablehlo.dot_general %92, %93, batching_dims = [0] x [0], contracting_dims = [2] x [1] {mhlo.sharding = "{manual}"} : (tensor<1x64x1xf32>, tensor<1x1x32xf32>) -> tensor<1x64x32xf32>
    %95 = stablehlo.transpose %94, dims = [0, 2, 1] {mhlo.sharding = "{manual}", result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,64]{1,2,0}"} : (tensor<1x64x32xf32>) -> tensor<1x32x64xf32>
    %96 = stablehlo.concatenate %95, %95, dim = 2 {mhlo.sharding = "{manual}"} : (tensor<1x32x64xf32>, tensor<1x32x64xf32>) -> tensor<1x32x128xf32>
    %97 = stablehlo.cosine %96 {mhlo.sharding = "{manual}"} : tensor<1x32x128xf32>
    %98 = stablehlo.convert %97 {mhlo.sharding = "{manual}"} : (tensor<1x32x128xf32>) -> tensor<1x32x128xbf16>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 2, 3] {mhlo.sharding = "{manual}"} : (tensor<1x32x128xbf16>) -> tensor<1x4x32x128xbf16>
    %100 = stablehlo.multiply %90, %99 {mhlo.sharding = "{manual}"} : tensor<1x4x32x128xbf16>
    %101 = stablehlo.slice %90 [0:1, 0:4, 0:32, 64:128] {mhlo.sharding = "{manual}"} : (tensor<1x4x32x128xbf16>) -> tensor<1x4x32x64xbf16>
    %102 = stablehlo.negate %101 {mhlo.sharding = "{manual}"} : tensor<1x4x32x64xbf16>
    %103 = stablehlo.slice %90 [0:1, 0:4, 0:32, 0:64] {mhlo.sharding = "{manual}"} : (tensor<1x4x32x128xbf16>) -> tensor<1x4x32x64xbf16>
    %104 = stablehlo.concatenate %102, %103, dim = 3 {mhlo.sharding = "{manual}"} : (tensor<1x4x32x64xbf16>, tensor<1x4x32x64xbf16>) -> tensor<1x4x32x128xbf16>
    %105 = stablehlo.sine %96 {mhlo.sharding = "{manual}"} : tensor<1x32x128xf32>
    %106 = stablehlo.convert %105 {mhlo.sharding = "{manual}"} : (tensor<1x32x128xf32>) -> tensor<1x32x128xbf16>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 2, 3] {mhlo.sharding = "{manual}"} : (tensor<1x32x128xbf16>) -> tensor<1x4x32x128xbf16>
    %108 = stablehlo.multiply %104, %107 {mhlo.sharding = "{manual}"} : tensor<1x4x32x128xbf16>
    %109 = stablehlo.add %100, %108 {mhlo.sharding = "{manual}"} : tensor<1x4x32x128xbf16>
    %110 = "stablehlo.scatter"(%13, %58, %109) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg19: tensor<bf16>, %arg20: tensor<bf16>):
      stablehlo.return %arg20 : tensor<bf16>
    }) {mhlo.sharding = "{manual}"} : (tensor<1x4x128x128xbf16>, tensor<32x1xi64>, tensor<1x4x32x128xbf16>) -> tensor<1x4x128x128xbf16>
    %111 = stablehlo.reshape %15 {mhlo.sharding = "{manual}"} : (tensor<512x3072xbf16>) -> tensor<1x512x3072xbf16>
    %112 = stablehlo.reshape %111 {mhlo.sharding = "{manual}"} : (tensor<1x512x3072xbf16>) -> tensor<512x3072xbf16>
    %113 = stablehlo.transpose %112, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,1024]{0,1}"} : (tensor<512x3072xbf16>) -> tensor<3072x512xbf16>
    %114 = stablehlo.dot_general %84, %113, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x512xbf16>) -> tensor<32x512xbf16>
    %115 = stablehlo.reshape %114 {mhlo.sharding = "{manual}"} : (tensor<32x512xbf16>) -> tensor<1x32x4x128xbf16>
    %116 = stablehlo.transpose %115, dims = [0, 2, 1, 3] {mhlo.sharding = "{manual}", result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,8,32,128]{3,1,2,0}"} : (tensor<1x32x4x128xbf16>) -> tensor<1x4x32x128xbf16>
    %117 = "stablehlo.scatter"(%17, %58, %116) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg19: tensor<bf16>, %arg20: tensor<bf16>):
      stablehlo.return %arg20 : tensor<bf16>
    }) {mhlo.sharding = "{manual}"} : (tensor<1x4x128x128xbf16>, tensor<32x1xi64>, tensor<1x4x32x128xbf16>) -> tensor<1x4x128x128xbf16>
    %118 = stablehlo.reshape %37 {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
    %119 = stablehlo.reshape %118 {mhlo.sharding = "{manual}"} : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
    %120 = stablehlo.broadcast_in_dim %119, dims = [2] {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x32x3072xbf16>
    %121 = stablehlo.reshape %31 {mhlo.sharding = "{manual}"} : (tensor<1536x3072xbf16>) -> tensor<1x1536x3072xbf16>
    %122 = stablehlo.reshape %121 {mhlo.sharding = "{manual}"} : (tensor<1x1536x3072xbf16>) -> tensor<1536x3072xbf16>
    %123 = stablehlo.transpose %122, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<1536x3072xbf16>) -> tensor<3072x1536xbf16>
    %124 = stablehlo.dot_general %84, %123, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x1536xbf16>) -> tensor<32x1536xbf16>
    %125 = stablehlo.reshape %124 {mhlo.sharding = "{manual}"} : (tensor<32x1536xbf16>) -> tensor<1x32x12x128xbf16>
    %126 = stablehlo.transpose %125, dims = [0, 2, 1, 3] {mhlo.sharding = "{manual}", result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,24,32,128]{3,1,2,0}"} : (tensor<1x32x12x128xbf16>) -> tensor<1x12x32x128xbf16>
    %127 = stablehlo.broadcast_in_dim %98, dims = [0, 2, 3] {mhlo.sharding = "{manual}"} : (tensor<1x32x128xbf16>) -> tensor<1x12x32x128xbf16>
    %128 = stablehlo.multiply %126, %127 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xbf16>
    %129 = stablehlo.slice %126 [0:1, 0:12, 0:32, 64:128] {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xbf16>) -> tensor<1x12x32x64xbf16>
    %130 = stablehlo.negate %129 {mhlo.sharding = "{manual}"} : tensor<1x12x32x64xbf16>
    %131 = stablehlo.slice %126 [0:1, 0:12, 0:32, 0:64] {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xbf16>) -> tensor<1x12x32x64xbf16>
    %132 = stablehlo.concatenate %130, %131, dim = 3 {mhlo.sharding = "{manual}"} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>) -> tensor<1x12x32x128xbf16>
    %133 = stablehlo.broadcast_in_dim %106, dims = [0, 2, 3] {mhlo.sharding = "{manual}"} : (tensor<1x32x128xbf16>) -> tensor<1x12x32x128xbf16>
    %134 = stablehlo.multiply %132, %133 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xbf16>
    %135 = stablehlo.add %128, %134 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xbf16>
    %136 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 3, 4] {mhlo.sharding = "{manual}"} : (tensor<1x4x128x128xbf16>) -> tensor<1x4x3x128x128xbf16>
    %137 = stablehlo.reshape %136 {mhlo.sharding = "{manual}"} : (tensor<1x4x3x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %138 = stablehlo.transpose %137, dims = [0, 1, 3, 2] {mhlo.sharding = "{manual}", result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,24,128,128]{2,3,1,0}"} : (tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %139 = stablehlo.dot_general %135, %138, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}, mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x32x128xbf16>
    %140 = stablehlo.multiply %139, %47 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xbf16>
    %141 = stablehlo.broadcast_in_dim %29, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i1>) -> tensor<128xi1>
    %142 = stablehlo.and %141, %c_7 {mhlo.sharding = "{manual}"} : tensor<128xi1>
    %143 = stablehlo.reshape %142 {mhlo.sharding = "{manual}"} : (tensor<128xi1>) -> tensor<1x1x1x128xi1>
    %144 = stablehlo.reshape %142 {mhlo.sharding = "{manual}"} : (tensor<128xi1>) -> tensor<1x1x128xi1>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1, 3] {mhlo.sharding = "{manual}"} : (tensor<1x1x128xi1>) -> tensor<1x1x32x128xi1>
    %146 = stablehlo.not %143 {mhlo.sharding = "{manual}"} : tensor<1x1x1x128xi1>
    %147 = stablehlo.reshape %146 {mhlo.sharding = "{manual}"} : (tensor<1x1x1x128xi1>) -> tensor<1x1x128xi1>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 3] {mhlo.sharding = "{manual}"} : (tensor<1x1x128xi1>) -> tensor<1x1x32x128xi1>
    %149 = stablehlo.broadcast_in_dim %c, dims = [1] {mhlo.sharding = "{manual}"} : (tensor<128xi64>) -> tensor<32x128xi64>
    %150 = stablehlo.broadcast_in_dim %c_0, dims = [0] {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<32x128xi64>
    %151 = stablehlo.broadcast_in_dim %c, dims = [1] {mhlo.sharding = "{manual}"} : (tensor<128xi64>) -> tensor<32x128xi64>
    %152 = stablehlo.subtract %151, %150 {mhlo.sharding = "{manual}"} : tensor<32x128xi64>
    %153 = stablehlo.compare  GE, %152, %45 {mhlo.sharding = "{manual}"} : (tensor<32x128xi64>, tensor<32x128xi64>) -> tensor<32x128xi1>
    %154 = stablehlo.select %153, %44, %43 {mhlo.sharding = "{manual}"} : tensor<32x128xi1>, tensor<32x128xbf16>
    %155 = stablehlo.broadcast_in_dim %54, dims = [0] {mhlo.sharding = "{manual}"} : (tensor<32xi64>) -> tensor<32x128xi64>
    %156 = stablehlo.compare  GT, %149, %155 {mhlo.sharding = "{manual}"} : (tensor<32x128xi64>, tensor<32x128xi64>) -> tensor<32x128xi1>
    %157 = stablehlo.convert %156 {mhlo.sharding = "{manual}"} : (tensor<32x128xi1>) -> tensor<32x128xbf16>
    %158 = stablehlo.multiply %154, %157 {mhlo.sharding = "{manual}"} : tensor<32x128xbf16>
    %159 = stablehlo.reshape %158 {mhlo.sharding = "{manual}"} : (tensor<32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %160 = stablehlo.slice %159 [0:1, 0:1, 0:32, 0:32] {mhlo.sharding = "{manual}"} : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x32xbf16>
    %161 = stablehlo.reshape %27 {mhlo.sharding = "{manual}"} : (tensor<1x32xi64>) -> tensor<1x1x32xi64>
    %162 = stablehlo.reshape %161 {mhlo.sharding = "{manual}"} : (tensor<1x1x32xi64>) -> tensor<1x1x1x32xi64>
    %163 = stablehlo.convert %162 {mhlo.sharding = "{manual}"} : (tensor<1x1x1x32xi64>) -> tensor<1x1x1x32xbf16>
    %164 = stablehlo.reshape %163 {mhlo.sharding = "{manual}"} : (tensor<1x1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 3] {mhlo.sharding = "{manual}"} : (tensor<1x1x32xbf16>) -> tensor<1x1x32x32xbf16>
    %166 = stablehlo.add %160, %165 {mhlo.sharding = "{manual}"} : tensor<1x1x32x32xbf16>
    %167 = stablehlo.compare  EQ, %166, %42 {mhlo.sharding = "{manual}"} : (tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xi1>
    %168 = stablehlo.select %167, %41, %160 {mhlo.sharding = "{manual}"} : tensor<1x1x32x32xi1>, tensor<1x1x32x32xbf16>
    %169 = stablehlo.floor %cst_12 {mhlo.sharding = "{manual}"} : tensor<128xf32>
    %170 = stablehlo.convert %169 {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xi64>
    %171 = stablehlo.broadcast_in_dim %c_11, dims = [] {mhlo.sharding = "{manual}"} : (tensor<i64>) -> tensor<128xi64>
    %172 = stablehlo.clamp %171, %170, %39 {mhlo.sharding = "{manual}"} : tensor<128xi64>
    %173 = stablehlo.compare  LT, %172, %40 {mhlo.sharding = "{manual}"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %174 = stablehlo.add %172, %38 {mhlo.sharding = "{manual}"} : tensor<128xi64>
    %175 = stablehlo.select %173, %174, %172 {mhlo.sharding = "{manual}"} : tensor<128xi1>, tensor<128xi64>
    %176 = stablehlo.reshape %175 {mhlo.sharding = "{manual}"} : (tensor<128xi64>) -> tensor<128x1xi64>
    %177 = "stablehlo.gather"(%168, %176) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 32, 1>}> {mhlo.sharding = "{manual}"} : (tensor<1x1x32x32xbf16>, tensor<128x1xi64>) -> tensor<1x1x32x128xbf16>
    %178 = stablehlo.select %148, %46, %177 {mhlo.sharding = "{manual}"} : tensor<1x1x32x128xi1>, tensor<1x1x32x128xbf16>
    %179 = stablehlo.select %145, %178, %159 {mhlo.sharding = "{manual}"} : tensor<1x1x32x128xi1>, tensor<1x1x32x128xbf16>
    %180 = stablehlo.reshape %179 {mhlo.sharding = "{manual}"} : (tensor<1x1x32x128xbf16>) -> tensor<1x32x128xbf16>
    %181 = stablehlo.broadcast_in_dim %180, dims = [0, 2, 3] {mhlo.sharding = "{manual}"} : (tensor<1x32x128xbf16>) -> tensor<1x12x32x128xbf16>
    %182 = stablehlo.add %140, %181 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xbf16>
    %183 = stablehlo.convert %182 {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xbf16>) -> tensor<1x12x32x128xf32>
    %184 = stablehlo.reduce(%183 init: %cst_1) applies stablehlo.maximum across dimensions = [3] {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xf32>, tensor<f32>) -> tensor<1x12x32xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2] {mhlo.sharding = "{manual}"} : (tensor<1x12x32xf32>) -> tensor<1x12x32x128xf32>
    %186 = stablehlo.subtract %183, %185 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xf32>
    %187 = stablehlo.exponential %186 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xf32>
    %188 = stablehlo.reduce(%187 init: %cst) applies stablehlo.add across dimensions = [3] {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xf32>, tensor<f32>) -> tensor<1x12x32xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1, 2] {mhlo.sharding = "{manual}"} : (tensor<1x12x32xf32>) -> tensor<1x12x32x128xf32>
    %190 = stablehlo.divide %187, %189 {mhlo.sharding = "{manual}"} : tensor<1x12x32x128xf32>
    %191 = stablehlo.convert %190 {mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xf32>) -> tensor<1x12x32x128xbf16>
    %192 = stablehlo.broadcast_in_dim %117, dims = [0, 1, 3, 4] {mhlo.sharding = "{manual}"} : (tensor<1x4x128x128xbf16>) -> tensor<1x4x3x128x128xbf16>
    %193 = stablehlo.reshape %192 {mhlo.sharding = "{manual}"} : (tensor<1x4x3x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %194 = stablehlo.dot_general %191, %193, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}, mhlo.sharding = "{manual}"} : (tensor<1x12x32x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x32x128xbf16>
    %195 = stablehlo.transpose %194, dims = [0, 2, 1, 3] {mhlo.sharding = "{manual}", result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,24,128]{3,1,2,0}"} : (tensor<1x12x32x128xbf16>) -> tensor<1x32x12x128xbf16>
    %196 = stablehlo.reshape %195 {mhlo.sharding = "{manual}"} : (tensor<1x32x12x128xbf16>) -> tensor<32x1536xbf16>
    %197 = stablehlo.reshape %25 {mhlo.sharding = "{manual}"} : (tensor<3072x1536xbf16>) -> tensor<1x3072x1536xbf16>
    %198 = stablehlo.reshape %197 {mhlo.sharding = "{manual}"} : (tensor<1x3072x1536xbf16>) -> tensor<3072x1536xbf16>
    %199 = stablehlo.transpose %198, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,3072]{0,1}"} : (tensor<3072x1536xbf16>) -> tensor<1536x3072xbf16>
    %200 = stablehlo.dot_general %196, %199, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x1536xbf16>, tensor<1536x3072xbf16>) -> tensor<32x3072xbf16>
    %201 = "stablehlo.all_reduce"(%200) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg19: tensor<bf16>, %arg20: tensor<bf16>):
      %270 = stablehlo.add %arg19, %arg20 {mhlo.sharding = "{manual}"} : tensor<bf16>
      stablehlo.return %270 : tensor<bf16>
    }) {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>) -> tensor<32x3072xbf16>
    %202 = stablehlo.reshape %201 {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>) -> tensor<1x32x3072xbf16>
    %203 = stablehlo.add %68, %202 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xbf16>
    %204 = stablehlo.reshape %33 {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x1x3072xbf16>
    %205 = stablehlo.reshape %204 {mhlo.sharding = "{manual}"} : (tensor<1x1x3072xbf16>) -> tensor<3072xbf16>
    %206 = stablehlo.broadcast_in_dim %205, dims = [2] {mhlo.sharding = "{manual}"} : (tensor<3072xbf16>) -> tensor<1x32x3072xbf16>
    %207 = stablehlo.convert %203 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<1x32x3072xf32>
    %208 = stablehlo.broadcast_in_dim %cst_3, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x3072xf32>
    %209 = stablehlo.power %207, %208 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %210 = stablehlo.reduce(%209 init: %cst) applies stablehlo.add across dimensions = [2] {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>, tensor<f32>) -> tensor<1x32xf32>
    %211 = stablehlo.broadcast_in_dim %cst_4, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32xf32>
    %212 = stablehlo.multiply %210, %211 {mhlo.sharding = "{manual}"} : tensor<1x32xf32>
    %213 = stablehlo.reshape %212 {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %214 = stablehlo.broadcast_in_dim %cst_5, dims = [] {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<1x32x1xf32>
    %215 = stablehlo.add %213, %214 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %216 = stablehlo.rsqrt %215 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %217 = stablehlo.reshape %216 {mhlo.sharding = "{manual}"} : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %218 = stablehlo.broadcast_in_dim %217, dims = [0, 1] {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x3072xf32>
    %219 = stablehlo.multiply %207, %218 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %220 = stablehlo.convert %219 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>) -> tensor<1x32x3072xbf16>
    %221 = stablehlo.multiply %206, %220 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xbf16>
    %222 = stablehlo.reshape %221 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<32x3072xbf16>
    %223 = stablehlo.reshape %35 {mhlo.sharding = "{manual}"} : (tensor<4096x3072xbf16>) -> tensor<1x4096x3072xbf16>
    %224 = stablehlo.reshape %223 {mhlo.sharding = "{manual}"} : (tensor<1x4096x3072xbf16>) -> tensor<4096x3072xbf16>
    %225 = stablehlo.transpose %224, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
    %226 = stablehlo.dot_general %222, %225, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<32x4096xbf16>
    %227 = stablehlo.reshape %226 {mhlo.sharding = "{manual}"} : (tensor<32x4096xbf16>) -> tensor<1x32x4096xbf16>
    %228 = stablehlo.logistic %227 {mhlo.sharding = "{manual}"} : tensor<1x32x4096xbf16>
    %229 = stablehlo.multiply %227, %228 {mhlo.sharding = "{manual}"} : tensor<1x32x4096xbf16>
    %230 = stablehlo.reshape %23 {mhlo.sharding = "{manual}"} : (tensor<4096x3072xbf16>) -> tensor<1x4096x3072xbf16>
    %231 = stablehlo.reshape %230 {mhlo.sharding = "{manual}"} : (tensor<1x4096x3072xbf16>) -> tensor<4096x3072xbf16>
    %232 = stablehlo.transpose %231, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,8192]{0,1}"} : (tensor<4096x3072xbf16>) -> tensor<3072x4096xbf16>
    %233 = stablehlo.dot_general %222, %232, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x4096xbf16>) -> tensor<32x4096xbf16>
    %234 = stablehlo.reshape %233 {mhlo.sharding = "{manual}"} : (tensor<32x4096xbf16>) -> tensor<1x32x4096xbf16>
    %235 = stablehlo.multiply %229, %234 {mhlo.sharding = "{manual}"} : tensor<1x32x4096xbf16>
    %236 = stablehlo.reshape %235 {mhlo.sharding = "{manual}"} : (tensor<1x32x4096xbf16>) -> tensor<32x4096xbf16>
    %237 = stablehlo.reshape %21 {mhlo.sharding = "{manual}"} : (tensor<3072x4096xbf16>) -> tensor<1x3072x4096xbf16>
    %238 = stablehlo.reshape %237 {mhlo.sharding = "{manual}"} : (tensor<1x3072x4096xbf16>) -> tensor<3072x4096xbf16>
    %239 = stablehlo.transpose %238, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[8192,3072]{0,1}"} : (tensor<3072x4096xbf16>) -> tensor<4096x3072xbf16>
    %240 = stablehlo.dot_general %236, %239, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x4096xbf16>, tensor<4096x3072xbf16>) -> tensor<32x3072xbf16>
    %241 = "stablehlo.all_reduce"(%240) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg19: tensor<bf16>, %arg20: tensor<bf16>):
      %270 = stablehlo.add %arg19, %arg20 {mhlo.sharding = "{manual}"} : tensor<bf16>
      stablehlo.return %270 : tensor<bf16>
    }) {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>) -> tensor<32x3072xbf16>
    %242 = stablehlo.reshape %241 {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>) -> tensor<1x32x3072xbf16>
    %243 = stablehlo.add %203, %242 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xbf16>
    %244 = stablehlo.convert %243 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<1x32x3072xf32>
    %245 = stablehlo.power %244, %50 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %246 = stablehlo.reduce(%245 init: %cst) applies stablehlo.add across dimensions = [2] {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>, tensor<f32>) -> tensor<1x32xf32>
    %247 = stablehlo.multiply %246, %49 {mhlo.sharding = "{manual}"} : tensor<1x32xf32>
    %248 = stablehlo.reshape %247 {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
    %249 = stablehlo.add %248, %48 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %250 = stablehlo.rsqrt %249 {mhlo.sharding = "{manual}"} : tensor<1x32x1xf32>
    %251 = stablehlo.reshape %250 {mhlo.sharding = "{manual}"} : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1] {mhlo.sharding = "{manual}"} : (tensor<1x32xf32>) -> tensor<1x32x3072xf32>
    %253 = stablehlo.multiply %244, %252 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xf32>
    %254 = stablehlo.convert %253 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xf32>) -> tensor<1x32x3072xbf16>
    %255 = stablehlo.multiply %120, %254 {mhlo.sharding = "{manual}"} : tensor<1x32x3072xbf16>
    %256 = stablehlo.reshape %255 {mhlo.sharding = "{manual}"} : (tensor<1x32x3072xbf16>) -> tensor<32x3072xbf16>
    %257 = stablehlo.reshape %19 {mhlo.sharding = "{manual}"} : (tensor<128256x3072xbf16>) -> tensor<1x128256x3072xbf16>
    %258 = stablehlo.reshape %257 {mhlo.sharding = "{manual}"} : (tensor<1x128256x3072xbf16>) -> tensor<128256x3072xbf16>
    %259 = stablehlo.transpose %258, dims = [1, 0] {mhlo.sharding = "{manual}", result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[3072,128256]{0,1}"} : (tensor<128256x3072xbf16>) -> tensor<3072x128256xbf16>
    %260 = stablehlo.dot_general %256, %259, contracting_dims = [1] x [0] {mhlo.sharding = "{manual}"} : (tensor<32x3072xbf16>, tensor<3072x128256xbf16>) -> tensor<32x128256xbf16>
    %261 = stablehlo.reshape %260 {mhlo.sharding = "{manual}"} : (tensor<32x128256xbf16>) -> tensor<1x32x128256xbf16>
    %262 = mhlo.copy %110 {mhlo.sharding = "{manual}"} : tensor<1x4x128x128xbf16>
    %263 = stablehlo.custom_call @SPMDShardToFullShape(%262) {mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : (tensor<1x4x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %264 = mhlo.copy %117 {mhlo.sharding = "{manual}"} : tensor<1x4x128x128xbf16>
    %265 = stablehlo.custom_call @SPMDShardToFullShape(%264) {mhlo.sharding = "{devices=[1,2,1,1]<=[2]}"} : (tensor<1x4x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %266 = mhlo.copy %260 {mhlo.sharding = "{manual}"} : tensor<32x128256xbf16>
    %267 = stablehlo.custom_call @SPMDShardToFullShape(%266) {mhlo.sharding = "{replicated}"} : (tensor<32x128256xbf16>) -> tensor<32x128256xbf16>
    %268 = mhlo.copy %261 {mhlo.sharding = "{manual}"} : tensor<1x32x128256xbf16>
    %269 = stablehlo.custom_call @SPMDShardToFullShape(%268) {mhlo.sharding = "{replicated}"} : (tensor<1x32x128256xbf16>) -> tensor<1x32x128256xbf16>
    return %263, %265, %267, %269 : tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>, tensor<32x128256xbf16>, tensor<1x32x128256xbf16>
  }
}

