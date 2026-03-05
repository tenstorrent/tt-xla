// SHLO
module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @main(%arg0: tensor<1x1x16xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x1x16xi64>>, ttir.name = "args_0"}, %arg1: tensor<1x16x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>, ttir.name = "args_1"}) -> (tensor<1x16x512xbf16> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>}) {
    %c = stablehlo.constant dense<0> : tensor<1x16x512x1xui32>
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x16xi64>) -> tensor<1x16xi64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x16xi64>) -> tensor<1x16x512xi64>
    %2 = stablehlo.convert %1 : (tensor<1x16x512xi64>) -> tensor<1x16x512xui32>
    %3 = stablehlo.reshape %2 : (tensor<1x16x512xui32>) -> tensor<1x16x512x1xui32>
    %4 = stablehlo.iota dim = 0 : tensor<512xui32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<512xui32>) -> tensor<1x16x512x1xui32>
    %6 = stablehlo.concatenate %c, %3, %5, dim = 3 : (tensor<1x16x512x1xui32>, tensor<1x16x512x1xui32>, tensor<1x16x512x1xui32>) -> tensor<1x16x512x3xui32>
    %7 = "stablehlo.gather"(%arg1, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x16x512xbf16>, tensor<1x16x512x3xui32>) -> tensor<1x16x512xbf16>
    return %7 : tensor<1x16x512xbf16>
  }
}

// ------------------------------------------------------------
// TTIR
module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<1x1x16xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x1x16xi64>>, ttir.name = "args_0"}, %arg1: tensor<1x16x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>, ttir.name = "args_1"}) -> (tensor<1x16x512xbf16> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>}) {
        %0 = "ttir.constant"() <{value = dense<0> : tensor<1x16x512x1xui32>}> : () -> tensor<1x16x512x1xui32>
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 16 : i32]}> : (tensor<1x1x16xi64>) -> tensor<1x16xi64>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 16 : i32, 1 : i32]}> : (tensor<1x16xi64>) -> tensor<1x16x1xi64>
        %3 = "ttir.broadcast"(%2) <{broadcast_dimensions = array<i64: 1, 1, 512>}> : (tensor<1x16x1xi64>) -> tensor<1x16x512xi64>
        %4 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<1x16x512xi64>) -> tensor<1x16x512xui32>
        %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 16 : i32, 512 : i32, 1 : i32]}> : (tensor<1x16x512xui32>) -> tensor<1x16x512x1xui32>
        %6 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 512 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<512xui32>
        %7 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xui32>) -> tensor<1x1x512x1xui32>
        %8 = "ttir.broadcast"(%7) <{broadcast_dimensions = array<i64: 1, 16, 1, 1>}> : (tensor<1x1x512x1xui32>) -> tensor<1x16x512x1xui32>
        %9 = "ttir.concat"(%0, %5, %8) <{dim = 3 : si32}> : (tensor<1x16x512x1xui32>, tensor<1x16x512x1xui32>, tensor<1x16x512x1xui32>) -> tensor<1x16x512x3xui32>
        %10 = "ttir.gather"(%arg1, %9) <{collapsed_slice_dims = array<i64: 0, 1, 2>, index_vector_dim = 3 : si64, indices_are_sorted = false, offset_dims = array<i64>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 1, 1>, start_index_map = array<i64: 0, 1, 2>, start_indices_batching_dims = array<i64>}> : (tensor<1x16x512xbf16>, tensor<1x16x512x3xui32>) -> tensor<1x16x512xbf16>
        return %10 : tensor<1x16x512xbf16>
      }
    }
  }
}

// ------------------------------------------------------------
// TTNN

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1, 2, 3, 4, 5, 6, 7], [1 : i32, 1 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], [ 0x0x0x0], [<[0, 0, 0], [3, 0, 0]>, <[0, 1, 0], [3, 1, 0]>, <[0, 9, 0], [4, 1, 0]>, <[0, 14, 0], [1, 14, 0]>, <[0, 15, 0], [1, 15, 0]>, <[1, 6, 0], [2, 6, 0]>, <[1, 7, 0], [2, 7, 0]>, <[1, 9, 0], [5, 1, 0]>, <[2, 9, 0], [7, 1, 0]>, <[2, 14, 0], [3, 14, 0]>, <[2, 15, 0], [3, 15, 0]>, <[3, 9, 0], [6, 1, 0]>, <[4, 6, 0], [5, 6, 0]>, <[6, 6, 0], [7, 6, 0]>]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 512 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 512 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x16xsi32, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 16 + d1, d2), <1x1>, memref<16x512xbf16, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<256x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 512 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x256x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x256x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8192xui32, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8192x1xbf16, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 8192 + d1, d2), <1x1>, memref<256x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.17 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>
      func.func private @main_const_eval_0() -> tensor<3x1xf32, #ttnn_layout> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[[5.120000e+02], [1.000000e+00], [1.000000e+00]]> : tensor<3x1xf32>}> : (!ttnn.device) -> tensor<3x1xf32, #ttnn_layout>
        return %1 : tensor<3x1xf32, #ttnn_layout>
      }
      func.func private @main_const_eval_1() -> tensor<1x16x512x1xbf16, #ttnn_layout1> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 0 : i32, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<1x16x512x1>}> : (!ttnn.device) -> tensor<1x16x512x1xbf16, #ttnn_layout1>
        return %1 : tensor<1x16x512x1xbf16, #ttnn_layout1>
      }
      func.func private @main_const_eval_2() -> tensor<1x16x512x1xbf16, #ttnn_layout1> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<u32>, end = 512 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<512xui32, #ttnn_layout2>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xui32, #ttnn_layout2>) -> tensor<1x1x512x1xui32, #ttnn_layout3>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512xui32, #ttnn_layout2>) -> ()
        %3 = "ttnn.repeat"(%2) <{repeat_dims = #ttnn.shape<1x16x1x1>}> : (tensor<1x1x512x1xui32, #ttnn_layout3>) -> tensor<1x16x512x1xui32, #ttnn_layout4>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x512x1xui32, #ttnn_layout3>) -> ()
        %4 = "ttnn.typecast"(%3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x512x1xui32, #ttnn_layout4>) -> tensor<1x16x512x1xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x16x512x1xui32, #ttnn_layout4>) -> ()
        return %4 : tensor<1x16x512x1xbf16, #ttnn_layout1>
      }
      func.func @main(%arg0: tensor<1x1x16xsi32, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x1x16xi64>>, ttir.name = "args_0"}, %arg1: tensor<1x16x512xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>, ttir.name = "args_1"}) -> (tensor<1x16x512xbf16, #ttnn_layout7> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1x16x512xbf16>>}) attributes {tt.function_type = "forward_device"} {
        %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<3x1xf32, #ttnn_layout>
        %1 = ttcore.load_cached(@main_const_eval_1, []) : () -> tensor<1x16x512x1xbf16, #ttnn_layout1>
        %2 = ttcore.load_cached(@main_const_eval_2, []) : () -> tensor<1x16x512x1xbf16, #ttnn_layout1>
        %3 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x16xsi32, #ttnn_layout5>) -> tensor<1x1x16xsi32, #ttnn_layout8>
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x16xsi32, #ttnn_layout5>) -> ()
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 16 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x16xsi32, #ttnn_layout8>) -> tensor<1x16x1x1xsi32, #ttnn_layout9>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x16xsi32, #ttnn_layout8>) -> ()
        %5 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x16x1x1xsi32, #ttnn_layout9>) -> tensor<1x16x1x1xui32, #ttnn_layout10>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x16x1x1xsi32, #ttnn_layout9>) -> ()
        %6 = "ttnn.repeat"(%5) <{repeat_dims = #ttnn.shape<1x1x512x1>}> : (tensor<1x16x1x1xui32, #ttnn_layout10>) -> tensor<1x16x512x1xui32, #ttnn_layout4>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x16x1x1xui32, #ttnn_layout10>) -> ()
        %7 = "ttnn.typecast"(%6) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x512x1xui32, #ttnn_layout4>) -> tensor<1x16x512x1xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x16x512x1xui32, #ttnn_layout4>) -> ()
        %8 = "ttnn.concat"(%1, %7, %2) <{dim = 3 : si32}> : (tensor<1x16x512x1xbf16, #ttnn_layout1>, tensor<1x16x512x1xbf16, #ttnn_layout1>, tensor<1x16x512x1xbf16, #ttnn_layout1>) -> tensor<1x16x512x3xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x16x512x1xbf16, #ttnn_layout1>) -> ()
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x16x512x1xbf16, #ttnn_layout1>) -> ()
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x16x512x1xbf16, #ttnn_layout1>) -> ()
        %9 = "ttnn.typecast"(%8) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x16x512x3xbf16, #ttnn_layout1>) -> tensor<1x16x512x3xui32, #ttnn_layout4>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x16x512x3xbf16, #ttnn_layout1>) -> ()
        %10 = "ttnn.to_layout"(%arg1) <{layout = #ttnn.layout<tile>}> : (tensor<1x16x512xbf16, #ttnn_layout6>) -> tensor<1x16x512xbf16, #ttnn_layout7>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x16x512xbf16, #ttnn_layout6>) -> ()
        %11 = "ttnn.permute"(%10) <{permutation = array<i64: 1, 2, 0>}> : (tensor<1x16x512xbf16, #ttnn_layout7>) -> tensor<16x512x1xbf16, #ttnn_layout11>
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x16x512xbf16, #ttnn_layout7>) -> ()
        %12 = "ttnn.reshape"(%11) <{shape = [8192 : i32, 1 : i32]}> : (tensor<16x512x1xbf16, #ttnn_layout11>) -> tensor<8192x1xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<16x512x1xbf16, #ttnn_layout11>) -> ()
        %13 = "ttnn.typecast"(%9) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x16x512x3xui32, #ttnn_layout4>) -> tensor<1x16x512x3xf32, #ttnn_layout13>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x16x512x3xui32, #ttnn_layout4>) -> ()
        %14 = "ttnn.matmul"(%13, %0) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x16x512x3xf32, #ttnn_layout13>, tensor<3x1xf32, #ttnn_layout>) -> tensor<1x16x512x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x16x512x3xf32, #ttnn_layout13>) -> ()
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<3x1xf32, #ttnn_layout>) -> ()
        %15 = "ttnn.reshape"(%14) <{shape = [1 : i32, 8192 : i32]}> : (tensor<1x16x512x1xf32, #ttnn_layout13>) -> tensor<1x8192xf32, #ttnn_layout14>
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x16x512x1xf32, #ttnn_layout13>) -> ()
        %16 = "ttnn.typecast"(%15) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x8192xf32, #ttnn_layout14>) -> tensor<1x8192xui32, #ttnn_layout15>
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1x8192xf32, #ttnn_layout14>) -> ()
        %17 = "ttnn.to_layout"(%16) <{layout = #ttnn.layout<row_major>}> : (tensor<1x8192xui32, #ttnn_layout15>) -> tensor<1x8192xui32, #ttnn_layout16>
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<1x8192xui32, #ttnn_layout15>) -> ()
        %18 = "ttnn.to_layout"(%12) <{layout = #ttnn.layout<row_major>}> : (tensor<8192x1xbf16, #ttnn_layout12>) -> tensor<8192x1xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<8192x1xbf16, #ttnn_layout12>) -> ()
        %19 = "ttnn.embedding"(%17, %18) : (tensor<1x8192xui32, #ttnn_layout16>, tensor<8192x1xbf16, #ttnn_layout17>) -> tensor<1x8192x1xbf16, #ttnn_layout18>
        "ttnn.deallocate"(%18) <{force = false}> : (tensor<8192x1xbf16, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1x8192xui32, #ttnn_layout16>) -> ()
        %20 = "ttnn.reshape"(%19) <{shape = [1 : i32, 16 : i32, 512 : i32]}> : (tensor<1x8192x1xbf16, #ttnn_layout18>) -> tensor<1x16x512xbf16, #ttnn_layout7>
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1x8192x1xbf16, #ttnn_layout18>) -> ()
        return %20 : tensor<1x16x512xbf16, #ttnn_layout7>
      }
    }
  }
}
