#dram = #ttnn.buffer_type<dram>
#loc = loc(unknown)
#loc1 = loc("p0.2")
#loc2 = loc("p1.9")
#loc3 = loc("p2.19")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136448, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 2), (3, 0), (4, 0), (2, 6), (7, 7), (0, 4), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 2), (3, 0), (4, 0), (2, 6), (7, 7), (0, 4), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136448, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 0), (0, 0), (2, 0), (3, 0), (0, 4), (7, 4), (1, 4), (5, 4), (4, 4), (6, 7), (2, 4), (3, 4)], dram_bank_to_logical_worker_noc1 = [(7, 0), (0, 0), (2, 0), (3, 0), (0, 4), (7, 4), (1, 4), (5, 4), (4, 4), (6, 7), (2, 4), (3, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136448, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (2, 6), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (2, 6), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136448, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (7, 7), (1, 4), (6, 4), (3, 6), (2, 4), (4, 4), (5, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (7, 7), (1, 4), (6, 4), (3, 6), (2, 4), (4, 4), (5, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073153760, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (3, 7), (1, 4), (7, 4), (6, 4), (2, 4), (4, 4), (5, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (3, 7), (1, 4), (7, 4), (6, 4), (2, 4), (4, 4), (5, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073153760, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (7, 7), (1, 4), (3, 6), (6, 4), (2, 4), (4, 4), (5, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (7, 7), (1, 4), (3, 6), (6, 4), (2, 4), (4, 4), (5, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073153760, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 0), (0, 0), (1, 3), (1, 2), (0, 4), (7, 4), (1, 4), (5, 4), (4, 4), (2, 4), (3, 7), (3, 6)], dram_bank_to_logical_worker_noc1 = [(7, 0), (0, 0), (1, 3), (1, 2), (0, 4), (7, 4), (1, 4), (5, 4), (4, 4), (2, 4), (3, 7), (3, 6)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103776, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073153760, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)]}], [0, 1, 2, 3, 4, 5, 6, 7], [1 : i32, 1 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], [ 0x0x0x0], [<[0, 6, 0], [3, 6, 0]>, <[0, 7, 0], [3, 7, 0]>, <[0, 9, 0], [4, 1, 0]>, <[0, 14, 0], [1, 14, 0]>, <[0, 15, 0], [1, 15, 0]>, <[1, 0, 0], [2, 0, 0]>, <[1, 1, 0], [2, 1, 0]>, <[1, 9, 0], [6, 1, 0]>, <[2, 9, 0], [7, 1, 0]>, <[2, 14, 0], [3, 14, 0]>, <[2, 15, 0], [3, 15, 0]>, <[3, 9, 0], [5, 1, 0]>, <[4, 6, 0], [6, 6, 0]>, <[5, 6, 0], [7, 6, 0]>]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x128xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x128xf32, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 4 + d2, d3), <1x1>, memref<128x1xf32, #system_memory>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 163840 + d1 * 1280 + d2, d3), <1x1>, memref<163840x720xbf16, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 163840 + d1 * 1280 + d2, d3), <1x1>, memref<5120x23x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 163840 + d1 * 1280 + d2, d3), <1x1>, memref<5120x23x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x28800x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x28800x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.104 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.104 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc, ttnn.l1_const_eval_usage = 1024 : ui64} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func private @main_const_eval_0() -> tensor<1x1x1x1xf32, #ttnn_layout> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 9.99999997E-7 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #ttnn_layout> loc(#loc)
        return %1 : tensor<1x1x1x1xf32, #ttnn_layout> loc(#loc)
      } loc(#loc)
      func.func private @main_const_eval_1(%arg0: tensor<128xbf16, #ttnn_layout1> loc(unknown)) -> tensor<1x32x4x1xf32, #ttnn_layout2> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128xbf16, #ttnn_layout1>) -> tensor<128xf32, #ttnn_layout3> loc(#loc)
        %2 = call @cpu_hoisted_const_eval_68e0ee1a(%1) {ttir.cpu_hoist_call} : (tensor<128xf32, #ttnn_layout3>) -> tensor<1x32x4x1xf32, #ttnn_layout4> loc(#loc)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<128xf32, #ttnn_layout3>) -> () loc(#loc)
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x4x1xf32, #ttnn_layout4>) -> tensor<1x32x4x1xf32, #ttnn_layout5> loc(#loc)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout4>) -> () loc(#loc)
        %4 = "ttnn.to_device"(%3, %0) : (tensor<1x32x4x1xf32, #ttnn_layout5>, !ttnn.device) -> tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout5>) -> () loc(#loc)
        return %4 : tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
      } loc(#loc)
      func.func private @main_const_eval_2(%arg0: tensor<128xbf16, #ttnn_layout1> loc(unknown)) -> tensor<1x32x4x1xf32, #ttnn_layout2> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc)
        %1 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128xbf16, #ttnn_layout1>) -> tensor<128xf32, #ttnn_layout3> loc(#loc)
        %2 = call @cpu_hoisted_const_eval_68e0ee1a(%1) {ttir.cpu_hoist_call} : (tensor<128xf32, #ttnn_layout3>) -> tensor<1x32x4x1xf32, #ttnn_layout4> loc(#loc)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<128xf32, #ttnn_layout3>) -> () loc(#loc)
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x4x1xf32, #ttnn_layout4>) -> tensor<1x32x4x1xf32, #ttnn_layout5> loc(#loc)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout4>) -> () loc(#loc)
        %4 = "ttnn.to_device"(%3, %0) : (tensor<1x32x4x1xf32, #ttnn_layout5>, !ttnn.device) -> tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout5>) -> () loc(#loc)
        return %4 : tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
      } loc(#loc)
      func.func @main(%arg0: tensor<128xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "norm.bias"} loc("p0.2"), %arg1: tensor<128xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "norm.weight"} loc("p1.9"), %arg2: tensor<1x128x1280x720xbf16, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x1280x720xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"} loc("p2.19")) -> (tensor<1x128x1280x720xbf16, #ttnn_layout7> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x1280x720xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) attributes {tt.function_type = "forward_device"} {
        %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<1x1x1x1xf32, #ttnn_layout> loc(#loc)
        %1 = ttcore.load_cached(@main_const_eval_1, [%arg0]) : (tensor<128xbf16, #ttnn_layout1>) -> tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<128xbf16, #ttnn_layout1>) -> () loc(#loc)
        %2 = ttcore.load_cached(@main_const_eval_2, [%arg1]) : (tensor<128xbf16, #ttnn_layout1>) -> tensor<1x32x4x1xf32, #ttnn_layout2> loc(#loc)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128xbf16, #ttnn_layout1>) -> () loc(#loc)
        %3 = "ttnn.to_layout"(%arg2) <{layout = #ttnn.layout<tile>}> : (tensor<1x128x1280x720xbf16, #ttnn_layout6>) -> tensor<1x128x1280x720xbf16, #ttnn_layout7> loc(#loc4)
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x128x1280x720xbf16, #ttnn_layout6>) -> () loc(#loc4)
        %4 = "ttnn.typecast"(%3) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1280x720xbf16, #ttnn_layout7>) -> tensor<1x128x1280x720xf32, #ttnn_layout8> loc(#loc5)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x128x1280x720xbf16, #ttnn_layout7>) -> () loc(#loc5)
        %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 32 : i32, 4 : i32, 921600 : i32]}> : (tensor<1x128x1280x720xf32, #ttnn_layout8>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc5)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x128x1280x720xf32, #ttnn_layout8>) -> () loc(#loc5)
        %6 = "ttnn.mean"(%5) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> tensor<1x32x1x1xf32, #ttnn_layout2> loc(#loc19)
        %7 = "ttnn.subtract"(%5, %6) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>, tensor<1x32x1x1xf32, #ttnn_layout2>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc7)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x32x1x1xf32, #ttnn_layout2>) -> () loc(#loc7)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc7)
        %8 = "ttnn.multiply"(%7, %7) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>, tensor<1x32x4x921600xf32, #ttnn_layout9>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc8)
        %9 = "ttnn.mean"(%8) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32, 3 : i32], keep_dim = true}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> tensor<1x32x1x1xf32, #ttnn_layout2> loc(#loc20)
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc20)
        %10 = "ttnn.add"(%9, %0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x1x1xf32, #ttnn_layout2>, tensor<1x1x1x1xf32, #ttnn_layout>) -> tensor<1x32x1x1xf32, #ttnn_layout2> loc(#loc10)
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x32x1x1xf32, #ttnn_layout2>) -> () loc(#loc10)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout>) -> () loc(#loc10)
        %11 = "ttnn.rsqrt"(%10) : (tensor<1x32x1x1xf32, #ttnn_layout2>) -> tensor<1x32x1x1xf32, #ttnn_layout2> loc(#loc11)
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x32x1x1xf32, #ttnn_layout2>) -> () loc(#loc11)
        %12 = "ttnn.multiply"(%7, %11) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>, tensor<1x32x1x1xf32, #ttnn_layout2>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc12)
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x32x1x1xf32, #ttnn_layout2>) -> () loc(#loc12)
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc12)
        %13 = "ttnn.multiply"(%12, %2) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>, tensor<1x32x4x1xf32, #ttnn_layout2>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc13)
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc13)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout2>) -> () loc(#loc13)
        %14 = "ttnn.add"(%13, %1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>, tensor<1x32x4x1xf32, #ttnn_layout2>) -> tensor<1x32x4x921600xf32, #ttnn_layout9> loc(#loc14)
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc14)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x32x4x1xf32, #ttnn_layout2>) -> () loc(#loc14)
        %15 = "ttnn.typecast"(%14) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> tensor<1x32x4x921600xbf16, #ttnn_layout10> loc(#loc15)
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x32x4x921600xf32, #ttnn_layout9>) -> () loc(#loc15)
        %16 = "ttnn.reshape"(%15) <{shape = [1 : i32, 128 : i32, 1280 : i32, 720 : i32]}> : (tensor<1x32x4x921600xbf16, #ttnn_layout10>) -> tensor<1x128x1280x720xbf16, #ttnn_layout7> loc(#loc15)
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1x32x4x921600xbf16, #ttnn_layout10>) -> () loc(#loc15)
        return %16 : tensor<1x128x1280x720xbf16, #ttnn_layout7> loc(#loc)
      } loc(#loc)
      func.func private @cpu_hoisted_const_eval_68e0ee1a(tensor<128xf32, #ttnn_layout3>) -> tensor<1x32x4x1xf32, #ttnn_layout4> attributes {func_hash = "68e0ee1af3c379f04f5470982c3134ba168b824c621b08132d013b9dcce24c1a", tt.function_type = "forward_cpu_declaration"} loc(#loc)
    } loc(#loc)
  } loc(#loc)
  ttcore.cpu_module {
    builtin.module {
      func.func @cpu_hoisted_const_eval_68e0ee1a(%arg0: tensor<128xf32> {bufferization.access = "read"} loc(unknown)) -> tensor<1x32x4x1xf32> attributes {arg_ranks = [1], func_hash = "68e0ee1af3c379f04f5470982c3134ba168b824c621b08132d013b9dcce24c1a", result_ranks = [4], tt.function_type = "forward_cpu"} {
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xf32>) -> tensor<1x128x1x1xf32> loc(#loc21)
        %1 = "ttir.typecast"(%0) <{conservative_folding = false}> : (tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32> loc(#loc18)
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 4 : i32, 1 : i32]}> : (tensor<1x128x1x1xf32>) -> tensor<1x32x4x1xf32> loc(#loc13)
        return %2 : tensor<1x32x4x1xf32> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc4 = loc("convert.21_in_0_layout")
#loc5 = loc("convert.21")
#loc6 = loc("reduce.68")
#loc7 = loc("subtract.88")
#loc8 = loc("multiply.44")
#loc9 = loc("reduce.51")
#loc10 = loc("add.81")
#loc11 = loc("rsqrt.82")
#loc12 = loc("multiply.91")
#loc13 = loc("multiply.95")
#loc14 = loc("add.101")
#loc15 = loc("convert.102")
#loc16 = loc("reshape.8")
#loc17 = loc("reshape.3")
#loc18 = loc("convert.96")
#loc19 = loc("reduce.68_mean"(#loc6))
#loc20 = loc("reduce.51_mean"(#loc9))
#loc21 = loc(fused[#loc16, #loc17])
