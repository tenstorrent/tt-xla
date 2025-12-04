#loc1 = loc("p0.3")
module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=8]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  vhlo.func_v1 @main(%arg0: !vhlo.tensor_v1<32x32x!vhlo.bf16_v1> loc("p0.3")) -> (!vhlo.tensor_v1<32x32x!vhlo.bf16_v1>) {
    %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<1.000000e+00> : tensor<bf16>>}> : () -> !vhlo.tensor_v1<!vhlo.bf16_v1> loc(#loc2)
    %1 = "vhlo.broadcast_in_dim_v1"(%0) <{broadcast_dimensions = #vhlo.tensor_v1<dense<> : tensor<0xi64>>}> : (!vhlo.tensor_v1<!vhlo.bf16_v1>) -> !vhlo.tensor_v1<32x32x!vhlo.bf16_v1> loc(#loc2)
    %2 = "vhlo.add_v1"(%arg0, %1) : (!vhlo.tensor_v1<32x32x!vhlo.bf16_v1>, !vhlo.tensor_v1<32x32x!vhlo.bf16_v1>) -> !vhlo.tensor_v1<32x32x!vhlo.bf16_v1> loc(#loc3)
    "vhlo.return_v1"(%2) : (!vhlo.tensor_v1<32x32x!vhlo.bf16_v1>) -> () loc(#loc)
  } {arg_attrs = #vhlo.array_v1<[#vhlo.dict_v1<{#vhlo.string_v1<"mhlo.frontend_attributes"> = #vhlo.dict_v1<{#vhlo.string_v1<"xla.sdy.sharding"> = #vhlo.string_v1<"#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>">}>, #vhlo.string_v1<"mhlo.sharding"> = #vhlo.string_v1<"{devices=[1,8]<=[8]}">}>]>, res_attrs = #vhlo.array_v1<[#vhlo.dict_v1<{#vhlo.string_v1<"mhlo.frontend_attributes"> = #vhlo.dict_v1<{#vhlo.string_v1<"xla.sdy.sharding"> = #vhlo.string_v1<"#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>">}>, #vhlo.string_v1<"mhlo.sharding"> = #vhlo.string_v1<"{devices=[1,8]<=[8]}">}>]>, sym_visibility = #vhlo.string_v1<"">} loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("broadcast.5")
#loc3 = loc("add.6")