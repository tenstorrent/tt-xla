// -----------------------------------------------------------------------------
// START SHLO MODULE
// -----------------------------------------------------------------------------
module @SyncTensorsGraph.683 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.bias"}, %arg1: tensor<512x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.weight"}, %arg2: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_input_layernorm_weight"}, %arg3: tensor<1x128xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg4: tensor<201088x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_embed_tokens.weight"}, %arg5: tensor<32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_rotary_emb_inv_freq"}, %arg6: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.bias"}, %arg7: tensor<512x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.weight"}, %arg8: tensor<201088x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___lm_head.weight"}, %arg9: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_norm_weight"}, %arg10: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_bias"}, %arg11: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_weight"}, %arg12: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_post_attention_layernorm_weight"}, %arg13: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.bias"}, %arg14: tensor<2880x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.weight"}, %arg15: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_sinks"}, %arg16: tensor<bf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<bf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L['self'].model.lifted_tensor_1"}, %arg17: tensor<1x128xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg18: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg19: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.bias"}, %arg20: tensor<4096x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.weight"}, %arg21: tensor<1x1x32x8xi64> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x32x8xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_expert_mapping"}, %arg22: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj_bias"}, %arg23: tensor<32x2880x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj"}, %arg24: tensor<32x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj_bias"}, %arg25: tensor<32x2880x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj"}) -> (tensor<1x8x127x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x8x127x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x128x201088xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x201088xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
    %0:3 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25) in_shardings=[<@mesh, [{"_axis_1"}]>, <@mesh, [{"_axis_1"}, {"_axis_0"}]>, <@mesh, [{"_axis_0"}]>, <@mesh, [{}, {}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}]>, <@mesh, [{"_axis_1"}]>, <@mesh, [{"_axis_1"}, {"_axis_0"}]>, <@mesh, [{}, {}]>, <@mesh, [{"_axis_0"}]>, <@mesh, [{}]>, <@mesh, [{}, {"_axis_0"}]>, <@mesh, [{"_axis_0"}]>, <@mesh, [{"_axis_0"}]>, <@mesh, [{"_axis_0"}, {"_axis_1"}]>, <@mesh, [{}]>, <@mesh, []>, <@mesh, [{}, {}]>, <@mesh, []>, <@mesh, [{"_axis_1"}]>, <@mesh, [{"_axis_1"}, {"_axis_0"}]>, <@mesh, [{}, {}, {}, {}]>, <@mesh, [{"_axis_0", "_axis_1"}, {}]>, <@mesh, [{"_axis_0", "_axis_1"}, {}, {}]>, <@mesh, [{"_axis_0", "_axis_1"}, {}]>, <@mesh, [{"_axis_0", "_axis_1"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_1"}, {}, {}]>, <@mesh, [{}, {"_axis_1"}, {}, {}]>, <@mesh, [{}, {}, {}]>] manual_axes={"_axis_0", "_axis_1"} (%arg26: tensor<128xbf16>, %arg27: tensor<128x1440xbf16>, %arg28: tensor<1440xbf16>, %arg29: tensor<1x128xi64>, %arg30: tensor<201088x1440xbf16>, %arg31: tensor<32xf32>, %arg32: tensor<128xbf16>, %arg33: tensor<128x1440xbf16>, %arg34: tensor<201088x2880xbf16>, %arg35: tensor<1440xbf16>, %arg36: tensor<32xbf16>, %arg37: tensor<32x1440xbf16>, %arg38: tensor<1440xbf16>, %arg39: tensor<1440xbf16>, %arg40: tensor<1440x1024xbf16>, %arg41: tensor<64xbf16>, %arg42: tensor<bf16>, %arg43: tensor<1x128xi64>, %arg44: tensor<i1>, %arg45: tensor<1024xbf16>, %arg46: tensor<1024x1440xbf16>, %arg47: tensor<1x1x32x8xi64>, %arg48: tensor<4x2880xbf16>, %arg49: tensor<4x2880x2880xbf16>, %arg50: tensor<4x5760xbf16>, %arg51: tensor<4x2880x5760xbf16>) {
      %cst = stablehlo.constant {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} dense<9.99999974E-6> : tensor<f32>
      %cst_0 = stablehlo.constant {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} dense<3.47222231E-4> : tensor<f32>
      %cst_1 = stablehlo.constant {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} dense<2.000000e+00> : tensor<f32>
      %cst_2 = stablehlo.constant {reoutline.comp_attrs = {epsilon = 9.99999974E-6 : f32, normalized_shape = dense<2880> : tensor<1xi64>}, reoutline.group = "composite_tenstorrent.rms_norm.impl_1", reoutline.orig_name = "tenstorrent.rms_norm", reoutline.seed} dense<0.000000e+00> : tensor<f32>
      %c = stablehlo.constant dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<128xi64>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %c_4 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>
      %cst_5 = stablehlo.constant dense<"0x000000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C42000080420000824200008442000086420000884200008A4200008C4200008E42000090420000924200009442000096420000984200009A4200009C4200009E420000A0420000A2420000A4420000A6420000A8420000AA420000AC420000AE420000B0420000B2420000B4420000B6420000B8420000BA420000BC420000BE420000C0420000C2420000C4420000C6420000C8420000CA420000CC420000CE420000D0420000D2420000D4420000D6420000D8420000DA420000DC420000DE420000E0420000E2420000E4420000E6420000E8420000EA420000EC420000EE420000F0420000F2420000F4420000F6420000F8420000FA420000FC420000FE42"> : tensor<1x1x128xf32>
      %cst_6 = stablehlo.constant dense<1.34657359> : tensor<f32>
      %cst_7 = stablehlo.constant dense<1.250000e-01> : tensor<bf16>
      %c_8 = stablehlo.constant dense<"0x80FFFFFFFFFFFFFF81FFFFFFFFFFFFFF82FFFFFFFFFFFFFF83FFFFFFFFFFFFFF84FFFFFFFFFFFFFF85FFFFFFFFFFFFFF86FFFFFFFFFFFFFF87FFFFFFFFFFFFFF88FFFFFFFFFFFFFF89FFFFFFFFFFFFFF8AFFFFFFFFFFFFFF8BFFFFFFFFFFFFFF8CFFFFFFFFFFFFFF8DFFFFFFFFFFFFFF8EFFFFFFFFFFFFFF8FFFFFFFFFFFFFFF90FFFFFFFFFFFFFF91FFFFFFFFFFFFFF92FFFFFFFFFFFFFF93FFFFFFFFFFFFFF94FFFFFFFFFFFFFF95FFFFFFFFFFFFFF96FFFFFFFFFFFFFF97FFFFFFFFFFFFFF98FFFFFFFFFFFFFF99FFFFFFFFFFFFFF9AFFFFFFFFFFFFFF9BFFFFFFFFFFFFFF9CFFFFFFFFFFFFFF9DFFFFFFFFFFFFFF9EFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFA0FFFFFFFFFFFFFFA1FFFFFFFFFFFFFFA2FFFFFFFFFFFFFFA3FFFFFFFFFFFFFFA4FFFFFFFFFFFFFFA5FFFFFFFFFFFFFFA6FFFFFFFFFFFFFFA7FFFFFFFFFFFFFFA8FFFFFFFFFFFFFFA9FFFFFFFFFFFFFFAAFFFFFFFFFFFFFFABFFFFFFFFFFFFFFACFFFFFFFFFFFFFFADFFFFFFFFFFFFFFAEFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFB0FFFFFFFFFFFFFFB1FFFFFFFFFFFFFFB2FFFFFFFFFFFFFFB3FFFFFFFFFFFFFFB4FFFFFFFFFFFFFFB5FFFFFFFFFFFFFFB6FFFFFFFFFFFFFFB7FFFFFFFFFFFFFFB8FFFFFFFFFFFFFFB9FFFFFFFFFFFFFFBAFFFFFFFFFFFFFFBBFFFFFFFFFFFFFFBCFFFFFFFFFFFFFFBDFFFFFFFFFFFFFFBEFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFC0FFFFFFFFFFFFFFC1FFFFFFFFFFFFFFC2FFFFFFFFFFFFFFC3FFFFFFFFFFFFFFC4FFFFFFFFFFFFFFC5FFFFFFFFFFFFFFC6FFFFFFFFFFFFFFC7FFFFFFFFFFFFFFC8FFFFFFFFFFFFFFC9FFFFFFFFFFFFFFCAFFFFFFFFFFFFFFCBFFFFFFFFFFFFFFCCFFFFFFFFFFFFFFCDFFFFFFFFFFFFFFCEFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFD0FFFFFFFFFFFFFFD1FFFFFFFFFFFFFFD2FFFFFFFFFFFFFFD3FFFFFFFFFFFFFFD4FFFFFFFFFFFFFFD5FFFFFFFFFFFFFFD6FFFFFFFFFFFFFFD7FFFFFFFFFFFFFFD8FFFFFFFFFFFFFFD9FFFFFFFFFFFFFFDAFFFFFFFFFFFFFFDBFFFFFFFFFFFFFFDCFFFFFFFFFFFFFFDDFFFFFFFFFFFFFFDEFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFE0FFFFFFFFFFFFFFE1FFFFFFFFFFFFFFE2FFFFFFFFFFFFFFE3FFFFFFFFFFFFFFE4FFFFFFFFFFFFFFE5FFFFFFFFFFFFFFE6FFFFFFFFFFFFFFE7FFFFFFFFFFFFFFE8FFFFFFFFFFFFFFE9FFFFFFFFFFFFFFEAFFFFFFFFFFFFFFEBFFFFFFFFFFFFFFECFFFFFFFFFFFFFFEDFFFFFFFFFFFFFFEEFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF0FFFFFFFFFFFFFFF1FFFFFFFFFFFFFFF2FFFFFFFFFFFFFFF3FFFFFFFFFFFFFFF4FFFFFFFFFFFFFFF5FFFFFFFFFFFFFFF6FFFFFFFFFFFFFFF7FFFFFFFFFFFFFFF8FFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"> : tensor<128xi64>
      %c_9 = stablehlo.constant dense<"0x0000000000000000000000000000000000000000000000000100000000000000000000000000000002000000000000000000000000000000030000000000000000000000000000000400000000000000000000000000000005000000000000000000000000000000060000000000000000000000000000000700000000000000000000000000000008000000000000000000000000000000090000000000000000000000000000000A0000000000000000000000000000000B0000000000000000000000000000000C0000000000000000000000000000000D0000000000000000000000000000000E0000000000000000000000000000000F000000000000000000000000000000100000000000000000000000000000001100000000000000000000000000000012000000000000000000000000000000130000000000000000000000000000001400000000000000000000000000000015000000000000000000000000000000160000000000000000000000000000001700000000000000000000000000000018000000000000000000000000000000190000000000000000000000000000001A0000000000000000000000000000001B0000000000000000000000000000001C0000000000000000000000000000001D0000000000000000000000000000001E0000000000000000000000000000001F000000000000000000000000000000200000000000000000000000000000002100000000000000000000000000000022000000000000000000000000000000230000000000000000000000000000002400000000000000000000000000000025000000000000000000000000000000260000000000000000000000000000002700000000000000000000000000000028000000000000000000000000000000290000000000000000000000000000002A0000000000000000000000000000002B0000000000000000000000000000002C0000000000000000000000000000002D0000000000000000000000000000002E0000000000000000000000000000002F000000000000000000000000000000300000000000000000000000000000003100000000000000000000000000000032000000000000000000000000000000330000000000000000000000000000003400000000000000000000000000000035000000000000000000000000000000360000000000000000000000000000003700000000000000000000000000000038000000000000000000000000000000390000000000000000000000000000003A0000000000000000000000000000003B0000000000000000000000000000003C0000000000000000000000000000003D0000000000000000000000000000003E0000000000000000000000000000003F000000000000000000000000000000400000000000000000000000000000004100000000000000000000000000000042000000000000000000000000000000430000000000000000000000000000004400000000000000000000000000000045000000000000000000000000000000460000000000000000000000000000004700000000000000000000000000000048000000000000000000000000000000490000000000000000000000000000004A0000000000000000000000000000004B0000000000000000000000000000004C0000000000000000000000000000004D0000000000000000000000000000004E0000000000000000000000000000004F000000000000000000000000000000500000000000000000000000000000005100000000000000000000000000000052000000000000000000000000000000530000000000000000000000000000005400000000000000000000000000000055000000000000000000000000000000560000000000000000000000000000005700000000000000000000000000000058000000000000000000000000000000590000000000000000000000000000005A0000000000000000000000000000005B0000000000000000000000000000005C0000000000000000000000000000005D0000000000000000000000000000005E0000000000000000000000000000005F000000000000000000000000000000600000000000000000000000000000006100000000000000000000000000000062000000000000000000000000000000630000000000000000000000000000006400000000000000000000000000000065000000000000000000000000000000660000000000000000000000000000006700000000000000000000000000000068000000000000000000000000000000690000000000000000000000000000006A0000000000000000000000000000006B0000000000000000000000000000006C0000000000000000000000000000006D0000000000000000000000000000006E0000000000000000000000000000006F000000000000000000000000000000700000000000000000000000000000007100000000000000000000000000000072000000000000000000000000000000730000000000000000000000000000007400000000000000000000000000000075000000000000000000000000000000760000000000000000000000000000007700000000000000000000000000000078000000000000000000000000000000790000000000000000000000000000007A0000000000000000000000000000007B0000000000000000000000000000007C0000000000000000000000000000007D0000000000000000000000000000007E0000000000000000000000000000007F00000000000000"> : tensor<1x128x2xi64>
      %cst_10 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %cst_12 = stablehlo.constant dense<-7.000000e+00> : tensor<bf16>
      %cst_13 = stablehlo.constant dense<7.000000e+00> : tensor<bf16>
      %cst_14 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
      %cst_15 = stablehlo.constant dense<0xFF80> : tensor<bf16>
      %cst_16 = stablehlo.constant dense<1.703130e+00> : tensor<bf16>
      %1 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %2 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %3 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %4 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %5 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %6 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<bf16>) -> tensor<128x32xbf16>
      %7 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<bf16>) -> tensor<1x1x128x128xbf16>
      %8 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<1x16x128x128xbf16>
      %9 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x32xf32>
      %10 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<1x128xi64>
      %11 = stablehlo.reshape %arg30 : (tensor<201088x1440xbf16>) -> tensor<1x201088x1440xbf16>
      %12 = stablehlo.reshape %11 : (tensor<1x201088x1440xbf16>) -> tensor<201088x1440xbf16>
      %13 = stablehlo.reshape %arg29 : (tensor<1x128xi64>) -> tensor<1x1x128xi64>
      %14 = stablehlo.reshape %13 : (tensor<1x1x128xi64>) -> tensor<128xi64>
      %15 = stablehlo.convert %14 : (tensor<128xi64>) -> tensor<128xui32>
      %16 = "stablehlo.gather"(%12, %15) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1440>}> : (tensor<201088x1440xbf16>, tensor<128xui32>) -> tensor<128x1440xbf16>
      %17 = stablehlo.reshape %16 : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
      %18 = stablehlo.reshape %arg28 : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
      %19 = stablehlo.reshape %18 : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
      %20 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<f32>) -> tensor<1x128x1xf32>
      %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<f32>) -> tensor<1x128xf32>
      %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<f32>) -> tensor<1x128x1440xf32>
      %23 = stablehlo.convert %17 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
      %24 = stablehlo.power %23, %22 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128x1440xf32>
      %25 = stablehlo.reduce(%24 init: %cst_2) applies stablehlo.add across dimensions = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128x1440xf32>, tensor<f32>) -> tensor<1x128xf32>
      %26 = "stablehlo.all_reduce"(%25) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<f32>, %arg53: tensor<f32>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<f32>
        stablehlo.return %304 : tensor<f32>
      }) : (tensor<1x128xf32>) -> tensor<1x128xf32>
      %27 = stablehlo.multiply %26, %21 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128xf32>
      %28 = stablehlo.reshape %27 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
      %29 = stablehlo.add %28, %20 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128x1xf32>
      %30 = stablehlo.rsqrt %29 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128x1xf32>
      %31 = stablehlo.reshape %30 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
      %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128xf32>) -> tensor<1x128x1440xf32>
      %33 = stablehlo.multiply %23, %32 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128x1440xf32>
      %34 = stablehlo.convert %19 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1440xbf16>) -> tensor<1440xf32>
      %35 = stablehlo.broadcast_in_dim %34, dims = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1440xf32>) -> tensor<1x128x1440xf32>
      %36 = stablehlo.multiply %33, %35 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : tensor<1x128x1440xf32>
      %37 = stablehlo.convert %36 {reoutline.group = "composite_tenstorrent.rms_norm.impl_1"} : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
      %38 = stablehlo.reshape %37 : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
      %39 = stablehlo.reshape %arg46 : (tensor<1024x1440xbf16>) -> tensor<1x1024x1440xbf16>
      %40 = stablehlo.reshape %39 : (tensor<1x1024x1440xbf16>) -> tensor<1024x1440xbf16>
      %41 = stablehlo.transpose %40, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2880,4096]{0,1}"} : (tensor<1024x1440xbf16>) -> tensor<1440x1024xbf16>
      %42 = stablehlo.dot_general %38, %41, contracting_dims = [1] x [0] : (tensor<128x1440xbf16>, tensor<1440x1024xbf16>) -> tensor<128x1024xbf16>
      %43 = "stablehlo.all_reduce"(%42) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<bf16>
        stablehlo.return %304 : tensor<bf16>
      }) : (tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
      %44 = stablehlo.reshape %43 : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
      %45 = stablehlo.reshape %arg45 : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
      %46 = stablehlo.reshape %45 : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
      %47 = stablehlo.broadcast_in_dim %46, dims = [2] : (tensor<1024xbf16>) -> tensor<1x128x1024xbf16>
      %48 = stablehlo.add %44, %47 : tensor<1x128x1024xbf16>
      %49 = stablehlo.reshape %48 : (tensor<1x128x1024xbf16>) -> tensor<1x128x16x64xbf16>
      %50 = stablehlo.transpose %49, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,64,128,64]{3,1,2,0}"} : (tensor<1x128x16x64xbf16>) -> tensor<1x16x128x64xbf16>
      %51 = stablehlo.slice %50 [0:1, 0:16, 0:128, 0:32] : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
      %52 = stablehlo.reshape %arg31 : (tensor<32xf32>) -> tensor<1x1x32xf32>
      %53 = stablehlo.reshape %52 : (tensor<1x1x32xf32>) -> tensor<1x32x1xf32>
      %54 = stablehlo.dot_general %53, %cst_5, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x32x1xf32>, tensor<1x1x128xf32>) -> tensor<1x32x128xf32>
      %55 = stablehlo.transpose %54, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,128,32]{1,2,0}"} : (tensor<1x32x128xf32>) -> tensor<1x128x32xf32>
      %56 = stablehlo.cosine %55 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,128,32]{1,2,0}"} : tensor<1x128x32xf32>
      %57 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x128x32xf32>
      %58 = stablehlo.multiply %56, %57 : tensor<1x128x32xf32>
      %59 = stablehlo.convert %58 : (tensor<1x128x32xf32>) -> tensor<1x128x32xbf16>
      %60 = stablehlo.broadcast_in_dim %59, dims = [0, 2, 3] : (tensor<1x128x32xbf16>) -> tensor<1x16x128x32xbf16>
      %61 = stablehlo.multiply %51, %60 : tensor<1x16x128x32xbf16>
      %62 = stablehlo.slice %50 [0:1, 0:16, 0:128, 32:64] : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
      %63 = stablehlo.sine %55 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,128,32]{1,2,0}"} : tensor<1x128x32xf32>
      %64 = stablehlo.multiply %63, %9 : tensor<1x128x32xf32>
      %65 = stablehlo.convert %64 : (tensor<1x128x32xf32>) -> tensor<1x128x32xbf16>
      %66 = stablehlo.broadcast_in_dim %65, dims = [0, 2, 3] : (tensor<1x128x32xbf16>) -> tensor<1x16x128x32xbf16>
      %67 = stablehlo.multiply %62, %66 : tensor<1x16x128x32xbf16>
      %68 = stablehlo.subtract %61, %67 : tensor<1x16x128x32xbf16>
      %69 = stablehlo.multiply %62, %60 : tensor<1x16x128x32xbf16>
      %70 = stablehlo.multiply %51, %66 : tensor<1x16x128x32xbf16>
      %71 = stablehlo.add %69, %70 : tensor<1x16x128x32xbf16>
      %72 = stablehlo.concatenate %68, %71, dim = 3 : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x64xbf16>
      %73 = stablehlo.reshape %arg33 : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
      %74 = stablehlo.reshape %73 : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
      %75 = stablehlo.transpose %74, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2880,512]{0,1}"} : (tensor<128x1440xbf16>) -> tensor<1440x128xbf16>
      %76 = stablehlo.dot_general %38, %75, contracting_dims = [1] x [0] : (tensor<128x1440xbf16>, tensor<1440x128xbf16>) -> tensor<128x128xbf16>
      %77 = "stablehlo.all_reduce"(%76) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<bf16>
        stablehlo.return %304 : tensor<bf16>
      }) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %78 = stablehlo.reshape %77 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %79 = stablehlo.reshape %arg32 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %80 = stablehlo.reshape %79 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %81 = stablehlo.broadcast_in_dim %80, dims = [2] : (tensor<128xbf16>) -> tensor<1x128x128xbf16>
      %82 = stablehlo.add %78, %81 : tensor<1x128x128xbf16>
      %83 = stablehlo.reshape %82 : (tensor<1x128x128xbf16>) -> tensor<1x128x2x64xbf16>
      %84 = stablehlo.transpose %83, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,8,128,64]{3,1,2,0}"} : (tensor<1x128x2x64xbf16>) -> tensor<1x2x128x64xbf16>
      %85 = stablehlo.slice %84 [0:1, 0:2, 0:128, 0:32] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x128x32xbf16>
      %86 = stablehlo.broadcast_in_dim %59, dims = [0, 2, 3] : (tensor<1x128x32xbf16>) -> tensor<1x2x128x32xbf16>
      %87 = stablehlo.multiply %85, %86 : tensor<1x2x128x32xbf16>
      %88 = stablehlo.slice %84 [0:1, 0:2, 0:128, 32:64] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x128x32xbf16>
      %89 = stablehlo.broadcast_in_dim %65, dims = [0, 2, 3] : (tensor<1x128x32xbf16>) -> tensor<1x2x128x32xbf16>
      %90 = stablehlo.multiply %88, %89 : tensor<1x2x128x32xbf16>
      %91 = stablehlo.subtract %87, %90 : tensor<1x2x128x32xbf16>
      %92 = stablehlo.multiply %88, %86 : tensor<1x2x128x32xbf16>
      %93 = stablehlo.multiply %85, %89 : tensor<1x2x128x32xbf16>
      %94 = stablehlo.add %92, %93 : tensor<1x2x128x32xbf16>
      %95 = stablehlo.concatenate %91, %94, dim = 3 : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x64xbf16>
      %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 3, 4] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x8x128x64xbf16>
      %97 = stablehlo.reshape %96 : (tensor<1x2x8x128x64xbf16>) -> tensor<1x16x128x64xbf16>
      %98 = stablehlo.transpose %97, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "bf16[1,64,64,128]{2,3,1,0}"} : (tensor<1x16x128x64xbf16>) -> tensor<1x16x64x128xbf16>
      %99 = stablehlo.dot_general %72, %98, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x16x128x64xbf16>, tensor<1x16x64x128xbf16>) -> tensor<1x16x128x128xbf16>
      %100 = stablehlo.multiply %99, %8 : tensor<1x16x128x128xbf16>
      %101 = stablehlo.broadcast_in_dim %arg44, dims = [] : (tensor<i1>) -> tensor<128x128xi1>
      %102 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<128xi64>) -> tensor<128x128xi64>
      %103 = stablehlo.broadcast_in_dim %c_8, dims = [0] : (tensor<128xi64>) -> tensor<128x128xi64>
      %104 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<128xi64>) -> tensor<128x128xi64>
      %105 = stablehlo.compare  GT, %104, %103 : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
      %106 = stablehlo.and %101, %105 : tensor<128x128xi1>
      %107 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<128xi64>) -> tensor<128x128xi64>
      %108 = stablehlo.compare  LE, %102, %107 : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
      %109 = stablehlo.and %106, %108 : tensor<128x128xi1>
      %110 = stablehlo.and %101, %109 : tensor<128x128xi1>
      %111 = stablehlo.reshape %110 : (tensor<128x128xi1>) -> tensor<1x128x128xi1>
      %112 = stablehlo.reshape %arg43 : (tensor<1x128xi64>) -> tensor<1x1x128xi64>
      %113 = stablehlo.reshape %112 : (tensor<1x1x128xi64>) -> tensor<1x128xi64>
      %114 = stablehlo.compare  NE, %113, %10 : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1>
      %115 = "stablehlo.gather"(%114, %c_9) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<1x128xi1>, tensor<1x128x2xi64>) -> tensor<1x128xi1>
      %116 = stablehlo.broadcast_in_dim %115, dims = [0, 2] : (tensor<1x128xi1>) -> tensor<1x128x128xi1>
      %117 = stablehlo.and %111, %116 : tensor<1x128x128xi1>
      %118 = stablehlo.reshape %117 : (tensor<1x128x128xi1>) -> tensor<1x1x128x128xi1>
      %119 = stablehlo.reshape %arg42 : (tensor<bf16>) -> tensor<1x1x1xbf16>
      %120 = stablehlo.reshape %119 : (tensor<1x1x1xbf16>) -> tensor<1x1xbf16>
      %121 = stablehlo.broadcast_in_dim %120, dims = [0, 1] : (tensor<1x1xbf16>) -> tensor<1x1x128x128xbf16>
      %122 = stablehlo.select %118, %121, %7 : tensor<1x1x128x128xi1>, tensor<1x1x128x128xbf16>
      %123 = stablehlo.reshape %122 : (tensor<1x1x128x128xbf16>) -> tensor<1x128x128xbf16>
      %124 = stablehlo.broadcast_in_dim %123, dims = [0, 2, 3] : (tensor<1x128x128xbf16>) -> tensor<1x16x128x128xbf16>
      %125 = stablehlo.add %100, %124 : tensor<1x16x128x128xbf16>
      %126 = stablehlo.composite "sdy.all_slice" %arg41 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1"}]>}, decomposition = @sdy.all_slice1} : (tensor<64xbf16>) -> tensor<16xbf16>
      %127 = stablehlo.reshape %126 : (tensor<16xbf16>) -> tensor<1x1x16xbf16>
      %128 = stablehlo.reshape %127 : (tensor<1x1x16xbf16>) -> tensor<1x16x1xbf16>
      %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 3] : (tensor<1x16x1xbf16>) -> tensor<1x16x128x1xbf16>
      %130 = stablehlo.concatenate %125, %129, dim = 3 : (tensor<1x16x128x128xbf16>, tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
      %131 = stablehlo.iota dim = 0 : tensor<128xi64>
      %132 = stablehlo.broadcast_in_dim %131, dims = [0] : (tensor<128xi64>) -> tensor<128x4x1xi64>
      %133 = stablehlo.reduce(%130 init: %cst_15) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x128x129xbf16>, tensor<bf16>) -> tensor<1x16x128xbf16>
      %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<1x16x128xbf16>) -> tensor<1x16x128x129xbf16>
      %135 = stablehlo.subtract %130, %134 : tensor<1x16x128x129xbf16>
      %136 = stablehlo.reduce(%135 init: %cst_15) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x128x129xbf16>, tensor<bf16>) -> tensor<1x16x128xbf16>
      %137 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x16x128xbf16>) -> tensor<1x16x128x129xbf16>
      %138 = stablehlo.subtract %135, %137 : tensor<1x16x128x129xbf16>
      %139 = stablehlo.exponential %138 : tensor<1x16x128x129xbf16>
      %140 = stablehlo.reduce(%139 init: %cst_11) applies stablehlo.add across dimensions = [3] : (tensor<1x16x128x129xbf16>, tensor<bf16>) -> tensor<1x16x128xbf16>
      %141 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x16x128xbf16>) -> tensor<1x16x128x129xbf16>
      %142 = stablehlo.divide %139, %141 : tensor<1x16x128x129xbf16>
      %143 = stablehlo.slice %142 [0:1, 0:16, 0:128, 0:128] : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128x128xbf16>
      %144 = stablehlo.reshape %arg27 : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
      %145 = stablehlo.reshape %144 : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
      %146 = stablehlo.transpose %145, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2880,512]{0,1}"} : (tensor<128x1440xbf16>) -> tensor<1440x128xbf16>
      %147 = stablehlo.dot_general %38, %146, contracting_dims = [1] x [0] : (tensor<128x1440xbf16>, tensor<1440x128xbf16>) -> tensor<128x128xbf16>
      %148 = "stablehlo.all_reduce"(%147) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<bf16>
        stablehlo.return %304 : tensor<bf16>
      }) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %149 = stablehlo.reshape %148 : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
      %150 = stablehlo.reshape %arg26 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
      %151 = stablehlo.reshape %150 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
      %152 = stablehlo.broadcast_in_dim %151, dims = [2] : (tensor<128xbf16>) -> tensor<1x128x128xbf16>
      %153 = stablehlo.add %149, %152 : tensor<1x128x128xbf16>
      %154 = stablehlo.reshape %153 : (tensor<1x128x128xbf16>) -> tensor<1x128x2x64xbf16>
      %155 = stablehlo.transpose %154, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,8,128,64]{3,1,2,0}"} : (tensor<1x128x2x64xbf16>) -> tensor<1x2x128x64xbf16>
      %156 = stablehlo.broadcast_in_dim %155, dims = [0, 1, 3, 4] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x8x128x64xbf16>
      %157 = stablehlo.reshape %156 : (tensor<1x2x8x128x64xbf16>) -> tensor<1x16x128x64xbf16>
      %158 = stablehlo.dot_general %143, %157, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x16x128x128xbf16>, tensor<1x16x128x64xbf16>) -> tensor<1x16x128x64xbf16>
      %159 = stablehlo.transpose %158, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,128,64,64]{3,1,2,0}"} : (tensor<1x16x128x64xbf16>) -> tensor<1x128x16x64xbf16>
      %160 = stablehlo.reshape %159 : (tensor<1x128x16x64xbf16>) -> tensor<128x1024xbf16>
      %161 = stablehlo.reshape %arg40 : (tensor<1440x1024xbf16>) -> tensor<1x1440x1024xbf16>
      %162 = stablehlo.reshape %161 : (tensor<1x1440x1024xbf16>) -> tensor<1440x1024xbf16>
      %163 = stablehlo.transpose %162, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[4096,2880]{0,1}"} : (tensor<1440x1024xbf16>) -> tensor<1024x1440xbf16>
      %164 = stablehlo.dot_general %160, %163, contracting_dims = [1] x [0] : (tensor<128x1024xbf16>, tensor<1024x1440xbf16>) -> tensor<128x1440xbf16>
      %165 = "stablehlo.all_reduce"(%164) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<bf16>
        stablehlo.return %304 : tensor<bf16>
      }) : (tensor<128x1440xbf16>) -> tensor<128x1440xbf16>
      %166 = stablehlo.reshape %165 : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
      %167 = stablehlo.reshape %arg39 : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
      %168 = stablehlo.reshape %167 : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
      %169 = stablehlo.broadcast_in_dim %168, dims = [2] : (tensor<1440xbf16>) -> tensor<1x128x1440xbf16>
      %170 = stablehlo.add %166, %169 : tensor<1x128x1440xbf16>
      %171 = stablehlo.add %17, %170 : tensor<1x128x1440xbf16>
      %172 = stablehlo.reshape %arg38 : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
      %173 = stablehlo.reshape %172 : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
      %174 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<f32>) -> tensor<1x128x1xf32>
      %175 = stablehlo.broadcast_in_dim %cst_0, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<f32>) -> tensor<1x128xf32>
      %176 = stablehlo.broadcast_in_dim %cst_1, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<f32>) -> tensor<1x128x1440xf32>
      %177 = stablehlo.convert %171 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
      %178 = stablehlo.power %177, %176 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128x1440xf32>
      %179 = stablehlo.reduce(%178 init: %cst_2) applies stablehlo.add across dimensions = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128x1440xf32>, tensor<f32>) -> tensor<1x128xf32>
      %180 = "stablehlo.all_reduce"(%179) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<f32>, %arg53: tensor<f32>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<f32>
        stablehlo.return %304 : tensor<f32>
      }) : (tensor<1x128xf32>) -> tensor<1x128xf32>
      %181 = stablehlo.multiply %180, %175 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128xf32>
      %182 = stablehlo.reshape %181 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
      %183 = stablehlo.add %182, %174 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128x1xf32>
      %184 = stablehlo.rsqrt %183 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128x1xf32>
      %185 = stablehlo.reshape %184 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
      %186 = stablehlo.broadcast_in_dim %185, dims = [0, 1] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128xf32>) -> tensor<1x128x1440xf32>
      %187 = stablehlo.multiply %177, %186 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128x1440xf32>
      %188 = stablehlo.convert %173 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1440xbf16>) -> tensor<1440xf32>
      %189 = stablehlo.broadcast_in_dim %188, dims = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1440xf32>) -> tensor<1x128x1440xf32>
      %190 = stablehlo.multiply %187, %189 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : tensor<1x128x1440xf32>
      %191 = stablehlo.convert %190 {reoutline.group = "composite_tenstorrent.rms_norm.impl_0"} : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
      %192 = stablehlo.reshape %191 : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
      %193 = stablehlo.convert %192 : (tensor<128x1440xbf16>) -> tensor<128x1440xf32>
      %194 = stablehlo.reshape %arg37 : (tensor<32x1440xbf16>) -> tensor<1x32x1440xbf16>
      %195 = stablehlo.reshape %194 : (tensor<1x32x1440xbf16>) -> tensor<32x1440xbf16>
      %196 = stablehlo.transpose %195, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2880,32]{0,1}"} : (tensor<32x1440xbf16>) -> tensor<1440x32xbf16>
      %197 = stablehlo.convert %196 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2880,32]{0,1}"} : (tensor<1440x32xbf16>) -> tensor<1440x32xf32>
      %198 = stablehlo.dot_general %193, %197, contracting_dims = [1] x [0] : (tensor<128x1440xf32>, tensor<1440x32xf32>) -> tensor<128x32xf32>
      %199 = "stablehlo.all_reduce"(%198) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<f32>, %arg53: tensor<f32>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<f32>
        stablehlo.return %304 : tensor<f32>
      }) : (tensor<128x32xf32>) -> tensor<128x32xf32>
      %200 = stablehlo.reshape %arg36 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
      %201 = stablehlo.reshape %200 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
      %202 = stablehlo.convert %201 : (tensor<32xbf16>) -> tensor<32xf32>
      %203 = stablehlo.broadcast_in_dim %202, dims = [1] : (tensor<32xf32>) -> tensor<128x32xf32>
      %204 = stablehlo.add %199, %203 : tensor<128x32xf32>
      %205 = stablehlo.convert %204 : (tensor<128x32xf32>) -> tensor<128x32xbf16>
      %206 = stablehlo.iota dim = 0 : tensor<32xi32>
      %207 = stablehlo.broadcast_in_dim %206, dims = [1] : (tensor<32xi32>) -> tensor<128x32xi32>
      %208:2 = "stablehlo.sort"(%205, %207) <{dimension = 1 : i64}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>, %arg54: tensor<i32>, %arg55: tensor<i32>):
        %304 = stablehlo.compare  GT, %arg52, %arg53,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
        stablehlo.return %304 : tensor<i1>
      }) : (tensor<128x32xbf16>, tensor<128x32xi32>) -> (tensor<128x32xbf16>, tensor<128x32xi32>)
      %209 = stablehlo.slice %208#1 [0:128, 0:4] : (tensor<128x32xi32>) -> tensor<128x4xi32>
      %210 = stablehlo.convert %209 : (tensor<128x4xi32>) -> tensor<128x4xi64>
      %211 = stablehlo.reshape %210 : (tensor<128x4xi64>) -> tensor<128x4x1xi64>
      %212 = stablehlo.concatenate %132, %211, dim = 2 : (tensor<128x4x1xi64>, tensor<128x4x1xi64>) -> tensor<128x4x2xi64>
      %213 = stablehlo.slice %208#0 [0:128, 0:4] : (tensor<128x32xbf16>) -> tensor<128x4xbf16>
      %214 = stablehlo.reduce(%213 init: %cst_15) applies stablehlo.maximum across dimensions = [1] : (tensor<128x4xbf16>, tensor<bf16>) -> tensor<128xbf16>
      %215 = stablehlo.broadcast_in_dim %214, dims = [0] : (tensor<128xbf16>) -> tensor<128x4xbf16>
      %216 = stablehlo.subtract %213, %215 : tensor<128x4xbf16>
      %217 = stablehlo.exponential %216 : tensor<128x4xbf16>
      %218 = stablehlo.reduce(%217 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<128x4xbf16>, tensor<bf16>) -> tensor<128xbf16>
      %219 = stablehlo.broadcast_in_dim %218, dims = [0] : (tensor<128xbf16>) -> tensor<128x4xbf16>
      %220 = stablehlo.divide %217, %219 : tensor<128x4xbf16>
      %221 = "stablehlo.scatter"(%6, %212, %220) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        stablehlo.return %arg53 : tensor<bf16>
      }) : (tensor<128x32xbf16>, tensor<128x4x2xi64>, tensor<128x4xbf16>) -> tensor<128x32xbf16>
      %222 = stablehlo.broadcast_in_dim %221, dims = [1, 2] : (tensor<128x32xbf16>) -> tensor<2x128x32xbf16>
      %223 = stablehlo.reshape %222 : (tensor<2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
      %224 = stablehlo.reshape %191 : (tensor<1x128x1440xbf16>) -> tensor<1x1x128x1440xbf16>
      %225 = stablehlo.reshape %210 : (tensor<128x4xi64>) -> tensor<1x1x128x4xi64>
      %226 = "stablehlo.all_gather"(%224) <{all_gather_dim = 3 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x128x1440xbf16>) -> tensor<1x1x128x2880xbf16>
      %227:2 = stablehlo.custom_call @tt.all_to_all_dispatch(%226, %225, %arg47) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2"}, xla_shape = "(bf16[1,2,128,2880]{3,2,1,0}, s64[1,2,128,4]{3,2,1,0})"} : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
      %228:2 = stablehlo.custom_call @tt.moe_expert_token_remap(%223, %arg47, %227#1) {api_version = 0 : i32, mhlo.frontend_attributes = {reduction_size = "32"}, xla_shape = "(bf16[1,2,128,32]{3,2,1,0}, bf16[1,1,8,32]{3,2,1,0})"} : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
      %229 = stablehlo.slice %155 [0:1, 0:2, 1:128, 0:64] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x127x64xbf16>
      %230 = stablehlo.slice %95 [0:1, 0:2, 1:128, 0:64] : (tensor<1x2x128x64xbf16>) -> tensor<1x2x127x64xbf16>
      %231 = stablehlo.reshape %227#0 : (tensor<1x2x128x2880xbf16>) -> tensor<2x4x32x2880xbf16>
      %232 = stablehlo.reshape %arg51 : (tensor<4x2880x5760xbf16>) -> tensor<1x4x2880x5760xbf16>
      %233 = stablehlo.reshape %228#1 : (tensor<1x1x8x4xbf16>) -> tensor<2x4x1x4xbf16>
      %234 = stablehlo.custom_call @tt.sparse_matmul(%231, %232, %233) {api_version = 0 : i32, mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}} : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
      %235 = stablehlo.reshape %234 : (tensor<2x4x1x4x32x5760xbf16>) -> tensor<2x4x4x32x5760xbf16>
      %236 = stablehlo.transpose %235, dims = [0, 1, 3, 2, 4] {result_layout = dense<[4, 2, 3, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[2,4,32,32,5760]{4,2,3,1,0}"} : (tensor<2x4x4x32x5760xbf16>) -> tensor<2x4x32x4x5760xbf16>
      %237 = stablehlo.reshape %arg50 : (tensor<4x5760xbf16>) -> tensor<1x4x5760xbf16>
      %238 = stablehlo.reshape %237 : (tensor<1x4x5760xbf16>) -> tensor<4x5760xbf16>
      %239 = stablehlo.broadcast_in_dim %238, dims = [3, 4] : (tensor<4x5760xbf16>) -> tensor<2x4x32x4x5760xbf16>
      %240 = stablehlo.add %236, %239 : tensor<2x4x32x4x5760xbf16>
      %241 = stablehlo.slice %240 [0:2, 0:4, 0:32, 0:4, 1:5760:2] : (tensor<2x4x32x4x5760xbf16>) -> tensor<2x4x32x4x2880xbf16>
      %242 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<bf16>) -> tensor<2x4x32x4x2880xbf16>
      %243 = stablehlo.clamp %5, %241, %242 : tensor<2x4x32x4x2880xbf16>
      %244 = stablehlo.add %243, %3 : tensor<2x4x32x4x2880xbf16>
      %245 = stablehlo.slice %240 [0:2, 0:4, 0:32, 0:4, 0:5760:2] : (tensor<2x4x32x4x5760xbf16>) -> tensor<2x4x32x4x2880xbf16>
      %246 = stablehlo.clamp %2, %245, %4 : tensor<2x4x32x4x2880xbf16>
      %247 = stablehlo.multiply %246, %1 : tensor<2x4x32x4x2880xbf16>
      %248 = stablehlo.logistic %247 : tensor<2x4x32x4x2880xbf16>
      %249 = stablehlo.multiply %246, %248 : tensor<2x4x32x4x2880xbf16>
      %250 = stablehlo.multiply %244, %249 : tensor<2x4x32x4x2880xbf16>
      %251 = stablehlo.transpose %250, dims = [0, 1, 3, 2, 4] {result_layout = dense<[4, 2, 3, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[2,4,32,32,2880]{4,2,3,1,0}"} : (tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x4x32x2880xbf16>
      %252 = stablehlo.reshape %251 : (tensor<2x4x4x32x2880xbf16>) -> tensor<8x4x32x2880xbf16>
      %253 = stablehlo.reshape %arg49 : (tensor<4x2880x2880xbf16>) -> tensor<1x4x2880x2880xbf16>
      %254 = stablehlo.custom_call @tt.sparse_matmul(%252, %253, %228#1) {api_version = 0 : i32, mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}} : (tensor<8x4x32x2880xbf16>, tensor<1x4x2880x2880xbf16>, tensor<1x1x8x4xbf16>) -> tensor<8x4x32x2880xbf16>
      %255 = stablehlo.reshape %254 : (tensor<8x4x32x2880xbf16>) -> tensor<2x4x4x32x2880xbf16>
      %256 = stablehlo.transpose %255, dims = [0, 1, 3, 2, 4] {result_layout = dense<[4, 2, 3, 1, 0]> : tensor<5xindex>, xla_shape = "bf16[2,4,32,32,2880]{4,2,3,1,0}"} : (tensor<2x4x4x32x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
      %257 = stablehlo.reshape %arg48 : (tensor<4x2880xbf16>) -> tensor<1x4x2880xbf16>
      %258 = stablehlo.reshape %257 : (tensor<1x4x2880xbf16>) -> tensor<4x2880xbf16>
      %259 = stablehlo.broadcast_in_dim %258, dims = [3, 4] : (tensor<4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
      %260 = stablehlo.add %256, %259 : tensor<2x4x32x4x2880xbf16>
      %261 = stablehlo.transpose %260, dims = [3, 0, 1, 2, 4] {result_layout = dense<[4, 0, 3, 2, 1]> : tensor<5xindex>, xla_shape = "bf16[32,2,4,32,2880]{4,0,3,2,1}"} : (tensor<2x4x32x4x2880xbf16>) -> tensor<4x2x4x32x2880xbf16>
      %262 = stablehlo.reshape %261 : (tensor<4x2x4x32x2880xbf16>) -> tensor<4x2x128x2880xbf16>
      %263 = stablehlo.custom_call @tt.all_to_all_combine(%262, %227#1, %arg47) {api_version = 0 : i32, mhlo.frontend_attributes = {cluster_axis = "0", num_devices = "2", num_experts_per_tok = "4", output_shard_dim = "1"}} : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
      %264 = stablehlo.composite "sdy.all_slice" %263 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"_axis_0"}]>}, decomposition = @sdy.all_slice2} : (tensor<4x1x128x2880xbf16>) -> tensor<4x1x128x1440xbf16>
      %265 = stablehlo.broadcast_in_dim %210, dims = [0, 1] : (tensor<128x4xi64>) -> tensor<128x4x32xi64>
      %266 = stablehlo.broadcast_in_dim %c_4, dims = [2] : (tensor<32xi64>) -> tensor<128x4x32xi64>
      %267 = stablehlo.compare  EQ, %265, %266 : (tensor<128x4x32xi64>, tensor<128x4x32xi64>) -> tensor<128x4x32xi1>
      %268 = stablehlo.convert %267 : (tensor<128x4x32xi1>) -> tensor<128x4x32xbf16>
      %269 = stablehlo.dot_general %268, %221, batching_dims = [0] x [0], contracting_dims = [2] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<128x4x32xbf16>, tensor<128x32xbf16>) -> tensor<128x4xbf16>
      %270 = stablehlo.reshape %269 : (tensor<128x4xbf16>) -> tensor<1x128x4xbf16>
      %271 = stablehlo.transpose %270, dims = [2, 0, 1] {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "bf16[4,1,128]{0,2,1}"} : (tensor<1x128x4xbf16>) -> tensor<4x1x128xbf16>
      %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<4x1x128xbf16>) -> tensor<4x1x128x1440xbf16>
      %273 = stablehlo.multiply %264, %272 : tensor<4x1x128x1440xbf16>
      %274 = stablehlo.reduce(%273 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<4x1x128x1440xbf16>, tensor<bf16>) -> tensor<1x128x1440xbf16>
      %275 = stablehlo.add %171, %274 : tensor<1x128x1440xbf16>
      %276 = stablehlo.reshape %arg35 : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
      %277 = stablehlo.reshape %276 : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
      %278 = stablehlo.broadcast_in_dim %cst, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<f32>) -> tensor<1x128x1xf32>
      %279 = stablehlo.broadcast_in_dim %cst_0, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<f32>) -> tensor<1x128xf32>
      %280 = stablehlo.broadcast_in_dim %cst_1, dims = [] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<f32>) -> tensor<1x128x1440xf32>
      %281 = stablehlo.convert %275 {reoutline.arg_operand_indices = array<i64: 0>, reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
      %282 = stablehlo.power %281, %280 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128x1440xf32>
      %283 = stablehlo.reduce(%282 init: %cst_2) applies stablehlo.add across dimensions = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128x1440xf32>, tensor<f32>) -> tensor<1x128xf32>
      %284 = "stablehlo.all_reduce"(%283) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<f32>, %arg53: tensor<f32>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<f32>
        stablehlo.return %304 : tensor<f32>
      }) : (tensor<1x128xf32>) -> tensor<1x128xf32>
      %285 = stablehlo.multiply %284, %279 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128xf32>
      %286 = stablehlo.reshape %285 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
      %287 = stablehlo.add %286, %278 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128x1xf32>
      %288 = stablehlo.rsqrt %287 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128x1xf32>
      %289 = stablehlo.reshape %288 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
      %290 = stablehlo.broadcast_in_dim %289, dims = [0, 1] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128xf32>) -> tensor<1x128x1440xf32>
      %291 = stablehlo.multiply %281, %290 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128x1440xf32>
      %292 = stablehlo.convert %277 {reoutline.arg_operand_indices = array<i64: 1>, reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1440xbf16>) -> tensor<1440xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [2] {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1440xf32>) -> tensor<1x128x1440xf32>
      %294 = stablehlo.multiply %291, %293 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : tensor<1x128x1440xf32>
      %295 = stablehlo.convert %294 {reoutline.group = "composite_tenstorrent.rms_norm.impl"} : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
      %296 = stablehlo.reshape %295 : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
      %297 = stablehlo.composite "sdy.all_slice" %arg34 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}, decomposition = @sdy.all_slice3} : (tensor<201088x2880xbf16>) -> tensor<201088x1440xbf16>
      %298 = stablehlo.reshape %297 : (tensor<201088x1440xbf16>) -> tensor<1x201088x1440xbf16>
      %299 = stablehlo.reshape %298 : (tensor<1x201088x1440xbf16>) -> tensor<201088x1440xbf16>
      %300 = stablehlo.transpose %299, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2880,201088]{0,1}"} : (tensor<201088x1440xbf16>) -> tensor<1440x201088xbf16>
      %301 = stablehlo.dot_general %296, %300, contracting_dims = [1] x [0] : (tensor<128x1440xbf16>, tensor<1440x201088xbf16>) -> tensor<128x201088xbf16>
      %302 = "stablehlo.all_reduce"(%301) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> ({
      ^bb0(%arg52: tensor<bf16>, %arg53: tensor<bf16>):
        %304 = stablehlo.add %arg52, %arg53 : tensor<bf16>
        stablehlo.return %304 : tensor<bf16>
      }) : (tensor<128x201088xbf16>) -> tensor<128x201088xbf16>
      %303 = stablehlo.reshape %302 : (tensor<128x201088xbf16>) -> tensor<1x128x201088xbf16>
      sdy.return %229, %230, %303 : tensor<1x2x127x64xbf16>, tensor<1x2x127x64xbf16>, tensor<1x128x201088xbf16>
    } : (tensor<512xbf16>, tensor<512x2880xbf16>, tensor<2880xbf16>, tensor<1x128xi64>, tensor<201088x2880xbf16>, tensor<32xf32>, tensor<512xbf16>, tensor<512x2880xbf16>, tensor<201088x2880xbf16>, tensor<2880xbf16>, tensor<32xbf16>, tensor<32x2880xbf16>, tensor<2880xbf16>, tensor<2880xbf16>, tensor<2880x4096xbf16>, tensor<64xbf16>, tensor<bf16>, tensor<1x128xi64>, tensor<i1>, tensor<4096xbf16>, tensor<4096x2880xbf16>, tensor<1x1x32x8xi64>, tensor<32x2880xbf16>, tensor<32x2880x2880xbf16>, tensor<32x5760xbf16>, tensor<32x2880x5760xbf16>) -> (tensor<1x8x127x64xbf16>, tensor<1x8x127x64xbf16>, tensor<1x128x201088xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x8x127x64xbf16>, tensor<1x8x127x64xbf16>, tensor<1x128x201088xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64xbf16>) -> tensor<16xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<64xbf16>) -> tensor<4x16xbf16>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x16xbf16>) -> tensor<4x16xbf16>
    %2 = stablehlo.slice %1 [0:1, 0:16] : (tensor<4x16xbf16>) -> tensor<1x16xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x16xbf16>) -> tensor<16xbf16>
    return %3 : tensor<16xbf16>
  }
  func.func private @sdy.all_slice2(%arg0: tensor<4x1x128x2880xbf16>) -> tensor<4x1x128x1440xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<4x1x128x2880xbf16>) -> tensor<4x1x128x2x1440xbf16>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 3 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 3 : i64}> : (tensor<4x1x128x2x1440xbf16>) -> tensor<4x1x128x2x1440xbf16>
    %2 = stablehlo.slice %1 [0:4, 0:1, 0:128, 0:1, 0:1440] : (tensor<4x1x128x2x1440xbf16>) -> tensor<4x1x128x1x1440xbf16>
    %3 = stablehlo.reshape %2 : (tensor<4x1x128x1x1440xbf16>) -> tensor<4x1x128x1440xbf16>
    return %3 : tensor<4x1x128x1440xbf16>
  }
  func.func private @sdy.all_slice3(%arg0: tensor<201088x2880xbf16>) -> tensor<201088x1440xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<201088x2880xbf16>) -> tensor<201088x2x1440xbf16>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<201088x2x1440xbf16>) -> tensor<201088x2x1440xbf16>
    %2 = stablehlo.slice %1 [0:201088, 0:1, 0:1440] : (tensor<201088x2x1440xbf16>) -> tensor<201088x1x1440xbf16>
    %3 = stablehlo.reshape %2 : (tensor<201088x1x1440xbf16>) -> tensor<201088x1440xbf16>
    return %3 : tensor<201088x1440xbf16>
  }
}
// -----------------------------------------------------------------------------
// END SHLO MODULE
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// START TTIR MODULE
// -----------------------------------------------------------------------------
module @SyncTensorsGraph.683 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.683 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
      func.func @main(%arg0: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.bias"}, %arg1: tensor<512x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.weight"}, %arg2: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_input_layernorm_weight"}, %arg3: tensor<1x128xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg4: tensor<201088x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_embed_tokens.weight"}, %arg5: tensor<32xf32> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_rotary_emb_inv_freq"}, %arg6: tensor<512xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.bias"}, %arg7: tensor<512x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.weight"}, %arg8: tensor<201088x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___lm_head.weight"}, %arg9: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_norm_weight"}, %arg10: tensor<32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_bias"}, %arg11: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_weight"}, %arg12: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_post_attention_layernorm_weight"}, %arg13: tensor<2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.bias"}, %arg14: tensor<2880x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.weight"}, %arg15: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_sinks"}, %arg16: tensor<bf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<bf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L['self'].model.lifted_tensor_1"}, %arg17: tensor<1x128xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg18: tensor<i1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg19: tensor<4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.bias"}, %arg20: tensor<4096x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.weight"}, %arg21: tensor<1x1x32x8xi64> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x32x8xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_expert_mapping"}, %arg22: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj_bias"}, %arg23: tensor<32x2880x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj"}, %arg24: tensor<32x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj_bias"}, %arg25: tensor<32x2880x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj"}) -> (tensor<1x8x127x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x8x127x64xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x128x201088xbf16> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x201088xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16>) -> tensor<128xbf16>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512x2880xbf16>) -> tensor<128x1440xbf16>
        %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16>) -> tensor<1440xbf16>
        %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x128xi64>) -> tensor<1x128xi64>
        %4 = "ttir.mesh_shard"(%arg4) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<201088x2880xbf16>) -> tensor<201088x1440xbf16>
        %5 = "ttir.mesh_shard"(%arg5) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32xf32>) -> tensor<32xf32>
        %6 = "ttir.mesh_shard"(%arg6) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16>) -> tensor<128xbf16>
        %7 = "ttir.mesh_shard"(%arg7) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512x2880xbf16>) -> tensor<128x1440xbf16>
        %8 = "ttir.mesh_shard"(%arg8) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<201088x2880xbf16>) -> tensor<201088x2880xbf16>
        %9 = "ttir.mesh_shard"(%arg9) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16>) -> tensor<1440xbf16>
        %10 = "ttir.mesh_shard"(%arg10) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32xbf16>) -> tensor<32xbf16>
        %11 = "ttir.mesh_shard"(%arg11) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880xbf16>) -> tensor<32x1440xbf16>
        %12 = "ttir.mesh_shard"(%arg12) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16>) -> tensor<1440xbf16>
        %13 = "ttir.mesh_shard"(%arg13) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16>) -> tensor<1440xbf16>
        %14 = "ttir.mesh_shard"(%arg14) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880x4096xbf16>) -> tensor<1440x1024xbf16>
        %15 = "ttir.mesh_shard"(%arg15) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64xbf16>) -> tensor<64xbf16>
        %16 = "ttir.mesh_shard"(%arg16) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<bf16>) -> tensor<bf16>
        %17 = "ttir.mesh_shard"(%arg17) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x128xi64>) -> tensor<1x128xi64>
        %18 = "ttir.mesh_shard"(%arg18) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<i1>) -> tensor<i1>
        %19 = "ttir.mesh_shard"(%arg19) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xbf16>) -> tensor<1024xbf16>
        %20 = "ttir.mesh_shard"(%arg20) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x2880xbf16>) -> tensor<1024x1440xbf16>
        %21 = "ttir.mesh_shard"(%arg21) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x1x32x8xi64>) -> tensor<1x1x32x8xi64>
        %22 = "ttir.mesh_shard"(%arg22) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880xbf16>) -> tensor<4x2880xbf16>
        %23 = "ttir.mesh_shard"(%arg23) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880x2880xbf16>) -> tensor<4x2880x2880xbf16>
        %24 = "ttir.mesh_shard"(%arg24) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x5760xbf16>) -> tensor<4x5760xbf16>
        %25 = "ttir.mesh_shard"(%arg25) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880x5760xbf16>) -> tensor<4x2880x5760xbf16>
        %26 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
        %27 = "ttir.constant"() <{value = dense<3.47222231E-4> : tensor<f32>}> : () -> tensor<f32>
        %28 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %29 = "ttir.constant"() <{value = dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F00000000000000"> : tensor<128xi64>}> : () -> tensor<128xi64>
        %30 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %31 = "ttir.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>}> : () -> tensor<32xi64>
        %32 = "ttir.constant"() <{value = dense<"0x000000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C42000080420000824200008442000086420000884200008A4200008C4200008E42000090420000924200009442000096420000984200009A4200009C4200009E420000A0420000A2420000A4420000A6420000A8420000AA420000AC420000AE420000B0420000B2420000B4420000B6420000B8420000BA420000BC420000BE420000C0420000C2420000C4420000C6420000C8420000CA420000CC420000CE420000D0420000D2420000D4420000D6420000D8420000DA420000DC420000DE420000E0420000E2420000E4420000E6420000E8420000EA420000EC420000EE420000F0420000F2420000F4420000F6420000F8420000FA420000FC420000FE42"> : tensor<1x1x128xf32>}> : () -> tensor<1x1x128xf32>
        %33 = "ttir.constant"() <{value = dense<1.34657359> : tensor<f32>}> : () -> tensor<f32>
        %34 = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
        %35 = "ttir.constant"() <{value = dense<"0x80FFFFFFFFFFFFFF81FFFFFFFFFFFFFF82FFFFFFFFFFFFFF83FFFFFFFFFFFFFF84FFFFFFFFFFFFFF85FFFFFFFFFFFFFF86FFFFFFFFFFFFFF87FFFFFFFFFFFFFF88FFFFFFFFFFFFFF89FFFFFFFFFFFFFF8AFFFFFFFFFFFFFF8BFFFFFFFFFFFFFF8CFFFFFFFFFFFFFF8DFFFFFFFFFFFFFF8EFFFFFFFFFFFFFF8FFFFFFFFFFFFFFF90FFFFFFFFFFFFFF91FFFFFFFFFFFFFF92FFFFFFFFFFFFFF93FFFFFFFFFFFFFF94FFFFFFFFFFFFFF95FFFFFFFFFFFFFF96FFFFFFFFFFFFFF97FFFFFFFFFFFFFF98FFFFFFFFFFFFFF99FFFFFFFFFFFFFF9AFFFFFFFFFFFFFF9BFFFFFFFFFFFFFF9CFFFFFFFFFFFFFF9DFFFFFFFFFFFFFF9EFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFA0FFFFFFFFFFFFFFA1FFFFFFFFFFFFFFA2FFFFFFFFFFFFFFA3FFFFFFFFFFFFFFA4FFFFFFFFFFFFFFA5FFFFFFFFFFFFFFA6FFFFFFFFFFFFFFA7FFFFFFFFFFFFFFA8FFFFFFFFFFFFFFA9FFFFFFFFFFFFFFAAFFFFFFFFFFFFFFABFFFFFFFFFFFFFFACFFFFFFFFFFFFFFADFFFFFFFFFFFFFFAEFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFB0FFFFFFFFFFFFFFB1FFFFFFFFFFFFFFB2FFFFFFFFFFFFFFB3FFFFFFFFFFFFFFB4FFFFFFFFFFFFFFB5FFFFFFFFFFFFFFB6FFFFFFFFFFFFFFB7FFFFFFFFFFFFFFB8FFFFFFFFFFFFFFB9FFFFFFFFFFFFFFBAFFFFFFFFFFFFFFBBFFFFFFFFFFFFFFBCFFFFFFFFFFFFFFBDFFFFFFFFFFFFFFBEFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFC0FFFFFFFFFFFFFFC1FFFFFFFFFFFFFFC2FFFFFFFFFFFFFFC3FFFFFFFFFFFFFFC4FFFFFFFFFFFFFFC5FFFFFFFFFFFFFFC6FFFFFFFFFFFFFFC7FFFFFFFFFFFFFFC8FFFFFFFFFFFFFFC9FFFFFFFFFFFFFFCAFFFFFFFFFFFFFFCBFFFFFFFFFFFFFFCCFFFFFFFFFFFFFFCDFFFFFFFFFFFFFFCEFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFD0FFFFFFFFFFFFFFD1FFFFFFFFFFFFFFD2FFFFFFFFFFFFFFD3FFFFFFFFFFFFFFD4FFFFFFFFFFFFFFD5FFFFFFFFFFFFFFD6FFFFFFFFFFFFFFD7FFFFFFFFFFFFFFD8FFFFFFFFFFFFFFD9FFFFFFFFFFFFFFDAFFFFFFFFFFFFFFDBFFFFFFFFFFFFFFDCFFFFFFFFFFFFFFDDFFFFFFFFFFFFFFDEFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFE0FFFFFFFFFFFFFFE1FFFFFFFFFFFFFFE2FFFFFFFFFFFFFFE3FFFFFFFFFFFFFFE4FFFFFFFFFFFFFFE5FFFFFFFFFFFFFFE6FFFFFFFFFFFFFFE7FFFFFFFFFFFFFFE8FFFFFFFFFFFFFFE9FFFFFFFFFFFFFFEAFFFFFFFFFFFFFFEBFFFFFFFFFFFFFFECFFFFFFFFFFFFFFEDFFFFFFFFFFFFFFEEFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF0FFFFFFFFFFFFFFF1FFFFFFFFFFFFFFF2FFFFFFFFFFFFFFF3FFFFFFFFFFFFFFF4FFFFFFFFFFFFFFF5FFFFFFFFFFFFFFF6FFFFFFFFFFFFFFF7FFFFFFFFFFFFFFF8FFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"> : tensor<128xi64>}> : () -> tensor<128xi64>
        %36 = "ttir.constant"() <{value = dense<"0x0000000000000000000000000000000000000000000000000100000000000000000000000000000002000000000000000000000000000000030000000000000000000000000000000400000000000000000000000000000005000000000000000000000000000000060000000000000000000000000000000700000000000000000000000000000008000000000000000000000000000000090000000000000000000000000000000A0000000000000000000000000000000B0000000000000000000000000000000C0000000000000000000000000000000D0000000000000000000000000000000E0000000000000000000000000000000F000000000000000000000000000000100000000000000000000000000000001100000000000000000000000000000012000000000000000000000000000000130000000000000000000000000000001400000000000000000000000000000015000000000000000000000000000000160000000000000000000000000000001700000000000000000000000000000018000000000000000000000000000000190000000000000000000000000000001A0000000000000000000000000000001B0000000000000000000000000000001C0000000000000000000000000000001D0000000000000000000000000000001E0000000000000000000000000000001F000000000000000000000000000000200000000000000000000000000000002100000000000000000000000000000022000000000000000000000000000000230000000000000000000000000000002400000000000000000000000000000025000000000000000000000000000000260000000000000000000000000000002700000000000000000000000000000028000000000000000000000000000000290000000000000000000000000000002A0000000000000000000000000000002B0000000000000000000000000000002C0000000000000000000000000000002D0000000000000000000000000000002E0000000000000000000000000000002F000000000000000000000000000000300000000000000000000000000000003100000000000000000000000000000032000000000000000000000000000000330000000000000000000000000000003400000000000000000000000000000035000000000000000000000000000000360000000000000000000000000000003700000000000000000000000000000038000000000000000000000000000000390000000000000000000000000000003A0000000000000000000000000000003B0000000000000000000000000000003C0000000000000000000000000000003D0000000000000000000000000000003E0000000000000000000000000000003F000000000000000000000000000000400000000000000000000000000000004100000000000000000000000000000042000000000000000000000000000000430000000000000000000000000000004400000000000000000000000000000045000000000000000000000000000000460000000000000000000000000000004700000000000000000000000000000048000000000000000000000000000000490000000000000000000000000000004A0000000000000000000000000000004B0000000000000000000000000000004C0000000000000000000000000000004D0000000000000000000000000000004E0000000000000000000000000000004F000000000000000000000000000000500000000000000000000000000000005100000000000000000000000000000052000000000000000000000000000000530000000000000000000000000000005400000000000000000000000000000055000000000000000000000000000000560000000000000000000000000000005700000000000000000000000000000058000000000000000000000000000000590000000000000000000000000000005A0000000000000000000000000000005B0000000000000000000000000000005C0000000000000000000000000000005D0000000000000000000000000000005E0000000000000000000000000000005F000000000000000000000000000000600000000000000000000000000000006100000000000000000000000000000062000000000000000000000000000000630000000000000000000000000000006400000000000000000000000000000065000000000000000000000000000000660000000000000000000000000000006700000000000000000000000000000068000000000000000000000000000000690000000000000000000000000000006A0000000000000000000000000000006B0000000000000000000000000000006C0000000000000000000000000000006D0000000000000000000000000000006E0000000000000000000000000000006F000000000000000000000000000000700000000000000000000000000000007100000000000000000000000000000072000000000000000000000000000000730000000000000000000000000000007400000000000000000000000000000075000000000000000000000000000000760000000000000000000000000000007700000000000000000000000000000078000000000000000000000000000000790000000000000000000000000000007A0000000000000000000000000000007B0000000000000000000000000000007C0000000000000000000000000000007D0000000000000000000000000000007E0000000000000000000000000000007F00000000000000"> : tensor<1x128x2xi64>}> : () -> tensor<1x128x2xi64>
        %37 = "ttir.constant"() <{value = dense<-3.389530e+38> : tensor<bf16>}> : () -> tensor<bf16>
        %38 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %39 = "ttir.constant"() <{value = dense<-7.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %40 = "ttir.constant"() <{value = dense<7.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %41 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %42 = "ttir.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
        %43 = "ttir.constant"() <{value = dense<1.703130e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %44 = "ttir.reshape"(%43) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1x1xbf16>
        %45 = "ttir.broadcast"(%44) <{broadcast_dimensions = array<i64: 2, 4, 32, 4, 2880>}> : (tensor<1x1x1x1x1xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %46 = "ttir.reshape"(%42) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1x1xbf16>
        %47 = "ttir.broadcast"(%46) <{broadcast_dimensions = array<i64: 2, 4, 32, 4, 2880>}> : (tensor<1x1x1x1x1xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %48 = "ttir.reshape"(%41) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1x1xbf16>
        %49 = "ttir.broadcast"(%48) <{broadcast_dimensions = array<i64: 2, 4, 32, 4, 2880>}> : (tensor<1x1x1x1x1xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %50 = "ttir.reshape"(%40) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1x1xbf16>
        %51 = "ttir.broadcast"(%50) <{broadcast_dimensions = array<i64: 2, 4, 32, 4, 2880>}> : (tensor<1x1x1x1x1xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %52 = "ttir.reshape"(%39) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1x1xbf16>
        %53 = "ttir.broadcast"(%52) <{broadcast_dimensions = array<i64: 2, 4, 32, 4, 2880>}> : (tensor<1x1x1x1x1xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %54 = "ttir.reshape"(%38) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1xbf16>
        %55 = "ttir.broadcast"(%54) <{broadcast_dimensions = array<i64: 128, 32>}> : (tensor<1x1xbf16>) -> tensor<128x32xbf16>
        %56 = "ttir.reshape"(%37) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %57 = "ttir.broadcast"(%56) <{broadcast_dimensions = array<i64: 1, 1, 128, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x1x128x128xbf16>
        %58 = "ttir.reshape"(%34) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %59 = "ttir.broadcast"(%58) <{broadcast_dimensions = array<i64: 1, 16, 128, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x16x128x128xbf16>
        %60 = "ttir.reshape"(%33) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %61 = "ttir.broadcast"(%60) <{broadcast_dimensions = array<i64: 1, 128, 32>}> : (tensor<1x1x1xf32>) -> tensor<1x128x32xf32>
        %62 = "ttir.reshape"(%30) <{shape = [1 : i32, 1 : i32]}> : (tensor<i64>) -> tensor<1x1xi64>
        %63 = "ttir.broadcast"(%62) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<1x1xi64>) -> tensor<1x128xi64>
        %64 = "ttir.reshape"(%4) <{shape = [1 : i32, 201088 : i32, 1440 : i32]}> : (tensor<201088x1440xbf16>) -> tensor<1x201088x1440xbf16>
        %65 = "ttir.reshape"(%64) <{shape = [201088 : i32, 1440 : i32]}> : (tensor<1x201088x1440xbf16>) -> tensor<201088x1440xbf16>
        %66 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xi64>) -> tensor<1x1x128xi64>
        %67 = "ttir.reshape"(%66) <{shape = [128 : i32]}> : (tensor<1x1x128xi64>) -> tensor<128xi64>
        %68 = "ttir.typecast"(%67) <{conservative_folding = false}> : (tensor<128xi64>) -> tensor<128xui32>
        %69 = "ttir.gather"(%65, %68) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 1440>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<201088x1440xbf16>, tensor<128xui32>) -> tensor<128x1440xbf16>
        %70 = "ttir.reshape"(%69) <{shape = [1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %71 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
        %72 = "ttir.reshape"(%71) <{shape = [1440 : i32]}> : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
        %73 = "ttir.reshape"(%26) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %74 = "ttir.broadcast"(%73) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x128x1xf32>
        %75 = "ttir.reshape"(%27) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
        %76 = "ttir.broadcast"(%75) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<1x1xf32>) -> tensor<1x128xf32>
        %77 = "ttir.reshape"(%28) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %78 = "ttir.broadcast"(%77) <{broadcast_dimensions = array<i64: 1, 128, 1440>}> : (tensor<1x1x1xf32>) -> tensor<1x128x1440xf32>
        %79 = "ttir.typecast"(%70) <{conservative_folding = false}> : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
        %80 = "ttir.pow"(%79, %78) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %81 = "ttir.sum"(%80) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128xf32>
        %82 = "ttir.all_reduce"(%81) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %83 = "ttir.multiply"(%82, %76) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
        %84 = "ttir.reshape"(%83) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %85 = "ttir.add"(%84, %74) : (tensor<1x128x1xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %86 = "ttir.rsqrt"(%85) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %87 = "ttir.reshape"(%86) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
        %88 = "ttir.reshape"(%87) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %89 = "ttir.broadcast"(%88) <{broadcast_dimensions = array<i64: 1, 1, 1440>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1440xf32>
        %90 = "ttir.multiply"(%79, %89) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %91 = "ttir.typecast"(%72) <{conservative_folding = false}> : (tensor<1440xbf16>) -> tensor<1440xf32>
        %92 = "ttir.reshape"(%91) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xf32>) -> tensor<1x1x1440xf32>
        %93 = "ttir.broadcast"(%92) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1440xf32>) -> tensor<1x128x1440xf32>
        %94 = "ttir.multiply"(%90, %93) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %95 = "ttir.typecast"(%94) <{conservative_folding = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
        %96 = "ttir.reshape"(%95) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
        %97 = "ttir.reshape"(%20) <{shape = [1 : i32, 1024 : i32, 1440 : i32]}> : (tensor<1024x1440xbf16>) -> tensor<1x1024x1440xbf16>
        %98 = "ttir.reshape"(%97) <{shape = [1024 : i32, 1440 : i32]}> : (tensor<1x1024x1440xbf16>) -> tensor<1024x1440xbf16>
        %99 = "ttir.permute"(%98) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1440xbf16>) -> tensor<1440x1024xbf16>
        %100 = "ttir.dot_general"(%96, %99) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1440xbf16>, tensor<1440x1024xbf16>) -> tensor<128x1024xbf16>
        %101 = "ttir.all_reduce"(%100) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
        %102 = "ttir.reshape"(%101) <{shape = [1 : i32, 128 : i32, 1024 : i32]}> : (tensor<128x1024xbf16>) -> tensor<1x128x1024xbf16>
        %103 = "ttir.reshape"(%19) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %104 = "ttir.reshape"(%103) <{shape = [1024 : i32]}> : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
        %105 = "ttir.reshape"(%104) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %106 = "ttir.broadcast"(%105) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1024xbf16>) -> tensor<1x128x1024xbf16>
        %107 = "ttir.add"(%102, %106) : (tensor<1x128x1024xbf16>, tensor<1x128x1024xbf16>) -> tensor<1x128x1024xbf16>
        %108 = "ttir.reshape"(%107) <{shape = [1 : i32, 128 : i32, 16 : i32, 64 : i32]}> : (tensor<1x128x1024xbf16>) -> tensor<1x128x16x64xbf16>
        %109 = "ttir.permute"(%108) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x16x64xbf16>) -> tensor<1x16x128x64xbf16>
        %110 = "ttir.slice_static"(%109) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
        %111 = "ttir.reshape"(%5) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xf32>) -> tensor<1x1x32xf32>
        %112 = "ttir.reshape"(%111) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x1x32xf32>) -> tensor<1x32x1xf32>
        %113 = "ttir.dot_general"(%112, %32) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x32x1xf32>, tensor<1x1x128xf32>) -> tensor<1x32x128xf32>
        %114 = "ttir.permute"(%113) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x32x128xf32>) -> tensor<1x128x32xf32>
        %115 = "ttir.cos"(%114) : (tensor<1x128x32xf32>) -> tensor<1x128x32xf32>
        %116 = "ttir.multiply"(%115, %61) : (tensor<1x128x32xf32>, tensor<1x128x32xf32>) -> tensor<1x128x32xf32>
        %117 = "ttir.typecast"(%116) <{conservative_folding = false}> : (tensor<1x128x32xf32>) -> tensor<1x128x32xbf16>
        %118 = "ttir.reshape"(%117) <{shape = [1 : i32, 1 : i32, 128 : i32, 32 : i32]}> : (tensor<1x128x32xbf16>) -> tensor<1x1x128x32xbf16>
        %119 = "ttir.broadcast"(%118) <{broadcast_dimensions = array<i64: 1, 16, 1, 1>}> : (tensor<1x1x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %120 = "ttir.multiply"(%110, %119) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %121 = "ttir.slice_static"(%109) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16>) -> tensor<1x16x128x32xbf16>
        %122 = "ttir.sin"(%114) : (tensor<1x128x32xf32>) -> tensor<1x128x32xf32>
        %123 = "ttir.multiply"(%122, %61) : (tensor<1x128x32xf32>, tensor<1x128x32xf32>) -> tensor<1x128x32xf32>
        %124 = "ttir.typecast"(%123) <{conservative_folding = false}> : (tensor<1x128x32xf32>) -> tensor<1x128x32xbf16>
        %125 = "ttir.reshape"(%124) <{shape = [1 : i32, 1 : i32, 128 : i32, 32 : i32]}> : (tensor<1x128x32xbf16>) -> tensor<1x1x128x32xbf16>
        %126 = "ttir.broadcast"(%125) <{broadcast_dimensions = array<i64: 1, 16, 1, 1>}> : (tensor<1x1x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %127 = "ttir.multiply"(%121, %126) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %128 = "ttir.subtract"(%120, %127) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %129 = "ttir.multiply"(%121, %119) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %130 = "ttir.multiply"(%110, %126) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %131 = "ttir.add"(%129, %130) : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x32xbf16>
        %132 = "ttir.concat"(%128, %131) <{dim = 3 : si32}> : (tensor<1x16x128x32xbf16>, tensor<1x16x128x32xbf16>) -> tensor<1x16x128x64xbf16>
        %133 = "ttir.reshape"(%7) <{shape = [1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %134 = "ttir.reshape"(%133) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
        %135 = "ttir.permute"(%134) <{permutation = array<i64: 1, 0>}> : (tensor<128x1440xbf16>) -> tensor<1440x128xbf16>
        %136 = "ttir.dot_general"(%96, %135) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1440xbf16>, tensor<1440x128xbf16>) -> tensor<128x128xbf16>
        %137 = "ttir.all_reduce"(%136) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %138 = "ttir.reshape"(%137) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
        %139 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %140 = "ttir.reshape"(%139) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %141 = "ttir.reshape"(%140) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %142 = "ttir.broadcast"(%141) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x128xbf16>) -> tensor<1x128x128xbf16>
        %143 = "ttir.add"(%138, %142) : (tensor<1x128x128xbf16>, tensor<1x128x128xbf16>) -> tensor<1x128x128xbf16>
        %144 = "ttir.reshape"(%143) <{shape = [1 : i32, 128 : i32, 2 : i32, 64 : i32]}> : (tensor<1x128x128xbf16>) -> tensor<1x128x2x64xbf16>
        %145 = "ttir.permute"(%144) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x2x64xbf16>) -> tensor<1x2x128x64xbf16>
        %146 = "ttir.slice_static"(%145) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x128x32xbf16>
        %147 = "ttir.broadcast"(%118) <{broadcast_dimensions = array<i64: 1, 2, 1, 1>}> : (tensor<1x1x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %148 = "ttir.multiply"(%146, %147) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %149 = "ttir.slice_static"(%145) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x128x32xbf16>
        %150 = "ttir.broadcast"(%125) <{broadcast_dimensions = array<i64: 1, 2, 1, 1>}> : (tensor<1x1x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %151 = "ttir.multiply"(%149, %150) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %152 = "ttir.subtract"(%148, %151) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %153 = "ttir.multiply"(%149, %147) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %154 = "ttir.multiply"(%146, %150) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %155 = "ttir.add"(%153, %154) : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %156 = "ttir.concat"(%152, %155) <{dim = 3 : si32}> : (tensor<1x2x128x32xbf16>, tensor<1x2x128x32xbf16>) -> tensor<1x2x128x64xbf16>
        %157 = "ttir.reshape"(%156) <{shape = [1 : i32, 2 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x1x128x64xbf16>
        %158 = "ttir.broadcast"(%157) <{broadcast_dimensions = array<i64: 1, 1, 8, 1, 1>}> : (tensor<1x2x1x128x64xbf16>) -> tensor<1x2x8x128x64xbf16>
        %159 = "ttir.reshape"(%158) <{shape = [1 : i32, 16 : i32, 128 : i32, 64 : i32]}> : (tensor<1x2x8x128x64xbf16>) -> tensor<1x16x128x64xbf16>
        %160 = "ttir.permute"(%159) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x16x128x64xbf16>) -> tensor<1x16x64x128xbf16>
        %161 = "ttir.dot_general"(%132, %160) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<1x16x128x64xbf16>, tensor<1x16x64x128xbf16>) -> tensor<1x16x128x128xbf16>
        %162 = "ttir.multiply"(%161, %59) : (tensor<1x16x128x128xbf16>, tensor<1x16x128x128xbf16>) -> tensor<1x16x128x128xbf16>
        %163 = "ttir.reshape"(%18) <{shape = [1 : i32, 1 : i32]}> : (tensor<i1>) -> tensor<1x1xi1>
        %164 = "ttir.broadcast"(%163) <{broadcast_dimensions = array<i64: 128, 128>}> : (tensor<1x1xi1>) -> tensor<128x128xi1>
        %165 = "ttir.reshape"(%29) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xi64>) -> tensor<1x128xi64>
        %166 = "ttir.broadcast"(%165) <{broadcast_dimensions = array<i64: 128, 1>}> : (tensor<1x128xi64>) -> tensor<128x128xi64>
        %167 = "ttir.reshape"(%35) <{shape = [128 : i32, 1 : i32]}> : (tensor<128xi64>) -> tensor<128x1xi64>
        %168 = "ttir.broadcast"(%167) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<128x1xi64>) -> tensor<128x128xi64>
        %169 = "ttir.gt"(%166, %168) : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
        %170 = "ttir.logical_and"(%164, %169) : (tensor<128x128xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
        %171 = "ttir.reshape"(%29) <{shape = [128 : i32, 1 : i32]}> : (tensor<128xi64>) -> tensor<128x1xi64>
        %172 = "ttir.broadcast"(%171) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<128x1xi64>) -> tensor<128x128xi64>
        %173 = "ttir.le"(%166, %172) : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
        %174 = "ttir.logical_and"(%170, %173) : (tensor<128x128xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
        %175 = "ttir.logical_and"(%164, %174) : (tensor<128x128xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
        %176 = "ttir.reshape"(%175) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xi1>) -> tensor<1x128x128xi1>
        %177 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xi64>) -> tensor<1x1x128xi64>
        %178 = "ttir.reshape"(%177) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x1x128xi64>) -> tensor<1x128xi64>
        %179 = "ttir.ne"(%178, %63) : (tensor<1x128xi64>, tensor<1x128xi64>) -> tensor<1x128xi1>
        %180 = "ttir.gather"(%179, %36) <{collapsed_slice_dims = array<i64: 0, 1>, index_vector_dim = 2 : si64, indices_are_sorted = false, offset_dims = array<i64>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 1>, start_index_map = array<i64: 0, 1>, start_indices_batching_dims = array<i64>}> : (tensor<1x128xi1>, tensor<1x128x2xi64>) -> tensor<1x128xi1>
        %181 = "ttir.reshape"(%180) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xi1>) -> tensor<1x1x128xi1>
        %182 = "ttir.broadcast"(%181) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x128xi1>) -> tensor<1x128x128xi1>
        %183 = "ttir.logical_and"(%176, %182) : (tensor<1x128x128xi1>, tensor<1x128x128xi1>) -> tensor<1x128x128xi1>
        %184 = "ttir.reshape"(%183) <{shape = [1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<1x128x128xi1>) -> tensor<1x1x128x128xi1>
        %185 = "ttir.reshape"(%16) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1xbf16>
        %186 = "ttir.reshape"(%185) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1x1xbf16>) -> tensor<1x1xbf16>
        %187 = "ttir.reshape"(%186) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1xbf16>) -> tensor<1x1x1x1xbf16>
        %188 = "ttir.broadcast"(%187) <{broadcast_dimensions = array<i64: 1, 1, 128, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x1x128x128xbf16>
        %189 = "ttir.where"(%184, %188, %57) : (tensor<1x1x128x128xi1>, tensor<1x1x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x1x128x128xbf16>
        %190 = "ttir.reshape"(%189) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<1x1x128x128xbf16>) -> tensor<1x128x128xbf16>
        %191 = "ttir.reshape"(%190) <{shape = [1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<1x128x128xbf16>) -> tensor<1x1x128x128xbf16>
        %192 = "ttir.broadcast"(%191) <{broadcast_dimensions = array<i64: 1, 16, 1, 1>}> : (tensor<1x1x128x128xbf16>) -> tensor<1x16x128x128xbf16>
        %193 = "ttir.add"(%162, %192) : (tensor<1x16x128x128xbf16>, tensor<1x16x128x128xbf16>) -> tensor<1x16x128x128xbf16>
        %194 = "ttir.mesh_partition"(%15) <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<64xbf16>) -> tensor<16xbf16>
        %195 = "ttir.reshape"(%194) <{shape = [1 : i32, 1 : i32, 16 : i32]}> : (tensor<16xbf16>) -> tensor<1x1x16xbf16>
        %196 = "ttir.reshape"(%195) <{shape = [1 : i32, 16 : i32, 1 : i32]}> : (tensor<1x1x16xbf16>) -> tensor<1x16x1xbf16>
        %197 = "ttir.reshape"(%196) <{shape = [1 : i32, 16 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x1xbf16>) -> tensor<1x16x1x1xbf16>
        %198 = "ttir.broadcast"(%197) <{broadcast_dimensions = array<i64: 1, 1, 128, 1>}> : (tensor<1x16x1x1xbf16>) -> tensor<1x16x128x1xbf16>
        %199 = "ttir.concat"(%193, %198) <{dim = 3 : si32}> : (tensor<1x16x128x128xbf16>, tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
        %200 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<128xi64>
        %201 = "ttir.reshape"(%200) <{shape = [128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xi64>) -> tensor<128x1x1xi64>
        %202 = "ttir.broadcast"(%201) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<128x1x1xi64>) -> tensor<128x4x1xi64>
        %203 = "ttir.max"(%199) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16>
        %204 = "ttir.reshape"(%203) <{shape = [1 : i32, 16 : i32, 128 : i32, 1 : i32]}> : (tensor<1x16x128xbf16>) -> tensor<1x16x128x1xbf16>
        %205 = "ttir.broadcast"(%204) <{broadcast_dimensions = array<i64: 1, 1, 1, 129>}> : (tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
        %206 = "ttir.subtract"(%199, %205) : (tensor<1x16x128x129xbf16>, tensor<1x16x128x129xbf16>) -> tensor<1x16x128x129xbf16>
        %207 = "ttir.max"(%206) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16>
        %208 = "ttir.reshape"(%207) <{shape = [1 : i32, 16 : i32, 128 : i32, 1 : i32]}> : (tensor<1x16x128xbf16>) -> tensor<1x16x128x1xbf16>
        %209 = "ttir.broadcast"(%208) <{broadcast_dimensions = array<i64: 1, 1, 1, 129>}> : (tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
        %210 = "ttir.subtract"(%206, %209) : (tensor<1x16x128x129xbf16>, tensor<1x16x128x129xbf16>) -> tensor<1x16x128x129xbf16>
        %211 = "ttir.exp"(%210) : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128x129xbf16>
        %212 = "ttir.sum"(%211) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128xbf16>
        %213 = "ttir.reshape"(%212) <{shape = [1 : i32, 16 : i32, 128 : i32, 1 : i32]}> : (tensor<1x16x128xbf16>) -> tensor<1x16x128x1xbf16>
        %214 = "ttir.broadcast"(%213) <{broadcast_dimensions = array<i64: 1, 1, 1, 129>}> : (tensor<1x16x128x1xbf16>) -> tensor<1x16x128x129xbf16>
        %215 = "ttir.div"(%211, %214) : (tensor<1x16x128x129xbf16>, tensor<1x16x128x129xbf16>) -> tensor<1x16x128x129xbf16>
        %216 = "ttir.slice_static"(%215) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x129xbf16>) -> tensor<1x16x128x128xbf16>
        %217 = "ttir.reshape"(%1) <{shape = [1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %218 = "ttir.reshape"(%217) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
        %219 = "ttir.permute"(%218) <{permutation = array<i64: 1, 0>}> : (tensor<128x1440xbf16>) -> tensor<1440x128xbf16>
        %220 = "ttir.dot_general"(%96, %219) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1440xbf16>, tensor<1440x128xbf16>) -> tensor<128x128xbf16>
        %221 = "ttir.all_reduce"(%220) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %222 = "ttir.reshape"(%221) <{shape = [1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16>) -> tensor<1x128x128xbf16>
        %223 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %224 = "ttir.reshape"(%223) <{shape = [128 : i32]}> : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
        %225 = "ttir.reshape"(%224) <{shape = [1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
        %226 = "ttir.broadcast"(%225) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x128xbf16>) -> tensor<1x128x128xbf16>
        %227 = "ttir.add"(%222, %226) : (tensor<1x128x128xbf16>, tensor<1x128x128xbf16>) -> tensor<1x128x128xbf16>
        %228 = "ttir.reshape"(%227) <{shape = [1 : i32, 128 : i32, 2 : i32, 64 : i32]}> : (tensor<1x128x128xbf16>) -> tensor<1x128x2x64xbf16>
        %229 = "ttir.permute"(%228) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x2x64xbf16>) -> tensor<1x2x128x64xbf16>
        %230 = "ttir.reshape"(%229) <{shape = [1 : i32, 2 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x1x128x64xbf16>
        %231 = "ttir.broadcast"(%230) <{broadcast_dimensions = array<i64: 1, 1, 8, 1, 1>}> : (tensor<1x2x1x128x64xbf16>) -> tensor<1x2x8x128x64xbf16>
        %232 = "ttir.reshape"(%231) <{shape = [1 : i32, 16 : i32, 128 : i32, 64 : i32]}> : (tensor<1x2x8x128x64xbf16>) -> tensor<1x16x128x64xbf16>
        %233 = "ttir.dot_general"(%216, %232) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<1x16x128x128xbf16>, tensor<1x16x128x64xbf16>) -> tensor<1x16x128x64xbf16>
        %234 = "ttir.permute"(%233) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x16x128x64xbf16>) -> tensor<1x128x16x64xbf16>
        %235 = "ttir.reshape"(%234) <{shape = [128 : i32, 1024 : i32]}> : (tensor<1x128x16x64xbf16>) -> tensor<128x1024xbf16>
        %236 = "ttir.reshape"(%14) <{shape = [1 : i32, 1440 : i32, 1024 : i32]}> : (tensor<1440x1024xbf16>) -> tensor<1x1440x1024xbf16>
        %237 = "ttir.reshape"(%236) <{shape = [1440 : i32, 1024 : i32]}> : (tensor<1x1440x1024xbf16>) -> tensor<1440x1024xbf16>
        %238 = "ttir.permute"(%237) <{permutation = array<i64: 1, 0>}> : (tensor<1440x1024xbf16>) -> tensor<1024x1440xbf16>
        %239 = "ttir.dot_general"(%235, %238) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1024xbf16>, tensor<1024x1440xbf16>) -> tensor<128x1440xbf16>
        %240 = "ttir.all_reduce"(%239) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x1440xbf16>) -> tensor<128x1440xbf16>
        %241 = "ttir.reshape"(%240) <{shape = [1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %242 = "ttir.reshape"(%13) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
        %243 = "ttir.reshape"(%242) <{shape = [1440 : i32]}> : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
        %244 = "ttir.reshape"(%243) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
        %245 = "ttir.broadcast"(%244) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1440xbf16>) -> tensor<1x128x1440xbf16>
        %246 = "ttir.add"(%241, %245) : (tensor<1x128x1440xbf16>, tensor<1x128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %247 = "ttir.add"(%70, %246) : (tensor<1x128x1440xbf16>, tensor<1x128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %248 = "ttir.reshape"(%12) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
        %249 = "ttir.reshape"(%248) <{shape = [1440 : i32]}> : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
        %250 = "ttir.typecast"(%247) <{conservative_folding = false}> : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
        %251 = "ttir.pow"(%250, %78) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %252 = "ttir.sum"(%251) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128xf32>
        %253 = "ttir.all_reduce"(%252) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %254 = "ttir.multiply"(%253, %76) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
        %255 = "ttir.reshape"(%254) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %256 = "ttir.add"(%255, %74) : (tensor<1x128x1xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %257 = "ttir.rsqrt"(%256) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %258 = "ttir.reshape"(%257) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
        %259 = "ttir.reshape"(%258) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %260 = "ttir.broadcast"(%259) <{broadcast_dimensions = array<i64: 1, 1, 1440>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1440xf32>
        %261 = "ttir.multiply"(%250, %260) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %262 = "ttir.typecast"(%249) <{conservative_folding = false}> : (tensor<1440xbf16>) -> tensor<1440xf32>
        %263 = "ttir.reshape"(%262) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xf32>) -> tensor<1x1x1440xf32>
        %264 = "ttir.broadcast"(%263) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1440xf32>) -> tensor<1x128x1440xf32>
        %265 = "ttir.multiply"(%261, %264) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %266 = "ttir.typecast"(%265) <{conservative_folding = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
        %267 = "ttir.reshape"(%266) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
        %268 = "ttir.typecast"(%267) <{conservative_folding = false}> : (tensor<128x1440xbf16>) -> tensor<128x1440xf32>
        %269 = "ttir.reshape"(%11) <{shape = [1 : i32, 32 : i32, 1440 : i32]}> : (tensor<32x1440xbf16>) -> tensor<1x32x1440xbf16>
        %270 = "ttir.reshape"(%269) <{shape = [32 : i32, 1440 : i32]}> : (tensor<1x32x1440xbf16>) -> tensor<32x1440xbf16>
        %271 = "ttir.permute"(%270) <{permutation = array<i64: 1, 0>}> : (tensor<32x1440xbf16>) -> tensor<1440x32xbf16>
        %272 = "ttir.typecast"(%271) <{conservative_folding = false}> : (tensor<1440x32xbf16>) -> tensor<1440x32xf32>
        %273 = "ttir.dot_general"(%268, %272) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1440xf32>, tensor<1440x32xf32>) -> tensor<128x32xf32>
        %274 = "ttir.all_reduce"(%273) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x32xf32>) -> tensor<128x32xf32>
        %275 = "ttir.reshape"(%10) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
        %276 = "ttir.reshape"(%275) <{shape = [32 : i32]}> : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
        %277 = "ttir.typecast"(%276) <{conservative_folding = false}> : (tensor<32xbf16>) -> tensor<32xf32>
        %278 = "ttir.reshape"(%277) <{shape = [1 : i32, 32 : i32]}> : (tensor<32xf32>) -> tensor<1x32xf32>
        %279 = "ttir.broadcast"(%278) <{broadcast_dimensions = array<i64: 128, 1>}> : (tensor<1x32xf32>) -> tensor<128x32xf32>
        %280 = "ttir.add"(%274, %279) : (tensor<128x32xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
        %281 = "ttir.typecast"(%280) <{conservative_folding = false}> : (tensor<128x32xf32>) -> tensor<128x32xbf16>
        %282 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 32 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<32xi32>
        %283 = "ttir.reshape"(%282) <{shape = [1 : i32, 32 : i32]}> : (tensor<32xi32>) -> tensor<1x32xi32>
        %values, %indices = "ttir.sort"(%281) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<128x32xbf16>) -> (tensor<128x32xbf16>, tensor<128x32xi32>)
        %284 = "ttir.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x32xi32>) -> tensor<128x4xi32>
        %285 = "ttir.typecast"(%284) <{conservative_folding = false}> : (tensor<128x4xi32>) -> tensor<128x4xi64>
        %286 = "ttir.reshape"(%285) <{shape = [128 : i32, 4 : i32, 1 : i32]}> : (tensor<128x4xi64>) -> tensor<128x4x1xi64>
        %287 = "ttir.concat"(%202, %286) <{dim = 2 : si32}> : (tensor<128x4x1xi64>, tensor<128x4x1xi64>) -> tensor<128x4x2xi64>
        %288 = "ttir.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x32xbf16>) -> tensor<128x4xbf16>
        %289 = "ttir.max"(%288) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x4xbf16>) -> tensor<128xbf16>
        %290 = "ttir.reshape"(%289) <{shape = [128 : i32, 1 : i32]}> : (tensor<128xbf16>) -> tensor<128x1xbf16>
        %291 = "ttir.broadcast"(%290) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<128x1xbf16>) -> tensor<128x4xbf16>
        %292 = "ttir.subtract"(%288, %291) : (tensor<128x4xbf16>, tensor<128x4xbf16>) -> tensor<128x4xbf16>
        %293 = "ttir.exp"(%292) : (tensor<128x4xbf16>) -> tensor<128x4xbf16>
        %294 = "ttir.sum"(%293) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x4xbf16>) -> tensor<128xbf16>
        %295 = "ttir.reshape"(%294) <{shape = [128 : i32, 1 : i32]}> : (tensor<128xbf16>) -> tensor<128x1xbf16>
        %296 = "ttir.broadcast"(%295) <{broadcast_dimensions = array<i64: 1, 4>}> : (tensor<128x1xbf16>) -> tensor<128x4xbf16>
        %297 = "ttir.div"(%293, %296) : (tensor<128x4xbf16>, tensor<128x4xbf16>) -> tensor<128x4xbf16>
        %298 = "ttir.reshape"(%287) <{shape = [512 : i32, 2 : i32]}> : (tensor<128x4x2xi64>) -> tensor<512x2xi64>
        %299 = "ttir.reshape"(%298) <{shape = [512 : i32, 1 : i32, 2 : i32]}> : (tensor<512x2xi64>) -> tensor<512x1x2xi64>
        %300 = "ttir.repeat"(%299) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<512x1x2xi64>) -> tensor<512x1x2xi64>
        %301 = "ttir.reshape"(%300) <{shape = [512 : i32, 2 : i32]}> : (tensor<512x1x2xi64>) -> tensor<512x2xi64>
        %302 = "ttir.slice_static"(%301) <{begins = [0 : i32, 0 : i32], ends = [512 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xi64>) -> tensor<512x1xi64>
        %303 = "ttir.slice_static"(%301) <{begins = [0 : i32, 1 : i32], ends = [512 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xi64>) -> tensor<512x1xi64>
        %304 = "ttir.full"() <{fill_value = 32 : i32, shape = array<i32: 512, 1>}> : () -> tensor<512x1xi64>
        %305 = "ttir.multiply"(%302, %304) : (tensor<512x1xi64>, tensor<512x1xi64>) -> tensor<512x1xi64>
        %306 = "ttir.add"(%305, %303) : (tensor<512x1xi64>, tensor<512x1xi64>) -> tensor<512x1xi64>
        %307 = "ttir.reshape"(%306) <{shape = [512 : i32]}> : (tensor<512x1xi64>) -> tensor<512xi64>
        %308 = "ttir.reshape"(%55) <{shape = [4096 : i32]}> : (tensor<128x32xbf16>) -> tensor<4096xbf16>
        %309 = "ttir.reshape"(%297) <{shape = [512 : i32]}> : (tensor<128x4xbf16>) -> tensor<512xbf16>
        %310 = "ttir.scatter"(%308, %307, %309) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<4096xbf16>, tensor<512xi64>, tensor<512xbf16>) -> tensor<4096xbf16>
        %311 = "ttir.reshape"(%310) <{shape = [128 : i32, 32 : i32]}> : (tensor<4096xbf16>) -> tensor<128x32xbf16>
        %312 = "ttir.reshape"(%311) <{shape = [1 : i32, 128 : i32, 32 : i32]}> : (tensor<128x32xbf16>) -> tensor<1x128x32xbf16>
        %313 = "ttir.broadcast"(%312) <{broadcast_dimensions = array<i64: 2, 1, 1>}> : (tensor<1x128x32xbf16>) -> tensor<2x128x32xbf16>
        %314 = "ttir.reshape"(%313) <{shape = [1 : i32, 2 : i32, 128 : i32, 32 : i32]}> : (tensor<2x128x32xbf16>) -> tensor<1x2x128x32xbf16>
        %315 = "ttir.reshape"(%266) <{shape = [1 : i32, 1 : i32, 128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<1x1x128x1440xbf16>
        %316 = "ttir.reshape"(%285) <{shape = [1 : i32, 1 : i32, 128 : i32, 4 : i32]}> : (tensor<128x4xi64>) -> tensor<1x1x128x4xi64>
        %317 = "ttir.all_gather"(%315) <{all_gather_dim = 3 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x128x1440xbf16>) -> tensor<1x1x128x2880xbf16>
        %dispatched, %metadata = "ttir.all_to_all_dispatch"(%317, %316, %21) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16>, tensor<1x1x128x4xi64>, tensor<1x1x32x8xi64>) -> (tensor<1x2x128x2880xbf16>, tensor<1x2x128x4xi64>)
        %mapping, %reduced = "ttir.moe_expert_token_remap"(%314, %21, %metadata) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
        %318 = "ttir.slice_static"(%229) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x127x64xbf16>
        %319 = "ttir.slice_static"(%156) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16>) -> tensor<1x2x127x64xbf16>
        %320 = "ttir.reshape"(%dispatched) <{shape = [2 : i32, 4 : i32, 32 : i32, 2880 : i32]}> : (tensor<1x2x128x2880xbf16>) -> tensor<2x4x32x2880xbf16>
        %321 = "ttir.reshape"(%25) <{shape = [1 : i32, 4 : i32, 2880 : i32, 5760 : i32]}> : (tensor<4x2880x5760xbf16>) -> tensor<1x4x2880x5760xbf16>
        %322 = "ttir.reshape"(%reduced) <{shape = [2 : i32, 4 : i32, 1 : i32, 4 : i32]}> : (tensor<1x1x8x4xbf16>) -> tensor<2x4x1x4xbf16>
        %323 = "ttir.sparse_matmul"(%320, %321, %322) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
        %324 = "ttir.reshape"(%323) <{shape = [2 : i32, 4 : i32, 4 : i32, 32 : i32, 5760 : i32]}> : (tensor<2x4x1x4x32x5760xbf16>) -> tensor<2x4x4x32x5760xbf16>
        %325 = "ttir.permute"(%324) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<2x4x4x32x5760xbf16>) -> tensor<2x4x32x4x5760xbf16>
        %326 = "ttir.reshape"(%24) <{shape = [1 : i32, 4 : i32, 5760 : i32]}> : (tensor<4x5760xbf16>) -> tensor<1x4x5760xbf16>
        %327 = "ttir.reshape"(%326) <{shape = [4 : i32, 5760 : i32]}> : (tensor<1x4x5760xbf16>) -> tensor<4x5760xbf16>
        %328 = "ttir.reshape"(%327) <{shape = [1 : i32, 1 : i32, 1 : i32, 4 : i32, 5760 : i32]}> : (tensor<4x5760xbf16>) -> tensor<1x1x1x4x5760xbf16>
        %329 = "ttir.broadcast"(%328) <{broadcast_dimensions = array<i64: 2, 4, 32, 1, 1>}> : (tensor<1x1x1x4x5760xbf16>) -> tensor<2x4x32x4x5760xbf16>
        %330 = "ttir.add"(%325, %329) : (tensor<2x4x32x4x5760xbf16>, tensor<2x4x32x4x5760xbf16>) -> tensor<2x4x32x4x5760xbf16>
        %331 = "ttir.slice_static"(%330) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 4 : i32, 32 : i32, 4 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x4x32x4x5760xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %332 = "ttir.clamp_tensor"(%331, %53, %51) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %333 = "ttir.add"(%332, %49) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %334 = "ttir.slice_static"(%330) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4 : i32, 32 : i32, 4 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<2x4x32x4x5760xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %335 = "ttir.clamp_tensor"(%334, %47, %51) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %336 = "ttir.multiply"(%335, %45) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %337 = "ttir.sigmoid"(%336) : (tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %338 = "ttir.multiply"(%335, %337) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %339 = "ttir.multiply"(%333, %338) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %340 = "ttir.permute"(%339) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x4x32x2880xbf16>
        %341 = "ttir.reshape"(%340) <{shape = [8 : i32, 4 : i32, 32 : i32, 2880 : i32]}> : (tensor<2x4x4x32x2880xbf16>) -> tensor<8x4x32x2880xbf16>
        %342 = "ttir.reshape"(%23) <{shape = [1 : i32, 4 : i32, 2880 : i32, 2880 : i32]}> : (tensor<4x2880x2880xbf16>) -> tensor<1x4x2880x2880xbf16>
        %343 = "ttir.sparse_matmul"(%341, %342, %reduced) <{is_input_a_sparse = true, is_input_b_sparse = false, nnz = 0 : i64}> : (tensor<8x4x32x2880xbf16>, tensor<1x4x2880x2880xbf16>, tensor<1x1x8x4xbf16>) -> tensor<8x4x32x2880xbf16>
        %344 = "ttir.reshape"(%343) <{shape = [2 : i32, 4 : i32, 4 : i32, 32 : i32, 2880 : i32]}> : (tensor<8x4x32x2880xbf16>) -> tensor<2x4x4x32x2880xbf16>
        %345 = "ttir.permute"(%344) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<2x4x4x32x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %346 = "ttir.reshape"(%22) <{shape = [1 : i32, 4 : i32, 2880 : i32]}> : (tensor<4x2880xbf16>) -> tensor<1x4x2880xbf16>
        %347 = "ttir.reshape"(%346) <{shape = [4 : i32, 2880 : i32]}> : (tensor<1x4x2880xbf16>) -> tensor<4x2880xbf16>
        %348 = "ttir.reshape"(%347) <{shape = [1 : i32, 1 : i32, 1 : i32, 4 : i32, 2880 : i32]}> : (tensor<4x2880xbf16>) -> tensor<1x1x1x4x2880xbf16>
        %349 = "ttir.broadcast"(%348) <{broadcast_dimensions = array<i64: 2, 4, 32, 1, 1>}> : (tensor<1x1x1x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %350 = "ttir.add"(%345, %349) : (tensor<2x4x32x4x2880xbf16>, tensor<2x4x32x4x2880xbf16>) -> tensor<2x4x32x4x2880xbf16>
        %351 = "ttir.permute"(%350) <{permutation = array<i64: 3, 0, 1, 2, 4>}> : (tensor<2x4x32x4x2880xbf16>) -> tensor<4x2x4x32x2880xbf16>
        %352 = "ttir.reshape"(%351) <{shape = [4 : i32, 2 : i32, 128 : i32, 2880 : i32]}> : (tensor<4x2x4x32x2880xbf16>) -> tensor<4x2x128x2880xbf16>
        %353 = "ttir.all_to_all_combine"(%352, %metadata, %21) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
        %354 = "ttir.all_reduce"(%353) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4x1x128x2880xbf16>) -> tensor<4x1x128x2880xbf16>
        %355 = "ttir.mesh_partition"(%354) <{cluster_axis = 0 : ui32, dim = 3 : si32}> : (tensor<4x1x128x2880xbf16>) -> tensor<4x1x128x1440xbf16>
        %356 = "ttir.broadcast"(%286) <{broadcast_dimensions = array<i64: 1, 1, 32>}> : (tensor<128x4x1xi64>) -> tensor<128x4x32xi64>
        %357 = "ttir.reshape"(%31) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xi64>) -> tensor<1x1x32xi64>
        %358 = "ttir.broadcast"(%357) <{broadcast_dimensions = array<i64: 128, 4, 1>}> : (tensor<1x1x32xi64>) -> tensor<128x4x32xi64>
        %359 = "ttir.eq"(%356, %358) : (tensor<128x4x32xi64>, tensor<128x4x32xi64>) -> tensor<128x4x32xi1>
        %360 = "ttir.typecast"(%359) <{conservative_folding = false}> : (tensor<128x4x32xi1>) -> tensor<128x4x32xbf16>
        %361 = "ttir.dot_general"(%360, %311) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<128x4x32xbf16>, tensor<128x32xbf16>) -> tensor<128x4xbf16>
        %362 = "ttir.reshape"(%361) <{shape = [1 : i32, 128 : i32, 4 : i32]}> : (tensor<128x4xbf16>) -> tensor<1x128x4xbf16>
        %363 = "ttir.permute"(%362) <{permutation = array<i64: 2, 0, 1>}> : (tensor<1x128x4xbf16>) -> tensor<4x1x128xbf16>
        %364 = "ttir.reshape"(%363) <{shape = [4 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<4x1x128xbf16>) -> tensor<4x1x128x1xbf16>
        %365 = "ttir.broadcast"(%364) <{broadcast_dimensions = array<i64: 1, 1, 1, 1440>}> : (tensor<4x1x128x1xbf16>) -> tensor<4x1x128x1440xbf16>
        %366 = "ttir.multiply"(%355, %365) : (tensor<4x1x128x1440xbf16>, tensor<4x1x128x1440xbf16>) -> tensor<4x1x128x1440xbf16>
        %367 = "ttir.sum"(%366) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<4x1x128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %368 = "ttir.add"(%247, %367) : (tensor<1x128x1440xbf16>, tensor<1x128x1440xbf16>) -> tensor<1x128x1440xbf16>
        %369 = "ttir.reshape"(%9) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16>) -> tensor<1x1x1440xbf16>
        %370 = "ttir.reshape"(%369) <{shape = [1440 : i32]}> : (tensor<1x1x1440xbf16>) -> tensor<1440xbf16>
        %371 = "ttir.typecast"(%368) <{conservative_folding = false}> : (tensor<1x128x1440xbf16>) -> tensor<1x128x1440xf32>
        %372 = "ttir.pow"(%371, %78) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %373 = "ttir.sum"(%372) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128xf32>
        %374 = "ttir.all_reduce"(%373) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %375 = "ttir.multiply"(%374, %76) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
        %376 = "ttir.reshape"(%375) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %377 = "ttir.add"(%376, %74) : (tensor<1x128x1xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %378 = "ttir.rsqrt"(%377) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
        %379 = "ttir.reshape"(%378) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128x1xf32>) -> tensor<1x128xf32>
        %380 = "ttir.reshape"(%379) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
        %381 = "ttir.broadcast"(%380) <{broadcast_dimensions = array<i64: 1, 1, 1440>}> : (tensor<1x128x1xf32>) -> tensor<1x128x1440xf32>
        %382 = "ttir.multiply"(%371, %381) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %383 = "ttir.typecast"(%370) <{conservative_folding = false}> : (tensor<1440xbf16>) -> tensor<1440xf32>
        %384 = "ttir.reshape"(%383) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xf32>) -> tensor<1x1x1440xf32>
        %385 = "ttir.broadcast"(%384) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x1440xf32>) -> tensor<1x128x1440xf32>
        %386 = "ttir.multiply"(%382, %385) : (tensor<1x128x1440xf32>, tensor<1x128x1440xf32>) -> tensor<1x128x1440xf32>
        %387 = "ttir.typecast"(%386) <{conservative_folding = false}> : (tensor<1x128x1440xf32>) -> tensor<1x128x1440xbf16>
        %388 = "ttir.reshape"(%387) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16>) -> tensor<128x1440xbf16>
        %389 = "ttir.mesh_partition"(%8) <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<201088x2880xbf16>) -> tensor<201088x1440xbf16>
        %390 = "ttir.reshape"(%389) <{shape = [1 : i32, 201088 : i32, 1440 : i32]}> : (tensor<201088x1440xbf16>) -> tensor<1x201088x1440xbf16>
        %391 = "ttir.reshape"(%390) <{shape = [201088 : i32, 1440 : i32]}> : (tensor<1x201088x1440xbf16>) -> tensor<201088x1440xbf16>
        %392 = "ttir.permute"(%391) <{permutation = array<i64: 1, 0>}> : (tensor<201088x1440xbf16>) -> tensor<1440x201088xbf16>
        %393 = "ttir.dot_general"(%388, %392) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x1440xbf16>, tensor<1440x201088xbf16>) -> tensor<128x201088xbf16>
        %394 = "ttir.all_reduce"(%393) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x201088xbf16>) -> tensor<128x201088xbf16>
        %395 = "ttir.reshape"(%394) <{shape = [1 : i32, 128 : i32, 201088 : i32]}> : (tensor<128x201088xbf16>) -> tensor<1x128x201088xbf16>
        %396 = "ttir.mesh_shard"(%318) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 4, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x2x127x64xbf16>) -> tensor<1x8x127x64xbf16>
        %397 = "ttir.mesh_shard"(%319) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 4, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x2x127x64xbf16>) -> tensor<1x8x127x64xbf16>
        %398 = "ttir.mesh_shard"(%395) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x128x201088xbf16>) -> tensor<1x128x201088xbf16>
        return %396, %397, %398 : tensor<1x8x127x64xbf16>, tensor<1x8x127x64xbf16>, tensor<1x128x201088xbf16>
      }
    }
  }
}
// -----------------------------------------------------------------------------
// END TTIR MODULE
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// START TTNN MODULE
// -----------------------------------------------------------------------------
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073136832, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073154112, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0, 1, 2, 3, 4, 5, 6, 7], [1 : i32, 1 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], [ 0x0x0x0], [<[0, 6, 0], [3, 6, 0]>, <[0, 7, 0], [3, 7, 0]>, <[0, 9, 0], [4, 1, 0]>, <[0, 14, 0], [1, 14, 0]>, <[0, 15, 0], [1, 15, 0]>, <[1, 0, 0], [2, 0, 0]>, <[1, 1, 0], [2, 1, 0]>, <[1, 9, 0], [6, 1, 0]>, <[2, 9, 0], [7, 1, 0]>, <[2, 14, 0], [3, 14, 0]>, <[2, 15, 0], [3, 15, 0]>, <[3, 9, 0], [5, 1, 0]>, <[4, 6, 0], [6, 6, 0]>, <[5, 6, 0], [7, 6, 0]>]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<201088x2880xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<201088x1440xbf16, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<201088x2880xbf16, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<6284x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<6284x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<128x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<92160x5760xbf16, #system_memory>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 11520 + d1 * 2880 + d2, d3), <1x1>, memref<360x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<92160x5760xbf16, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<2880x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<360x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2880xbf16, #system_memory>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2880xbf16, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xf32, #system_memory>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xf32, #dram>, <interleaved>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4096x2880xbf16, #system_memory>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4096x2880xbf16, #dram>, <interleaved>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout34 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2880xbf16, #system_memory>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<45x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2880xbf16, #dram>, <interleaved>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<45x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<92160x2880xbf16, #system_memory>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 11520 + d1 * 2880 + d2, d3), <1x1>, memref<360x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<92160x2880xbf16, #dram>, <interleaved>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<2880x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<360x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout46 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout47 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x128xui32, #dram>, <interleaved>>
#ttnn_layout48 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout49 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout50 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout51 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x512xbf16, #system_memory>>
#ttnn_layout52 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout53 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x512xbf16, #dram>, <interleaved>>
#ttnn_layout54 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout55 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout56 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2880x4096xbf16, #system_memory>>
#ttnn_layout57 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<45x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout58 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2880x4096xbf16, #dram>, <interleaved>>
#ttnn_layout59 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<90x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout60 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4096xbf16, #system_memory>>
#ttnn_layout61 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout62 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4096xbf16, #dram>, <interleaved>>
#ttnn_layout63 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout64 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout65 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout66 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 128 + d2, d3), <1x1>, memref<64x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout67 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64xbf16, #dram>, <interleaved>>
#ttnn_layout68 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16xbf16, #dram>, <interleaved>>
#ttnn_layout69 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout70 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout71 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x5760xbf16, #system_memory>>
#ttnn_layout72 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout73 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x5760xbf16, #dram>, <interleaved>>
#ttnn_layout74 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout75 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout76 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1 * 128 + d2 * 32 + d3, d4), <1x1>, memref<4x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout77 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x8xsi32, #system_memory>>
#ttnn_layout78 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x8xui16, #dram>, <interleaved>>
#ttnn_layout79 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x8xui16, #system_memory>>
#ttnn_layout80 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x45x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout81 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xbf16, #system_memory>>
#ttnn_layout82 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32xbf16, #dram>, <interleaved>>
#ttnn_layout83 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout84 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout85 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout86 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1 * 128 + d2 * 32 + d3, d4), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout87 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xbf16, #system_memory>>
#ttnn_layout88 = #ttnn.ttnn_layout<() -> (0, 0), <1x1>, memref<1x1xbf16, #dram>, <interleaved>>
#ttnn_layout89 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<512x2880xbf16, #system_memory>>
#ttnn_layout90 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout91 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<512x2880xbf16, #dram>, <interleaved>>
#ttnn_layout92 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout93 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x45x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout94 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout95 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout96 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout97 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout98 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout99 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout100 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x128xsi32, #dram>, <interleaved>>
#ttnn_layout101 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout102 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x6284x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout103 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout104 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x45x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout105 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout106 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout107 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout108 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout109 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout110 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x45x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout111 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout112 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout113 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout114 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout115 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 32 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout116 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 128 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout117 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout118 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout119 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout120 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout121 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout122 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 128 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout123 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout124 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout125 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x1xbf16, #dram>, <interleaved>>
#ttnn_layout126 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout127 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout128 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 128 + d2, d3), <1x1>, memref<64x5x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout129 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout130 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout131 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout132 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout133 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout134 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout135 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout136 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x8x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout137 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout138 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x256xsi32, #dram>, <interleaved>>
#ttnn_layout139 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x256xbf16, #dram>, <interleaved>>
#ttnn_layout140 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout141 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>
#ttnn_layout142 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout143 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, u16>, #system_memory>>
#ttnn_layout144 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x4xui16, #system_memory>>
#ttnn_layout145 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x4xui16, #dram>, <interleaved>>
#ttnn_layout146 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<256x2880xbf16, #dram>, <interleaved>>
#ttnn_layout147 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<256x4xui16, #dram>, <interleaved>>
#ttnn_layout148 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout149 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<256x32xbf16, #dram>, <interleaved>>
#ttnn_layout150 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<256x4xbf16, #dram>, <interleaved>>
#ttnn_layout151 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<8x4xui16, #dram>, <interleaved>>
#ttnn_layout152 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout153 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<8x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout154 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout155 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 + d2, d3), <1x1>, memref<8x4xbf16, #dram>, <interleaved>>
#ttnn_layout156 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 512 + d1 * 128 + d2 * 128 + d3 * 32 + d4, d5), <1x1>, memref<32x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout157 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<32x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout158 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout159 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<8x4xui16, #system_memory>>
#ttnn_layout160 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<8x4xbf16, #system_memory>>
#ttnn_layout161 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<8x4xbf16, #dram>, <interleaved>>
#ttnn_layout162 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 512 + d1 * 128 + d2 * 32 + d3, d4), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout163 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 256 + d1 * 128 + d2 * 32 + d3, d4), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout164 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout165 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<1024x2880xbf16, #dram>, <interleaved>>
#ttnn_layout166 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<512x2880xbf16, #dram>, <interleaved>>
#ttnn_layout167 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<16x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout168 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout169 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<512x1440xbf16, #dram>, <interleaved>>
#ttnn_layout170 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<16x45x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout171 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<128x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout172 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout173 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout174 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x6284x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout175 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x6284x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout176 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x3142x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout177 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x3142x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module @SyncTensorsGraph.683 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.683 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 2x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7], meshTopology = [linear, ring]>
      func.func private @main_const_eval_0(%arg0: tensor<201088x2880xbf16, #ttnn_layout>) -> tensor<201088x1440xbf16, #ttnn_layout1> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<201088x2880xbf16, #ttnn_layout>, !ttnn.device) -> tensor<201088x2880xbf16, #ttnn_layout2>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<201088x2880xbf16, #ttnn_layout2>) -> tensor<201088x2880xbf16, #ttnn_layout3>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<201088x2880xbf16, #ttnn_layout2>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<201088x2880xbf16, #ttnn_layout3>, !ttnn.device) -> tensor<201088x1440xbf16, #ttnn_layout4>
        %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<201088x1440xbf16, #ttnn_layout4>) -> tensor<201088x1440xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<201088x2880xbf16, #ttnn_layout3>) -> ()
        return %4 : tensor<201088x1440xbf16, #ttnn_layout1>
      }
      func.func private @main_const_eval_1() -> tensor<512x1xsi32, #ttnn_layout5> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, end = 128 : i64, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<128xsi32, #ttnn_layout6>
        %2 = "ttnn.reshape"(%1) <{shape = [128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xsi32, #ttnn_layout6>) -> tensor<128x1x1xsi32, #ttnn_layout7>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<128xsi32, #ttnn_layout6>) -> ()
        %3 = "ttnn.repeat"(%2) <{repeat_dims = #ttnn.shape<1x4x1>}> : (tensor<128x1x1xsi32, #ttnn_layout7>) -> tensor<128x4x1xsi32, #ttnn_layout7>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<128x1x1xsi32, #ttnn_layout7>) -> ()
        %4 = "ttnn.reshape"(%3) <{shape = [512 : i32, 1 : i32]}> : (tensor<128x4x1xsi32, #ttnn_layout7>) -> tensor<512x1xsi32, #ttnn_layout5>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<128x4x1xsi32, #ttnn_layout7>) -> ()
        return %4 : tensor<512x1xsi32, #ttnn_layout5>
      }
      func.func private @main_const_eval_2() -> (tensor<1x1x1xf32, #ttnn_layout8>, tensor<1x1xf32, #ttnn_layout9>) attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 9.99999974E-6 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<f32, #ttnn_layout10>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32, #ttnn_layout10>) -> tensor<1x1x1xf32, #ttnn_layout8>
        %3 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32, #ttnn_layout10>) -> tensor<1x1xf32, #ttnn_layout9>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<f32, #ttnn_layout10>) -> ()
        return %2, %3 : tensor<1x1x1xf32, #ttnn_layout8>, tensor<1x1xf32, #ttnn_layout9>
      }
      func.func private @main_const_eval_3(%arg0: tensor<32x2880x5760xbf16, #ttnn_layout11>) -> tensor<1x4x2880x5760xbf16, #ttnn_layout12> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2880x5760xbf16, #ttnn_layout11>, !ttnn.device) -> tensor<32x2880x5760xbf16, #ttnn_layout13>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x2880x5760xbf16, #ttnn_layout13>) -> tensor<32x2880x5760xbf16, #ttnn_layout14>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x2880x5760xbf16, #ttnn_layout13>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880x5760xbf16, #ttnn_layout14>, !ttnn.device) -> tensor<4x2880x5760xbf16, #ttnn_layout15>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 4 : i32, 2880 : i32, 5760 : i32]}> : (tensor<4x2880x5760xbf16, #ttnn_layout15>) -> tensor<1x4x2880x5760xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x2880x5760xbf16, #ttnn_layout14>) -> ()
        return %4 : tensor<1x4x2880x5760xbf16, #ttnn_layout12>
      }
      func.func private @main_const_eval_4(%arg0: tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xbf16, #ttnn_layout17> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<2880xbf16, #ttnn_layout16>, !ttnn.device) -> tensor<2880xbf16, #ttnn_layout18>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<2880xbf16, #ttnn_layout18>) -> tensor<2880xbf16, #ttnn_layout19>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2880xbf16, #ttnn_layout18>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16, #ttnn_layout19>, !ttnn.device) -> tensor<1440xbf16, #ttnn_layout20>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1440 : i32]}> : (tensor<1440xbf16, #ttnn_layout20>) -> tensor<1x1440xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<2880xbf16, #ttnn_layout19>) -> ()
        return %4 : tensor<1x1440xbf16, #ttnn_layout17>
      }
      func.func private @main_const_eval_5(%arg0: tensor<32xf32, #ttnn_layout21>) -> (tensor<1x1x128x32xbf16, #ttnn_layout22>, tensor<1x1x128x32xbf16, #ttnn_layout22>) attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xf32, #ttnn_layout21>, !ttnn.device) -> tensor<32xf32, #ttnn_layout23>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32xf32, #ttnn_layout23>) -> tensor<32xf32, #ttnn_layout24>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32xf32, #ttnn_layout23>) -> ()
        %3 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 1.34657359 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<f32, #ttnn_layout10>
        %4 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<"0x000000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C42000080420000824200008442000086420000884200008A4200008C4200008E42000090420000924200009442000096420000984200009A4200009C4200009E420000A0420000A2420000A4420000A6420000A8420000AA420000AC420000AE420000B0420000B2420000B4420000B6420000B8420000BA420000BC420000BE420000C0420000C2420000C4420000C6420000C8420000CA420000CC420000CE420000D0420000D2420000D4420000D6420000D8420000DA420000DC420000DE420000E0420000E2420000E4420000E6420000E8420000EA420000EC420000EE420000F0420000F2420000F4420000F6420000F8420000FA420000FC420000FE42"> : tensor<1x1x128xf32>}> : (!ttnn.device) -> tensor<1x1x128xf32, #ttnn_layout25>
        %5 = "ttnn.reshape"(%2) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<32xf32, #ttnn_layout24>) -> tensor<1x32x1xf32, #ttnn_layout8>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32xf32, #ttnn_layout24>) -> ()
        %6 = "ttnn.matmul"(%5, %4) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x1xf32, #ttnn_layout8>, tensor<1x1x128xf32, #ttnn_layout25>) -> tensor<1x32x128xf32, #ttnn_layout25>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x32x1xf32, #ttnn_layout8>) -> ()
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x128xf32, #ttnn_layout25>) -> ()
        %7 = "ttnn.permute"(%6) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x32x128xf32, #ttnn_layout25>) -> tensor<1x128x32xf32, #ttnn_layout26>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x32x128xf32, #ttnn_layout25>) -> ()
        %8 = "ttnn.reshape"(%7) <{shape = [1 : i32, 1 : i32, 128 : i32, 32 : i32]}> : (tensor<1x128x32xf32, #ttnn_layout26>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x128x32xf32, #ttnn_layout26>) -> ()
        %9 = "ttnn.cos"(%8) : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        %10 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32, #ttnn_layout10>) -> tensor<1x1x1x1xf32, #ttnn_layout28>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<f32, #ttnn_layout10>) -> ()
        %11 = "ttnn.multiply"(%9, %10) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x128x32xf32, #ttnn_layout27>, tensor<1x1x1x1xf32, #ttnn_layout28>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        %12 = "ttnn.typecast"(%11) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> tensor<1x1x128x32xbf16, #ttnn_layout22>
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        %13 = "ttnn.sin"(%8) : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        %14 = "ttnn.multiply"(%13, %10) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x128x32xf32, #ttnn_layout27>, tensor<1x1x1x1xf32, #ttnn_layout28>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout28>) -> ()
        %15 = "ttnn.typecast"(%14) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> tensor<1x1x128x32xbf16, #ttnn_layout22>
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        return %12, %15 : tensor<1x1x128x32xbf16, #ttnn_layout22>, tensor<1x1x128x32xbf16, #ttnn_layout22>
      }
      func.func private @main_const_eval_6(%arg0: tensor<4096x2880xbf16, #ttnn_layout29>) -> tensor<1024x1440xbf16, #ttnn_layout30> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<4096x2880xbf16, #ttnn_layout29>, !ttnn.device) -> tensor<4096x2880xbf16, #ttnn_layout31>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<4096x2880xbf16, #ttnn_layout31>) -> tensor<4096x2880xbf16, #ttnn_layout32>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<4096x2880xbf16, #ttnn_layout31>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096x2880xbf16, #ttnn_layout32>, !ttnn.device) -> tensor<1024x1440xbf16, #ttnn_layout30>
        return %3 : tensor<1024x1440xbf16, #ttnn_layout30>
      }
      func.func private @main_const_eval_7() -> tensor<1x1x1x1xbf16, #ttnn_layout33> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<bf16, #ttnn_layout34>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        return %2 : tensor<1x1x1x1xbf16, #ttnn_layout33>
      }
      func.func private @main_const_eval_8(%arg0: tensor<32x2880xbf16, #ttnn_layout35>) -> tensor<1440x32xf32, #ttnn_layout36> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2880xbf16, #ttnn_layout35>, !ttnn.device) -> tensor<32x2880xbf16, #ttnn_layout37>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x2880xbf16, #ttnn_layout37>) -> tensor<32x2880xbf16, #ttnn_layout38>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout37>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880xbf16, #ttnn_layout38>, !ttnn.device) -> tensor<32x1440xbf16, #ttnn_layout17>
        %4 = "ttnn.permute"(%3) <{permutation = array<i64: 1, 0>}> : (tensor<32x1440xbf16, #ttnn_layout17>) -> tensor<1440x32xbf16, #ttnn_layout39>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout38>) -> ()
        %5 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1440x32xbf16, #ttnn_layout39>) -> tensor<1440x32xf32, #ttnn_layout36>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1440x32xbf16, #ttnn_layout39>) -> ()
        return %5 : tensor<1440x32xf32, #ttnn_layout36>
      }
      func.func private @main_const_eval_9(%arg0: tensor<32x2880x2880xbf16, #ttnn_layout40>) -> tensor<1x4x2880x2880xbf16, #ttnn_layout41> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2880x2880xbf16, #ttnn_layout40>, !ttnn.device) -> tensor<32x2880x2880xbf16, #ttnn_layout42>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x2880x2880xbf16, #ttnn_layout42>) -> tensor<32x2880x2880xbf16, #ttnn_layout43>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x2880x2880xbf16, #ttnn_layout42>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880x2880xbf16, #ttnn_layout43>, !ttnn.device) -> tensor<4x2880x2880xbf16, #ttnn_layout44>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 4 : i32, 2880 : i32, 2880 : i32]}> : (tensor<4x2880x2880xbf16, #ttnn_layout44>) -> tensor<1x4x2880x2880xbf16, #ttnn_layout41>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x2880x2880xbf16, #ttnn_layout43>) -> ()
        return %4 : tensor<1x4x2880x2880xbf16, #ttnn_layout41>
      }
      func.func private @main_const_eval_10() -> tensor<1x1xsi32, #ttnn_layout45> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 0 : i32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<si32, #ttnn_layout46>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32]}> : (tensor<si32, #ttnn_layout46>) -> tensor<1x1xsi32, #ttnn_layout45>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<si32, #ttnn_layout46>) -> ()
        return %2 : tensor<1x1xsi32, #ttnn_layout45>
      }
      func.func private @main_const_eval_11() -> tensor<1x128xui32, #ttnn_layout47> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<"0x0000000000000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F000000"> : tensor<1x128x2xsi32>}> : (!ttnn.device) -> tensor<1x128x2xsi32, #ttnn_layout48>
        %2 = "ttnn.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 128 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x128x2xsi32, #ttnn_layout48>) -> tensor<1x128x1xsi32, #ttnn_layout48>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x128x2xsi32, #ttnn_layout48>) -> ()
        %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 128 : i32]}> : (tensor<1x128x1xsi32, #ttnn_layout48>) -> tensor<1x128xsi32, #ttnn_layout49>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x128x1xsi32, #ttnn_layout48>) -> ()
        %4 = "ttnn.typecast"(%3) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x128xsi32, #ttnn_layout49>) -> tensor<1x128xui32, #ttnn_layout50>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x128xsi32, #ttnn_layout49>) -> ()
        %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<1x128xui32, #ttnn_layout50>) -> tensor<1x128xui32, #ttnn_layout47>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x128xui32, #ttnn_layout50>) -> ()
        return %5 : tensor<1x128xui32, #ttnn_layout47>
      }
      func.func private @main_const_eval_12(%arg0: tensor<512xbf16, #ttnn_layout51>) -> tensor<1x128xbf16, #ttnn_layout52> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512xbf16, #ttnn_layout51>, !ttnn.device) -> tensor<512xbf16, #ttnn_layout53>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<512xbf16, #ttnn_layout53>) -> tensor<512xbf16, #ttnn_layout54>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512xbf16, #ttnn_layout53>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16, #ttnn_layout54>, !ttnn.device) -> tensor<128xbf16, #ttnn_layout55>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xbf16, #ttnn_layout55>) -> tensor<1x128xbf16, #ttnn_layout52>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<512xbf16, #ttnn_layout54>) -> ()
        return %4 : tensor<1x128xbf16, #ttnn_layout52>
      }
      func.func private @main_const_eval_13(%arg0: tensor<2880x4096xbf16, #ttnn_layout56>) -> tensor<1440x1024xbf16, #ttnn_layout57> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<2880x4096xbf16, #ttnn_layout56>, !ttnn.device) -> tensor<2880x4096xbf16, #ttnn_layout58>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<2880x4096xbf16, #ttnn_layout58>) -> tensor<2880x4096xbf16, #ttnn_layout59>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2880x4096xbf16, #ttnn_layout58>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880x4096xbf16, #ttnn_layout59>, !ttnn.device) -> tensor<1440x1024xbf16, #ttnn_layout57>
        return %3 : tensor<1440x1024xbf16, #ttnn_layout57>
      }
      func.func private @main_const_eval_14(%arg0: tensor<4096xbf16, #ttnn_layout60>) -> tensor<1x1024xbf16, #ttnn_layout61> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<4096xbf16, #ttnn_layout60>, !ttnn.device) -> tensor<4096xbf16, #ttnn_layout62>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<4096xbf16, #ttnn_layout62>) -> tensor<4096xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<4096xbf16, #ttnn_layout62>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<4096xbf16, #ttnn_layout63>, !ttnn.device) -> tensor<1024xbf16, #ttnn_layout64>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xbf16, #ttnn_layout64>) -> tensor<1x1024xbf16, #ttnn_layout61>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<4096xbf16, #ttnn_layout63>) -> ()
        return %4 : tensor<1x1024xbf16, #ttnn_layout61>
      }
      func.func private @main_const_eval_15(%arg0: tensor<64xbf16, #ttnn_layout65>) -> tensor<1x16x128x1xbf16, #ttnn_layout66> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64xbf16, #ttnn_layout65>, !ttnn.device) -> tensor<64xbf16, #ttnn_layout67>
        %2 = "ttnn.mesh_partition"(%1) <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<64xbf16, #ttnn_layout67>) -> tensor<16xbf16, #ttnn_layout68>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<64xbf16, #ttnn_layout67>) -> ()
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>}> : (tensor<16xbf16, #ttnn_layout68>) -> tensor<16xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<16xbf16, #ttnn_layout68>) -> ()
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 16 : i32, 1 : i32, 1 : i32]}> : (tensor<16xbf16, #ttnn_layout69>) -> tensor<1x16x1x1xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<16xbf16, #ttnn_layout69>) -> ()
        %5 = "ttnn.repeat"(%4) <{repeat_dims = #ttnn.shape<1x1x128x1>}> : (tensor<1x16x1x1xbf16, #ttnn_layout70>) -> tensor<1x16x128x1xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x16x1x1xbf16, #ttnn_layout70>) -> ()
        return %5 : tensor<1x16x128x1xbf16, #ttnn_layout66>
      }
      func.func private @main_const_eval_16() -> tensor<4096xbf16, #ttnn_layout62> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 0.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<bf16, #ttnn_layout34>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        %3 = "ttnn.repeat"(%2) <{repeat_dims = #ttnn.shape<4096>}> : (tensor<1xbf16, #ttnn_layout69>) -> tensor<4096xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1xbf16, #ttnn_layout69>) -> ()
        %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<4096xbf16, #ttnn_layout63>) -> tensor<4096xbf16, #ttnn_layout62>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<4096xbf16, #ttnn_layout63>) -> ()
        return %4 : tensor<4096xbf16, #ttnn_layout62>
      }
      func.func private @main_const_eval_17(%arg0: tensor<32x5760xbf16, #ttnn_layout71>) -> tensor<1x4x1x5760xbf16, #ttnn_layout72> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x5760xbf16, #ttnn_layout71>, !ttnn.device) -> tensor<32x5760xbf16, #ttnn_layout73>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x5760xbf16, #ttnn_layout73>) -> tensor<32x5760xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x5760xbf16, #ttnn_layout73>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x5760xbf16, #ttnn_layout74>, !ttnn.device) -> tensor<4x5760xbf16, #ttnn_layout74>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32, 4 : i32, 5760 : i32]}> : (tensor<4x5760xbf16, #ttnn_layout74>) -> tensor<1x1x1x4x5760xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x5760xbf16, #ttnn_layout74>) -> ()
        %5 = "ttnn.permute"(%4) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<1x1x1x4x5760xbf16, #ttnn_layout75>) -> tensor<1x1x4x1x5760xbf16, #ttnn_layout76>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1x4x5760xbf16, #ttnn_layout75>) -> ()
        %6 = "ttnn.reshape"(%5) <{shape = [1 : i32, 4 : i32, 1 : i32, 5760 : i32]}> : (tensor<1x1x4x1x5760xbf16, #ttnn_layout76>) -> tensor<1x4x1x5760xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x4x1x5760xbf16, #ttnn_layout76>) -> ()
        return %6 : tensor<1x4x1x5760xbf16, #ttnn_layout72>
      }
      func.func private @main_const_eval_18() -> (tensor<1x1xf32, #ttnn_layout9>, tensor<1x1x1xf32, #ttnn_layout8>) attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, fill_value = 3.47222231E-4 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<f32, #ttnn_layout10>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32, #ttnn_layout10>) -> tensor<1x1xf32, #ttnn_layout9>
        %3 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32, #ttnn_layout10>) -> tensor<1x1x1xf32, #ttnn_layout8>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<f32, #ttnn_layout10>) -> ()
        return %2, %3 : tensor<1x1xf32, #ttnn_layout9>, tensor<1x1x1xf32, #ttnn_layout8>
      }
      func.func private @main_const_eval_19(%arg0: tensor<1x1x32x8xsi32, #ttnn_layout77>) -> tensor<1x1x32x8xui16, #ttnn_layout78> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<u16>}> : (tensor<1x1x32x8xsi32, #ttnn_layout77>) -> tensor<1x1x32x8xui16, #ttnn_layout79>
        %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x32x8xui16, #ttnn_layout79>, !ttnn.device) -> tensor<1x1x32x8xui16, #ttnn_layout78>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x32x8xui16, #ttnn_layout79>) -> ()
        return %2 : tensor<1x1x32x8xui16, #ttnn_layout78>
      }
      func.func private @main_const_eval_20(%arg0: tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xf32, #ttnn_layout80> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<2880xbf16, #ttnn_layout16>, !ttnn.device) -> tensor<2880xbf16, #ttnn_layout18>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<2880xbf16, #ttnn_layout18>) -> tensor<2880xbf16, #ttnn_layout19>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2880xbf16, #ttnn_layout18>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16, #ttnn_layout19>, !ttnn.device) -> tensor<1440xbf16, #ttnn_layout20>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1440 : i32]}> : (tensor<1440xbf16, #ttnn_layout20>) -> tensor<1x1440xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<2880xbf16, #ttnn_layout19>) -> ()
        %5 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1440xbf16, #ttnn_layout17>) -> tensor<1x1440xf32, #ttnn_layout80>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1440xbf16, #ttnn_layout17>) -> ()
        return %5 : tensor<1x1440xf32, #ttnn_layout80>
      }
      func.func private @main_const_eval_21(%arg0: tensor<32xbf16, #ttnn_layout81>) -> tensor<1x32xf32, #ttnn_layout9> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32xbf16, #ttnn_layout81>, !ttnn.device) -> tensor<32xbf16, #ttnn_layout82>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32xbf16, #ttnn_layout82>) -> tensor<32xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32xbf16, #ttnn_layout82>) -> ()
        %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 32 : i32]}> : (tensor<32xbf16, #ttnn_layout69>) -> tensor<1x32xbf16, #ttnn_layout83>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32xbf16, #ttnn_layout69>) -> ()
        %4 = "ttnn.typecast"(%3) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x32xbf16, #ttnn_layout83>) -> tensor<1x32xf32, #ttnn_layout9>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32xbf16, #ttnn_layout83>) -> ()
        return %4 : tensor<1x32xf32, #ttnn_layout9>
      }
      func.func private @main_const_eval_22(%arg0: tensor<32x2880xbf16, #ttnn_layout35>) -> tensor<1x4x1x2880xbf16, #ttnn_layout84> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<32x2880xbf16, #ttnn_layout35>, !ttnn.device) -> tensor<32x2880xbf16, #ttnn_layout37>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<32x2880xbf16, #ttnn_layout37>) -> tensor<32x2880xbf16, #ttnn_layout38>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout37>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 8, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x2880xbf16, #ttnn_layout38>, !ttnn.device) -> tensor<4x2880xbf16, #ttnn_layout38>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32, 4 : i32, 2880 : i32]}> : (tensor<4x2880xbf16, #ttnn_layout38>) -> tensor<1x1x1x4x2880xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout38>) -> ()
        %5 = "ttnn.permute"(%4) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<1x1x1x4x2880xbf16, #ttnn_layout85>) -> tensor<1x1x4x1x2880xbf16, #ttnn_layout86>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1x4x2880xbf16, #ttnn_layout85>) -> ()
        %6 = "ttnn.reshape"(%5) <{shape = [1 : i32, 4 : i32, 1 : i32, 2880 : i32]}> : (tensor<1x1x4x1x2880xbf16, #ttnn_layout86>) -> tensor<1x4x1x2880xbf16, #ttnn_layout84>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x4x1x2880xbf16, #ttnn_layout86>) -> ()
        return %6 : tensor<1x4x1x2880xbf16, #ttnn_layout84>
      }
      func.func private @main_const_eval_23(%arg0: tensor<201088x2880xbf16, #ttnn_layout>) -> tensor<201088x1440xbf16, #ttnn_layout4> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<201088x2880xbf16, #ttnn_layout>, !ttnn.device) -> tensor<201088x2880xbf16, #ttnn_layout2>
        %2 = "ttnn.mesh_partition"(%1) <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<201088x2880xbf16, #ttnn_layout2>) -> tensor<201088x1440xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<201088x2880xbf16, #ttnn_layout2>) -> ()
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<tile>}> : (tensor<201088x1440xbf16, #ttnn_layout1>) -> tensor<201088x1440xbf16, #ttnn_layout4>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<201088x1440xbf16, #ttnn_layout1>) -> ()
        return %3 : tensor<201088x1440xbf16, #ttnn_layout4>
      }
      func.func private @main_const_eval_24(%arg0: tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xf32, #ttnn_layout80> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<2880xbf16, #ttnn_layout16>, !ttnn.device) -> tensor<2880xbf16, #ttnn_layout18>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<2880xbf16, #ttnn_layout18>) -> tensor<2880xbf16, #ttnn_layout19>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2880xbf16, #ttnn_layout18>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16, #ttnn_layout19>, !ttnn.device) -> tensor<1440xbf16, #ttnn_layout20>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1440 : i32]}> : (tensor<1440xbf16, #ttnn_layout20>) -> tensor<1x1440xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<2880xbf16, #ttnn_layout19>) -> ()
        %5 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1440xbf16, #ttnn_layout17>) -> tensor<1x1440xf32, #ttnn_layout80>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1440xbf16, #ttnn_layout17>) -> ()
        return %5 : tensor<1x1440xf32, #ttnn_layout80>
      }
      func.func private @main_const_eval_25() -> tensor<1x1x1x1xbf16, #ttnn_layout33> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.250000e-01 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<bf16, #ttnn_layout34>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        return %2 : tensor<1x1x1x1xbf16, #ttnn_layout33>
      }
      func.func private @main_const_eval_26() -> tensor<1x1xsi32, #ttnn_layout45> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 32 : i32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x1>}> : (!ttnn.device) -> tensor<1x1xsi32, #ttnn_layout45>
        return %1 : tensor<1x1xsi32, #ttnn_layout45>
      }
      func.func private @main_const_eval_27() -> tensor<1x1x1x1xbf16, #ttnn_layout33> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = -3.38953139E+38 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<bf16, #ttnn_layout34>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        return %2 : tensor<1x1x1x1xbf16, #ttnn_layout33>
      }
      func.func private @main_const_eval_28(%arg0: tensor<bf16, #ttnn_layout87>) -> tensor<1x1x1x1xbf16, #ttnn_layout33> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<bf16, #ttnn_layout87>, !ttnn.device) -> tensor<bf16, #ttnn_layout88>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<bf16, #ttnn_layout88>) -> tensor<bf16, #ttnn_layout34>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout88>) -> ()
        %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        return %3 : tensor<1x1x1x1xbf16, #ttnn_layout33>
      }
      func.func private @main_const_eval_29(%arg0: tensor<512x2880xbf16, #ttnn_layout89>) -> tensor<128x1440xbf16, #ttnn_layout90> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512x2880xbf16, #ttnn_layout89>, !ttnn.device) -> tensor<512x2880xbf16, #ttnn_layout91>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<512x2880xbf16, #ttnn_layout91>) -> tensor<512x2880xbf16, #ttnn_layout92>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512x2880xbf16, #ttnn_layout91>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512x2880xbf16, #ttnn_layout92>, !ttnn.device) -> tensor<128x1440xbf16, #ttnn_layout90>
        return %3 : tensor<128x1440xbf16, #ttnn_layout90>
      }
      func.func private @main_const_eval_30(%arg0: tensor<512xbf16, #ttnn_layout51>) -> tensor<1x128xbf16, #ttnn_layout52> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512xbf16, #ttnn_layout51>, !ttnn.device) -> tensor<512xbf16, #ttnn_layout53>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<512xbf16, #ttnn_layout53>) -> tensor<512xbf16, #ttnn_layout54>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512xbf16, #ttnn_layout53>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512xbf16, #ttnn_layout54>, !ttnn.device) -> tensor<128xbf16, #ttnn_layout55>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xbf16, #ttnn_layout55>) -> tensor<1x128xbf16, #ttnn_layout52>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<512xbf16, #ttnn_layout54>) -> ()
        return %4 : tensor<1x128xbf16, #ttnn_layout52>
      }
      func.func private @main_const_eval_31(%arg0: tensor<512x2880xbf16, #ttnn_layout89>) -> tensor<128x1440xbf16, #ttnn_layout90> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<512x2880xbf16, #ttnn_layout89>, !ttnn.device) -> tensor<512x2880xbf16, #ttnn_layout91>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<512x2880xbf16, #ttnn_layout91>) -> tensor<512x2880xbf16, #ttnn_layout92>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512x2880xbf16, #ttnn_layout91>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 4, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<512x2880xbf16, #ttnn_layout92>, !ttnn.device) -> tensor<128x1440xbf16, #ttnn_layout90>
        return %3 : tensor<128x1440xbf16, #ttnn_layout90>
      }
      func.func private @main_const_eval_32(%arg0: tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1x1440xf32, #ttnn_layout93> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.to_device"(%arg0, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<2880xbf16, #ttnn_layout16>, !ttnn.device) -> tensor<2880xbf16, #ttnn_layout18>
        %2 = "ttnn.to_layout"(%1) <{layout = #ttnn.layout<tile>}> : (tensor<2880xbf16, #ttnn_layout18>) -> tensor<2880xbf16, #ttnn_layout19>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2880xbf16, #ttnn_layout18>) -> ()
        %3 = "ttnn.mesh_shard"(%2, %0) <{shard_dims = array<i64: 0, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2880xbf16, #ttnn_layout19>, !ttnn.device) -> tensor<1440xbf16, #ttnn_layout20>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1440 : i32]}> : (tensor<1440xbf16, #ttnn_layout20>) -> tensor<1x1x1440xbf16, #ttnn_layout94>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<2880xbf16, #ttnn_layout19>) -> ()
        %5 = "ttnn.typecast"(%4) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x1440xbf16, #ttnn_layout94>) -> tensor<1x1x1440xf32, #ttnn_layout93>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1440xbf16, #ttnn_layout94>) -> ()
        return %5 : tensor<1x1x1440xf32, #ttnn_layout93>
      }
      func.func private @main_const_eval_33() -> tensor<1x1x32xsi32, #ttnn_layout95> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xsi32>}> : (!ttnn.device) -> tensor<32xsi32, #ttnn_layout96>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xsi32, #ttnn_layout96>) -> tensor<1x1x32xsi32, #ttnn_layout95>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32xsi32, #ttnn_layout96>) -> ()
        return %2 : tensor<1x1x32xsi32, #ttnn_layout95>
      }
      func.func private @main_const_eval_34() -> (tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x128x128xbf16, #ttnn_layout97>) attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<"0x80FFFFFF81FFFFFF82FFFFFF83FFFFFF84FFFFFF85FFFFFF86FFFFFF87FFFFFF88FFFFFF89FFFFFF8AFFFFFF8BFFFFFF8CFFFFFF8DFFFFFF8EFFFFFF8FFFFFFF90FFFFFF91FFFFFF92FFFFFF93FFFFFF94FFFFFF95FFFFFF96FFFFFF97FFFFFF98FFFFFF99FFFFFF9AFFFFFF9BFFFFFF9CFFFFFF9DFFFFFF9EFFFFFF9FFFFFFFA0FFFFFFA1FFFFFFA2FFFFFFA3FFFFFFA4FFFFFFA5FFFFFFA6FFFFFFA7FFFFFFA8FFFFFFA9FFFFFFAAFFFFFFABFFFFFFACFFFFFFADFFFFFFAEFFFFFFAFFFFFFFB0FFFFFFB1FFFFFFB2FFFFFFB3FFFFFFB4FFFFFFB5FFFFFFB6FFFFFFB7FFFFFFB8FFFFFFB9FFFFFFBAFFFFFFBBFFFFFFBCFFFFFFBDFFFFFFBEFFFFFFBFFFFFFFC0FFFFFFC1FFFFFFC2FFFFFFC3FFFFFFC4FFFFFFC5FFFFFFC6FFFFFFC7FFFFFFC8FFFFFFC9FFFFFFCAFFFFFFCBFFFFFFCCFFFFFFCDFFFFFFCEFFFFFFCFFFFFFFD0FFFFFFD1FFFFFFD2FFFFFFD3FFFFFFD4FFFFFFD5FFFFFFD6FFFFFFD7FFFFFFD8FFFFFFD9FFFFFFDAFFFFFFDBFFFFFFDCFFFFFFDDFFFFFFDEFFFFFFDFFFFFFFE0FFFFFFE1FFFFFFE2FFFFFFE3FFFFFFE4FFFFFFE5FFFFFFE6FFFFFFE7FFFFFFE8FFFFFFE9FFFFFFEAFFFFFFEBFFFFFFECFFFFFFEDFFFFFFEEFFFFFFEFFFFFFFF0FFFFFFF1FFFFFFF2FFFFFFF3FFFFFFF4FFFFFFF5FFFFFFF6FFFFFFF7FFFFFFF8FFFFFFF9FFFFFFFAFFFFFFFBFFFFFFFCFFFFFFFDFFFFFFFEFFFFFFFFFFFFFF"> : tensor<128xsi32>}> : (!ttnn.device) -> tensor<128xsi32, #ttnn_layout6>
        %2 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : tensor<128xsi32>}> : (!ttnn.device) -> tensor<128xsi32, #ttnn_layout6>
        %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<128xsi32, #ttnn_layout6>) -> tensor<1x1x1x128xsi32, #ttnn_layout98>
        %4 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<128xsi32, #ttnn_layout6>) -> tensor<1x1x128x1xsi32, #ttnn_layout99>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<128xsi32, #ttnn_layout6>) -> ()
        %5 = "ttnn.gt"(%3, %4) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x1x128xsi32, #ttnn_layout98>, tensor<1x1x128x1xsi32, #ttnn_layout99>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x128x1xsi32, #ttnn_layout99>) -> ()
        %6 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<128xsi32, #ttnn_layout6>) -> tensor<1x1x128x1xsi32, #ttnn_layout99>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<128xsi32, #ttnn_layout6>) -> ()
        %7 = "ttnn.ge"(%6, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x128x1xsi32, #ttnn_layout99>, tensor<1x1x1x128xsi32, #ttnn_layout98>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x128x1xsi32, #ttnn_layout99>) -> ()
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1x128xsi32, #ttnn_layout98>) -> ()
        return %5, %7 : tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x128x128xbf16, #ttnn_layout97>
      }
      func.func private @main_const_eval_35() -> tensor<1x1x1x1xbf16, #ttnn_layout33> attributes {tt.function_type = "const_eval"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.703125 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<>}> : (!ttnn.device) -> tensor<bf16, #ttnn_layout34>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        return %2 : tensor<1x1x1x1xbf16, #ttnn_layout33>
      }
      func.func @main(%arg0: tensor<512xbf16, #ttnn_layout51> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.bias"}, %arg1: tensor<512x2880xbf16, #ttnn_layout89> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_v_proj.weight"}, %arg2: tensor<2880xbf16, #ttnn_layout16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_input_layernorm_weight"}, %arg3: tensor<1x128xsi32, #ttnn_layout100> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_0"}, %arg4: tensor<201088x2880xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_embed_tokens.weight"}, %arg5: tensor<32xf32, #ttnn_layout21> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_rotary_emb_inv_freq"}, %arg6: tensor<512xbf16, #ttnn_layout51> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.bias"}, %arg7: tensor<512x2880xbf16, #ttnn_layout89> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_k_proj.weight"}, %arg8: tensor<201088x2880xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<201088x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___lm_head.weight"}, %arg9: tensor<2880xbf16, #ttnn_layout16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_norm_weight"}, %arg10: tensor<32xbf16, #ttnn_layout81> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_bias"}, %arg11: tensor<32x2880xbf16, #ttnn_layout35> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_router_weight"}, %arg12: tensor<2880xbf16, #ttnn_layout16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_post_attention_layernorm_weight"}, %arg13: tensor<2880xbf16, #ttnn_layout16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.bias"}, %arg14: tensor<2880x4096xbf16, #ttnn_layout56> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1440x1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_o_proj.weight"}, %arg15: tensor<64xbf16, #ttnn_layout65> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_sinks"}, %arg16: tensor<bf16, #ttnn_layout87> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<bf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L['self'].model.lifted_tensor_1"}, %arg17: tensor<1x128xsi32, #ttnn_layout100> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "args_1"}, %arg18: tensor<bf16, #ttnn_layout88> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<i1>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg19: tensor<4096xbf16, #ttnn_layout60> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.bias"}, %arg20: tensor<4096x2880xbf16, #ttnn_layout29> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1024x1440xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_self_attn_q_proj.weight"}, %arg21: tensor<1x1x32x8xsi32, #ttnn_layout77> {ttcore.argument_type = #ttcore.argument_type<constant>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x32x8xi64>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_expert_mapping"}, %arg22: tensor<32x2880xbf16, #ttnn_layout35> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj_bias"}, %arg23: tensor<32x2880x2880xbf16, #ttnn_layout40> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x2880xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_down_proj"}, %arg24: tensor<32x5760xbf16, #ttnn_layout71> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj_bias"}, %arg25: tensor<32x2880x5760xbf16, #ttnn_layout11> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<4x2880x5760xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttir.name = "L__self___model_layers_0_mlp_experts_gate_up_proj"}) -> (tensor<1x8x127x64xbf16, #ttnn_layout101> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x8x127x64xbf16, #ttnn_layout101> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x2x127x64xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, tensor<1x128x201088xbf16, #ttnn_layout102> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x128x201088xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) attributes {tt.function_type = "forward_device"} {
        %0 = ttcore.load_cached(@main_const_eval_0, [%arg4]) : (tensor<201088x2880xbf16, #ttnn_layout>) -> tensor<201088x1440xbf16, #ttnn_layout1>
        "ttnn.deallocate"(%arg4) <{force = false}> : (tensor<201088x2880xbf16, #ttnn_layout>) -> ()
        %1 = ttcore.load_cached(@main_const_eval_1, []) : () -> tensor<512x1xsi32, #ttnn_layout5>
        %2:2 = ttcore.load_cached(@main_const_eval_2, []) : () -> (tensor<1x1x1xf32, #ttnn_layout8>, tensor<1x1xf32, #ttnn_layout9>)
        %3 = ttcore.load_cached(@main_const_eval_3, [%arg25]) : (tensor<32x2880x5760xbf16, #ttnn_layout11>) -> tensor<1x4x2880x5760xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%arg25) <{force = false}> : (tensor<32x2880x5760xbf16, #ttnn_layout11>) -> ()
        %4 = ttcore.load_cached(@main_const_eval_4, [%arg13]) : (tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%arg13) <{force = false}> : (tensor<2880xbf16, #ttnn_layout16>) -> ()
        %5:2 = ttcore.load_cached(@main_const_eval_5, [%arg5]) : (tensor<32xf32, #ttnn_layout21>) -> (tensor<1x1x128x32xbf16, #ttnn_layout22>, tensor<1x1x128x32xbf16, #ttnn_layout22>)
        "ttnn.deallocate"(%arg5) <{force = false}> : (tensor<32xf32, #ttnn_layout21>) -> ()
        %6 = ttcore.load_cached(@main_const_eval_6, [%arg20]) : (tensor<4096x2880xbf16, #ttnn_layout29>) -> tensor<1024x1440xbf16, #ttnn_layout30>
        "ttnn.deallocate"(%arg20) <{force = false}> : (tensor<4096x2880xbf16, #ttnn_layout29>) -> ()
        %7 = ttcore.load_cached(@main_const_eval_7, []) : () -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        %8 = ttcore.load_cached(@main_const_eval_8, [%arg11]) : (tensor<32x2880xbf16, #ttnn_layout35>) -> tensor<1440x32xf32, #ttnn_layout36>
        "ttnn.deallocate"(%arg11) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout35>) -> ()
        %9 = ttcore.load_cached(@main_const_eval_9, [%arg23]) : (tensor<32x2880x2880xbf16, #ttnn_layout40>) -> tensor<1x4x2880x2880xbf16, #ttnn_layout41>
        "ttnn.deallocate"(%arg23) <{force = false}> : (tensor<32x2880x2880xbf16, #ttnn_layout40>) -> ()
        %10 = ttcore.load_cached(@main_const_eval_10, []) : () -> tensor<1x1xsi32, #ttnn_layout45>
        %11 = ttcore.load_cached(@main_const_eval_11, []) : () -> tensor<1x128xui32, #ttnn_layout47>
        %12 = ttcore.load_cached(@main_const_eval_12, [%arg6]) : (tensor<512xbf16, #ttnn_layout51>) -> tensor<1x128xbf16, #ttnn_layout52>
        "ttnn.deallocate"(%arg6) <{force = false}> : (tensor<512xbf16, #ttnn_layout51>) -> ()
        %13 = ttcore.load_cached(@main_const_eval_13, [%arg14]) : (tensor<2880x4096xbf16, #ttnn_layout56>) -> tensor<1440x1024xbf16, #ttnn_layout57>
        "ttnn.deallocate"(%arg14) <{force = false}> : (tensor<2880x4096xbf16, #ttnn_layout56>) -> ()
        %14 = ttcore.load_cached(@main_const_eval_14, [%arg19]) : (tensor<4096xbf16, #ttnn_layout60>) -> tensor<1x1024xbf16, #ttnn_layout61>
        "ttnn.deallocate"(%arg19) <{force = false}> : (tensor<4096xbf16, #ttnn_layout60>) -> ()
        %15 = ttcore.load_cached(@main_const_eval_15, [%arg15]) : (tensor<64xbf16, #ttnn_layout65>) -> tensor<1x16x128x1xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%arg15) <{force = false}> : (tensor<64xbf16, #ttnn_layout65>) -> ()
        %16 = ttcore.load_cached(@main_const_eval_16, []) : () -> tensor<4096xbf16, #ttnn_layout62>
        %17 = ttcore.load_cached(@main_const_eval_17, [%arg24]) : (tensor<32x5760xbf16, #ttnn_layout71>) -> tensor<1x4x1x5760xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%arg24) <{force = false}> : (tensor<32x5760xbf16, #ttnn_layout71>) -> ()
        %18:2 = ttcore.load_cached(@main_const_eval_18, []) : () -> (tensor<1x1xf32, #ttnn_layout9>, tensor<1x1x1xf32, #ttnn_layout8>)
        %19 = ttcore.load_cached(@main_const_eval_19, [%arg21]) : (tensor<1x1x32x8xsi32, #ttnn_layout77>) -> tensor<1x1x32x8xui16, #ttnn_layout78>
        "ttnn.deallocate"(%arg21) <{force = false}> : (tensor<1x1x32x8xsi32, #ttnn_layout77>) -> ()
        %20 = ttcore.load_cached(@main_const_eval_20, [%arg2]) : (tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xf32, #ttnn_layout80>
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<2880xbf16, #ttnn_layout16>) -> ()
        %21 = ttcore.load_cached(@main_const_eval_21, [%arg10]) : (tensor<32xbf16, #ttnn_layout81>) -> tensor<1x32xf32, #ttnn_layout9>
        "ttnn.deallocate"(%arg10) <{force = false}> : (tensor<32xbf16, #ttnn_layout81>) -> ()
        %22 = ttcore.load_cached(@main_const_eval_22, [%arg22]) : (tensor<32x2880xbf16, #ttnn_layout35>) -> tensor<1x4x1x2880xbf16, #ttnn_layout84>
        "ttnn.deallocate"(%arg22) <{force = false}> : (tensor<32x2880xbf16, #ttnn_layout35>) -> ()
        %23 = ttcore.load_cached(@main_const_eval_23, [%arg8]) : (tensor<201088x2880xbf16, #ttnn_layout>) -> tensor<201088x1440xbf16, #ttnn_layout4>
        "ttnn.deallocate"(%arg8) <{force = false}> : (tensor<201088x2880xbf16, #ttnn_layout>) -> ()
        %24 = ttcore.load_cached(@main_const_eval_24, [%arg9]) : (tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1440xf32, #ttnn_layout80>
        "ttnn.deallocate"(%arg9) <{force = false}> : (tensor<2880xbf16, #ttnn_layout16>) -> ()
        %25 = ttcore.load_cached(@main_const_eval_25, []) : () -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        %26 = ttcore.load_cached(@main_const_eval_26, []) : () -> tensor<1x1xsi32, #ttnn_layout45>
        %27 = ttcore.load_cached(@main_const_eval_27, []) : () -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        %28 = ttcore.load_cached(@main_const_eval_28, [%arg16]) : (tensor<bf16, #ttnn_layout87>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%arg16) <{force = false}> : (tensor<bf16, #ttnn_layout87>) -> ()
        %29 = ttcore.load_cached(@main_const_eval_29, [%arg1]) : (tensor<512x2880xbf16, #ttnn_layout89>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<512x2880xbf16, #ttnn_layout89>) -> ()
        %30 = ttcore.load_cached(@main_const_eval_30, [%arg0]) : (tensor<512xbf16, #ttnn_layout51>) -> tensor<1x128xbf16, #ttnn_layout52>
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<512xbf16, #ttnn_layout51>) -> ()
        %31 = ttcore.load_cached(@main_const_eval_31, [%arg7]) : (tensor<512x2880xbf16, #ttnn_layout89>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%arg7) <{force = false}> : (tensor<512x2880xbf16, #ttnn_layout89>) -> ()
        %32 = ttcore.load_cached(@main_const_eval_32, [%arg12]) : (tensor<2880xbf16, #ttnn_layout16>) -> tensor<1x1x1440xf32, #ttnn_layout93>
        "ttnn.deallocate"(%arg12) <{force = false}> : (tensor<2880xbf16, #ttnn_layout16>) -> ()
        %33 = ttcore.load_cached(@main_const_eval_33, []) : () -> tensor<1x1x32xsi32, #ttnn_layout95>
        %34:2 = ttcore.load_cached(@main_const_eval_34, []) : () -> (tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x128x128xbf16, #ttnn_layout97>)
        %35 = ttcore.load_cached(@main_const_eval_35, []) : () -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        %36 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 2x4>}> : () -> !ttnn.device
        %37 = "ttnn.to_layout"(%arg3) <{layout = #ttnn.layout<tile>}> : (tensor<1x128xsi32, #ttnn_layout100>) -> tensor<1x128xsi32, #ttnn_layout49>
        "ttnn.deallocate"(%arg3) <{force = false}> : (tensor<1x128xsi32, #ttnn_layout100>) -> ()
        %38 = "ttnn.typecast"(%37) <{dtype = #ttcore.supportedDataTypes<u32>}> : (tensor<1x128xsi32, #ttnn_layout49>) -> tensor<1x128xui32, #ttnn_layout50>
        "ttnn.deallocate"(%37) <{force = false}> : (tensor<1x128xsi32, #ttnn_layout49>) -> ()
        %39 = "ttnn.to_layout"(%38) <{layout = #ttnn.layout<row_major>}> : (tensor<1x128xui32, #ttnn_layout50>) -> tensor<1x128xui32, #ttnn_layout47>
        "ttnn.deallocate"(%38) <{force = false}> : (tensor<1x128xui32, #ttnn_layout50>) -> ()
        %40 = "ttnn.embedding"(%39, %0) : (tensor<1x128xui32, #ttnn_layout47>, tensor<201088x1440xbf16, #ttnn_layout1>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%39) <{force = false}> : (tensor<1x128xui32, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<201088x1440xbf16, #ttnn_layout1>) -> ()
        %41 = "ttnn.typecast"(%40) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %42 = "ttnn.pow_scalar"(%41) <{rhs = 2.000000e+00 : f32}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %43 = "ttnn.sum"(%42) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%42) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %44 = "ttnn.reshape"(%43) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<1x1x1x128xf32, #ttnn_layout106>
        "ttnn.deallocate"(%43) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %45 = "ttnn.reduce_scatter"(%44) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> tensor<1x1x1x64xf32, #ttnn_layout107>
        "ttnn.deallocate"(%44) <{force = false}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> ()
        %46 = "ttnn.reshape"(%45) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> tensor<1x64xf32, #ttnn_layout108>
        "ttnn.deallocate"(%45) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> ()
        %47 = "ttnn.all_gather"(%46) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x64xf32, #ttnn_layout108>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%46) <{force = false}> : (tensor<1x64xf32, #ttnn_layout108>) -> ()
        %48 = "ttnn.reshape"(%47) <{shape = [128 : i32, 1 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%47) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %49 = "ttnn.multiply"(%48, %18#0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1xf32, #ttnn_layout109>, tensor<1x1xf32, #ttnn_layout9>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%48) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %50 = "ttnn.add"(%49, %2#1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1xf32, #ttnn_layout109>, tensor<1x1xf32, #ttnn_layout9>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%49) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %51 = "ttnn.rsqrt"(%50) : (tensor<128x1xf32, #ttnn_layout109>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%50) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %52 = "ttnn.reshape"(%41) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%41) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %53 = "ttnn.multiply"(%52, %51) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1440xf32, #ttnn_layout110>, tensor<128x1xf32, #ttnn_layout109>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%52) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        "ttnn.deallocate"(%51) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %54 = "ttnn.multiply"(%53, %20) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1440xf32, #ttnn_layout110>, tensor<1x1440xf32, #ttnn_layout80>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%53) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        "ttnn.deallocate"(%20) <{force = false}> : (tensor<1x1440xf32, #ttnn_layout80>) -> ()
        %55 = "ttnn.typecast"(%54) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x1440xf32, #ttnn_layout110>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%54) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        %56 = "ttnn.matmul"(%55, %6) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<128x1440xbf16, #ttnn_layout90>, tensor<1024x1440xbf16, #ttnn_layout30>) -> tensor<128x1024xbf16, #ttnn_layout111>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1024x1440xbf16, #ttnn_layout30>) -> ()
        %57 = "ttnn.reshape"(%56) <{shape = [1 : i32, 1 : i32, 128 : i32, 1024 : i32]}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> tensor<1x1x128x1024xbf16, #ttnn_layout112>
        "ttnn.deallocate"(%56) <{force = false}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> ()
        %58 = "ttnn.reduce_scatter"(%57) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x1024xbf16, #ttnn_layout112>) -> tensor<1x1x128x512xbf16, #ttnn_layout113>
        "ttnn.deallocate"(%57) <{force = false}> : (tensor<1x1x128x1024xbf16, #ttnn_layout112>) -> ()
        %59 = "ttnn.reshape"(%58) <{shape = [128 : i32, 512 : i32]}> : (tensor<1x1x128x512xbf16, #ttnn_layout113>) -> tensor<128x512xbf16, #ttnn_layout114>
        "ttnn.deallocate"(%58) <{force = false}> : (tensor<1x1x128x512xbf16, #ttnn_layout113>) -> ()
        %60 = "ttnn.all_gather"(%59) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<128x512xbf16, #ttnn_layout114>) -> tensor<128x1024xbf16, #ttnn_layout111>
        "ttnn.deallocate"(%59) <{force = false}> : (tensor<128x512xbf16, #ttnn_layout114>) -> ()
        %61 = "ttnn.add"(%60, %14) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x1024xbf16, #ttnn_layout111>, tensor<1x1024xbf16, #ttnn_layout61>) -> tensor<128x1024xbf16, #ttnn_layout111>
        "ttnn.deallocate"(%60) <{force = false}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> ()
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x1024xbf16, #ttnn_layout61>) -> ()
        %62 = "ttnn.reshape"(%61) <{shape = [1 : i32, 128 : i32, 16 : i32, 64 : i32]}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> tensor<1x128x16x64xbf16, #ttnn_layout115>
        "ttnn.deallocate"(%61) <{force = false}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> ()
        %63 = "ttnn.permute"(%62) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x16x64xbf16, #ttnn_layout115>) -> tensor<1x16x128x64xbf16, #ttnn_layout116>
        "ttnn.deallocate"(%62) <{force = false}> : (tensor<1x128x16x64xbf16, #ttnn_layout115>) -> ()
        %64 = "ttnn.slice_static"(%63) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        %65 = "ttnn.multiply"(%64, %5#0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        %66 = "ttnn.slice_static"(%63) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%63) <{force = false}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> ()
        %67 = "ttnn.multiply"(%66, %5#1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        %68 = "ttnn.subtract"(%65, %67) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x16x128x32xbf16, #ttnn_layout66>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%67) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%65) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        %69 = "ttnn.multiply"(%66, %5#0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%66) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        %70 = "ttnn.multiply"(%64, %5#1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%64) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        %71 = "ttnn.add"(%69, %70) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x16x128x32xbf16, #ttnn_layout66>) -> tensor<1x16x128x32xbf16, #ttnn_layout66>
        "ttnn.deallocate"(%70) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%69) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        %72 = "ttnn.concat"(%68, %71) <{dim = 3 : si32}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>, tensor<1x16x128x32xbf16, #ttnn_layout66>) -> tensor<1x16x128x64xbf16, #ttnn_layout116>
        "ttnn.deallocate"(%71) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%68) <{force = false}> : (tensor<1x16x128x32xbf16, #ttnn_layout66>) -> ()
        %73 = "ttnn.matmul"(%55, %31) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<128x1440xbf16, #ttnn_layout90>, tensor<128x1440xbf16, #ttnn_layout90>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%31) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        %74 = "ttnn.reshape"(%73) <{shape = [1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16, #ttnn_layout117>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%73) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        %75 = "ttnn.reduce_scatter"(%74) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x1x128x64xbf16, #ttnn_layout118>
        "ttnn.deallocate"(%74) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        %76 = "ttnn.reshape"(%75) <{shape = [128 : i32, 64 : i32]}> : (tensor<1x1x128x64xbf16, #ttnn_layout118>) -> tensor<128x64xbf16, #ttnn_layout119>
        "ttnn.deallocate"(%75) <{force = false}> : (tensor<1x1x128x64xbf16, #ttnn_layout118>) -> ()
        %77 = "ttnn.all_gather"(%76) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<128x64xbf16, #ttnn_layout119>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%76) <{force = false}> : (tensor<128x64xbf16, #ttnn_layout119>) -> ()
        %78 = "ttnn.add"(%77, %12) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x128xbf16, #ttnn_layout117>, tensor<1x128xbf16, #ttnn_layout52>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%77) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1x128xbf16, #ttnn_layout52>) -> ()
        %79 = "ttnn.reshape"(%78) <{shape = [1 : i32, 128 : i32, 2 : i32, 64 : i32]}> : (tensor<128x128xbf16, #ttnn_layout117>) -> tensor<1x128x2x64xbf16, #ttnn_layout115>
        "ttnn.deallocate"(%78) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        %80 = "ttnn.permute"(%79) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x2x64xbf16, #ttnn_layout115>) -> tensor<1x2x128x64xbf16, #ttnn_layout120>
        "ttnn.deallocate"(%79) <{force = false}> : (tensor<1x128x2x64xbf16, #ttnn_layout115>) -> ()
        %81 = "ttnn.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        %82 = "ttnn.multiply"(%81, %5#0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        %83 = "ttnn.slice_static"(%80) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%80) <{force = false}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> ()
        %84 = "ttnn.multiply"(%83, %5#1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        %85 = "ttnn.subtract"(%82, %84) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x2x128x32xbf16, #ttnn_layout121>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%84) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        "ttnn.deallocate"(%82) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        %86 = "ttnn.multiply"(%83, %5#0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%83) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        "ttnn.deallocate"(%5#0) <{force = false}> : (tensor<1x1x128x32xbf16, #ttnn_layout22>) -> ()
        %87 = "ttnn.multiply"(%81, %5#1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%81) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        "ttnn.deallocate"(%5#1) <{force = false}> : (tensor<1x1x128x32xbf16, #ttnn_layout22>) -> ()
        %88 = "ttnn.add"(%86, %87) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x2x128x32xbf16, #ttnn_layout121>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%87) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        "ttnn.deallocate"(%86) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        %89 = "ttnn.concat"(%85, %88) <{dim = 3 : si32}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>, tensor<1x2x128x32xbf16, #ttnn_layout121>) -> tensor<1x2x128x64xbf16, #ttnn_layout120>
        "ttnn.deallocate"(%88) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        "ttnn.deallocate"(%85) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        %90 = "ttnn.repeat_interleave"(%89) <{dim = 1 : si32, repeats = 8 : ui32}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x16x128x64xbf16, #ttnn_layout116>
        %91 = "ttnn.matmul"(%72, %90) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>, tensor<1x16x128x64xbf16, #ttnn_layout116>) -> tensor<1x16x128x128xbf16, #ttnn_layout122>
        "ttnn.deallocate"(%90) <{force = false}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> ()
        "ttnn.deallocate"(%72) <{force = false}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> ()
        %92 = "ttnn.multiply"(%91, %25) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>, tensor<1x1x1x1xbf16, #ttnn_layout33>) -> tensor<1x16x128x128xbf16, #ttnn_layout122>
        "ttnn.deallocate"(%91) <{force = false}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>) -> ()
        "ttnn.deallocate"(%25) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        %93 = "ttnn.to_layout"(%arg18) <{layout = #ttnn.layout<tile>}> : (tensor<bf16, #ttnn_layout88>) -> tensor<bf16, #ttnn_layout34>
        "ttnn.deallocate"(%arg18) <{force = false}> : (tensor<bf16, #ttnn_layout88>) -> ()
        %94 = "ttnn.reshape"(%93) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16, #ttnn_layout34>) -> tensor<1x1x1x1xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%93) <{force = false}> : (tensor<bf16, #ttnn_layout34>) -> ()
        %95 = "ttnn.logical_and"(%94, %34#0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>, tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%34#0) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        %96 = "ttnn.logical_and"(%95, %34#1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%95) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        "ttnn.deallocate"(%34#1) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        %97 = "ttnn.logical_and"(%94, %96) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>, tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%96) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        "ttnn.deallocate"(%94) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        %98 = "ttnn.to_layout"(%arg17) <{layout = #ttnn.layout<tile>}> : (tensor<1x128xsi32, #ttnn_layout100>) -> tensor<1x128xsi32, #ttnn_layout49>
        "ttnn.deallocate"(%arg17) <{force = false}> : (tensor<1x128xsi32, #ttnn_layout100>) -> ()
        %99 = "ttnn.typecast"(%98) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128xsi32, #ttnn_layout49>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%98) <{force = false}> : (tensor<1x128xsi32, #ttnn_layout49>) -> ()
        %100 = "ttnn.permute"(%99) <{permutation = array<i64: 1, 0>}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%99) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %101 = "ttnn.typecast"(%100) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<128x1xf32, #ttnn_layout109>) -> tensor<128x1xsi32, #ttnn_layout123>
        "ttnn.deallocate"(%100) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %102 = "ttnn.ne"(%101, %10) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x1xsi32, #ttnn_layout123>, tensor<1x1xsi32, #ttnn_layout45>) -> tensor<128x1xbf16, #ttnn_layout124>
        "ttnn.deallocate"(%101) <{force = false}> : (tensor<128x1xsi32, #ttnn_layout123>) -> ()
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x1xsi32, #ttnn_layout45>) -> ()
        %103 = "ttnn.to_layout"(%102) <{layout = #ttnn.layout<row_major>}> : (tensor<128x1xbf16, #ttnn_layout124>) -> tensor<128x1xbf16, #ttnn_layout125>
        "ttnn.deallocate"(%102) <{force = false}> : (tensor<128x1xbf16, #ttnn_layout124>) -> ()
        %104 = "ttnn.embedding"(%11, %103) : (tensor<1x128xui32, #ttnn_layout47>, tensor<128x1xbf16, #ttnn_layout125>) -> tensor<1x128x1xbf16, #ttnn_layout126>
        "ttnn.deallocate"(%103) <{force = false}> : (tensor<128x1xbf16, #ttnn_layout125>) -> ()
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x128xui32, #ttnn_layout47>) -> ()
        %105 = "ttnn.reshape"(%104) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128x1xbf16, #ttnn_layout126>) -> tensor<1x1x1x128xbf16, #ttnn_layout127>
        "ttnn.deallocate"(%104) <{force = false}> : (tensor<1x128x1xbf16, #ttnn_layout126>) -> ()
        %106 = "ttnn.logical_and"(%97, %105) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x1x128xbf16, #ttnn_layout127>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%105) <{force = false}> : (tensor<1x1x1x128xbf16, #ttnn_layout127>) -> ()
        "ttnn.deallocate"(%97) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        %107 = "ttnn.where"(%106, %28, %27) : (tensor<1x1x128x128xbf16, #ttnn_layout97>, tensor<1x1x1x1xbf16, #ttnn_layout33>, tensor<1x1x1x1xbf16, #ttnn_layout33>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%106) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        "ttnn.deallocate"(%28) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        "ttnn.deallocate"(%27) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        %108 = "ttnn.add"(%92, %107) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>, tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x16x128x128xbf16, #ttnn_layout122>
        "ttnn.deallocate"(%107) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        "ttnn.deallocate"(%92) <{force = false}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>) -> ()
        %109 = "ttnn.concat"(%108, %15) <{dim = 3 : si32}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>, tensor<1x16x128x1xbf16, #ttnn_layout66>) -> tensor<1x16x128x129xbf16, #ttnn_layout128>
        "ttnn.deallocate"(%108) <{force = false}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>) -> ()
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<1x16x128x1xbf16, #ttnn_layout66>) -> ()
        %110 = "ttnn.softmax"(%109) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x16x128x129xbf16, #ttnn_layout128>) -> tensor<1x16x128x129xbf16, #ttnn_layout128>
        "ttnn.deallocate"(%109) <{force = false}> : (tensor<1x16x128x129xbf16, #ttnn_layout128>) -> ()
        %111 = "ttnn.slice_static"(%110) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 128 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x128x129xbf16, #ttnn_layout128>) -> tensor<1x16x128x128xbf16, #ttnn_layout122>
        "ttnn.deallocate"(%110) <{force = false}> : (tensor<1x16x128x129xbf16, #ttnn_layout128>) -> ()
        %112 = "ttnn.matmul"(%55, %29) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<128x1440xbf16, #ttnn_layout90>, tensor<128x1440xbf16, #ttnn_layout90>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%55) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        "ttnn.deallocate"(%29) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        %113 = "ttnn.reshape"(%112) <{shape = [1 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<128x128xbf16, #ttnn_layout117>) -> tensor<1x1x128x128xbf16, #ttnn_layout97>
        "ttnn.deallocate"(%112) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        %114 = "ttnn.reduce_scatter"(%113) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> tensor<1x1x128x64xbf16, #ttnn_layout118>
        "ttnn.deallocate"(%113) <{force = false}> : (tensor<1x1x128x128xbf16, #ttnn_layout97>) -> ()
        %115 = "ttnn.reshape"(%114) <{shape = [128 : i32, 64 : i32]}> : (tensor<1x1x128x64xbf16, #ttnn_layout118>) -> tensor<128x64xbf16, #ttnn_layout119>
        "ttnn.deallocate"(%114) <{force = false}> : (tensor<1x1x128x64xbf16, #ttnn_layout118>) -> ()
        %116 = "ttnn.all_gather"(%115) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<128x64xbf16, #ttnn_layout119>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%115) <{force = false}> : (tensor<128x64xbf16, #ttnn_layout119>) -> ()
        %117 = "ttnn.add"(%116, %30) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x128xbf16, #ttnn_layout117>, tensor<1x128xbf16, #ttnn_layout52>) -> tensor<128x128xbf16, #ttnn_layout117>
        "ttnn.deallocate"(%116) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        "ttnn.deallocate"(%30) <{force = false}> : (tensor<1x128xbf16, #ttnn_layout52>) -> ()
        %118 = "ttnn.reshape"(%117) <{shape = [1 : i32, 128 : i32, 2 : i32, 64 : i32]}> : (tensor<128x128xbf16, #ttnn_layout117>) -> tensor<1x128x2x64xbf16, #ttnn_layout115>
        "ttnn.deallocate"(%117) <{force = false}> : (tensor<128x128xbf16, #ttnn_layout117>) -> ()
        %119 = "ttnn.permute"(%118) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x2x64xbf16, #ttnn_layout115>) -> tensor<1x2x128x64xbf16, #ttnn_layout120>
        "ttnn.deallocate"(%118) <{force = false}> : (tensor<1x128x2x64xbf16, #ttnn_layout115>) -> ()
        %120 = "ttnn.repeat_interleave"(%119) <{dim = 1 : si32, repeats = 8 : ui32}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x16x128x64xbf16, #ttnn_layout116>
        %121 = "ttnn.matmul"(%111, %120) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>, tensor<1x16x128x64xbf16, #ttnn_layout116>) -> tensor<1x16x128x64xbf16, #ttnn_layout116>
        "ttnn.deallocate"(%120) <{force = false}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> ()
        "ttnn.deallocate"(%111) <{force = false}> : (tensor<1x16x128x128xbf16, #ttnn_layout122>) -> ()
        %122 = "ttnn.concatenate_heads"(%121) : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> tensor<1x128x1024xbf16, #ttnn_layout129>
        "ttnn.deallocate"(%121) <{force = false}> : (tensor<1x16x128x64xbf16, #ttnn_layout116>) -> ()
        %123 = "ttnn.reshape"(%122) <{shape = [128 : i32, 1024 : i32]}> : (tensor<1x128x1024xbf16, #ttnn_layout129>) -> tensor<128x1024xbf16, #ttnn_layout111>
        "ttnn.deallocate"(%122) <{force = false}> : (tensor<1x128x1024xbf16, #ttnn_layout129>) -> ()
        %124 = "ttnn.matmul"(%123, %13) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<128x1024xbf16, #ttnn_layout111>, tensor<1440x1024xbf16, #ttnn_layout57>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%123) <{force = false}> : (tensor<128x1024xbf16, #ttnn_layout111>) -> ()
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1440x1024xbf16, #ttnn_layout57>) -> ()
        %125 = "ttnn.reshape"(%124) <{shape = [1 : i32, 1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> tensor<1x1x128x1440xbf16, #ttnn_layout130>
        "ttnn.deallocate"(%124) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        %126 = "ttnn.reduce_scatter"(%125) <{cluster_axis = 1 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 2 : si32, topology = #ttcore.topology<linear>}> : (tensor<1x1x128x1440xbf16, #ttnn_layout130>) -> tensor<1x1x32x1440xbf16, #ttnn_layout131>
        "ttnn.deallocate"(%125) <{force = false}> : (tensor<1x1x128x1440xbf16, #ttnn_layout130>) -> ()
        %127 = "ttnn.reshape"(%126) <{shape = [32 : i32, 1440 : i32]}> : (tensor<1x1x32x1440xbf16, #ttnn_layout131>) -> tensor<32x1440xbf16, #ttnn_layout17>
        "ttnn.deallocate"(%126) <{force = false}> : (tensor<1x1x32x1440xbf16, #ttnn_layout131>) -> ()
        %128 = "ttnn.all_gather"(%127) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32, topology = #ttcore.topology<linear>}> : (tensor<32x1440xbf16, #ttnn_layout17>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%127) <{force = false}> : (tensor<32x1440xbf16, #ttnn_layout17>) -> ()
        %129 = "ttnn.add"(%128, %4) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x1440xbf16, #ttnn_layout90>, tensor<1x1440xbf16, #ttnn_layout17>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%128) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1440xbf16, #ttnn_layout17>) -> ()
        %130 = "ttnn.reshape"(%129) <{shape = [1 : i32, 128 : i32, 1440 : i32]}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%129) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        %131 = "ttnn.add"(%40, %130) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>, tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%130) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        "ttnn.deallocate"(%40) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        %132 = "ttnn.typecast"(%131) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %133 = "ttnn.pow_scalar"(%132) <{rhs = 2.000000e+00 : f32}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %134 = "ttnn.sum"(%133) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%133) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %135 = "ttnn.reshape"(%134) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<1x1x1x128xf32, #ttnn_layout106>
        "ttnn.deallocate"(%134) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %136 = "ttnn.reduce_scatter"(%135) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> tensor<1x1x1x64xf32, #ttnn_layout107>
        "ttnn.deallocate"(%135) <{force = false}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> ()
        %137 = "ttnn.reshape"(%136) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> tensor<1x64xf32, #ttnn_layout108>
        "ttnn.deallocate"(%136) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> ()
        %138 = "ttnn.all_gather"(%137) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x64xf32, #ttnn_layout108>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%137) <{force = false}> : (tensor<1x64xf32, #ttnn_layout108>) -> ()
        %139 = "ttnn.reshape"(%138) <{shape = [1 : i32, 128 : i32, 1 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<1x128x1xf32, #ttnn_layout26>
        "ttnn.deallocate"(%138) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %140 = "ttnn.multiply"(%139, %18#1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1xf32, #ttnn_layout26>, tensor<1x1x1xf32, #ttnn_layout8>) -> tensor<1x128x1xf32, #ttnn_layout26>
        "ttnn.deallocate"(%139) <{force = false}> : (tensor<1x128x1xf32, #ttnn_layout26>) -> ()
        "ttnn.deallocate"(%18#1) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout8>) -> ()
        %141 = "ttnn.add"(%140, %2#0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1xf32, #ttnn_layout26>, tensor<1x1x1xf32, #ttnn_layout8>) -> tensor<1x128x1xf32, #ttnn_layout26>
        "ttnn.deallocate"(%140) <{force = false}> : (tensor<1x128x1xf32, #ttnn_layout26>) -> ()
        "ttnn.deallocate"(%2#0) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout8>) -> ()
        %142 = "ttnn.rsqrt"(%141) : (tensor<1x128x1xf32, #ttnn_layout26>) -> tensor<1x128x1xf32, #ttnn_layout26>
        "ttnn.deallocate"(%141) <{force = false}> : (tensor<1x128x1xf32, #ttnn_layout26>) -> ()
        %143 = "ttnn.multiply"(%132, %142) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xf32, #ttnn_layout104>, tensor<1x128x1xf32, #ttnn_layout26>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        "ttnn.deallocate"(%142) <{force = false}> : (tensor<1x128x1xf32, #ttnn_layout26>) -> ()
        "ttnn.deallocate"(%132) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %144 = "ttnn.multiply"(%143, %32) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xf32, #ttnn_layout104>, tensor<1x1x1440xf32, #ttnn_layout93>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        "ttnn.deallocate"(%143) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        "ttnn.deallocate"(%32) <{force = false}> : (tensor<1x1x1440xf32, #ttnn_layout93>) -> ()
        %145 = "ttnn.typecast"(%144) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%144) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %146 = "ttnn.typecast"(%145) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %147 = "ttnn.reshape"(%146) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%146) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %148 = "ttnn.matmul"(%147, %8) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<128x1440xf32, #ttnn_layout110>, tensor<1440x32xf32, #ttnn_layout36>) -> tensor<128x32xf32, #ttnn_layout109>
        "ttnn.deallocate"(%147) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1440x32xf32, #ttnn_layout36>) -> ()
        %149 = "ttnn.reshape"(%148) <{shape = [1 : i32, 1 : i32, 128 : i32, 32 : i32]}> : (tensor<128x32xf32, #ttnn_layout109>) -> tensor<1x1x128x32xf32, #ttnn_layout27>
        "ttnn.deallocate"(%148) <{force = false}> : (tensor<128x32xf32, #ttnn_layout109>) -> ()
        %150 = "ttnn.reduce_scatter"(%149) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 2 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> tensor<1x1x64x32xf32, #ttnn_layout132>
        "ttnn.deallocate"(%149) <{force = false}> : (tensor<1x1x128x32xf32, #ttnn_layout27>) -> ()
        %151 = "ttnn.reshape"(%150) <{shape = [64 : i32, 32 : i32]}> : (tensor<1x1x64x32xf32, #ttnn_layout132>) -> tensor<64x32xf32, #ttnn_layout133>
        "ttnn.deallocate"(%150) <{force = false}> : (tensor<1x1x64x32xf32, #ttnn_layout132>) -> ()
        %152 = "ttnn.all_gather"(%151) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<64x32xf32, #ttnn_layout133>) -> tensor<128x32xf32, #ttnn_layout109>
        "ttnn.deallocate"(%151) <{force = false}> : (tensor<64x32xf32, #ttnn_layout133>) -> ()
        %153 = "ttnn.add"(%152, %21) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x32xf32, #ttnn_layout109>, tensor<1x32xf32, #ttnn_layout9>) -> tensor<128x32xf32, #ttnn_layout109>
        "ttnn.deallocate"(%152) <{force = false}> : (tensor<128x32xf32, #ttnn_layout109>) -> ()
        "ttnn.deallocate"(%21) <{force = false}> : (tensor<1x32xf32, #ttnn_layout9>) -> ()
        %154 = "ttnn.typecast"(%153) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x32xf32, #ttnn_layout109>) -> tensor<128x32xbf16, #ttnn_layout124>
        "ttnn.deallocate"(%153) <{force = false}> : (tensor<128x32xf32, #ttnn_layout109>) -> ()
        %values, %indices = "ttnn.sort"(%154) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<128x32xbf16, #ttnn_layout124>) -> (tensor<128x32xbf16, #ttnn_layout124>, tensor<128x32xui16, #ttnn_layout134>)
        "ttnn.deallocate"(%154) <{force = false}> : (tensor<128x32xbf16, #ttnn_layout124>) -> ()
        %155 = "ttnn.typecast"(%indices) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<128x32xui16, #ttnn_layout134>) -> tensor<128x32xsi32, #ttnn_layout123>
        "ttnn.deallocate"(%indices) <{force = false}> : (tensor<128x32xui16, #ttnn_layout134>) -> ()
        %156 = "ttnn.slice_static"(%155) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x32xsi32, #ttnn_layout123>) -> tensor<128x4xsi32, #ttnn_layout123>
        "ttnn.deallocate"(%155) <{force = false}> : (tensor<128x32xsi32, #ttnn_layout123>) -> ()
        %157 = "ttnn.reshape"(%156) <{shape = [128 : i32, 4 : i32, 1 : i32]}> : (tensor<128x4xsi32, #ttnn_layout123>) -> tensor<128x4x1xsi32, #ttnn_layout7>
        %158 = "ttnn.reshape"(%156) <{shape = [512 : i32, 1 : i32]}> : (tensor<128x4xsi32, #ttnn_layout123>) -> tensor<512x1xsi32, #ttnn_layout5>
        %159 = "ttnn.concat"(%1, %158) <{dim = 1 : si32}> : (tensor<512x1xsi32, #ttnn_layout5>, tensor<512x1xsi32, #ttnn_layout5>) -> tensor<512x2xsi32, #ttnn_layout5>
        "ttnn.deallocate"(%158) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        %160 = "ttnn.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [128 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x32xbf16, #ttnn_layout124>) -> tensor<128x4xbf16, #ttnn_layout124>
        "ttnn.deallocate"(%values) <{force = false}> : (tensor<128x32xbf16, #ttnn_layout124>) -> ()
        %161 = "ttnn.softmax"(%160) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 1 : si32, numericStable = true}> : (tensor<128x4xbf16, #ttnn_layout124>) -> tensor<128x4xbf16, #ttnn_layout124>
        "ttnn.deallocate"(%160) <{force = false}> : (tensor<128x4xbf16, #ttnn_layout124>) -> ()
        %162 = "ttnn.slice_static"(%159) <{begins = [0 : i32, 0 : i32], ends = [512 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xsi32, #ttnn_layout5>) -> tensor<512x1xsi32, #ttnn_layout5>
        %163 = "ttnn.slice_static"(%159) <{begins = [0 : i32, 1 : i32], ends = [512 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xsi32, #ttnn_layout5>) -> tensor<512x1xsi32, #ttnn_layout5>
        "ttnn.deallocate"(%159) <{force = false}> : (tensor<512x2xsi32, #ttnn_layout5>) -> ()
        %164 = "ttnn.multiply"(%162, %26) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<512x1xsi32, #ttnn_layout5>, tensor<1x1xsi32, #ttnn_layout45>) -> tensor<512x1xsi32, #ttnn_layout5>
        "ttnn.deallocate"(%162) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        "ttnn.deallocate"(%26) <{force = false}> : (tensor<1x1xsi32, #ttnn_layout45>) -> ()
        %165 = "ttnn.add"(%164, %163) <{dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<512x1xsi32, #ttnn_layout5>, tensor<512x1xsi32, #ttnn_layout5>) -> tensor<512x1xsi32, #ttnn_layout5>
        "ttnn.deallocate"(%164) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        "ttnn.deallocate"(%163) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        %166 = "ttnn.reshape"(%165) <{shape = [512 : i32]}> : (tensor<512x1xsi32, #ttnn_layout5>) -> tensor<512xsi32, #ttnn_layout135>
        "ttnn.deallocate"(%165) <{force = false}> : (tensor<512x1xsi32, #ttnn_layout5>) -> ()
        %167 = "ttnn.reshape"(%161) <{shape = [512 : i32]}> : (tensor<128x4xbf16, #ttnn_layout124>) -> tensor<512xbf16, #ttnn_layout54>
        "ttnn.deallocate"(%161) <{force = false}> : (tensor<128x4xbf16, #ttnn_layout124>) -> ()
        %168 = "ttnn.slice_static"(%166) <{begins = [0 : i32], ends = [256 : i32], step = [1 : i32]}> : (tensor<512xsi32, #ttnn_layout135>) -> tensor<256xsi32, #ttnn_layout136>
        %169 = "ttnn.slice_static"(%167) <{begins = [0 : i32], ends = [256 : i32], step = [1 : i32]}> : (tensor<512xbf16, #ttnn_layout54>) -> tensor<256xbf16, #ttnn_layout137>
        %170 = "ttnn.to_layout"(%168) <{layout = #ttnn.layout<row_major>}> : (tensor<256xsi32, #ttnn_layout136>) -> tensor<256xsi32, #ttnn_layout138>
        "ttnn.deallocate"(%168) <{force = false}> : (tensor<256xsi32, #ttnn_layout136>) -> ()
        %171 = "ttnn.to_layout"(%169) <{layout = #ttnn.layout<row_major>}> : (tensor<256xbf16, #ttnn_layout137>) -> tensor<256xbf16, #ttnn_layout139>
        "ttnn.deallocate"(%169) <{force = false}> : (tensor<256xbf16, #ttnn_layout137>) -> ()
        %172 = "ttnn.scatter"(%16, %170, %171) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<4096xbf16, #ttnn_layout62>, tensor<256xsi32, #ttnn_layout138>, tensor<256xbf16, #ttnn_layout139>) -> tensor<4096xbf16, #ttnn_layout62>
        "ttnn.deallocate"(%171) <{force = false}> : (tensor<256xbf16, #ttnn_layout139>) -> ()
        "ttnn.deallocate"(%170) <{force = false}> : (tensor<256xsi32, #ttnn_layout138>) -> ()
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<4096xbf16, #ttnn_layout62>) -> ()
        %173 = "ttnn.slice_static"(%166) <{begins = [256 : i32], ends = [512 : i32], step = [1 : i32]}> : (tensor<512xsi32, #ttnn_layout135>) -> tensor<256xsi32, #ttnn_layout136>
        "ttnn.deallocate"(%166) <{force = false}> : (tensor<512xsi32, #ttnn_layout135>) -> ()
        %174 = "ttnn.slice_static"(%167) <{begins = [256 : i32], ends = [512 : i32], step = [1 : i32]}> : (tensor<512xbf16, #ttnn_layout54>) -> tensor<256xbf16, #ttnn_layout137>
        "ttnn.deallocate"(%167) <{force = false}> : (tensor<512xbf16, #ttnn_layout54>) -> ()
        %175 = "ttnn.to_layout"(%173) <{layout = #ttnn.layout<row_major>}> : (tensor<256xsi32, #ttnn_layout136>) -> tensor<256xsi32, #ttnn_layout138>
        "ttnn.deallocate"(%173) <{force = false}> : (tensor<256xsi32, #ttnn_layout136>) -> ()
        %176 = "ttnn.to_layout"(%174) <{layout = #ttnn.layout<row_major>}> : (tensor<256xbf16, #ttnn_layout137>) -> tensor<256xbf16, #ttnn_layout139>
        "ttnn.deallocate"(%174) <{force = false}> : (tensor<256xbf16, #ttnn_layout137>) -> ()
        %177 = "ttnn.scatter"(%172, %175, %176) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<4096xbf16, #ttnn_layout62>, tensor<256xsi32, #ttnn_layout138>, tensor<256xbf16, #ttnn_layout139>) -> tensor<4096xbf16, #ttnn_layout62>
        "ttnn.deallocate"(%176) <{force = false}> : (tensor<256xbf16, #ttnn_layout139>) -> ()
        "ttnn.deallocate"(%175) <{force = false}> : (tensor<256xsi32, #ttnn_layout138>) -> ()
        "ttnn.deallocate"(%172) <{force = false}> : (tensor<4096xbf16, #ttnn_layout62>) -> ()
        %178 = "ttnn.to_layout"(%177) <{layout = #ttnn.layout<tile>}> : (tensor<4096xbf16, #ttnn_layout62>) -> tensor<4096xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%177) <{force = false}> : (tensor<4096xbf16, #ttnn_layout62>) -> ()
        %179 = "ttnn.reshape"(%178) <{shape = [1 : i32, 1 : i32, 128 : i32, 32 : i32]}> : (tensor<4096xbf16, #ttnn_layout63>) -> tensor<1x1x128x32xbf16, #ttnn_layout22>
        %180 = "ttnn.repeat"(%179) <{repeat_dims = #ttnn.shape<1x2x1x1>}> : (tensor<1x1x128x32xbf16, #ttnn_layout22>) -> tensor<1x2x128x32xbf16, #ttnn_layout121>
        "ttnn.deallocate"(%179) <{force = false}> : (tensor<1x1x128x32xbf16, #ttnn_layout22>) -> ()
        %181 = "ttnn.reshape"(%145) <{shape = [1 : i32, 1 : i32, 128 : i32, 1440 : i32]}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x1x128x1440xbf16, #ttnn_layout130>
        "ttnn.deallocate"(%145) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        %182 = "ttnn.reshape"(%156) <{shape = [1 : i32, 1 : i32, 128 : i32, 4 : i32]}> : (tensor<128x4xsi32, #ttnn_layout123>) -> tensor<1x1x128x4xsi32, #ttnn_layout99>
        "ttnn.deallocate"(%156) <{force = false}> : (tensor<128x4xsi32, #ttnn_layout123>) -> ()
        %183 = "ttnn.all_gather"(%181) <{all_gather_dim = 3 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x1440xbf16, #ttnn_layout130>) -> tensor<1x1x128x2880xbf16, #ttnn_layout140>
        "ttnn.deallocate"(%181) <{force = false}> : (tensor<1x1x128x1440xbf16, #ttnn_layout130>) -> ()
        %184 = "ttnn.to_layout"(%183) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x128x2880xbf16, #ttnn_layout140>) -> tensor<1x1x128x2880xbf16, #ttnn_layout141>
        "ttnn.deallocate"(%183) <{force = false}> : (tensor<1x1x128x2880xbf16, #ttnn_layout140>) -> ()
        %185 = "ttnn.typecast"(%182) <{dtype = #ttcore.supportedDataTypes<u16>}> : (tensor<1x1x128x4xsi32, #ttnn_layout99>) -> tensor<1x1x128x4xui16, #ttnn_layout142>
        "ttnn.deallocate"(%182) <{force = false}> : (tensor<1x1x128x4xsi32, #ttnn_layout99>) -> ()
        %186 = "ttnn.from_device"(%185) : (tensor<1x1x128x4xui16, #ttnn_layout142>) -> tensor<1x1x128x4xui16, #ttnn_layout143>
        "ttnn.deallocate"(%185) <{force = false}> : (tensor<1x1x128x4xui16, #ttnn_layout142>) -> ()
        %187 = "ttnn.to_layout"(%186) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x128x4xui16, #ttnn_layout143>) -> tensor<1x1x128x4xui16, #ttnn_layout144>
        "ttnn.deallocate"(%186) <{force = false}> : (tensor<1x1x128x4xui16, #ttnn_layout143>) -> ()
        %188 = "ttnn.to_device"(%187, %36) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x128x4xui16, #ttnn_layout144>, !ttnn.device) -> tensor<1x1x128x4xui16, #ttnn_layout145>
        "ttnn.deallocate"(%187) <{force = false}> : (tensor<1x1x128x4xui16, #ttnn_layout144>) -> ()
        %dispatched, %metadata = "ttnn.all_to_all_dispatch"(%184, %188, %19) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16, #ttnn_layout141>, tensor<1x1x128x4xui16, #ttnn_layout145>, tensor<1x1x32x8xui16, #ttnn_layout78>) -> (tensor<1x2x128x2880xbf16, #ttnn_layout146>, tensor<1x2x128x4xui16, #ttnn_layout147>)
        "ttnn.deallocate"(%188) <{force = false}> : (tensor<1x1x128x4xui16, #ttnn_layout145>) -> ()
        "ttnn.deallocate"(%184) <{force = false}> : (tensor<1x1x128x2880xbf16, #ttnn_layout141>) -> ()
        %189 = "ttnn.to_layout"(%dispatched) <{layout = #ttnn.layout<tile>}> : (tensor<1x2x128x2880xbf16, #ttnn_layout146>) -> tensor<1x2x128x2880xbf16, #ttnn_layout148>
        "ttnn.deallocate"(%dispatched) <{force = false}> : (tensor<1x2x128x2880xbf16, #ttnn_layout146>) -> ()
        %190 = "ttnn.to_layout"(%180) <{layout = #ttnn.layout<row_major>}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> tensor<1x2x128x32xbf16, #ttnn_layout149>
        "ttnn.deallocate"(%180) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout121>) -> ()
        %mapping, %reduced = "ttnn.moe_expert_token_remap"(%190, %19, %metadata) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16, #ttnn_layout149>, tensor<1x1x32x8xui16, #ttnn_layout78>, tensor<1x2x128x4xui16, #ttnn_layout147>) -> (tensor<1x2x128x4xbf16, #ttnn_layout150>, tensor<1x1x8x4xui16, #ttnn_layout151>)
        "ttnn.deallocate"(%mapping) <{force = false}> : (tensor<1x2x128x4xbf16, #ttnn_layout150>) -> ()
        "ttnn.deallocate"(%190) <{force = false}> : (tensor<1x2x128x32xbf16, #ttnn_layout149>) -> ()
        %191 = "ttnn.to_layout"(%reduced) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x8x4xui16, #ttnn_layout151>) -> tensor<1x1x8x4xui16, #ttnn_layout152>
        %192 = "ttnn.typecast"(%191) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x8x4xui16, #ttnn_layout152>) -> tensor<1x1x8x4xbf16, #ttnn_layout33>
        "ttnn.deallocate"(%191) <{force = false}> : (tensor<1x1x8x4xui16, #ttnn_layout152>) -> ()
        %193 = "ttnn.slice_static"(%119) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x2x127x64xbf16, #ttnn_layout120>
        "ttnn.deallocate"(%119) <{force = false}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> ()
        %194 = "ttnn.slice_static"(%89) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 2 : i32, 128 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> tensor<1x2x127x64xbf16, #ttnn_layout120>
        "ttnn.deallocate"(%89) <{force = false}> : (tensor<1x2x128x64xbf16, #ttnn_layout120>) -> ()
        %195 = "ttnn.reshape"(%189) <{shape = [2 : i32, 4 : i32, 32 : i32, 2880 : i32]}> : (tensor<1x2x128x2880xbf16, #ttnn_layout148>) -> tensor<2x4x32x2880xbf16, #ttnn_layout153>
        "ttnn.deallocate"(%189) <{force = false}> : (tensor<1x2x128x2880xbf16, #ttnn_layout148>) -> ()
        %196 = "ttnn.reshape"(%192) <{shape = [2 : i32, 4 : i32, 1 : i32, 4 : i32]}> : (tensor<1x1x8x4xbf16, #ttnn_layout33>) -> tensor<2x4x1x4xbf16, #ttnn_layout154>
        "ttnn.deallocate"(%192) <{force = false}> : (tensor<1x1x8x4xbf16, #ttnn_layout33>) -> ()
        %197 = "ttnn.to_layout"(%196) <{layout = #ttnn.layout<row_major>}> : (tensor<2x4x1x4xbf16, #ttnn_layout154>) -> tensor<2x4x1x4xbf16, #ttnn_layout155>
        "ttnn.deallocate"(%196) <{force = false}> : (tensor<2x4x1x4xbf16, #ttnn_layout154>) -> ()
        %198 = "ttnn.sparse_matmul"(%195, %3, %197) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64, program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 1, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 1, out_block_w = 1, per_core_m = 1, per_core_n = 23, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>}> : (tensor<2x4x32x2880xbf16, #ttnn_layout153>, tensor<1x4x2880x5760xbf16, #ttnn_layout12>, tensor<2x4x1x4xbf16, #ttnn_layout155>) -> tensor<2x4x1x4x32x5760xbf16, #ttnn_layout156>
        "ttnn.deallocate"(%197) <{force = false}> : (tensor<2x4x1x4xbf16, #ttnn_layout155>) -> ()
        "ttnn.deallocate"(%195) <{force = false}> : (tensor<2x4x32x2880xbf16, #ttnn_layout153>) -> ()
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x4x2880x5760xbf16, #ttnn_layout12>) -> ()
        %199 = "ttnn.reshape"(%198) <{shape = [8 : i32, 4 : i32, 32 : i32, 5760 : i32]}> : (tensor<2x4x1x4x32x5760xbf16, #ttnn_layout156>) -> tensor<8x4x32x5760xbf16, #ttnn_layout157>
        "ttnn.deallocate"(%198) <{force = false}> : (tensor<2x4x1x4x32x5760xbf16, #ttnn_layout156>) -> ()
        %200 = "ttnn.add"(%199, %17) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x5760xbf16, #ttnn_layout157>, tensor<1x4x1x5760xbf16, #ttnn_layout72>) -> tensor<8x4x32x5760xbf16, #ttnn_layout157>
        "ttnn.deallocate"(%199) <{force = false}> : (tensor<8x4x32x5760xbf16, #ttnn_layout157>) -> ()
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1x4x1x5760xbf16, #ttnn_layout72>) -> ()
        %201 = "ttnn.slice_static"(%200) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [8 : i32, 4 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<8x4x32x5760xbf16, #ttnn_layout157>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        %202 = "ttnn.clamp_scalar"(%201) <{max = 7.000000e+00 : f32, min = -7.000000e+00 : f32}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%201) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %203 = "ttnn.add"(%202, %7) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<1x1x1x1xbf16, #ttnn_layout33>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%202) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        %204 = "ttnn.slice_static"(%200) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 4 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<8x4x32x5760xbf16, #ttnn_layout157>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%200) <{force = false}> : (tensor<8x4x32x5760xbf16, #ttnn_layout157>) -> ()
        %205 = "ttnn.clamp_scalar"(%204) <{max = 7.000000e+00 : f32, min = 0xFF800000 : f32}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%204) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %206 = "ttnn.multiply"(%205, %35) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<1x1x1x1xbf16, #ttnn_layout33>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%35) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout33>) -> ()
        %207 = "ttnn.sigmoid"(%206) : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%206) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %208 = "ttnn.multiply"(%205, %207) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%207) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        "ttnn.deallocate"(%205) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %209 = "ttnn.multiply"(%203, %208) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%208) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        "ttnn.deallocate"(%203) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %210 = "ttnn.from_device"(%reduced) : (tensor<1x1x8x4xui16, #ttnn_layout151>) -> tensor<1x1x8x4xui16, #ttnn_layout159>
        "ttnn.deallocate"(%reduced) <{force = false}> : (tensor<1x1x8x4xui16, #ttnn_layout151>) -> ()
        %211 = "ttnn.typecast"(%210) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x8x4xui16, #ttnn_layout159>) -> tensor<1x1x8x4xbf16, #ttnn_layout160>
        "ttnn.deallocate"(%210) <{force = false}> : (tensor<1x1x8x4xui16, #ttnn_layout159>) -> ()
        %212 = "ttnn.to_device"(%211, %36) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x8x4xbf16, #ttnn_layout160>, !ttnn.device) -> tensor<1x1x8x4xbf16, #ttnn_layout161>
        "ttnn.deallocate"(%211) <{force = false}> : (tensor<1x1x8x4xbf16, #ttnn_layout160>) -> ()
        %213 = "ttnn.sparse_matmul"(%209, %9, %212) <{is_input_a_sparse = true, is_input_b_sparse = false, nnz = 0 : i64, program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 1, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 1, out_block_w = 1, per_core_m = 1, per_core_n = 12, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<1x4x2880x2880xbf16, #ttnn_layout41>, tensor<1x1x8x4xbf16, #ttnn_layout161>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%212) <{force = false}> : (tensor<1x1x8x4xbf16, #ttnn_layout161>) -> ()
        "ttnn.deallocate"(%209) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x4x2880x2880xbf16, #ttnn_layout41>) -> ()
        %214 = "ttnn.add"(%213, %22) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>, tensor<1x4x1x2880xbf16, #ttnn_layout84>) -> tensor<8x4x32x2880xbf16, #ttnn_layout158>
        "ttnn.deallocate"(%213) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        "ttnn.deallocate"(%22) <{force = false}> : (tensor<1x4x1x2880xbf16, #ttnn_layout84>) -> ()
        %215 = "ttnn.reshape"(%214) <{shape = [2 : i32, 4 : i32, 4 : i32, 32 : i32, 2880 : i32]}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> tensor<2x4x4x32x2880xbf16, #ttnn_layout162>
        "ttnn.deallocate"(%214) <{force = false}> : (tensor<8x4x32x2880xbf16, #ttnn_layout158>) -> ()
        %216 = "ttnn.permute"(%215) <{permutation = array<i64: 2, 0, 1, 3, 4>}> : (tensor<2x4x4x32x2880xbf16, #ttnn_layout162>) -> tensor<4x2x4x32x2880xbf16, #ttnn_layout163>
        "ttnn.deallocate"(%215) <{force = false}> : (tensor<2x4x4x32x2880xbf16, #ttnn_layout162>) -> ()
        %217 = "ttnn.reshape"(%216) <{shape = [4 : i32, 2 : i32, 128 : i32, 2880 : i32]}> : (tensor<4x2x4x32x2880xbf16, #ttnn_layout163>) -> tensor<4x2x128x2880xbf16, #ttnn_layout164>
        "ttnn.deallocate"(%216) <{force = false}> : (tensor<4x2x4x32x2880xbf16, #ttnn_layout163>) -> ()
        %218 = "ttnn.to_layout"(%217) <{layout = #ttnn.layout<row_major>}> : (tensor<4x2x128x2880xbf16, #ttnn_layout164>) -> tensor<4x2x128x2880xbf16, #ttnn_layout165>
        "ttnn.deallocate"(%217) <{force = false}> : (tensor<4x2x128x2880xbf16, #ttnn_layout164>) -> ()
        %219 = "ttnn.all_to_all_combine"(%218, %metadata, %19) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16, #ttnn_layout165>, tensor<1x2x128x4xui16, #ttnn_layout147>, tensor<1x1x32x8xui16, #ttnn_layout78>) -> tensor<4x1x128x2880xbf16, #ttnn_layout166>
        "ttnn.deallocate"(%218) <{force = false}> : (tensor<4x2x128x2880xbf16, #ttnn_layout165>) -> ()
        "ttnn.deallocate"(%metadata) <{force = false}> : (tensor<1x2x128x4xui16, #ttnn_layout147>) -> ()
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1x1x32x8xui16, #ttnn_layout78>) -> ()
        %220 = "ttnn.to_layout"(%219) <{layout = #ttnn.layout<tile>}> : (tensor<4x1x128x2880xbf16, #ttnn_layout166>) -> tensor<4x1x128x2880xbf16, #ttnn_layout167>
        "ttnn.deallocate"(%219) <{force = false}> : (tensor<4x1x128x2880xbf16, #ttnn_layout166>) -> ()
        %221 = "ttnn.reduce_scatter"(%220) <{cluster_axis = 1 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 2 : si32, topology = #ttcore.topology<linear>}> : (tensor<4x1x128x2880xbf16, #ttnn_layout167>) -> tensor<4x1x32x2880xbf16, #ttnn_layout168>
        "ttnn.deallocate"(%220) <{force = false}> : (tensor<4x1x128x2880xbf16, #ttnn_layout167>) -> ()
        %222 = "ttnn.all_gather"(%221) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32, topology = #ttcore.topology<linear>}> : (tensor<4x1x32x2880xbf16, #ttnn_layout168>) -> tensor<4x1x128x2880xbf16, #ttnn_layout167>
        "ttnn.deallocate"(%221) <{force = false}> : (tensor<4x1x32x2880xbf16, #ttnn_layout168>) -> ()
        %223 = "ttnn.to_layout"(%222) <{layout = #ttnn.layout<row_major>}> : (tensor<4x1x128x2880xbf16, #ttnn_layout167>) -> tensor<4x1x128x2880xbf16, #ttnn_layout166>
        "ttnn.deallocate"(%222) <{force = false}> : (tensor<4x1x128x2880xbf16, #ttnn_layout167>) -> ()
        %224 = "ttnn.mesh_partition"(%223) <{cluster_axis = 0 : ui32, dim = 3 : si32}> : (tensor<4x1x128x2880xbf16, #ttnn_layout166>) -> tensor<4x1x128x1440xbf16, #ttnn_layout169>
        "ttnn.deallocate"(%223) <{force = false}> : (tensor<4x1x128x2880xbf16, #ttnn_layout166>) -> ()
        %225 = "ttnn.to_layout"(%224) <{layout = #ttnn.layout<tile>}> : (tensor<4x1x128x1440xbf16, #ttnn_layout169>) -> tensor<4x1x128x1440xbf16, #ttnn_layout170>
        "ttnn.deallocate"(%224) <{force = false}> : (tensor<4x1x128x1440xbf16, #ttnn_layout169>) -> ()
        %226 = "ttnn.eq"(%157, %33) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x4x1xsi32, #ttnn_layout7>, tensor<1x1x32xsi32, #ttnn_layout95>) -> tensor<128x4x32xbf16, #ttnn_layout171>
        "ttnn.deallocate"(%157) <{force = false}> : (tensor<128x4x1xsi32, #ttnn_layout7>) -> ()
        "ttnn.deallocate"(%33) <{force = false}> : (tensor<1x1x32xsi32, #ttnn_layout95>) -> ()
        %227 = "ttnn.reshape"(%178) <{shape = [128 : i32, 32 : i32, 1 : i32]}> : (tensor<4096xbf16, #ttnn_layout63>) -> tensor<128x32x1xbf16, #ttnn_layout171>
        "ttnn.deallocate"(%178) <{force = false}> : (tensor<4096xbf16, #ttnn_layout63>) -> ()
        %228 = "ttnn.matmul"(%226, %227) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<128x4x32xbf16, #ttnn_layout171>, tensor<128x32x1xbf16, #ttnn_layout171>) -> tensor<128x4x1xbf16, #ttnn_layout171>
        "ttnn.deallocate"(%227) <{force = false}> : (tensor<128x32x1xbf16, #ttnn_layout171>) -> ()
        "ttnn.deallocate"(%226) <{force = false}> : (tensor<128x4x32xbf16, #ttnn_layout171>) -> ()
        %229 = "ttnn.reshape"(%228) <{shape = [1 : i32, 128 : i32, 4 : i32]}> : (tensor<128x4x1xbf16, #ttnn_layout171>) -> tensor<1x128x4xbf16, #ttnn_layout126>
        "ttnn.deallocate"(%228) <{force = false}> : (tensor<128x4x1xbf16, #ttnn_layout171>) -> ()
        %230 = "ttnn.permute"(%229) <{permutation = array<i64: 2, 0, 1>}> : (tensor<1x128x4xbf16, #ttnn_layout126>) -> tensor<4x1x128xbf16, #ttnn_layout172>
        "ttnn.deallocate"(%229) <{force = false}> : (tensor<1x128x4xbf16, #ttnn_layout126>) -> ()
        %231 = "ttnn.reshape"(%230) <{shape = [4 : i32, 1 : i32, 128 : i32, 1 : i32]}> : (tensor<4x1x128xbf16, #ttnn_layout172>) -> tensor<4x1x128x1xbf16, #ttnn_layout173>
        "ttnn.deallocate"(%230) <{force = false}> : (tensor<4x1x128xbf16, #ttnn_layout172>) -> ()
        %232 = "ttnn.multiply"(%225, %231) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<4x1x128x1440xbf16, #ttnn_layout170>, tensor<4x1x128x1xbf16, #ttnn_layout173>) -> tensor<4x1x128x1440xbf16, #ttnn_layout170>
        "ttnn.deallocate"(%231) <{force = false}> : (tensor<4x1x128x1xbf16, #ttnn_layout173>) -> ()
        "ttnn.deallocate"(%225) <{force = false}> : (tensor<4x1x128x1440xbf16, #ttnn_layout170>) -> ()
        %233 = "ttnn.sum"(%232) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [0 : i32], keep_dim = false}> : (tensor<4x1x128x1440xbf16, #ttnn_layout170>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%232) <{force = false}> : (tensor<4x1x128x1440xbf16, #ttnn_layout170>) -> ()
        %234 = "ttnn.add"(%131, %233) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>, tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xbf16, #ttnn_layout103>
        "ttnn.deallocate"(%233) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        "ttnn.deallocate"(%131) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        %235 = "ttnn.typecast"(%234) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        "ttnn.deallocate"(%234) <{force = false}> : (tensor<1x128x1440xbf16, #ttnn_layout103>) -> ()
        %236 = "ttnn.pow_scalar"(%235) <{rhs = 2.000000e+00 : f32}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128x1440xf32, #ttnn_layout104>
        %237 = "ttnn.sum"(%236) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%236) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %238 = "ttnn.reshape"(%237) <{shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<1x1x1x128xf32, #ttnn_layout106>
        "ttnn.deallocate"(%237) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %239 = "ttnn.reduce_scatter"(%238) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> tensor<1x1x1x64xf32, #ttnn_layout107>
        "ttnn.deallocate"(%238) <{force = false}> : (tensor<1x1x1x128xf32, #ttnn_layout106>) -> ()
        %240 = "ttnn.reshape"(%239) <{shape = [1 : i32, 64 : i32]}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> tensor<1x64xf32, #ttnn_layout108>
        "ttnn.deallocate"(%239) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout107>) -> ()
        %241 = "ttnn.all_gather"(%240) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<1x64xf32, #ttnn_layout108>) -> tensor<1x128xf32, #ttnn_layout105>
        "ttnn.deallocate"(%240) <{force = false}> : (tensor<1x64xf32, #ttnn_layout108>) -> ()
        %242 = "ttnn.reshape"(%241) <{shape = [128 : i32, 1 : i32]}> : (tensor<1x128xf32, #ttnn_layout105>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%241) <{force = false}> : (tensor<1x128xf32, #ttnn_layout105>) -> ()
        %243 = "ttnn.multiply"(%242, %18#0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1xf32, #ttnn_layout109>, tensor<1x1xf32, #ttnn_layout9>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%242) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        "ttnn.deallocate"(%18#0) <{force = false}> : (tensor<1x1xf32, #ttnn_layout9>) -> ()
        %244 = "ttnn.add"(%243, %2#1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1xf32, #ttnn_layout109>, tensor<1x1xf32, #ttnn_layout9>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%243) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        "ttnn.deallocate"(%2#1) <{force = false}> : (tensor<1x1xf32, #ttnn_layout9>) -> ()
        %245 = "ttnn.rsqrt"(%244) : (tensor<128x1xf32, #ttnn_layout109>) -> tensor<128x1xf32, #ttnn_layout109>
        "ttnn.deallocate"(%244) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %246 = "ttnn.reshape"(%235) <{shape = [128 : i32, 1440 : i32]}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%235) <{force = false}> : (tensor<1x128x1440xf32, #ttnn_layout104>) -> ()
        %247 = "ttnn.multiply"(%246, %245) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1440xf32, #ttnn_layout110>, tensor<128x1xf32, #ttnn_layout109>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%246) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        "ttnn.deallocate"(%245) <{force = false}> : (tensor<128x1xf32, #ttnn_layout109>) -> ()
        %248 = "ttnn.multiply"(%247, %24) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<128x1440xf32, #ttnn_layout110>, tensor<1x1440xf32, #ttnn_layout80>) -> tensor<128x1440xf32, #ttnn_layout110>
        "ttnn.deallocate"(%247) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        "ttnn.deallocate"(%24) <{force = false}> : (tensor<1x1440xf32, #ttnn_layout80>) -> ()
        %249 = "ttnn.typecast"(%248) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<128x1440xf32, #ttnn_layout110>) -> tensor<128x1440xbf16, #ttnn_layout90>
        "ttnn.deallocate"(%248) <{force = false}> : (tensor<128x1440xf32, #ttnn_layout110>) -> ()
        %250 = "ttnn.matmul"(%249, %23) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<128x1440xbf16, #ttnn_layout90>, tensor<201088x1440xbf16, #ttnn_layout4>) -> tensor<128x201088xbf16, #ttnn_layout174>
        "ttnn.deallocate"(%249) <{force = false}> : (tensor<128x1440xbf16, #ttnn_layout90>) -> ()
        "ttnn.deallocate"(%23) <{force = false}> : (tensor<201088x1440xbf16, #ttnn_layout4>) -> ()
        %251 = "ttnn.reshape"(%250) <{shape = [1 : i32, 1 : i32, 128 : i32, 201088 : i32]}> : (tensor<128x201088xbf16, #ttnn_layout174>) -> tensor<1x1x128x201088xbf16, #ttnn_layout175>
        "ttnn.deallocate"(%250) <{force = false}> : (tensor<128x201088xbf16, #ttnn_layout174>) -> ()
        %252 = "ttnn.reduce_scatter"(%251) <{cluster_axis = 0 : ui32, compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32, topology = #ttcore.topology<ring>}> : (tensor<1x1x128x201088xbf16, #ttnn_layout175>) -> tensor<1x1x128x100544xbf16, #ttnn_layout176>
        "ttnn.deallocate"(%251) <{force = false}> : (tensor<1x1x128x201088xbf16, #ttnn_layout175>) -> ()
        %253 = "ttnn.reshape"(%252) <{shape = [128 : i32, 100544 : i32]}> : (tensor<1x1x128x100544xbf16, #ttnn_layout176>) -> tensor<128x100544xbf16, #ttnn_layout177>
        "ttnn.deallocate"(%252) <{force = false}> : (tensor<1x1x128x100544xbf16, #ttnn_layout176>) -> ()
        %254 = "ttnn.all_gather"(%253) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, topology = #ttcore.topology<ring>}> : (tensor<128x100544xbf16, #ttnn_layout177>) -> tensor<128x201088xbf16, #ttnn_layout174>
        "ttnn.deallocate"(%253) <{force = false}> : (tensor<128x100544xbf16, #ttnn_layout177>) -> ()
        %255 = "ttnn.reshape"(%254) <{shape = [1 : i32, 128 : i32, 201088 : i32]}> : (tensor<128x201088xbf16, #ttnn_layout174>) -> tensor<1x128x201088xbf16, #ttnn_layout102>
        "ttnn.deallocate"(%254) <{force = false}> : (tensor<128x201088xbf16, #ttnn_layout174>) -> ()
        %256 = "ttnn.mesh_shard"(%193, %36) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 4, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x2x127x64xbf16, #ttnn_layout120>, !ttnn.device) -> tensor<1x8x127x64xbf16, #ttnn_layout101>
        %257 = "ttnn.mesh_shard"(%194, %36) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 4, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x2x127x64xbf16, #ttnn_layout120>, !ttnn.device) -> tensor<1x8x127x64xbf16, #ttnn_layout101>
        return %256, %257, %255 : tensor<1x8x127x64xbf16, #ttnn_layout101>, tensor<1x8x127x64xbf16, #ttnn_layout101>, tensor<1x128x201088xbf16, #ttnn_layout102>
      }
    }
  }
}
// -----------------------------------------------------------------------------
// END TTNN MODULE
// -----------------------------------------------------------------------------
