# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model wrapper for CLIP Vision Encoder + IP-Adapter Resampler."""

import ttnn
import utils
from consteval import run_const_evals
from models.common.lightweightmodule import LightweightModule
from weights_loader import load_inputs_for__main


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    def __init__(self, weights, cache, device):
        self.device = device
        self.weights = weights
        self.cer = run_const_evals(weights, cache)

    def forward(self, pixel_values):
        # Move input to device
        assert pixel_values.device() is None, "pixel_values must be on host"
        pixel_values = ttnn.to_device(
            pixel_values,
            self.device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        CLIPVisionEmbeddings_0_0_0 = self.CLIPVisionEmbeddings_0_0(
            pixel_values,
            self.cer["utils_constEvalFuncWrapper_142_0"],
            self.cer["utils_constEvalFuncWrapper_88_0"],
            self.cer["utils_constEvalFuncWrapper_66_0"],
        )
        LayerNorm_1_0_0 = self.LayerNorm_1_0(
            self.weights[385], self.weights[386], CLIPVisionEmbeddings_0_0_0
        )
        CLIPEncoderLayer_2_0_0 = self.CLIPEncoderLayer_2_0(
            self.weights[383], LayerNorm_1_0_0, self.weights[384]
        )
        CLIPAttention_3_0_0 = self.CLIPAttention_3_0(
            self.weights[380],
            self.cer["utils_constEvalFuncWrapper_47_0"],
            self.cer["utils_constEvalFuncWrapper_124_0"],
            self.cer["utils_constEvalFuncWrapper_70_0"],
            CLIPEncoderLayer_2_0_0,
        )
        v_163, v_164 = self.CLIPEncoderLayer_4_0(
            self.weights[377], self.weights[378], LayerNorm_1_0_0, CLIPAttention_3_0_0
        )
        CLIPMLP_5_0_0 = self.CLIPMLP_5_0(
            v_163,
            self.cer["utils_constEvalFuncWrapper_73_0"],
            self.cer["utils_constEvalFuncWrapper_42_0"],
            self.weights[374],
            self.weights[376],
        )
        v_165, v_166 = self.CLIPEncoderLayer_6_0(
            CLIPMLP_5_0_0, self.weights[372], v_164, self.weights[371]
        )
        CLIPAttention_7_0_0 = self.CLIPAttention_7_0(
            self.cer["utils_constEvalFuncWrapper_62_0"],
            self.cer["utils_constEvalFuncWrapper_55_0"],
            v_166,
            self.weights[368],
            self.cer["utils_constEvalFuncWrapper_157_0"],
        )
        v_167, v_168 = self.CLIPEncoderLayer_8_0(
            self.weights[366], v_165, self.weights[365], CLIPAttention_7_0_0
        )
        CLIPMLP_9_0_0 = self.CLIPMLP_9_0(
            self.cer["utils_constEvalFuncWrapper_13_0"],
            self.weights[362],
            self.cer["utils_constEvalFuncWrapper_21_0"],
            self.weights[364],
            v_168,
        )
        v_169, v_170 = self.CLIPEncoderLayer_10_0(
            v_167, self.weights[360], self.weights[359], CLIPMLP_9_0_0
        )
        CLIPAttention_11_0_0 = self.CLIPAttention_11_0(
            self.cer["utils_constEvalFuncWrapper_80_0"],
            self.cer["utils_constEvalFuncWrapper_25_0"],
            v_170,
            self.cer["utils_constEvalFuncWrapper_122_0"],
            self.weights[356],
        )
        v_171, v_172 = self.CLIPEncoderLayer_12_0(
            v_169, self.weights[353], CLIPAttention_11_0_0, self.weights[354]
        )
        CLIPMLP_13_0_0 = self.CLIPMLP_13_0(
            self.cer["utils_constEvalFuncWrapper_81_0"],
            self.weights[352],
            self.weights[350],
            v_172,
            self.cer["utils_constEvalFuncWrapper_146_0"],
        )
        v_173, v_174 = self.CLIPEncoderLayer_14_0(
            self.weights[347], CLIPMLP_13_0_0, v_171, self.weights[348]
        )
        CLIPAttention_15_0_0 = self.CLIPAttention_15_0(
            self.cer["utils_constEvalFuncWrapper_90_0"],
            self.weights[344],
            self.cer["utils_constEvalFuncWrapper_132_0"],
            v_174,
            self.cer["utils_constEvalFuncWrapper_26_0"],
        )
        v_175, v_176 = self.CLIPEncoderLayer_16_0(
            CLIPAttention_15_0_0, v_173, self.weights[342], self.weights[341]
        )
        CLIPMLP_17_0_0 = self.CLIPMLP_17_0(
            v_175,
            self.cer["utils_constEvalFuncWrapper_145_0"],
            self.weights[338],
            self.weights[340],
            self.cer["utils_constEvalFuncWrapper_10_0"],
        )
        v_177, v_178 = self.CLIPEncoderLayer_18_0(
            CLIPMLP_17_0_0, self.weights[335], v_176, self.weights[336]
        )
        CLIPAttention_19_0_0 = self.CLIPAttention_19_0(
            self.cer["utils_constEvalFuncWrapper_97_0"],
            v_177,
            self.cer["utils_constEvalFuncWrapper_43_0"],
            self.cer["utils_constEvalFuncWrapper_127_0"],
            self.weights[332],
        )
        v_179, v_180 = self.CLIPEncoderLayer_20_0(
            self.weights[329], v_178, CLIPAttention_19_0_0, self.weights[330]
        )
        CLIPMLP_21_0_0 = self.CLIPMLP_21_0(
            self.cer["utils_constEvalFuncWrapper_150_0"],
            self.weights[326],
            self.weights[328],
            v_179,
            self.cer["utils_constEvalFuncWrapper_149_0"],
        )
        v_181, v_182 = self.CLIPEncoderLayer_22_0(
            CLIPMLP_21_0_0, self.weights[324], v_180, self.weights[323]
        )
        CLIPAttention_23_0_0 = self.CLIPAttention_23_0(
            self.weights[320],
            self.cer["utils_constEvalFuncWrapper_158_0"],
            self.cer["utils_constEvalFuncWrapper_69_0"],
            v_182,
            self.cer["utils_constEvalFuncWrapper_96_0"],
        )
        v_183, v_184 = self.CLIPEncoderLayer_24_0(
            self.weights[318], CLIPAttention_23_0_0, v_181, self.weights[317]
        )
        CLIPMLP_25_0_0 = self.CLIPMLP_25_0(
            self.cer["utils_constEvalFuncWrapper_91_0"],
            v_183,
            self.cer["utils_constEvalFuncWrapper_106_0"],
            self.weights[316],
            self.weights[314],
        )
        v_185, v_186 = self.CLIPEncoderLayer_26_0(
            self.weights[312], self.weights[311], CLIPMLP_25_0_0, v_184
        )
        CLIPAttention_27_0_0 = self.CLIPAttention_27_0(
            self.cer["utils_constEvalFuncWrapper_99_0"],
            v_185,
            self.cer["utils_constEvalFuncWrapper_46_0"],
            self.cer["utils_constEvalFuncWrapper_128_0"],
            self.weights[308],
        )
        v_187, v_188 = self.CLIPEncoderLayer_28_0(
            CLIPAttention_27_0_0, self.weights[305], self.weights[306], v_186
        )
        CLIPMLP_29_0_0 = self.CLIPMLP_29_0(
            self.cer["utils_constEvalFuncWrapper_53_0"],
            self.weights[304],
            self.weights[302],
            v_188,
            self.cer["utils_constEvalFuncWrapper_103_0"],
        )
        v_189, v_190 = self.CLIPEncoderLayer_30_0(
            CLIPMLP_29_0_0, v_187, self.weights[299], self.weights[300]
        )
        CLIPAttention_31_0_0 = self.CLIPAttention_31_0(
            self.cer["utils_constEvalFuncWrapper_120_0"],
            self.cer["utils_constEvalFuncWrapper_84_0"],
            v_189,
            self.weights[296],
            self.cer["utils_constEvalFuncWrapper_49_0"],
        )
        v_191, v_192 = self.CLIPEncoderLayer_32_0(
            self.weights[293], self.weights[294], CLIPAttention_31_0_0, v_190
        )
        CLIPMLP_33_0_0 = self.CLIPMLP_33_0(
            self.cer["utils_constEvalFuncWrapper_153_0"],
            self.cer["utils_constEvalFuncWrapper_27_0"],
            v_192,
            self.weights[290],
            self.weights[292],
        )
        v_193, v_194 = self.CLIPEncoderLayer_34_0(
            self.weights[288], self.weights[287], v_191, CLIPMLP_33_0_0
        )
        CLIPAttention_35_0_0 = self.CLIPAttention_35_0(
            v_193,
            self.cer["utils_constEvalFuncWrapper_40_0"],
            self.weights[284],
            self.cer["utils_constEvalFuncWrapper_74_0"],
            self.cer["utils_constEvalFuncWrapper_29_0"],
        )
        v_195, v_196 = self.CLIPEncoderLayer_36_0(
            CLIPAttention_35_0_0, self.weights[281], self.weights[282], v_194
        )
        CLIPMLP_37_0_0 = self.CLIPMLP_37_0(
            self.cer["utils_constEvalFuncWrapper_24_0"],
            self.weights[278],
            self.weights[280],
            self.cer["utils_constEvalFuncWrapper_93_0"],
            v_195,
        )
        v_197, v_198 = self.CLIPEncoderLayer_38_0(
            self.weights[276], self.weights[275], CLIPMLP_37_0_0, v_196
        )
        CLIPAttention_39_0_0 = self.CLIPAttention_39_0(
            v_197,
            self.cer["utils_constEvalFuncWrapper_119_0"],
            self.cer["utils_constEvalFuncWrapper_133_0"],
            self.cer["utils_constEvalFuncWrapper_113_0"],
            self.weights[272],
        )
        v_199, v_200 = self.CLIPEncoderLayer_40_0(
            self.weights[269], self.weights[270], v_198, CLIPAttention_39_0_0
        )
        CLIPMLP_41_0_0 = self.CLIPMLP_41_0(
            self.cer["utils_constEvalFuncWrapper_2_0"],
            v_200,
            self.weights[266],
            self.weights[268],
            self.cer["utils_constEvalFuncWrapper_155_0"],
        )
        v_201, v_202 = self.CLIPEncoderLayer_42_0(
            self.weights[264], CLIPMLP_41_0_0, v_199, self.weights[263]
        )
        CLIPAttention_43_0_0 = self.CLIPAttention_43_0(
            self.weights[260],
            self.cer["utils_constEvalFuncWrapper_152_0"],
            self.cer["utils_constEvalFuncWrapper_64_0"],
            v_202,
            self.cer["utils_constEvalFuncWrapper_71_0"],
        )
        v_203, v_204 = self.CLIPEncoderLayer_44_0(
            CLIPAttention_43_0_0, self.weights[258], v_201, self.weights[257]
        )
        CLIPMLP_45_0_0 = self.CLIPMLP_45_0(
            self.weights[256],
            self.cer["utils_constEvalFuncWrapper_95_0"],
            self.cer["utils_constEvalFuncWrapper_85_0"],
            v_204,
            self.weights[254],
        )
        v_205, v_206 = self.CLIPEncoderLayer_46_0(
            v_203, self.weights[252], self.weights[251], CLIPMLP_45_0_0
        )
        CLIPAttention_47_0_0 = self.CLIPAttention_47_0(
            self.cer["utils_constEvalFuncWrapper_67_0"],
            self.cer["utils_constEvalFuncWrapper_140_0"],
            v_205,
            self.cer["utils_constEvalFuncWrapper_116_0"],
            self.weights[248],
        )
        v_207, v_208 = self.CLIPEncoderLayer_48_0(
            self.weights[245], self.weights[246], CLIPAttention_47_0_0, v_206
        )
        CLIPMLP_49_0_0 = self.CLIPMLP_49_0(
            v_207,
            self.cer["utils_constEvalFuncWrapper_156_0"],
            self.cer["utils_constEvalFuncWrapper_151_0"],
            self.weights[244],
            self.weights[242],
        )
        v_209, v_210 = self.CLIPEncoderLayer_50_0(
            self.weights[240], v_208, CLIPMLP_49_0_0, self.weights[239]
        )
        CLIPAttention_51_0_0 = self.CLIPAttention_51_0(
            self.weights[236],
            self.cer["utils_constEvalFuncWrapper_87_0"],
            v_209,
            self.cer["utils_constEvalFuncWrapper_136_0"],
            self.cer["utils_constEvalFuncWrapper_68_0"],
        )
        v_211, v_212 = self.CLIPEncoderLayer_52_0(
            CLIPAttention_51_0_0, self.weights[233], v_210, self.weights[234]
        )
        CLIPMLP_53_0_0 = self.CLIPMLP_53_0(
            self.weights[232],
            self.cer["utils_constEvalFuncWrapper_5_0"],
            self.cer["utils_constEvalFuncWrapper_15_0"],
            v_211,
            self.weights[230],
        )
        v_213, v_214 = self.CLIPEncoderLayer_54_0(
            self.weights[227], CLIPMLP_53_0_0, self.weights[228], v_212
        )
        CLIPAttention_55_0_0 = self.CLIPAttention_55_0(
            self.weights[224],
            self.cer["utils_constEvalFuncWrapper_1_0"],
            self.cer["utils_constEvalFuncWrapper_126_0"],
            self.cer["utils_constEvalFuncWrapper_102_0"],
            v_214,
        )
        v_215, v_216 = self.CLIPEncoderLayer_56_0(
            CLIPAttention_55_0_0, self.weights[221], v_213, self.weights[222]
        )
        CLIPMLP_57_0_0 = self.CLIPMLP_57_0(
            self.cer["utils_constEvalFuncWrapper_92_0"],
            self.cer["utils_constEvalFuncWrapper_109_0"],
            self.weights[218],
            self.weights[220],
            v_216,
        )
        v_217, v_218 = self.CLIPEncoderLayer_58_0(
            self.weights[215], v_215, self.weights[216], CLIPMLP_57_0_0
        )
        CLIPAttention_59_0_0 = self.CLIPAttention_59_0(
            v_217,
            self.cer["utils_constEvalFuncWrapper_86_0"],
            self.weights[212],
            self.cer["utils_constEvalFuncWrapper_101_0"],
            self.cer["utils_constEvalFuncWrapper_11_0"],
        )
        v_219, v_220 = self.CLIPEncoderLayer_60_0(
            self.weights[209], self.weights[210], v_218, CLIPAttention_59_0_0
        )
        CLIPMLP_61_0_0 = self.CLIPMLP_61_0(
            self.weights[206],
            self.weights[208],
            self.cer["utils_constEvalFuncWrapper_18_0"],
            v_220,
            self.cer["utils_constEvalFuncWrapper_141_0"],
        )
        v_221, v_222 = self.CLIPEncoderLayer_62_0(
            CLIPMLP_61_0_0, self.weights[203], self.weights[204], v_219
        )
        CLIPAttention_63_0_0 = self.CLIPAttention_63_0(
            v_221,
            self.cer["utils_constEvalFuncWrapper_72_0"],
            self.cer["utils_constEvalFuncWrapper_114_0"],
            self.weights[200],
            self.cer["utils_constEvalFuncWrapper_23_0"],
        )
        v_223, v_224 = self.CLIPEncoderLayer_64_0(
            self.weights[197], CLIPAttention_63_0_0, self.weights[198], v_222
        )
        CLIPMLP_65_0_0 = self.CLIPMLP_65_0(
            self.weights[194],
            self.cer["utils_constEvalFuncWrapper_83_0"],
            self.cer["utils_constEvalFuncWrapper_154_0"],
            self.weights[196],
            v_224,
        )
        v_225, v_226 = self.CLIPEncoderLayer_66_0(
            self.weights[191], v_223, CLIPMLP_65_0_0, self.weights[192]
        )
        CLIPAttention_67_0_0 = self.CLIPAttention_67_0(
            self.cer["utils_constEvalFuncWrapper_118_0"],
            v_226,
            self.cer["utils_constEvalFuncWrapper_89_0"],
            self.weights[188],
            self.cer["utils_constEvalFuncWrapper_63_0"],
        )
        v_227, v_228 = self.CLIPEncoderLayer_68_0(
            v_225, CLIPAttention_67_0_0, self.weights[185], self.weights[186]
        )
        CLIPMLP_69_0_0 = self.CLIPMLP_69_0(
            self.weights[184],
            v_228,
            self.weights[182],
            self.cer["utils_constEvalFuncWrapper_130_0"],
            self.cer["utils_constEvalFuncWrapper_104_0"],
        )
        v_229, v_230 = self.CLIPEncoderLayer_70_0(
            v_227, CLIPMLP_69_0_0, self.weights[180], self.weights[179]
        )
        CLIPAttention_71_0_0 = self.CLIPAttention_71_0(
            self.cer["utils_constEvalFuncWrapper_34_0"],
            self.cer["utils_constEvalFuncWrapper_7_0"],
            v_230,
            self.weights[176],
            self.cer["utils_constEvalFuncWrapper_17_0"],
        )
        v_231, v_232 = self.CLIPEncoderLayer_72_0(
            self.weights[174], self.weights[173], CLIPAttention_71_0_0, v_229
        )
        CLIPMLP_73_0_0 = self.CLIPMLP_73_0(
            v_231,
            self.cer["utils_constEvalFuncWrapper_108_0"],
            self.cer["utils_constEvalFuncWrapper_19_0"],
            self.weights[172],
            self.weights[170],
        )
        v_233, v_234 = self.CLIPEncoderLayer_74_0(
            self.weights[168], self.weights[167], v_232, CLIPMLP_73_0_0
        )
        CLIPAttention_75_0_0 = self.CLIPAttention_75_0(
            self.cer["utils_constEvalFuncWrapper_134_0"],
            self.weights[164],
            self.cer["utils_constEvalFuncWrapper_112_0"],
            v_233,
            self.cer["utils_constEvalFuncWrapper_100_0"],
        )
        v_235, v_236 = self.CLIPEncoderLayer_76_0(
            self.weights[161], CLIPAttention_75_0_0, v_234, self.weights[162]
        )
        CLIPMLP_77_0_0 = self.CLIPMLP_77_0(
            v_235,
            self.weights[160],
            self.weights[158],
            self.cer["utils_constEvalFuncWrapper_94_0"],
            self.cer["utils_constEvalFuncWrapper_147_0"],
        )
        v_237, v_238 = self.CLIPEncoderLayer_78_0(
            self.weights[156], CLIPMLP_77_0_0, v_236, self.weights[155]
        )
        CLIPAttention_79_0_0 = self.CLIPAttention_79_0(
            self.weights[152],
            v_237,
            self.cer["utils_constEvalFuncWrapper_12_0"],
            self.cer["utils_constEvalFuncWrapper_50_0"],
            self.cer["utils_constEvalFuncWrapper_52_0"],
        )
        v_239, v_240 = self.CLIPEncoderLayer_80_0(
            self.weights[150], self.weights[149], v_238, CLIPAttention_79_0_0
        )
        CLIPMLP_81_0_0 = self.CLIPMLP_81_0(
            v_240,
            self.cer["utils_constEvalFuncWrapper_44_0"],
            self.weights[146],
            self.weights[148],
            self.cer["utils_constEvalFuncWrapper_28_0"],
        )
        v_241, v_242 = self.CLIPEncoderLayer_82_0(
            v_239, self.weights[144], self.weights[143], CLIPMLP_81_0_0
        )
        CLIPAttention_83_0_0 = self.CLIPAttention_83_0(
            self.cer["utils_constEvalFuncWrapper_78_0"],
            self.cer["utils_constEvalFuncWrapper_60_0"],
            self.weights[140],
            self.cer["utils_constEvalFuncWrapper_65_0"],
            v_242,
        )
        v_243, v_244 = self.CLIPEncoderLayer_84_0(
            CLIPAttention_83_0_0, self.weights[137], v_241, self.weights[138]
        )
        CLIPMLP_85_0_0 = self.CLIPMLP_85_0(
            self.cer["utils_constEvalFuncWrapper_82_0"],
            self.weights[134],
            self.weights[136],
            self.cer["utils_constEvalFuncWrapper_107_0"],
            v_244,
        )
        v_245, v_246 = self.CLIPEncoderLayer_86_0(
            v_243, CLIPMLP_85_0_0, self.weights[132], self.weights[131]
        )
        CLIPAttention_87_0_0 = self.CLIPAttention_87_0(
            self.weights[128],
            self.cer["utils_constEvalFuncWrapper_37_0"],
            self.cer["utils_constEvalFuncWrapper_20_0"],
            v_246,
            self.cer["utils_constEvalFuncWrapper_111_0"],
        )
        v_247, v_248 = self.CLIPEncoderLayer_88_0(
            CLIPAttention_87_0_0, v_245, self.weights[125], self.weights[126]
        )
        CLIPMLP_89_0_0 = self.CLIPMLP_89_0(
            self.weights[122],
            self.cer["utils_constEvalFuncWrapper_110_0"],
            v_247,
            self.weights[124],
            self.cer["utils_constEvalFuncWrapper_160_0"],
        )
        v_249, v_250 = self.CLIPEncoderLayer_90_0(
            self.weights[120], CLIPMLP_89_0_0, v_248, self.weights[119]
        )
        CLIPAttention_91_0_0 = self.CLIPAttention_91_0(
            v_249,
            self.cer["utils_constEvalFuncWrapper_148_0"],
            self.weights[116],
            self.cer["utils_constEvalFuncWrapper_57_0"],
            self.cer["utils_constEvalFuncWrapper_33_0"],
        )
        v_251, v_252 = self.CLIPEncoderLayer_92_0(
            CLIPAttention_91_0_0, self.weights[113], v_250, self.weights[114]
        )
        CLIPMLP_93_0_0 = self.CLIPMLP_93_0(
            self.weights[112],
            self.cer["utils_constEvalFuncWrapper_125_0"],
            self.weights[110],
            v_252,
            self.cer["utils_constEvalFuncWrapper_4_0"],
        )
        v_253, v_254 = self.CLIPEncoderLayer_94_0(
            self.weights[108], self.weights[107], v_251, CLIPMLP_93_0_0
        )
        CLIPAttention_95_0_0 = self.CLIPAttention_95_0(
            self.weights[104],
            v_254,
            self.cer["utils_constEvalFuncWrapper_36_0"],
            self.cer["utils_constEvalFuncWrapper_32_0"],
            self.cer["utils_constEvalFuncWrapper_51_0"],
        )
        v_255, v_256 = self.CLIPEncoderLayer_96_0(
            self.weights[102], CLIPAttention_95_0_0, v_253, self.weights[101]
        )
        CLIPMLP_97_0_0 = self.CLIPMLP_97_0(
            self.weights[98],
            self.cer["utils_constEvalFuncWrapper_0_0"],
            self.weights[100],
            self.cer["utils_constEvalFuncWrapper_22_0"],
            v_256,
        )
        v_257, v_258 = self.CLIPEncoderLayer_98_0(
            v_255, self.weights[95], CLIPMLP_97_0_0, self.weights[96]
        )
        CLIPAttention_99_0_0 = self.CLIPAttention_99_0(
            self.cer["utils_constEvalFuncWrapper_76_0"],
            v_258,
            self.cer["utils_constEvalFuncWrapper_143_0"],
            self.cer["utils_constEvalFuncWrapper_59_0"],
            self.weights[92],
        )
        v_259, v_260 = self.CLIPEncoderLayer_100_0(
            v_257, CLIPAttention_99_0_0, self.weights[89], self.weights[90]
        )
        CLIPMLP_101_0_0 = self.CLIPMLP_101_0(
            self.cer["utils_constEvalFuncWrapper_144_0"],
            self.cer["utils_constEvalFuncWrapper_139_0"],
            self.weights[88],
            self.weights[86],
            v_260,
        )
        v_261, v_262 = self.CLIPEncoderLayer_102_0(
            CLIPMLP_101_0_0, v_259, self.weights[83], self.weights[84]
        )
        CLIPAttention_103_0_0 = self.CLIPAttention_103_0(
            self.weights[80],
            self.cer["utils_constEvalFuncWrapper_61_0"],
            self.cer["utils_constEvalFuncWrapper_31_0"],
            self.cer["utils_constEvalFuncWrapper_58_0"],
            v_261,
        )
        v_263, v_264 = self.CLIPEncoderLayer_104_0(
            CLIPAttention_103_0_0, self.weights[78], self.weights[77], v_262
        )
        CLIPMLP_105_0_0 = self.CLIPMLP_105_0(
            self.weights[76],
            self.cer["utils_constEvalFuncWrapper_117_0"],
            self.weights[74],
            self.cer["utils_constEvalFuncWrapper_39_0"],
            v_264,
        )
        v_265, v_266 = self.CLIPEncoderLayer_106_0(
            v_263, CLIPMLP_105_0_0, self.weights[72], self.weights[71]
        )
        CLIPAttention_107_0_0 = self.CLIPAttention_107_0(
            self.cer["utils_constEvalFuncWrapper_77_0"],
            v_265,
            self.weights[68],
            self.cer["utils_constEvalFuncWrapper_105_0"],
            self.cer["utils_constEvalFuncWrapper_9_0"],
        )
        v_267, v_268 = self.CLIPEncoderLayer_108_0(
            CLIPAttention_107_0_0, self.weights[66], self.weights[65], v_266
        )
        CLIPMLP_109_0_0 = self.CLIPMLP_109_0(
            self.weights[62],
            v_267,
            self.cer["utils_constEvalFuncWrapper_123_0"],
            self.weights[64],
            self.cer["utils_constEvalFuncWrapper_98_0"],
        )
        v_269, v_270 = self.CLIPEncoderLayer_110_0(
            self.weights[59], self.weights[60], CLIPMLP_109_0_0, v_268
        )
        CLIPAttention_111_0_0 = self.CLIPAttention_111_0(
            v_270,
            self.weights[56],
            self.cer["utils_constEvalFuncWrapper_159_0"],
            self.cer["utils_constEvalFuncWrapper_8_0"],
            self.cer["utils_constEvalFuncWrapper_41_0"],
        )
        v_271, v_272 = self.CLIPEncoderLayer_112_0(
            self.weights[54], v_269, CLIPAttention_111_0_0, self.weights[53]
        )
        CLIPMLP_113_0_0 = self.CLIPMLP_113_0(
            v_271,
            self.weights[50],
            self.weights[52],
            self.cer["utils_constEvalFuncWrapper_129_0"],
            self.cer["utils_constEvalFuncWrapper_115_0"],
        )
        v_273, v_274 = self.CLIPEncoderLayer_114_0(
            self.weights[47], CLIPMLP_113_0_0, v_272, self.weights[48]
        )
        CLIPAttention_115_0_0 = self.CLIPAttention_115_0(
            v_274,
            self.weights[44],
            self.cer["utils_constEvalFuncWrapper_16_0"],
            self.cer["utils_constEvalFuncWrapper_3_0"],
            self.cer["utils_constEvalFuncWrapper_121_0"],
        )
        v_275, v_276 = self.CLIPEncoderLayer_116_0(
            v_273, self.weights[41], self.weights[42], CLIPAttention_115_0_0
        )
        CLIPMLP_117_0_0 = self.CLIPMLP_117_0(
            self.weights[38],
            self.cer["utils_constEvalFuncWrapper_56_0"],
            v_275,
            self.cer["utils_constEvalFuncWrapper_14_0"],
            self.weights[40],
        )
        v_277, v_278 = self.CLIPEncoderLayer_118_0(
            self.weights[36], CLIPMLP_117_0_0, self.weights[35], v_276
        )
        CLIPAttention_119_0_0 = self.CLIPAttention_119_0(
            self.cer["utils_constEvalFuncWrapper_45_0"],
            self.cer["utils_constEvalFuncWrapper_79_0"],
            v_278,
            self.weights[32],
            self.cer["utils_constEvalFuncWrapper_75_0"],
        )
        v_279, v_280 = self.CLIPEncoderLayer_120_0(
            v_277, self.weights[29], self.weights[30], CLIPAttention_119_0_0
        )
        CLIPMLP_121_0_0 = self.CLIPMLP_121_0(
            self.cer["utils_constEvalFuncWrapper_38_0"],
            self.cer["utils_constEvalFuncWrapper_35_0"],
            self.weights[26],
            v_280,
            self.weights[28],
        )
        v_281, v_282 = self.CLIPEncoderLayer_122_0(
            self.weights[23], self.weights[24], v_279, CLIPMLP_121_0_0
        )
        CLIPAttention_123_0_0 = self.CLIPAttention_123_0(
            self.weights[20],
            self.cer["utils_constEvalFuncWrapper_48_0"],
            v_281,
            self.cer["utils_constEvalFuncWrapper_138_0"],
            self.cer["utils_constEvalFuncWrapper_131_0"],
        )
        v_283, v_284 = self.CLIPEncoderLayer_124_0(
            CLIPAttention_123_0_0, v_282, self.weights[18], self.weights[17]
        )
        CLIPMLP_125_0_0 = self.CLIPMLP_125_0(
            v_284,
            self.weights[14],
            self.weights[16],
            self.cer["utils_constEvalFuncWrapper_135_0"],
            self.cer["utils_constEvalFuncWrapper_54_0"],
        )
        CLIPEncoderLayer_126_0_0 = self.CLIPEncoderLayer_126_0(v_283, CLIPMLP_125_0_0)
        Linear_127_0_0 = self.Linear_127_0(
            self.cer["utils_constEvalFuncWrapper_137_0"],
            self.weights[12],
            CLIPEncoderLayer_126_0_0,
        )
        v_285, v_286, v_287, v_288 = self.IPAdapterPlusImageProjectionBlock_128_0(
            self.cer["utils_constEvalFuncWrapper_30_0"],
            self.weights[539],
            self.weights[551],
            Linear_127_0_0,
            self.weights[527],
            self.weights[550],
            self.weights[10],
            self.weights[526],
            self.weights[9],
            self.weights[538],
        )
        Attention_129_0_0 = self.Attention_129_0(
            v_286,
            self.weights[5],
            self.weights[516],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights[6],
            self.cer["utils_constEvalFuncWrapper_30_1"],
        )
        v_289, v_290 = self.IPAdapterPlusImageProjectionBlock_130_0(
            self.weights[521], Attention_129_0_0, self.weights[4], self.weights[520]
        )
        Linear_131_0_0 = self.Linear_131_0(self.weights[519], v_290)
        Linear_132_0_0 = self.Linear_132_0(self.weights[518], Linear_131_0_0)
        v_291, v_292, v_293 = self.IPAdapterPlusImageProjectionBlock_133_0(
            v_289, Linear_132_0_0, self.weights[525], self.weights[524], v_288
        )
        Attention_134_0_0 = self.Attention_134_0(
            self.weights[529],
            self.weights[528],
            self.weights[522],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights[523],
            v_292,
            v_293,
        )
        v_294, v_295 = self.IPAdapterPlusImageProjectionBlock_135_0(
            v_291, self.weights[533], Attention_134_0_0, self.weights[532]
        )
        Linear_136_0_0 = self.Linear_136_0(self.weights[531], v_295)
        Linear_137_0_0 = self.Linear_137_0(Linear_136_0_0, self.weights[530])
        v_296, v_297, v_298 = self.IPAdapterPlusImageProjectionBlock_138_0(
            v_285, self.weights[537], v_294, Linear_137_0_0, self.weights[536]
        )
        Attention_139_0_0 = self.Attention_139_0(
            self.weights[540],
            self.weights[535],
            self.weights[541],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights[534],
            v_296,
            v_298,
        )
        v_299, v_300 = self.IPAdapterPlusImageProjectionBlock_140_0(
            self.weights[544], Attention_139_0_0, v_297, self.weights[545]
        )
        Linear_141_0_0 = self.Linear_141_0(self.weights[543], v_300)
        Linear_142_0_0 = self.Linear_142_0(Linear_141_0_0, self.weights[542])
        v_301, v_302, v_303 = self.IPAdapterPlusImageProjectionBlock_143_0(
            Linear_142_0_0, v_287, self.weights[548], self.weights[549], v_299
        )
        Attention_144_0_0 = self.Attention_144_0(
            self.weights[552],
            self.weights[553],
            self.cer["utils_constEvalFuncWrapperZeroArg_0_0"],
            self.weights[547],
            self.weights[546],
            v_301,
            v_302,
        )
        v_304, v_305 = self.IPAdapterPlusImageProjectionBlock_145_0(
            self.weights[556], self.weights[557], Attention_144_0_0, v_303
        )
        v_306, v_307 = self.Linear_147_0(v_304, self.weights[555])
        self.Linear_150_0(v_307)
        Linear_146_0_0 = self.Linear_146_0(v_305)
        IPAdapterPlusImageProjectionBlock_148_0_0 = (
            self.IPAdapterPlusImageProjectionBlock_148_0(
                Linear_146_0_0, v_306, self.weights[554]
            )
        )
        IPAdapterPlusImageProjection_149_0_0 = self.IPAdapterPlusImageProjection_149_0(
            v_305,
            self.weights[1],
            self.cer["utils_constEvalFuncWrapper_6_0"],
            IPAdapterPlusImageProjectionBlock_148_0_0,
            self.weights[3],
            self.weights[0],
        )
        self.IPAdapterPlusImageProjectionBlock_151_0(v_306)
        util_create_list_385 = [IPAdapterPlusImageProjection_149_0_0]
        return util_create_list_385

    def CLIPVisionEmbeddings_0_0(self, input_0, input_1, input_2, input_3):
        ttnn_to_layout_287 = ttnn.to_layout(
            input_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        utils_DeviceGetter_get_device_162 = utils.DeviceGetter.get_device((1, 1))
        ttnn_permute_3 = ttnn.permute(
            ttnn_to_layout_287,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_192 = ttnn.reshape(
            ttnn_permute_3,
            [1, 1, 50176, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_conv2d_0 = ttnn.conv2d(
            input_tensor=ttnn_reshape_192,
            weight_tensor=input_3,
            device=utils_DeviceGetter_get_device_162,
            in_channels=3,
            out_channels=1280,
            batch_size=1,
            input_height=224,
            input_width=224,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=None,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dL1Full, num_slices=0
            ),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_193 = ttnn.reshape(
            ttnn_conv2d_0,
            [1, 16, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_4 = ttnn.permute(
            ttnn_reshape_193,
            [0, 3, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_194 = ttnn.reshape(
            ttnn_permute_4,
            [1, 1280, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_386 = [input_1, ttnn_reshape_194]
        ttnn_concat_62 = ttnn.concat(
            util_create_list_386,
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_add_0 = ttnn.add(
            ttnn_concat_62,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_5 = ttnn.permute(
            ttnn_add_0,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        return ttnn_permute_5

    def LayerNorm_1_0(self, input_0, input_1, input_2):
        ttnn_layer_norm_1 = ttnn.layer_norm(
            input_2,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_1

    def CLIPEncoderLayer_2_0(self, input_0, input_1, input_2):
        ttnn_layer_norm_2 = ttnn.layer_norm(
            input_1,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_2

    def CLIPAttention_3_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_195 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_reshape_195,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_1 = ttnn.add(
            ttnn_matmul_1,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_0 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_1 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_2 = ttnn.slice(
            ttnn_add_1,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_196 = ttnn.reshape(
            ttnn_slice_0,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_197 = ttnn.reshape(
            ttnn_slice_1,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_198 = ttnn.reshape(
            ttnn_slice_2,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_6 = ttnn.permute(
            ttnn_reshape_197,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_7 = ttnn.permute(
            ttnn_reshape_198,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_8 = ttnn.permute(
            ttnn_reshape_196,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_0 = ttnn.pad(
            ttnn_permute_6,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_1 = ttnn.pad(
            ttnn_permute_7,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_2 = ttnn.pad(
            ttnn_permute_8,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_0 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_1,
                ttnn_pad_0,
                ttnn_pad_2,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_3 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_0,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_9 = ttnn.permute(
            ttnn_slice_3,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_199 = ttnn.reshape(
            ttnn_permute_9,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_reshape_199,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_2 = ttnn.add(
            ttnn_matmul_2,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_2

    def CLIPEncoderLayer_4_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_3 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_3 = ttnn.layer_norm(
            ttnn_add_3,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_3, ttnn_add_3

    def CLIPMLP_5_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_200 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_reshape_200,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_4 = ttnn.add(
            ttnn_matmul_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_0 = ttnn.gelu(
            ttnn_add_4,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_201 = ttnn.reshape(
            ttnn_gelu_0,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_reshape_201,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_5 = ttnn.add(
            ttnn_matmul_4,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_5

    def CLIPEncoderLayer_6_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_6 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_4 = ttnn.layer_norm(
            ttnn_add_6,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_6, ttnn_layer_norm_4

    def CLIPAttention_7_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_202 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_reshape_202,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_7 = ttnn.add(
            ttnn_matmul_5,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_4 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_5 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_6 = ttnn.slice(
            ttnn_add_7,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_203 = ttnn.reshape(
            ttnn_slice_4,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_204 = ttnn.reshape(
            ttnn_slice_5,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_205 = ttnn.reshape(
            ttnn_slice_6,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_10 = ttnn.permute(
            ttnn_reshape_204,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_11 = ttnn.permute(
            ttnn_reshape_205,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_12 = ttnn.permute(
            ttnn_reshape_203,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_3 = ttnn.pad(
            ttnn_permute_10,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_4 = ttnn.pad(
            ttnn_permute_11,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_5 = ttnn.pad(
            ttnn_permute_12,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_1 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_4,
                ttnn_pad_3,
                ttnn_pad_5,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_7 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_1,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_13 = ttnn.permute(
            ttnn_slice_7,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_206 = ttnn.reshape(
            ttnn_permute_13,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_6 = ttnn.matmul(
            ttnn_reshape_206,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_8 = ttnn.add(
            ttnn_matmul_6,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_8

    def CLIPEncoderLayer_8_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_9 = ttnn.add(
            input_1,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_5 = ttnn.layer_norm(
            ttnn_add_9,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_9, ttnn_layer_norm_5

    def CLIPMLP_9_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_207 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_7 = ttnn.matmul(
            ttnn_reshape_207,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_10 = ttnn.add(
            ttnn_matmul_7,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_1 = ttnn.gelu(
            ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_208 = ttnn.reshape(
            ttnn_gelu_1,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_reshape_208,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_11 = ttnn.add(
            ttnn_matmul_8,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_11

    def CLIPEncoderLayer_10_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_12 = ttnn.add(
            input_0,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_6 = ttnn.layer_norm(
            ttnn_add_12,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_12, ttnn_layer_norm_6

    def CLIPAttention_11_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_209 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_reshape_209,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_13 = ttnn.add(
            ttnn_matmul_9,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_8 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_9 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_10 = ttnn.slice(
            ttnn_add_13,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_210 = ttnn.reshape(
            ttnn_slice_8,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_211 = ttnn.reshape(
            ttnn_slice_9,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_212 = ttnn.reshape(
            ttnn_slice_10,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_14 = ttnn.permute(
            ttnn_reshape_211,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_15 = ttnn.permute(
            ttnn_reshape_212,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_16 = ttnn.permute(
            ttnn_reshape_210,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_6 = ttnn.pad(
            ttnn_permute_14,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_7 = ttnn.pad(
            ttnn_permute_15,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_8 = ttnn.pad(
            ttnn_permute_16,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_2 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_7,
                ttnn_pad_6,
                ttnn_pad_8,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_11 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_2,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_17 = ttnn.permute(
            ttnn_slice_11,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_213 = ttnn.reshape(
            ttnn_permute_17,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_reshape_213,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_14 = ttnn.add(
            ttnn_matmul_10,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_14

    def CLIPEncoderLayer_12_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_15 = ttnn.add(
            input_0,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_7 = ttnn.layer_norm(
            ttnn_add_15,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_15, ttnn_layer_norm_7

    def CLIPMLP_13_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_214 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_11 = ttnn.matmul(
            ttnn_reshape_214,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_16 = ttnn.add(
            ttnn_matmul_11,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_2 = ttnn.gelu(
            ttnn_add_16,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_215 = ttnn.reshape(
            ttnn_gelu_2,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_12 = ttnn.matmul(
            ttnn_reshape_215,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_17 = ttnn.add(
            ttnn_matmul_12,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_17

    def CLIPEncoderLayer_14_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_18 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_8 = ttnn.layer_norm(
            ttnn_add_18,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_18, ttnn_layer_norm_8

    def CLIPAttention_15_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_216 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_13 = ttnn.matmul(
            ttnn_reshape_216,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_19 = ttnn.add(
            ttnn_matmul_13,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_12 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_13 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_14 = ttnn.slice(
            ttnn_add_19,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_217 = ttnn.reshape(
            ttnn_slice_12,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_218 = ttnn.reshape(
            ttnn_slice_13,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_219 = ttnn.reshape(
            ttnn_slice_14,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_18 = ttnn.permute(
            ttnn_reshape_218,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_19 = ttnn.permute(
            ttnn_reshape_219,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_20 = ttnn.permute(
            ttnn_reshape_217,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_9 = ttnn.pad(
            ttnn_permute_18,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_10 = ttnn.pad(
            ttnn_permute_19,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_11 = ttnn.pad(
            ttnn_permute_20,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_3 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_10,
                ttnn_pad_9,
                ttnn_pad_11,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_15 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_3,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_21 = ttnn.permute(
            ttnn_slice_15,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_220 = ttnn.reshape(
            ttnn_permute_21,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_14 = ttnn.matmul(
            ttnn_reshape_220,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_20 = ttnn.add(
            ttnn_matmul_14,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_20

    def CLIPEncoderLayer_16_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_21 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_9 = ttnn.layer_norm(
            ttnn_add_21,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_9, ttnn_add_21

    def CLIPMLP_17_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_221 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_15 = ttnn.matmul(
            ttnn_reshape_221,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_22 = ttnn.add(
            ttnn_matmul_15,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_3 = ttnn.gelu(
            ttnn_add_22,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_222 = ttnn.reshape(
            ttnn_gelu_3,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_16 = ttnn.matmul(
            ttnn_reshape_222,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_23 = ttnn.add(
            ttnn_matmul_16,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_23

    def CLIPEncoderLayer_18_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_24 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_10 = ttnn.layer_norm(
            ttnn_add_24,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_10, ttnn_add_24

    def CLIPAttention_19_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_223 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_17 = ttnn.matmul(
            ttnn_reshape_223,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_25 = ttnn.add(
            ttnn_matmul_17,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_16 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_17 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_18 = ttnn.slice(
            ttnn_add_25,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_224 = ttnn.reshape(
            ttnn_slice_16,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_225 = ttnn.reshape(
            ttnn_slice_17,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_226 = ttnn.reshape(
            ttnn_slice_18,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_22 = ttnn.permute(
            ttnn_reshape_225,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_23 = ttnn.permute(
            ttnn_reshape_226,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_24 = ttnn.permute(
            ttnn_reshape_224,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_12 = ttnn.pad(
            ttnn_permute_22,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_13 = ttnn.pad(
            ttnn_permute_23,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_14 = ttnn.pad(
            ttnn_permute_24,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_4 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_13,
                ttnn_pad_12,
                ttnn_pad_14,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_19 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_4,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_25 = ttnn.permute(
            ttnn_slice_19,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_227 = ttnn.reshape(
            ttnn_permute_25,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_18 = ttnn.matmul(
            ttnn_reshape_227,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_26 = ttnn.add(
            ttnn_matmul_18,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_26

    def CLIPEncoderLayer_20_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_27 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_11 = ttnn.layer_norm(
            ttnn_add_27,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_11, ttnn_add_27

    def CLIPMLP_21_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_228 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_19 = ttnn.matmul(
            ttnn_reshape_228,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_28 = ttnn.add(
            ttnn_matmul_19,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_4 = ttnn.gelu(
            ttnn_add_28,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_229 = ttnn.reshape(
            ttnn_gelu_4,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_20 = ttnn.matmul(
            ttnn_reshape_229,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_29 = ttnn.add(
            ttnn_matmul_20,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_29

    def CLIPEncoderLayer_22_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_30 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_12 = ttnn.layer_norm(
            ttnn_add_30,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_30, ttnn_layer_norm_12

    def CLIPAttention_23_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_230 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_21 = ttnn.matmul(
            ttnn_reshape_230,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_31 = ttnn.add(
            ttnn_matmul_21,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_20 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_21 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_22 = ttnn.slice(
            ttnn_add_31,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_231 = ttnn.reshape(
            ttnn_slice_20,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_232 = ttnn.reshape(
            ttnn_slice_21,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_233 = ttnn.reshape(
            ttnn_slice_22,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_26 = ttnn.permute(
            ttnn_reshape_232,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_27 = ttnn.permute(
            ttnn_reshape_233,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_28 = ttnn.permute(
            ttnn_reshape_231,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_15 = ttnn.pad(
            ttnn_permute_26,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_16 = ttnn.pad(
            ttnn_permute_27,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_17 = ttnn.pad(
            ttnn_permute_28,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_5 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_16,
                ttnn_pad_15,
                ttnn_pad_17,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_23 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_5,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_29 = ttnn.permute(
            ttnn_slice_23,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_234 = ttnn.reshape(
            ttnn_permute_29,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_22 = ttnn.matmul(
            ttnn_reshape_234,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_32 = ttnn.add(
            ttnn_matmul_22,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_32

    def CLIPEncoderLayer_24_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_33 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_13 = ttnn.layer_norm(
            ttnn_add_33,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_13, ttnn_add_33

    def CLIPMLP_25_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_235 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_23 = ttnn.matmul(
            ttnn_reshape_235,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_34 = ttnn.add(
            ttnn_matmul_23,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_5 = ttnn.gelu(
            ttnn_add_34,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_236 = ttnn.reshape(
            ttnn_gelu_5,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_24 = ttnn.matmul(
            ttnn_reshape_236,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_35 = ttnn.add(
            ttnn_matmul_24,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_35

    def CLIPEncoderLayer_26_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_36 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_14 = ttnn.layer_norm(
            ttnn_add_36,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_14, ttnn_add_36

    def CLIPAttention_27_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_237 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_25 = ttnn.matmul(
            ttnn_reshape_237,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_37 = ttnn.add(
            ttnn_matmul_25,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_24 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_25 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_26 = ttnn.slice(
            ttnn_add_37,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_238 = ttnn.reshape(
            ttnn_slice_24,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_239 = ttnn.reshape(
            ttnn_slice_25,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_240 = ttnn.reshape(
            ttnn_slice_26,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_30 = ttnn.permute(
            ttnn_reshape_239,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_31 = ttnn.permute(
            ttnn_reshape_240,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_32 = ttnn.permute(
            ttnn_reshape_238,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_18 = ttnn.pad(
            ttnn_permute_30,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_19 = ttnn.pad(
            ttnn_permute_31,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_20 = ttnn.pad(
            ttnn_permute_32,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_6 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_19,
                ttnn_pad_18,
                ttnn_pad_20,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_27 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_6,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_33 = ttnn.permute(
            ttnn_slice_27,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_241 = ttnn.reshape(
            ttnn_permute_33,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_26 = ttnn.matmul(
            ttnn_reshape_241,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_38 = ttnn.add(
            ttnn_matmul_26,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_38

    def CLIPEncoderLayer_28_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_39 = ttnn.add(
            input_3,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_15 = ttnn.layer_norm(
            ttnn_add_39,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_39, ttnn_layer_norm_15

    def CLIPMLP_29_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_242 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_27 = ttnn.matmul(
            ttnn_reshape_242,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_40 = ttnn.add(
            ttnn_matmul_27,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_6 = ttnn.gelu(
            ttnn_add_40,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_243 = ttnn.reshape(
            ttnn_gelu_6,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_28 = ttnn.matmul(
            ttnn_reshape_243,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_41 = ttnn.add(
            ttnn_matmul_28,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_41

    def CLIPEncoderLayer_30_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_42 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_16 = ttnn.layer_norm(
            ttnn_add_42,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_16, ttnn_add_42

    def CLIPAttention_31_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_244 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_29 = ttnn.matmul(
            ttnn_reshape_244,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_43 = ttnn.add(
            ttnn_matmul_29,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_28 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_29 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_30 = ttnn.slice(
            ttnn_add_43,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_245 = ttnn.reshape(
            ttnn_slice_28,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_246 = ttnn.reshape(
            ttnn_slice_29,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_247 = ttnn.reshape(
            ttnn_slice_30,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_34 = ttnn.permute(
            ttnn_reshape_246,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_35 = ttnn.permute(
            ttnn_reshape_247,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_36 = ttnn.permute(
            ttnn_reshape_245,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_21 = ttnn.pad(
            ttnn_permute_34,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_22 = ttnn.pad(
            ttnn_permute_35,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_23 = ttnn.pad(
            ttnn_permute_36,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_7 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_22,
                ttnn_pad_21,
                ttnn_pad_23,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_31 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_7,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_37 = ttnn.permute(
            ttnn_slice_31,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_248 = ttnn.reshape(
            ttnn_permute_37,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_30 = ttnn.matmul(
            ttnn_reshape_248,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_44 = ttnn.add(
            ttnn_matmul_30,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_44

    def CLIPEncoderLayer_32_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_45 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_17 = ttnn.layer_norm(
            ttnn_add_45,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_45, ttnn_layer_norm_17

    def CLIPMLP_33_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_249 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_31 = ttnn.matmul(
            ttnn_reshape_249,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_46 = ttnn.add(
            ttnn_matmul_31,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_7 = ttnn.gelu(
            ttnn_add_46,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_250 = ttnn.reshape(
            ttnn_gelu_7,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_32 = ttnn.matmul(
            ttnn_reshape_250,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_47 = ttnn.add(
            ttnn_matmul_32,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_47

    def CLIPEncoderLayer_34_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_48 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_18 = ttnn.layer_norm(
            ttnn_add_48,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_18, ttnn_add_48

    def CLIPAttention_35_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_251 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_33 = ttnn.matmul(
            ttnn_reshape_251,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_49 = ttnn.add(
            ttnn_matmul_33,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_32 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_33 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_34 = ttnn.slice(
            ttnn_add_49,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_252 = ttnn.reshape(
            ttnn_slice_32,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_253 = ttnn.reshape(
            ttnn_slice_33,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_254 = ttnn.reshape(
            ttnn_slice_34,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_38 = ttnn.permute(
            ttnn_reshape_253,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_39 = ttnn.permute(
            ttnn_reshape_254,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_40 = ttnn.permute(
            ttnn_reshape_252,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_24 = ttnn.pad(
            ttnn_permute_38,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_25 = ttnn.pad(
            ttnn_permute_39,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_26 = ttnn.pad(
            ttnn_permute_40,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_8 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_25,
                ttnn_pad_24,
                ttnn_pad_26,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_35 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_8,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_41 = ttnn.permute(
            ttnn_slice_35,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_255 = ttnn.reshape(
            ttnn_permute_41,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_34 = ttnn.matmul(
            ttnn_reshape_255,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_50 = ttnn.add(
            ttnn_matmul_34,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_50

    def CLIPEncoderLayer_36_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_51 = ttnn.add(
            input_3,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_19 = ttnn.layer_norm(
            ttnn_add_51,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_19, ttnn_add_51

    def CLIPMLP_37_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_256 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_35 = ttnn.matmul(
            ttnn_reshape_256,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_52 = ttnn.add(
            ttnn_matmul_35,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_8 = ttnn.gelu(
            ttnn_add_52,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_257 = ttnn.reshape(
            ttnn_gelu_8,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_36 = ttnn.matmul(
            ttnn_reshape_257,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_53 = ttnn.add(
            ttnn_matmul_36,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_53

    def CLIPEncoderLayer_38_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_54 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_20 = ttnn.layer_norm(
            ttnn_add_54,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_20, ttnn_add_54

    def CLIPAttention_39_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_258 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_37 = ttnn.matmul(
            ttnn_reshape_258,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_55 = ttnn.add(
            ttnn_matmul_37,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_36 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_37 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_38 = ttnn.slice(
            ttnn_add_55,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_259 = ttnn.reshape(
            ttnn_slice_36,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_260 = ttnn.reshape(
            ttnn_slice_37,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_261 = ttnn.reshape(
            ttnn_slice_38,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_42 = ttnn.permute(
            ttnn_reshape_260,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_43 = ttnn.permute(
            ttnn_reshape_261,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_44 = ttnn.permute(
            ttnn_reshape_259,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_27 = ttnn.pad(
            ttnn_permute_42,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_28 = ttnn.pad(
            ttnn_permute_43,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_29 = ttnn.pad(
            ttnn_permute_44,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_9 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_28,
                ttnn_pad_27,
                ttnn_pad_29,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_39 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_9,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_45 = ttnn.permute(
            ttnn_slice_39,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_262 = ttnn.reshape(
            ttnn_permute_45,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_38 = ttnn.matmul(
            ttnn_reshape_262,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_56 = ttnn.add(
            ttnn_matmul_38,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_56

    def CLIPEncoderLayer_40_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_57 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_21 = ttnn.layer_norm(
            ttnn_add_57,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_57, ttnn_layer_norm_21

    def CLIPMLP_41_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_263 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_39 = ttnn.matmul(
            ttnn_reshape_263,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_58 = ttnn.add(
            ttnn_matmul_39,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_9 = ttnn.gelu(
            ttnn_add_58,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_264 = ttnn.reshape(
            ttnn_gelu_9,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_40 = ttnn.matmul(
            ttnn_reshape_264,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_59 = ttnn.add(
            ttnn_matmul_40,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_59

    def CLIPEncoderLayer_42_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_60 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_22 = ttnn.layer_norm(
            ttnn_add_60,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_60, ttnn_layer_norm_22

    def CLIPAttention_43_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_265 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_41 = ttnn.matmul(
            ttnn_reshape_265,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_61 = ttnn.add(
            ttnn_matmul_41,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_40 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_41 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_42 = ttnn.slice(
            ttnn_add_61,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_266 = ttnn.reshape(
            ttnn_slice_40,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_267 = ttnn.reshape(
            ttnn_slice_41,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_268 = ttnn.reshape(
            ttnn_slice_42,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_46 = ttnn.permute(
            ttnn_reshape_267,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_47 = ttnn.permute(
            ttnn_reshape_268,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_48 = ttnn.permute(
            ttnn_reshape_266,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_30 = ttnn.pad(
            ttnn_permute_46,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_31 = ttnn.pad(
            ttnn_permute_47,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_32 = ttnn.pad(
            ttnn_permute_48,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_10 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_31,
                ttnn_pad_30,
                ttnn_pad_32,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_43 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_10,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_49 = ttnn.permute(
            ttnn_slice_43,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_269 = ttnn.reshape(
            ttnn_permute_49,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_42 = ttnn.matmul(
            ttnn_reshape_269,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_62 = ttnn.add(
            ttnn_matmul_42,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_62

    def CLIPEncoderLayer_44_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_63 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_23 = ttnn.layer_norm(
            ttnn_add_63,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_63, ttnn_layer_norm_23

    def CLIPMLP_45_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_270 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_43 = ttnn.matmul(
            ttnn_reshape_270,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_64 = ttnn.add(
            ttnn_matmul_43,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_10 = ttnn.gelu(
            ttnn_add_64,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_271 = ttnn.reshape(
            ttnn_gelu_10,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_44 = ttnn.matmul(
            ttnn_reshape_271,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_65 = ttnn.add(
            ttnn_matmul_44,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_65

    def CLIPEncoderLayer_46_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_66 = ttnn.add(
            input_0,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_24 = ttnn.layer_norm(
            ttnn_add_66,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_24, ttnn_add_66

    def CLIPAttention_47_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_272 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_45 = ttnn.matmul(
            ttnn_reshape_272,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_67 = ttnn.add(
            ttnn_matmul_45,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_44 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_45 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_46 = ttnn.slice(
            ttnn_add_67,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_273 = ttnn.reshape(
            ttnn_slice_44,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_274 = ttnn.reshape(
            ttnn_slice_45,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_275 = ttnn.reshape(
            ttnn_slice_46,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_50 = ttnn.permute(
            ttnn_reshape_274,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_51 = ttnn.permute(
            ttnn_reshape_275,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_52 = ttnn.permute(
            ttnn_reshape_273,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_33 = ttnn.pad(
            ttnn_permute_50,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_34 = ttnn.pad(
            ttnn_permute_51,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_35 = ttnn.pad(
            ttnn_permute_52,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_11 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_34,
                ttnn_pad_33,
                ttnn_pad_35,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_47 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_11,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_53 = ttnn.permute(
            ttnn_slice_47,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_276 = ttnn.reshape(
            ttnn_permute_53,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_46 = ttnn.matmul(
            ttnn_reshape_276,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_68 = ttnn.add(
            ttnn_matmul_46,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_68

    def CLIPEncoderLayer_48_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_69 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_25 = ttnn.layer_norm(
            ttnn_add_69,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_25, ttnn_add_69

    def CLIPMLP_49_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_277 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_47 = ttnn.matmul(
            ttnn_reshape_277,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_70 = ttnn.add(
            ttnn_matmul_47,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_11 = ttnn.gelu(
            ttnn_add_70,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_278 = ttnn.reshape(
            ttnn_gelu_11,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_48 = ttnn.matmul(
            ttnn_reshape_278,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_71 = ttnn.add(
            ttnn_matmul_48,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_71

    def CLIPEncoderLayer_50_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_72 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_26 = ttnn.layer_norm(
            ttnn_add_72,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_26, ttnn_add_72

    def CLIPAttention_51_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_279 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_49 = ttnn.matmul(
            ttnn_reshape_279,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_73 = ttnn.add(
            ttnn_matmul_49,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_48 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_49 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_50 = ttnn.slice(
            ttnn_add_73,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_280 = ttnn.reshape(
            ttnn_slice_48,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_281 = ttnn.reshape(
            ttnn_slice_49,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_282 = ttnn.reshape(
            ttnn_slice_50,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_54 = ttnn.permute(
            ttnn_reshape_281,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_55 = ttnn.permute(
            ttnn_reshape_282,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_56 = ttnn.permute(
            ttnn_reshape_280,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_36 = ttnn.pad(
            ttnn_permute_54,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_37 = ttnn.pad(
            ttnn_permute_55,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_38 = ttnn.pad(
            ttnn_permute_56,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_12 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_37,
                ttnn_pad_36,
                ttnn_pad_38,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_51 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_12,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_57 = ttnn.permute(
            ttnn_slice_51,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_283 = ttnn.reshape(
            ttnn_permute_57,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_50 = ttnn.matmul(
            ttnn_reshape_283,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_74 = ttnn.add(
            ttnn_matmul_50,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_74

    def CLIPEncoderLayer_52_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_75 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_27 = ttnn.layer_norm(
            ttnn_add_75,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_27, ttnn_add_75

    def CLIPMLP_53_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_284 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_51 = ttnn.matmul(
            ttnn_reshape_284,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_76 = ttnn.add(
            ttnn_matmul_51,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_12 = ttnn.gelu(
            ttnn_add_76,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_285 = ttnn.reshape(
            ttnn_gelu_12,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_52 = ttnn.matmul(
            ttnn_reshape_285,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_77 = ttnn.add(
            ttnn_matmul_52,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_77

    def CLIPEncoderLayer_54_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_78 = ttnn.add(
            input_3,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_28 = ttnn.layer_norm(
            ttnn_add_78,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_78, ttnn_layer_norm_28

    def CLIPAttention_55_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_286 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_53 = ttnn.matmul(
            ttnn_reshape_286,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_79 = ttnn.add(
            ttnn_matmul_53,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_52 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_53 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_54 = ttnn.slice(
            ttnn_add_79,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_287 = ttnn.reshape(
            ttnn_slice_52,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_288 = ttnn.reshape(
            ttnn_slice_53,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_289 = ttnn.reshape(
            ttnn_slice_54,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_58 = ttnn.permute(
            ttnn_reshape_288,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_59 = ttnn.permute(
            ttnn_reshape_289,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_60 = ttnn.permute(
            ttnn_reshape_287,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_39 = ttnn.pad(
            ttnn_permute_58,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_40 = ttnn.pad(
            ttnn_permute_59,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_41 = ttnn.pad(
            ttnn_permute_60,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_13 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_40,
                ttnn_pad_39,
                ttnn_pad_41,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_55 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_13,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_61 = ttnn.permute(
            ttnn_slice_55,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_290 = ttnn.reshape(
            ttnn_permute_61,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_54 = ttnn.matmul(
            ttnn_reshape_290,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_80 = ttnn.add(
            ttnn_matmul_54,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_80

    def CLIPEncoderLayer_56_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_81 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_29 = ttnn.layer_norm(
            ttnn_add_81,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_81, ttnn_layer_norm_29

    def CLIPMLP_57_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_291 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_55 = ttnn.matmul(
            ttnn_reshape_291,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_82 = ttnn.add(
            ttnn_matmul_55,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_13 = ttnn.gelu(
            ttnn_add_82,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_292 = ttnn.reshape(
            ttnn_gelu_13,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_56 = ttnn.matmul(
            ttnn_reshape_292,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_83 = ttnn.add(
            ttnn_matmul_56,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_83

    def CLIPEncoderLayer_58_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_84 = ttnn.add(
            input_1,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_30 = ttnn.layer_norm(
            ttnn_add_84,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_30, ttnn_add_84

    def CLIPAttention_59_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_293 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_57 = ttnn.matmul(
            ttnn_reshape_293,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_85 = ttnn.add(
            ttnn_matmul_57,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_56 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_57 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_58 = ttnn.slice(
            ttnn_add_85,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_294 = ttnn.reshape(
            ttnn_slice_56,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_295 = ttnn.reshape(
            ttnn_slice_57,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_296 = ttnn.reshape(
            ttnn_slice_58,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_62 = ttnn.permute(
            ttnn_reshape_295,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_63 = ttnn.permute(
            ttnn_reshape_296,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_64 = ttnn.permute(
            ttnn_reshape_294,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_42 = ttnn.pad(
            ttnn_permute_62,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_43 = ttnn.pad(
            ttnn_permute_63,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_44 = ttnn.pad(
            ttnn_permute_64,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_14 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_43,
                ttnn_pad_42,
                ttnn_pad_44,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_59 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_14,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_65 = ttnn.permute(
            ttnn_slice_59,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_297 = ttnn.reshape(
            ttnn_permute_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_58 = ttnn.matmul(
            ttnn_reshape_297,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_86 = ttnn.add(
            ttnn_matmul_58,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_86

    def CLIPEncoderLayer_60_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_87 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_31 = ttnn.layer_norm(
            ttnn_add_87,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_87, ttnn_layer_norm_31

    def CLIPMLP_61_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_298 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_59 = ttnn.matmul(
            ttnn_reshape_298,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_88 = ttnn.add(
            ttnn_matmul_59,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_14 = ttnn.gelu(
            ttnn_add_88,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_299 = ttnn.reshape(
            ttnn_gelu_14,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_60 = ttnn.matmul(
            ttnn_reshape_299,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_89 = ttnn.add(
            ttnn_matmul_60,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_89

    def CLIPEncoderLayer_62_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_90 = ttnn.add(
            input_3,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_32 = ttnn.layer_norm(
            ttnn_add_90,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_32, ttnn_add_90

    def CLIPAttention_63_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_300 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_61 = ttnn.matmul(
            ttnn_reshape_300,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_91 = ttnn.add(
            ttnn_matmul_61,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_60 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_61 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_62 = ttnn.slice(
            ttnn_add_91,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_301 = ttnn.reshape(
            ttnn_slice_60,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_302 = ttnn.reshape(
            ttnn_slice_61,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_303 = ttnn.reshape(
            ttnn_slice_62,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_66 = ttnn.permute(
            ttnn_reshape_302,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_67 = ttnn.permute(
            ttnn_reshape_303,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_68 = ttnn.permute(
            ttnn_reshape_301,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_45 = ttnn.pad(
            ttnn_permute_66,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_46 = ttnn.pad(
            ttnn_permute_67,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_47 = ttnn.pad(
            ttnn_permute_68,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_15 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_46,
                ttnn_pad_45,
                ttnn_pad_47,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_63 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_15,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_69 = ttnn.permute(
            ttnn_slice_63,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_304 = ttnn.reshape(
            ttnn_permute_69,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_62 = ttnn.matmul(
            ttnn_reshape_304,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_92 = ttnn.add(
            ttnn_matmul_62,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_92

    def CLIPEncoderLayer_64_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_93 = ttnn.add(
            input_3,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_33 = ttnn.layer_norm(
            ttnn_add_93,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_93, ttnn_layer_norm_33

    def CLIPMLP_65_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_305 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_63 = ttnn.matmul(
            ttnn_reshape_305,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_94 = ttnn.add(
            ttnn_matmul_63,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_15 = ttnn.gelu(
            ttnn_add_94,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_306 = ttnn.reshape(
            ttnn_gelu_15,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_64 = ttnn.matmul(
            ttnn_reshape_306,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_95 = ttnn.add(
            ttnn_matmul_64,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_95

    def CLIPEncoderLayer_66_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_96 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_34 = ttnn.layer_norm(
            ttnn_add_96,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_96, ttnn_layer_norm_34

    def CLIPAttention_67_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_307 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_65 = ttnn.matmul(
            ttnn_reshape_307,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_97 = ttnn.add(
            ttnn_matmul_65,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_64 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_65 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_66 = ttnn.slice(
            ttnn_add_97,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_308 = ttnn.reshape(
            ttnn_slice_64,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_309 = ttnn.reshape(
            ttnn_slice_65,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_310 = ttnn.reshape(
            ttnn_slice_66,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_70 = ttnn.permute(
            ttnn_reshape_309,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_71 = ttnn.permute(
            ttnn_reshape_310,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_72 = ttnn.permute(
            ttnn_reshape_308,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_48 = ttnn.pad(
            ttnn_permute_70,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_49 = ttnn.pad(
            ttnn_permute_71,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_50 = ttnn.pad(
            ttnn_permute_72,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_16 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_49,
                ttnn_pad_48,
                ttnn_pad_50,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_67 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_16,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_73 = ttnn.permute(
            ttnn_slice_67,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_311 = ttnn.reshape(
            ttnn_permute_73,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_66 = ttnn.matmul(
            ttnn_reshape_311,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_98 = ttnn.add(
            ttnn_matmul_66,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_98

    def CLIPEncoderLayer_68_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_99 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_35 = ttnn.layer_norm(
            ttnn_add_99,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_99, ttnn_layer_norm_35

    def CLIPMLP_69_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_312 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_67 = ttnn.matmul(
            ttnn_reshape_312,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_100 = ttnn.add(
            ttnn_matmul_67,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_16 = ttnn.gelu(
            ttnn_add_100,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_313 = ttnn.reshape(
            ttnn_gelu_16,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_68 = ttnn.matmul(
            ttnn_reshape_313,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_101 = ttnn.add(
            ttnn_matmul_68,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_101

    def CLIPEncoderLayer_70_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_102 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_36 = ttnn.layer_norm(
            ttnn_add_102,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_102, ttnn_layer_norm_36

    def CLIPAttention_71_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_314 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_69 = ttnn.matmul(
            ttnn_reshape_314,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_103 = ttnn.add(
            ttnn_matmul_69,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_68 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_69 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_70 = ttnn.slice(
            ttnn_add_103,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_315 = ttnn.reshape(
            ttnn_slice_68,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_316 = ttnn.reshape(
            ttnn_slice_69,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_317 = ttnn.reshape(
            ttnn_slice_70,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_74 = ttnn.permute(
            ttnn_reshape_316,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_75 = ttnn.permute(
            ttnn_reshape_317,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_76 = ttnn.permute(
            ttnn_reshape_315,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_51 = ttnn.pad(
            ttnn_permute_74,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_52 = ttnn.pad(
            ttnn_permute_75,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_53 = ttnn.pad(
            ttnn_permute_76,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_17 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_52,
                ttnn_pad_51,
                ttnn_pad_53,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_71 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_17,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_77 = ttnn.permute(
            ttnn_slice_71,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_318 = ttnn.reshape(
            ttnn_permute_77,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_70 = ttnn.matmul(
            ttnn_reshape_318,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_104 = ttnn.add(
            ttnn_matmul_70,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_104

    def CLIPEncoderLayer_72_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_105 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_37 = ttnn.layer_norm(
            ttnn_add_105,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_37, ttnn_add_105

    def CLIPMLP_73_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_319 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_71 = ttnn.matmul(
            ttnn_reshape_319,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_106 = ttnn.add(
            ttnn_matmul_71,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_17 = ttnn.gelu(
            ttnn_add_106,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_320 = ttnn.reshape(
            ttnn_gelu_17,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_72 = ttnn.matmul(
            ttnn_reshape_320,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_107 = ttnn.add(
            ttnn_matmul_72,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_107

    def CLIPEncoderLayer_74_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_108 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_38 = ttnn.layer_norm(
            ttnn_add_108,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_38, ttnn_add_108

    def CLIPAttention_75_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_321 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_73 = ttnn.matmul(
            ttnn_reshape_321,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_109 = ttnn.add(
            ttnn_matmul_73,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_72 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_73 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_74 = ttnn.slice(
            ttnn_add_109,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_322 = ttnn.reshape(
            ttnn_slice_72,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_323 = ttnn.reshape(
            ttnn_slice_73,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_324 = ttnn.reshape(
            ttnn_slice_74,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_78 = ttnn.permute(
            ttnn_reshape_323,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_79 = ttnn.permute(
            ttnn_reshape_324,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_80 = ttnn.permute(
            ttnn_reshape_322,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_54 = ttnn.pad(
            ttnn_permute_78,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_55 = ttnn.pad(
            ttnn_permute_79,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_56 = ttnn.pad(
            ttnn_permute_80,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_18 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_55,
                ttnn_pad_54,
                ttnn_pad_56,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_75 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_18,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_81 = ttnn.permute(
            ttnn_slice_75,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_325 = ttnn.reshape(
            ttnn_permute_81,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_74 = ttnn.matmul(
            ttnn_reshape_325,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_110 = ttnn.add(
            ttnn_matmul_74,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_110

    def CLIPEncoderLayer_76_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_111 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_39 = ttnn.layer_norm(
            ttnn_add_111,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_39, ttnn_add_111

    def CLIPMLP_77_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_326 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_75 = ttnn.matmul(
            ttnn_reshape_326,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_112 = ttnn.add(
            ttnn_matmul_75,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_18 = ttnn.gelu(
            ttnn_add_112,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_327 = ttnn.reshape(
            ttnn_gelu_18,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_76 = ttnn.matmul(
            ttnn_reshape_327,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_113 = ttnn.add(
            ttnn_matmul_76,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_113

    def CLIPEncoderLayer_78_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_114 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_40 = ttnn.layer_norm(
            ttnn_add_114,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_40, ttnn_add_114

    def CLIPAttention_79_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_328 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_77 = ttnn.matmul(
            ttnn_reshape_328,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_115 = ttnn.add(
            ttnn_matmul_77,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_76 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_77 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_78 = ttnn.slice(
            ttnn_add_115,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_329 = ttnn.reshape(
            ttnn_slice_76,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_330 = ttnn.reshape(
            ttnn_slice_77,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_331 = ttnn.reshape(
            ttnn_slice_78,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_82 = ttnn.permute(
            ttnn_reshape_330,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_83 = ttnn.permute(
            ttnn_reshape_331,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_84 = ttnn.permute(
            ttnn_reshape_329,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_57 = ttnn.pad(
            ttnn_permute_82,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_58 = ttnn.pad(
            ttnn_permute_83,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_59 = ttnn.pad(
            ttnn_permute_84,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_19 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_58,
                ttnn_pad_57,
                ttnn_pad_59,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_79 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_19,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_85 = ttnn.permute(
            ttnn_slice_79,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_332 = ttnn.reshape(
            ttnn_permute_85,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_78 = ttnn.matmul(
            ttnn_reshape_332,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_116 = ttnn.add(
            ttnn_matmul_78,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_116

    def CLIPEncoderLayer_80_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_117 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_41 = ttnn.layer_norm(
            ttnn_add_117,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_117, ttnn_layer_norm_41

    def CLIPMLP_81_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_333 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_79 = ttnn.matmul(
            ttnn_reshape_333,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_118 = ttnn.add(
            ttnn_matmul_79,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_19 = ttnn.gelu(
            ttnn_add_118,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_334 = ttnn.reshape(
            ttnn_gelu_19,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_80 = ttnn.matmul(
            ttnn_reshape_334,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_119 = ttnn.add(
            ttnn_matmul_80,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_119

    def CLIPEncoderLayer_82_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_120 = ttnn.add(
            input_0,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_42 = ttnn.layer_norm(
            ttnn_add_120,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_120, ttnn_layer_norm_42

    def CLIPAttention_83_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_335 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_81 = ttnn.matmul(
            ttnn_reshape_335,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_121 = ttnn.add(
            ttnn_matmul_81,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_80 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_81 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_82 = ttnn.slice(
            ttnn_add_121,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_336 = ttnn.reshape(
            ttnn_slice_80,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_337 = ttnn.reshape(
            ttnn_slice_81,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_338 = ttnn.reshape(
            ttnn_slice_82,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_86 = ttnn.permute(
            ttnn_reshape_337,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_87 = ttnn.permute(
            ttnn_reshape_338,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_88 = ttnn.permute(
            ttnn_reshape_336,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_60 = ttnn.pad(
            ttnn_permute_86,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_61 = ttnn.pad(
            ttnn_permute_87,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_62 = ttnn.pad(
            ttnn_permute_88,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_20 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_61,
                ttnn_pad_60,
                ttnn_pad_62,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_83 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_20,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_89 = ttnn.permute(
            ttnn_slice_83,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_339 = ttnn.reshape(
            ttnn_permute_89,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_82 = ttnn.matmul(
            ttnn_reshape_339,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_122 = ttnn.add(
            ttnn_matmul_82,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_122

    def CLIPEncoderLayer_84_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_123 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_43 = ttnn.layer_norm(
            ttnn_add_123,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_123, ttnn_layer_norm_43

    def CLIPMLP_85_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_340 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_83 = ttnn.matmul(
            ttnn_reshape_340,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_124 = ttnn.add(
            ttnn_matmul_83,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_20 = ttnn.gelu(
            ttnn_add_124,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_341 = ttnn.reshape(
            ttnn_gelu_20,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_84 = ttnn.matmul(
            ttnn_reshape_341,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_125 = ttnn.add(
            ttnn_matmul_84,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_125

    def CLIPEncoderLayer_86_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_126 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_44 = ttnn.layer_norm(
            ttnn_add_126,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_126, ttnn_layer_norm_44

    def CLIPAttention_87_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_342 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_85 = ttnn.matmul(
            ttnn_reshape_342,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_127 = ttnn.add(
            ttnn_matmul_85,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_84 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_85 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_86 = ttnn.slice(
            ttnn_add_127,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_343 = ttnn.reshape(
            ttnn_slice_84,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_344 = ttnn.reshape(
            ttnn_slice_85,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_345 = ttnn.reshape(
            ttnn_slice_86,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_90 = ttnn.permute(
            ttnn_reshape_344,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_91 = ttnn.permute(
            ttnn_reshape_345,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_92 = ttnn.permute(
            ttnn_reshape_343,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_63 = ttnn.pad(
            ttnn_permute_90,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_64 = ttnn.pad(
            ttnn_permute_91,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_65 = ttnn.pad(
            ttnn_permute_92,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_21 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_64,
                ttnn_pad_63,
                ttnn_pad_65,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_87 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_21,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_93 = ttnn.permute(
            ttnn_slice_87,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_346 = ttnn.reshape(
            ttnn_permute_93,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_86 = ttnn.matmul(
            ttnn_reshape_346,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_128 = ttnn.add(
            ttnn_matmul_86,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_128

    def CLIPEncoderLayer_88_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_129 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_45 = ttnn.layer_norm(
            ttnn_add_129,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_45, ttnn_add_129

    def CLIPMLP_89_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_347 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_87 = ttnn.matmul(
            ttnn_reshape_347,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_130 = ttnn.add(
            ttnn_matmul_87,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_21 = ttnn.gelu(
            ttnn_add_130,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_348 = ttnn.reshape(
            ttnn_gelu_21,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_88 = ttnn.matmul(
            ttnn_reshape_348,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_131 = ttnn.add(
            ttnn_matmul_88,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_131

    def CLIPEncoderLayer_90_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_132 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_46 = ttnn.layer_norm(
            ttnn_add_132,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_46, ttnn_add_132

    def CLIPAttention_91_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_349 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_89 = ttnn.matmul(
            ttnn_reshape_349,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_133 = ttnn.add(
            ttnn_matmul_89,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_88 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_89 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_90 = ttnn.slice(
            ttnn_add_133,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_350 = ttnn.reshape(
            ttnn_slice_88,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_351 = ttnn.reshape(
            ttnn_slice_89,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_352 = ttnn.reshape(
            ttnn_slice_90,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_94 = ttnn.permute(
            ttnn_reshape_351,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_95 = ttnn.permute(
            ttnn_reshape_352,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_96 = ttnn.permute(
            ttnn_reshape_350,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_66 = ttnn.pad(
            ttnn_permute_94,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_67 = ttnn.pad(
            ttnn_permute_95,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_68 = ttnn.pad(
            ttnn_permute_96,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_22 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_67,
                ttnn_pad_66,
                ttnn_pad_68,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_91 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_22,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_97 = ttnn.permute(
            ttnn_slice_91,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_353 = ttnn.reshape(
            ttnn_permute_97,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_90 = ttnn.matmul(
            ttnn_reshape_353,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_134 = ttnn.add(
            ttnn_matmul_90,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_134

    def CLIPEncoderLayer_92_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_135 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_47 = ttnn.layer_norm(
            ttnn_add_135,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_135, ttnn_layer_norm_47

    def CLIPMLP_93_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_354 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_91 = ttnn.matmul(
            ttnn_reshape_354,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_136 = ttnn.add(
            ttnn_matmul_91,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_22 = ttnn.gelu(
            ttnn_add_136,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_355 = ttnn.reshape(
            ttnn_gelu_22,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_92 = ttnn.matmul(
            ttnn_reshape_355,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_137 = ttnn.add(
            ttnn_matmul_92,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_137

    def CLIPEncoderLayer_94_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_138 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_48 = ttnn.layer_norm(
            ttnn_add_138,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_138, ttnn_layer_norm_48

    def CLIPAttention_95_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_356 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_93 = ttnn.matmul(
            ttnn_reshape_356,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_139 = ttnn.add(
            ttnn_matmul_93,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_92 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_93 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_94 = ttnn.slice(
            ttnn_add_139,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_357 = ttnn.reshape(
            ttnn_slice_92,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_358 = ttnn.reshape(
            ttnn_slice_93,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_359 = ttnn.reshape(
            ttnn_slice_94,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_98 = ttnn.permute(
            ttnn_reshape_358,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_99 = ttnn.permute(
            ttnn_reshape_359,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_100 = ttnn.permute(
            ttnn_reshape_357,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_69 = ttnn.pad(
            ttnn_permute_98,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_70 = ttnn.pad(
            ttnn_permute_99,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_71 = ttnn.pad(
            ttnn_permute_100,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_23 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_70,
                ttnn_pad_69,
                ttnn_pad_71,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_95 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_23,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_101 = ttnn.permute(
            ttnn_slice_95,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_360 = ttnn.reshape(
            ttnn_permute_101,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_94 = ttnn.matmul(
            ttnn_reshape_360,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_140 = ttnn.add(
            ttnn_matmul_94,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_140

    def CLIPEncoderLayer_96_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_141 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_49 = ttnn.layer_norm(
            ttnn_add_141,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_141, ttnn_layer_norm_49

    def CLIPMLP_97_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_361 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_95 = ttnn.matmul(
            ttnn_reshape_361,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_142 = ttnn.add(
            ttnn_matmul_95,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_23 = ttnn.gelu(
            ttnn_add_142,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_362 = ttnn.reshape(
            ttnn_gelu_23,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_96 = ttnn.matmul(
            ttnn_reshape_362,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_143 = ttnn.add(
            ttnn_matmul_96,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_143

    def CLIPEncoderLayer_98_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_144 = ttnn.add(
            input_0,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_50 = ttnn.layer_norm(
            ttnn_add_144,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_144, ttnn_layer_norm_50

    def CLIPAttention_99_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_363 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_97 = ttnn.matmul(
            ttnn_reshape_363,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_145 = ttnn.add(
            ttnn_matmul_97,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_96 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_97 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_98 = ttnn.slice(
            ttnn_add_145,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_364 = ttnn.reshape(
            ttnn_slice_96,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_365 = ttnn.reshape(
            ttnn_slice_97,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_366 = ttnn.reshape(
            ttnn_slice_98,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_102 = ttnn.permute(
            ttnn_reshape_365,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_103 = ttnn.permute(
            ttnn_reshape_366,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_104 = ttnn.permute(
            ttnn_reshape_364,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_72 = ttnn.pad(
            ttnn_permute_102,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_73 = ttnn.pad(
            ttnn_permute_103,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_74 = ttnn.pad(
            ttnn_permute_104,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_24 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_73,
                ttnn_pad_72,
                ttnn_pad_74,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_99 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_24,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_105 = ttnn.permute(
            ttnn_slice_99,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_367 = ttnn.reshape(
            ttnn_permute_105,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_98 = ttnn.matmul(
            ttnn_reshape_367,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_146 = ttnn.add(
            ttnn_matmul_98,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_146

    def CLIPEncoderLayer_100_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_147 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_51 = ttnn.layer_norm(
            ttnn_add_147,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_147, ttnn_layer_norm_51

    def CLIPMLP_101_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_368 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_99 = ttnn.matmul(
            ttnn_reshape_368,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_148 = ttnn.add(
            ttnn_matmul_99,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_24 = ttnn.gelu(
            ttnn_add_148,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_369 = ttnn.reshape(
            ttnn_gelu_24,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_100 = ttnn.matmul(
            ttnn_reshape_369,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_149 = ttnn.add(
            ttnn_matmul_100,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_149

    def CLIPEncoderLayer_102_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_150 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_52 = ttnn.layer_norm(
            ttnn_add_150,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_52, ttnn_add_150

    def CLIPAttention_103_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_370 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_101 = ttnn.matmul(
            ttnn_reshape_370,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_151 = ttnn.add(
            ttnn_matmul_101,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_100 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_101 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_102 = ttnn.slice(
            ttnn_add_151,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_371 = ttnn.reshape(
            ttnn_slice_100,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_372 = ttnn.reshape(
            ttnn_slice_101,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_373 = ttnn.reshape(
            ttnn_slice_102,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_106 = ttnn.permute(
            ttnn_reshape_372,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_107 = ttnn.permute(
            ttnn_reshape_373,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_108 = ttnn.permute(
            ttnn_reshape_371,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_75 = ttnn.pad(
            ttnn_permute_106,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_76 = ttnn.pad(
            ttnn_permute_107,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_77 = ttnn.pad(
            ttnn_permute_108,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_25 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_76,
                ttnn_pad_75,
                ttnn_pad_77,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_103 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_25,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_109 = ttnn.permute(
            ttnn_slice_103,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_374 = ttnn.reshape(
            ttnn_permute_109,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_102 = ttnn.matmul(
            ttnn_reshape_374,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_152 = ttnn.add(
            ttnn_matmul_102,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_152

    def CLIPEncoderLayer_104_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_153 = ttnn.add(
            input_3,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_53 = ttnn.layer_norm(
            ttnn_add_153,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_153, ttnn_layer_norm_53

    def CLIPMLP_105_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_375 = ttnn.reshape(
            input_4,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_103 = ttnn.matmul(
            ttnn_reshape_375,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_154 = ttnn.add(
            ttnn_matmul_103,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_25 = ttnn.gelu(
            ttnn_add_154,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_376 = ttnn.reshape(
            ttnn_gelu_25,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_104 = ttnn.matmul(
            ttnn_reshape_376,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_155 = ttnn.add(
            ttnn_matmul_104,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_155

    def CLIPEncoderLayer_106_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_156 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_54 = ttnn.layer_norm(
            ttnn_add_156,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_54, ttnn_add_156

    def CLIPAttention_107_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_377 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_105 = ttnn.matmul(
            ttnn_reshape_377,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_157 = ttnn.add(
            ttnn_matmul_105,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_104 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_105 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_106 = ttnn.slice(
            ttnn_add_157,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_378 = ttnn.reshape(
            ttnn_slice_104,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_379 = ttnn.reshape(
            ttnn_slice_105,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_380 = ttnn.reshape(
            ttnn_slice_106,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_110 = ttnn.permute(
            ttnn_reshape_379,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_111 = ttnn.permute(
            ttnn_reshape_380,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_112 = ttnn.permute(
            ttnn_reshape_378,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_78 = ttnn.pad(
            ttnn_permute_110,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_79 = ttnn.pad(
            ttnn_permute_111,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_80 = ttnn.pad(
            ttnn_permute_112,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_26 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_79,
                ttnn_pad_78,
                ttnn_pad_80,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_107 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_26,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_113 = ttnn.permute(
            ttnn_slice_107,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_381 = ttnn.reshape(
            ttnn_permute_113,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_106 = ttnn.matmul(
            ttnn_reshape_381,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_158 = ttnn.add(
            ttnn_matmul_106,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_158

    def CLIPEncoderLayer_108_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_159 = ttnn.add(
            input_3,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_55 = ttnn.layer_norm(
            ttnn_add_159,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_55, ttnn_add_159

    def CLIPMLP_109_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_382 = ttnn.reshape(
            input_1,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_107 = ttnn.matmul(
            ttnn_reshape_382,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_160 = ttnn.add(
            ttnn_matmul_107,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_26 = ttnn.gelu(
            ttnn_add_160,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_383 = ttnn.reshape(
            ttnn_gelu_26,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_108 = ttnn.matmul(
            ttnn_reshape_383,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_161 = ttnn.add(
            ttnn_matmul_108,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_161

    def CLIPEncoderLayer_110_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_162 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_56 = ttnn.layer_norm(
            ttnn_add_162,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_162, ttnn_layer_norm_56

    def CLIPAttention_111_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_384 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_109 = ttnn.matmul(
            ttnn_reshape_384,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_163 = ttnn.add(
            ttnn_matmul_109,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_108 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_109 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_110 = ttnn.slice(
            ttnn_add_163,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_385 = ttnn.reshape(
            ttnn_slice_108,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_386 = ttnn.reshape(
            ttnn_slice_109,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_387 = ttnn.reshape(
            ttnn_slice_110,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_114 = ttnn.permute(
            ttnn_reshape_386,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_115 = ttnn.permute(
            ttnn_reshape_387,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_116 = ttnn.permute(
            ttnn_reshape_385,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_81 = ttnn.pad(
            ttnn_permute_114,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_82 = ttnn.pad(
            ttnn_permute_115,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_83 = ttnn.pad(
            ttnn_permute_116,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_27 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_82,
                ttnn_pad_81,
                ttnn_pad_83,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_111 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_27,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_117 = ttnn.permute(
            ttnn_slice_111,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_388 = ttnn.reshape(
            ttnn_permute_117,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_110 = ttnn.matmul(
            ttnn_reshape_388,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_164 = ttnn.add(
            ttnn_matmul_110,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_164

    def CLIPEncoderLayer_112_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_165 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_57 = ttnn.layer_norm(
            ttnn_add_165,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_57, ttnn_add_165

    def CLIPMLP_113_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_389 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_111 = ttnn.matmul(
            ttnn_reshape_389,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_166 = ttnn.add(
            ttnn_matmul_111,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_27 = ttnn.gelu(
            ttnn_add_166,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_390 = ttnn.reshape(
            ttnn_gelu_27,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_112 = ttnn.matmul(
            ttnn_reshape_390,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_167 = ttnn.add(
            ttnn_matmul_112,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_167

    def CLIPEncoderLayer_114_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_168 = ttnn.add(
            input_2,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_58 = ttnn.layer_norm(
            ttnn_add_168,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_168, ttnn_layer_norm_58

    def CLIPAttention_115_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_391 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_113 = ttnn.matmul(
            ttnn_reshape_391,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_169 = ttnn.add(
            ttnn_matmul_113,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_112 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_113 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_114 = ttnn.slice(
            ttnn_add_169,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_392 = ttnn.reshape(
            ttnn_slice_112,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_393 = ttnn.reshape(
            ttnn_slice_113,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_394 = ttnn.reshape(
            ttnn_slice_114,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_118 = ttnn.permute(
            ttnn_reshape_393,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_119 = ttnn.permute(
            ttnn_reshape_394,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_120 = ttnn.permute(
            ttnn_reshape_392,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_84 = ttnn.pad(
            ttnn_permute_118,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_85 = ttnn.pad(
            ttnn_permute_119,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_86 = ttnn.pad(
            ttnn_permute_120,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_28 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_85,
                ttnn_pad_84,
                ttnn_pad_86,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_115 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_28,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_121 = ttnn.permute(
            ttnn_slice_115,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_395 = ttnn.reshape(
            ttnn_permute_121,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_114 = ttnn.matmul(
            ttnn_reshape_395,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_170 = ttnn.add(
            ttnn_matmul_114,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_170

    def CLIPEncoderLayer_116_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_171 = ttnn.add(
            input_0,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_59 = ttnn.layer_norm(
            ttnn_add_171,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_59, ttnn_add_171

    def CLIPMLP_117_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_396 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_115 = ttnn.matmul(
            ttnn_reshape_396,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_172 = ttnn.add(
            ttnn_matmul_115,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_28 = ttnn.gelu(
            ttnn_add_172,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_397 = ttnn.reshape(
            ttnn_gelu_28,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_116 = ttnn.matmul(
            ttnn_reshape_397,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_173 = ttnn.add(
            ttnn_matmul_116,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_173

    def CLIPEncoderLayer_118_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_174 = ttnn.add(
            input_3,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_60 = ttnn.layer_norm(
            ttnn_add_174,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_174, ttnn_layer_norm_60

    def CLIPAttention_119_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_398 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_117 = ttnn.matmul(
            ttnn_reshape_398,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_175 = ttnn.add(
            ttnn_matmul_117,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_116 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_117 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_118 = ttnn.slice(
            ttnn_add_175,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_399 = ttnn.reshape(
            ttnn_slice_116,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_400 = ttnn.reshape(
            ttnn_slice_117,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_401 = ttnn.reshape(
            ttnn_slice_118,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_122 = ttnn.permute(
            ttnn_reshape_400,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_123 = ttnn.permute(
            ttnn_reshape_401,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_124 = ttnn.permute(
            ttnn_reshape_399,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_87 = ttnn.pad(
            ttnn_permute_122,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_88 = ttnn.pad(
            ttnn_permute_123,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_89 = ttnn.pad(
            ttnn_permute_124,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_29 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_88,
                ttnn_pad_87,
                ttnn_pad_89,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_119 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_29,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_125 = ttnn.permute(
            ttnn_slice_119,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_402 = ttnn.reshape(
            ttnn_permute_125,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_118 = ttnn.matmul(
            ttnn_reshape_402,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_176 = ttnn.add(
            ttnn_matmul_118,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_176

    def CLIPEncoderLayer_120_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_177 = ttnn.add(
            input_0,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_61 = ttnn.layer_norm(
            ttnn_add_177,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_1,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_177, ttnn_layer_norm_61

    def CLIPMLP_121_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_403 = ttnn.reshape(
            input_3,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_119 = ttnn.matmul(
            ttnn_reshape_403,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_178 = ttnn.add(
            ttnn_matmul_119,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_29 = ttnn.gelu(
            ttnn_add_178,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_404 = ttnn.reshape(
            ttnn_gelu_29,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_120 = ttnn.matmul(
            ttnn_reshape_404,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_179 = ttnn.add(
            ttnn_matmul_120,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_179

    def CLIPEncoderLayer_122_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_180 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_62 = ttnn.layer_norm(
            ttnn_add_180,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_62, ttnn_add_180

    def CLIPAttention_123_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_405 = ttnn.reshape(
            input_2,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_121 = ttnn.matmul(
            ttnn_reshape_405,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_181 = ttnn.add(
            ttnn_matmul_121,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_120 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_121 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_122 = ttnn.slice(
            ttnn_add_181,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_406 = ttnn.reshape(
            ttnn_slice_120,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_407 = ttnn.reshape(
            ttnn_slice_121,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_408 = ttnn.reshape(
            ttnn_slice_122,
            [1, 257, 16, 80],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_126 = ttnn.permute(
            ttnn_reshape_407,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_127 = ttnn.permute(
            ttnn_reshape_408,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_128 = ttnn.permute(
            ttnn_reshape_406,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_pad_90 = ttnn.pad(
            ttnn_permute_126,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_91 = ttnn.pad(
            ttnn_permute_127,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pad_92 = ttnn.pad(
            ttnn_permute_128,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_30 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_pad_91,
                ttnn_pad_90,
                ttnn_pad_92,
                attn_mask=None,
                is_causal=False,
                scale=0.11180340498685837,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_slice_123 = ttnn.slice(
            ttnn_transformer_scaled_dot_product_attention_30,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_129 = ttnn.permute(
            ttnn_slice_123,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_409 = ttnn.reshape(
            ttnn_permute_129,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_122 = ttnn.matmul(
            ttnn_reshape_409,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_182 = ttnn.add(
            ttnn_matmul_122,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_182

    def CLIPEncoderLayer_124_0(self, input_0, input_1, input_2, input_3):
        ttnn_add_183 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_63 = ttnn.layer_norm(
            ttnn_add_183,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_183, ttnn_layer_norm_63

    def CLIPMLP_125_0(self, input_0, input_1, input_2, input_3, input_4):
        ttnn_reshape_410 = ttnn.reshape(
            input_0,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_123 = ttnn.matmul(
            ttnn_reshape_410,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_184 = ttnn.add(
            ttnn_matmul_123,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_gelu_30 = ttnn.gelu(
            ttnn_add_184,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_411 = ttnn.reshape(
            ttnn_gelu_30,
            [257, 5120],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_124 = ttnn.matmul(
            ttnn_reshape_411,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_185 = ttnn.add(
            ttnn_matmul_124,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_185

    def CLIPEncoderLayer_126_0(self, input_0, input_1):
        ttnn_add_186 = ttnn.add(
            input_0,
            input_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_412 = ttnn.reshape(
            ttnn_add_186,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_412

    def Linear_127_0(self, input_0, input_1, input_2):
        ttnn_matmul_125 = ttnn.matmul(
            input_2,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_187 = ttnn.add(
            ttnn_matmul_125,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_187

    def IPAdapterPlusImageProjectionBlock_128_0(
        self,
        input_0,
        input_1,
        input_2,
        input_3,
        input_4,
        input_5,
        input_6,
        input_7,
        input_8,
        input_9,
    ):
        ttnn_layer_norm_64 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_4,
            bias=input_7,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_65 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_9,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_66 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_5,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_layer_norm_67 = ttnn.layer_norm(
            input_3,
            epsilon=9.9999997473787516e-06,
            weight=input_6,
            bias=input_8,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_413 = ttnn.reshape(
            ttnn_layer_norm_64,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_414 = ttnn.reshape(
            ttnn_layer_norm_65,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_415 = ttnn.reshape(
            ttnn_layer_norm_66,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_416 = ttnn.reshape(
            ttnn_layer_norm_67,
            [257, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_387 = [ttnn_reshape_416, input_0]
        ttnn_concat_63 = ttnn.concat(
            util_create_list_387,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_414, ttnn_concat_63, ttnn_reshape_415, ttnn_reshape_413

    def Attention_129_0(self, input_0, input_1, input_2, input_3, input_4, input_5):
        ttnn_matmul_126 = ttnn.matmul(
            input_0,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_127 = ttnn.matmul(
            input_0,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_3 = ttnn.typecast(
            ttnn_matmul_126,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_4 = ttnn.typecast(
            ttnn_matmul_127,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_417 = ttnn.reshape(
            ttnn_typecast_3,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_418 = ttnn.reshape(
            ttnn_typecast_4,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_130 = ttnn.permute(
            ttnn_reshape_417,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_131 = ttnn.permute(
            ttnn_reshape_418,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_132 = ttnn.permute(
            ttnn_permute_130,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_5 = ttnn.typecast(
            ttnn_permute_131,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_permute_132,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_31 = (
            ttnn.transformer.scaled_dot_product_attention(
                input_5,
                ttnn_typecast_6,
                ttnn_typecast_5,
                attn_mask=None,
                is_causal=False,
                scale=0.35355338454246521,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_31,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_419 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_128 = ttnn.matmul(
            ttnn_reshape_419,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_420 = ttnn.reshape(
            ttnn_matmul_128,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_0 = ttnn.divide(
            ttnn_reshape_420,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_0

    def IPAdapterPlusImageProjectionBlock_130_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_188 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_68 = ttnn.layer_norm(
            ttnn_add_188,
            epsilon=9.9999997473787516e-06,
            weight=input_0,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_188, ttnn_layer_norm_68

    def Linear_131_0(self, input_0, input_1):
        ttnn_reshape_421 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_129 = ttnn.matmul(
            ttnn_reshape_421,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_129

    def Linear_132_0(self, input_0, input_1):
        ttnn_matmul_130 = ttnn.matmul(
            input_1,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_422 = ttnn.reshape(
            ttnn_matmul_130,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_422

    def IPAdapterPlusImageProjectionBlock_133_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_189 = ttnn.add(
            input_1,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_69 = ttnn.layer_norm(
            ttnn_add_189,
            epsilon=9.9999997473787516e-06,
            weight=input_2,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_423 = ttnn.reshape(
            ttnn_layer_norm_69,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_388 = [input_4, ttnn_reshape_423]
        ttnn_concat_64 = ttnn.concat(
            util_create_list_388,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_189, ttnn_concat_64, ttnn_layer_norm_69

    def Attention_134_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_424 = ttnn.reshape(
            input_6,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_131 = ttnn.matmul(
            ttnn_reshape_424,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_132 = ttnn.matmul(
            input_5,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_7 = ttnn.typecast(
            ttnn_matmul_131,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_133 = ttnn.matmul(
            input_5,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_8 = ttnn.typecast(
            ttnn_matmul_132,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_425 = ttnn.reshape(
            ttnn_typecast_7,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_9 = ttnn.typecast(
            ttnn_matmul_133,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_426 = ttnn.reshape(
            ttnn_typecast_8,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_427 = ttnn.reshape(
            ttnn_typecast_9,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_133 = ttnn.permute(
            ttnn_reshape_425,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_134 = ttnn.permute(
            ttnn_reshape_426,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_135 = ttnn.permute(
            ttnn_reshape_427,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_10 = ttnn.typecast(
            ttnn_permute_133,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_11 = ttnn.typecast(
            ttnn_permute_134,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_136 = ttnn.permute(
            ttnn_permute_135,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_12 = ttnn.typecast(
            ttnn_permute_136,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_32 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_10,
                ttnn_typecast_12,
                ttnn_typecast_11,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_428 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_134 = ttnn.matmul(
            ttnn_reshape_428,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_429 = ttnn.reshape(
            ttnn_matmul_134,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_1 = ttnn.divide(
            ttnn_reshape_429,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_1

    def IPAdapterPlusImageProjectionBlock_135_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_190 = ttnn.add(
            input_2,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_70 = ttnn.layer_norm(
            ttnn_add_190,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_3,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_190, ttnn_layer_norm_70

    def Linear_136_0(self, input_0, input_1):
        ttnn_reshape_430 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_135 = ttnn.matmul(
            ttnn_reshape_430,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_135

    def Linear_137_0(self, input_0, input_1):
        ttnn_matmul_136 = ttnn.matmul(
            input_0,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_431 = ttnn.reshape(
            ttnn_matmul_136,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_431

    def IPAdapterPlusImageProjectionBlock_138_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_191 = ttnn.add(
            input_3,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_71 = ttnn.layer_norm(
            ttnn_add_191,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_4,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_432 = ttnn.reshape(
            ttnn_layer_norm_71,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_389 = [input_0, ttnn_reshape_432]
        ttnn_concat_65 = ttnn.concat(
            util_create_list_389,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_layer_norm_71, ttnn_add_191, ttnn_concat_65

    def Attention_139_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_433 = ttnn.reshape(
            input_5,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_137 = ttnn.matmul(
            ttnn_reshape_433,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_138 = ttnn.matmul(
            input_6,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_13 = ttnn.typecast(
            ttnn_matmul_137,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_139 = ttnn.matmul(
            input_6,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_14 = ttnn.typecast(
            ttnn_matmul_138,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_434 = ttnn.reshape(
            ttnn_typecast_13,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_15 = ttnn.typecast(
            ttnn_matmul_139,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_435 = ttnn.reshape(
            ttnn_typecast_14,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_436 = ttnn.reshape(
            ttnn_typecast_15,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_137 = ttnn.permute(
            ttnn_reshape_434,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_138 = ttnn.permute(
            ttnn_reshape_435,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_139 = ttnn.permute(
            ttnn_reshape_436,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_16 = ttnn.typecast(
            ttnn_permute_137,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_17 = ttnn.typecast(
            ttnn_permute_138,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_140 = ttnn.permute(
            ttnn_permute_139,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_18 = ttnn.typecast(
            ttnn_permute_140,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_33 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_16,
                ttnn_typecast_18,
                ttnn_typecast_17,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_33,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_437 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_2,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_140 = ttnn.matmul(
            ttnn_reshape_437,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_438 = ttnn.reshape(
            ttnn_matmul_140,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_2 = ttnn.divide(
            ttnn_reshape_438,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_2

    def IPAdapterPlusImageProjectionBlock_140_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_192 = ttnn.add(
            input_1,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_72 = ttnn.layer_norm(
            ttnn_add_192,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_add_192, ttnn_layer_norm_72

    def Linear_141_0(self, input_0, input_1):
        ttnn_reshape_439 = ttnn.reshape(
            input_1,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_141 = ttnn.matmul(
            ttnn_reshape_439,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_141

    def Linear_142_0(self, input_0, input_1):
        ttnn_matmul_142 = ttnn.matmul(
            input_0,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_440 = ttnn.reshape(
            ttnn_matmul_142,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_440

    def IPAdapterPlusImageProjectionBlock_143_0(
        self, input_0, input_1, input_2, input_3, input_4
    ):
        ttnn_add_193 = ttnn.add(
            input_0,
            input_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_73 = ttnn.layer_norm(
            ttnn_add_193,
            epsilon=9.9999997473787516e-06,
            weight=input_3,
            bias=input_2,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_441 = ttnn.reshape(
            ttnn_layer_norm_73,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        util_create_list_390 = [input_1, ttnn_reshape_441]
        ttnn_concat_66 = ttnn.concat(
            util_create_list_390,
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_layer_norm_73, ttnn_concat_66, ttnn_add_193

    def Attention_144_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5, input_6
    ):
        ttnn_reshape_442 = ttnn.reshape(
            input_5,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_143 = ttnn.matmul(
            ttnn_reshape_442,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_matmul_144 = ttnn.matmul(
            input_6,
            input_3,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_19 = ttnn.typecast(
            ttnn_matmul_143,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_145 = ttnn.matmul(
            input_6,
            input_0,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_typecast_20 = ttnn.typecast(
            ttnn_matmul_144,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_443 = ttnn.reshape(
            ttnn_typecast_19,
            [1, 16, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_21 = ttnn.typecast(
            ttnn_matmul_145,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_444 = ttnn.reshape(
            ttnn_typecast_20,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_445 = ttnn.reshape(
            ttnn_typecast_21,
            [1, 273, 20, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_141 = ttnn.permute(
            ttnn_reshape_443,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_142 = ttnn.permute(
            ttnn_reshape_444,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_permute_143 = ttnn.permute(
            ttnn_reshape_445,
            [0, 2, 3, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_22 = ttnn.typecast(
            ttnn_permute_141,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_23 = ttnn.typecast(
            ttnn_permute_142,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_144 = ttnn.permute(
            ttnn_permute_143,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_typecast_24 = ttnn.typecast(
            ttnn_permute_144,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_34 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_22,
                ttnn_typecast_24,
                ttnn_typecast_23,
                attn_mask=None,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_34,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_446 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_3,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_146 = ttnn.matmul(
            ttnn_reshape_446,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_reshape_447 = ttnn.reshape(
            ttnn_matmul_146,
            [1, 16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_divide_3 = ttnn.divide(
            ttnn_reshape_447,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_divide_3

    def IPAdapterPlusImageProjectionBlock_145_0(
        self, input_0, input_1, input_2, input_3
    ):
        ttnn_add_194 = ttnn.add(
            input_2,
            input_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_74 = ttnn.layer_norm(
            ttnn_add_194,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_0,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_74, ttnn_add_194

    def Linear_146_0(self, input):
        ttnn_reshape_448 = ttnn.reshape(
            input,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_reshape_448

    def Linear_147_0(self, input_0, input_1):
        ttnn_reshape_449 = ttnn.reshape(
            input_0,
            [16, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_147 = ttnn.matmul(
            ttnn_reshape_449,
            input_1,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation="gelu",
        )
        return ttnn_matmul_147, ttnn_reshape_449

    def IPAdapterPlusImageProjectionBlock_148_0(self, input_0, input_1, input_2):
        ttnn_matmul_148 = ttnn.matmul(
            input_1,
            input_2,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_195 = ttnn.add(
            ttnn_matmul_148,
            input_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return ttnn_add_195

    def IPAdapterPlusImageProjection_149_0(
        self, input_0, input_1, input_2, input_3, input_4, input_5
    ):
        ttnn_matmul_149 = ttnn.matmul(
            input_3,
            input_4,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=None,
            program_config=None,
            activation=None,
        )
        ttnn_add_196 = ttnn.add(
            ttnn_matmul_149,
            input_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_layer_norm_75 = ttnn.layer_norm(
            ttnn_add_196,
            epsilon=9.9999997473787516e-06,
            weight=input_1,
            bias=input_5,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        return ttnn_layer_norm_75

    def Linear_150_0(self, input):
        return

    def IPAdapterPlusImageProjectionBlock_151_0(self, input):
        return
