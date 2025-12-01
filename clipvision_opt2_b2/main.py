import ttnn
import utils
import time
import torch
import numpy as np
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from transformers.image_utils import load_image

from consteval import *



def _main(input):
    global CACHED_main_const_eval_132
    global CACHED_main_const_eval_131
    global CACHED_main_const_eval_130
    global CACHED_main_const_eval_129
    global CACHED_main_const_eval_128
    global CACHED_main_const_eval_127
    global CACHED_main_const_eval_126
    global CACHED_main_const_eval_125
    global CACHED_main_const_eval_124
    global CACHED_main_const_eval_123
    global CACHED_main_const_eval_122
    global CACHED_main_const_eval_121
    global CACHED_main_const_eval_120
    global CACHED_main_const_eval_119
    global CACHED_main_const_eval_118
    global CACHED_main_const_eval_117
    global CACHED_main_const_eval_116
    global CACHED_main_const_eval_115
    global CACHED_main_const_eval_114
    global CACHED_main_const_eval_113
    global CACHED_main_const_eval_112
    global CACHED_main_const_eval_111
    global CACHED_main_const_eval_110
    global CACHED_main_const_eval_109
    global CACHED_main_const_eval_108
    global CACHED_main_const_eval_107
    global CACHED_main_const_eval_106
    global CACHED_main_const_eval_105
    global CACHED_main_const_eval_104
    global CACHED_main_const_eval_103
    global CACHED_main_const_eval_102
    global CACHED_main_const_eval_101
    global CACHED_main_const_eval_100
    global CACHED_main_const_eval_99
    global CACHED_main_const_eval_98
    global CACHED_main_const_eval_97
    global CACHED_main_const_eval_96
    global CACHED_main_const_eval_95
    global CACHED_main_const_eval_94
    global CACHED_main_const_eval_93
    global CACHED_main_const_eval_92
    global CACHED_main_const_eval_91
    global CACHED_main_const_eval_90
    global CACHED_main_const_eval_89
    global CACHED_main_const_eval_88
    global CACHED_main_const_eval_87
    global CACHED_main_const_eval_86
    global CACHED_main_const_eval_85
    global CACHED_main_const_eval_84
    global CACHED_main_const_eval_83
    global CACHED_main_const_eval_82
    global CACHED_main_const_eval_81
    global CACHED_main_const_eval_80
    global CACHED_main_const_eval_79
    global CACHED_main_const_eval_78
    global CACHED_main_const_eval_77
    global CACHED_main_const_eval_76
    global CACHED_main_const_eval_75
    global CACHED_main_const_eval_74
    global CACHED_main_const_eval_73
    global CACHED_main_const_eval_72
    global CACHED_main_const_eval_71
    global CACHED_main_const_eval_70
    global CACHED_main_const_eval_69
    global CACHED_main_const_eval_68
    global CACHED_main_const_eval_67
    global CACHED_main_const_eval_66
    global CACHED_main_const_eval_65
    global CACHED_main_const_eval_64
    global CACHED_main_const_eval_63
    global CACHED_main_const_eval_62
    global CACHED_main_const_eval_61
    global CACHED_main_const_eval_60
    global CACHED_main_const_eval_59
    global CACHED_main_const_eval_58
    global CACHED_main_const_eval_57
    global CACHED_main_const_eval_56
    global CACHED_main_const_eval_55
    global CACHED_main_const_eval_54
    global CACHED_main_const_eval_53
    global CACHED_main_const_eval_52
    global CACHED_main_const_eval_51
    global CACHED_main_const_eval_50
    global CACHED_main_const_eval_49
    global CACHED_main_const_eval_48
    global CACHED_main_const_eval_47
    global CACHED_main_const_eval_46
    global CACHED_main_const_eval_45
    global CACHED_main_const_eval_44
    global CACHED_main_const_eval_43
    global CACHED_main_const_eval_42
    global CACHED_main_const_eval_41
    global CACHED_main_const_eval_40
    global CACHED_main_const_eval_39
    global CACHED_main_const_eval_38
    global CACHED_main_const_eval_37
    global CACHED_main_const_eval_36
    global CACHED_main_const_eval_35
    global CACHED_main_const_eval_34
    global CACHED_main_const_eval_33
    global CACHED_main_const_eval_32
    global CACHED_main_const_eval_31
    global CACHED_main_const_eval_30
    global CACHED_main_const_eval_29
    global CACHED_main_const_eval_28
    global CACHED_main_const_eval_27
    global CACHED_main_const_eval_26
    global CACHED_main_const_eval_25
    global CACHED_main_const_eval_24
    global CACHED_main_const_eval_23
    global CACHED_main_const_eval_22
    global CACHED_main_const_eval_21
    global CACHED_main_const_eval_20
    global CACHED_main_const_eval_19
    global CACHED_main_const_eval_18
    global CACHED_main_const_eval_17
    global CACHED_main_const_eval_16
    global CACHED_main_const_eval_15
    global CACHED_main_const_eval_14
    global CACHED_main_const_eval_13
    global CACHED_main_const_eval_12
    global CACHED_main_const_eval_11
    global CACHED_main_const_eval_10
    global CACHED_main_const_eval_9
    global CACHED_main_const_eval_8
    global CACHED_main_const_eval_7
    global CACHED_main_const_eval_6
    global CACHED_main_const_eval_5
    global CACHED_main_const_eval_4
    global CACHED_main_const_eval_3
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    input_6 = input[6]
    input_7 = input[7]
    input_8 = input[8]
    input_9 = input[9]
    input_10 = input[10]
    input_11 = input[11]
    input_12 = input[12]
    input_13 = input[13]
    input_14 = input[14]
    input_15 = input[15]
    input_16 = input[16]
    input_17 = input[17]
    input_18 = input[18]
    input_19 = input[19]
    input_20 = input[20]
    input_21 = input[21]
    input_22 = input[22]
    input_23 = input[23]
    input_24 = input[24]
    input_25 = input[25]
    input_26 = input[26]
    input_27 = input[27]
    input_28 = input[28]
    input_29 = input[29]
    input_30 = input[30]
    input_31 = input[31]
    input_32 = input[32]
    input_33 = input[33]
    input_34 = input[34]
    input_35 = input[35]
    input_36 = input[36]
    input_37 = input[37]
    input_38 = input[38]
    input_39 = input[39]
    input_40 = input[40]
    input_41 = input[41]
    input_42 = input[42]
    input_43 = input[43]
    input_44 = input[44]
    input_45 = input[45]
    input_46 = input[46]
    input_47 = input[47]
    input_48 = input[48]
    input_49 = input[49]
    input_50 = input[50]
    input_51 = input[51]
    input_52 = input[52]
    input_53 = input[53]
    input_54 = input[54]
    input_55 = input[55]
    input_56 = input[56]
    input_57 = input[57]
    input_58 = input[58]
    input_59 = input[59]
    input_60 = input[60]
    input_61 = input[61]
    input_62 = input[62]
    input_63 = input[63]
    input_64 = input[64]
    input_65 = input[65]
    input_66 = input[66]
    input_67 = input[67]
    input_68 = input[68]
    input_69 = input[69]
    input_70 = input[70]
    input_71 = input[71]
    input_72 = input[72]
    input_73 = input[73]
    input_74 = input[74]
    input_75 = input[75]
    input_76 = input[76]
    input_77 = input[77]
    input_78 = input[78]
    input_79 = input[79]
    input_80 = input[80]
    input_81 = input[81]
    input_82 = input[82]
    input_83 = input[83]
    input_84 = input[84]
    input_85 = input[85]
    input_86 = input[86]
    input_87 = input[87]
    input_88 = input[88]
    input_89 = input[89]
    input_90 = input[90]
    input_91 = input[91]
    input_92 = input[92]
    input_93 = input[93]
    input_94 = input[94]
    input_95 = input[95]
    input_96 = input[96]
    input_97 = input[97]
    input_98 = input[98]
    input_99 = input[99]
    input_100 = input[100]
    input_101 = input[101]
    input_102 = input[102]
    input_103 = input[103]
    input_104 = input[104]
    input_105 = input[105]
    input_106 = input[106]
    input_107 = input[107]
    input_108 = input[108]
    input_109 = input[109]
    input_110 = input[110]
    input_111 = input[111]
    input_112 = input[112]
    input_113 = input[113]
    input_114 = input[114]
    input_115 = input[115]
    input_116 = input[116]
    input_117 = input[117]
    input_118 = input[118]
    input_119 = input[119]
    input_120 = input[120]
    input_121 = input[121]
    input_122 = input[122]
    input_123 = input[123]
    input_124 = input[124]
    input_125 = input[125]
    input_126 = input[126]
    input_127 = input[127]
    input_128 = input[128]
    input_129 = input[129]
    input_130 = input[130]
    input_131 = input[131]
    input_132 = input[132]
    input_133 = input[133]
    input_134 = input[134]
    input_135 = input[135]
    input_136 = input[136]
    input_137 = input[137]
    input_138 = input[138]
    input_139 = input[139]
    input_140 = input[140]
    input_141 = input[141]
    input_142 = input[142]
    input_143 = input[143]
    input_144 = input[144]
    input_145 = input[145]
    input_146 = input[146]
    input_147 = input[147]
    input_148 = input[148]
    input_149 = input[149]
    input_150 = input[150]
    input_151 = input[151]
    input_152 = input[152]
    input_153 = input[153]
    input_154 = input[154]
    input_155 = input[155]
    input_156 = input[156]
    input_157 = input[157]
    input_158 = input[158]
    input_159 = input[159]
    input_160 = input[160]
    input_161 = input[161]
    input_162 = input[162]
    input_163 = input[163]
    input_164 = input[164]
    input_165 = input[165]
    input_166 = input[166]
    input_167 = input[167]
    input_168 = input[168]
    input_169 = input[169]
    input_170 = input[170]
    input_171 = input[171]
    input_172 = input[172]
    input_173 = input[173]
    input_174 = input[174]
    input_175 = input[175]
    input_176 = input[176]
    input_177 = input[177]
    input_178 = input[178]
    input_179 = input[179]
    input_180 = input[180]
    input_181 = input[181]
    input_182 = input[182]
    input_183 = input[183]
    input_184 = input[184]
    input_185 = input[185]
    input_186 = input[186]
    input_187 = input[187]
    input_188 = input[188]
    input_189 = input[189]
    input_190 = input[190]
    input_191 = input[191]
    input_192 = input[192]
    input_193 = input[193]
    input_194 = input[194]
    input_195 = input[195]
    input_196 = input[196]
    input_197 = input[197]
    input_198 = input[198]
    input_199 = input[199]
    input_200 = input[200]
    input_201 = input[201]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, CACHED_main_const_eval_0
    )
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    util_create_list_133 = [input_178]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_1, util_create_list_133, CACHED_main_const_eval_1
    )
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_2
    util_create_list_134 = [input_59]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_2, util_create_list_134, CACHED_main_const_eval_2
    )
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_3 = main_const_eval_3
    util_create_list_135 = [input_145]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_3, util_create_list_135, CACHED_main_const_eval_3
    )
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_4 = main_const_eval_4
    util_create_list_136 = [input_137]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_4, util_create_list_136, CACHED_main_const_eval_4
    )
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_5 = main_const_eval_5
    util_create_list_137 = [input_20]
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_5, util_create_list_137, CACHED_main_const_eval_5
    )
    CACHED_main_const_eval_5 = utils_constEvalFuncWrapper_4
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_6 = main_const_eval_6
    util_create_list_138 = [input_127]
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(
        const_6, util_create_list_138, CACHED_main_const_eval_6
    )
    CACHED_main_const_eval_6 = utils_constEvalFuncWrapper_5
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_7 = main_const_eval_7
    util_create_list_139 = [input_31]
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(
        const_7, util_create_list_139, CACHED_main_const_eval_7
    )
    CACHED_main_const_eval_7 = utils_constEvalFuncWrapper_6
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_8 = main_const_eval_8
    util_create_list_140 = [input_116]
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(
        const_8, util_create_list_140, CACHED_main_const_eval_8
    )
    CACHED_main_const_eval_8 = utils_constEvalFuncWrapper_7
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_9 = main_const_eval_9
    util_create_list_141 = [input_123]
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(
        const_9, util_create_list_141, CACHED_main_const_eval_9
    )
    CACHED_main_const_eval_9 = utils_constEvalFuncWrapper_8
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_10 = main_const_eval_10
    util_create_list_142 = [input_153]
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(
        const_10, util_create_list_142, CACHED_main_const_eval_10
    )
    CACHED_main_const_eval_10 = utils_constEvalFuncWrapper_9
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_11 = main_const_eval_11
    util_create_list_143 = [input_186]
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_11, util_create_list_143, CACHED_main_const_eval_11
    )
    CACHED_main_const_eval_11 = utils_constEvalFuncWrapper_10
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_12 = main_const_eval_12
    util_create_list_144 = [input_190]
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_12, util_create_list_144, CACHED_main_const_eval_12
    )
    CACHED_main_const_eval_12 = utils_constEvalFuncWrapper_11
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_13 = main_const_eval_13
    util_create_list_145 = [input_188]
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_13, util_create_list_145, CACHED_main_const_eval_13
    )
    CACHED_main_const_eval_13 = utils_constEvalFuncWrapper_12
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_14 = main_const_eval_14
    util_create_list_146 = [input_3]
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_14, util_create_list_146, CACHED_main_const_eval_14
    )
    CACHED_main_const_eval_14 = utils_constEvalFuncWrapper_13
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_15 = main_const_eval_15
    util_create_list_147 = [input_125]
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_15, util_create_list_147, CACHED_main_const_eval_15
    )
    CACHED_main_const_eval_15 = utils_constEvalFuncWrapper_14
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_16 = main_const_eval_16
    util_create_list_148 = [input_21]
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_16, util_create_list_148, CACHED_main_const_eval_16
    )
    CACHED_main_const_eval_16 = utils_constEvalFuncWrapper_15
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_17 = main_const_eval_17
    util_create_list_149 = [input_156]
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_17, util_create_list_149, CACHED_main_const_eval_17
    )
    CACHED_main_const_eval_17 = utils_constEvalFuncWrapper_16
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_18 = main_const_eval_18
    util_create_list_150 = [input_194]
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_18, util_create_list_150, CACHED_main_const_eval_18
    )
    CACHED_main_const_eval_18 = utils_constEvalFuncWrapper_17
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_19 = main_const_eval_19
    util_create_list_151 = [input_15]
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_19, util_create_list_151, CACHED_main_const_eval_19
    )
    CACHED_main_const_eval_19 = utils_constEvalFuncWrapper_18
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_20 = main_const_eval_20
    util_create_list_152 = [input_131]
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_20, util_create_list_152, CACHED_main_const_eval_20
    )
    CACHED_main_const_eval_20 = utils_constEvalFuncWrapper_19
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_21 = main_const_eval_21
    util_create_list_153 = [input_111]
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_21, util_create_list_153, CACHED_main_const_eval_21
    )
    CACHED_main_const_eval_21 = utils_constEvalFuncWrapper_20
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_22 = main_const_eval_22
    util_create_list_154 = [input_27]
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_22, util_create_list_154, CACHED_main_const_eval_22
    )
    CACHED_main_const_eval_22 = utils_constEvalFuncWrapper_21
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_23 = main_const_eval_23
    util_create_list_155 = [input_97]
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_23, util_create_list_155, CACHED_main_const_eval_23
    )
    CACHED_main_const_eval_23 = utils_constEvalFuncWrapper_22
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_24 = main_const_eval_24
    util_create_list_156 = [input_77]
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_24, util_create_list_156, CACHED_main_const_eval_24
    )
    CACHED_main_const_eval_24 = utils_constEvalFuncWrapper_23
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_25 = main_const_eval_25
    util_create_list_157 = [input_62]
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_25, util_create_list_157, CACHED_main_const_eval_25
    )
    CACHED_main_const_eval_25 = utils_constEvalFuncWrapper_24
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_26 = main_const_eval_26
    util_create_list_158 = [input_110]
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_26, util_create_list_158, CACHED_main_const_eval_26
    )
    CACHED_main_const_eval_26 = utils_constEvalFuncWrapper_25
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_27 = main_const_eval_27
    util_create_list_159 = [input_141]
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_27, util_create_list_159, CACHED_main_const_eval_27
    )
    CACHED_main_const_eval_27 = utils_constEvalFuncWrapper_26
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_28 = main_const_eval_28
    util_create_list_160 = [input_65]
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_28, util_create_list_160, CACHED_main_const_eval_28
    )
    CACHED_main_const_eval_28 = utils_constEvalFuncWrapper_27
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_29 = main_const_eval_29
    util_create_list_161 = [input_79]
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_29, util_create_list_161, CACHED_main_const_eval_29
    )
    CACHED_main_const_eval_29 = utils_constEvalFuncWrapper_28
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_30 = main_const_eval_30
    util_create_list_162 = [input_192]
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_30, util_create_list_162, CACHED_main_const_eval_30
    )
    CACHED_main_const_eval_30 = utils_constEvalFuncWrapper_29
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_31 = main_const_eval_31
    util_create_list_163 = [input_128]
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_31, util_create_list_163, CACHED_main_const_eval_31
    )
    CACHED_main_const_eval_31 = utils_constEvalFuncWrapper_30
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    const_32 = main_const_eval_32
    util_create_list_164 = [input_80]
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_32, util_create_list_164, CACHED_main_const_eval_32
    )
    CACHED_main_const_eval_32 = utils_constEvalFuncWrapper_31
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_33 = main_const_eval_33
    util_create_list_165 = [input_117]
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_33, util_create_list_165, CACHED_main_const_eval_33
    )
    CACHED_main_const_eval_33 = utils_constEvalFuncWrapper_32
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_34 = main_const_eval_34
    util_create_list_166 = [input_115]
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_34, util_create_list_166, CACHED_main_const_eval_34
    )
    CACHED_main_const_eval_34 = utils_constEvalFuncWrapper_33
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_35 = main_const_eval_35
    util_create_list_167 = [input_135]
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_35, util_create_list_167, CACHED_main_const_eval_35
    )
    CACHED_main_const_eval_35 = utils_constEvalFuncWrapper_34
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_36 = main_const_eval_36
    util_create_list_168 = [input_168]
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_36, util_create_list_168, CACHED_main_const_eval_36
    )
    CACHED_main_const_eval_36 = utils_constEvalFuncWrapper_35
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_37 = main_const_eval_37
    util_create_list_169 = [input_172]
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_37, util_create_list_169, CACHED_main_const_eval_37
    )
    CACHED_main_const_eval_37 = utils_constEvalFuncWrapper_36
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_38 = main_const_eval_38
    util_create_list_170 = [input_75]
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_38, util_create_list_170, CACHED_main_const_eval_38
    )
    CACHED_main_const_eval_38 = utils_constEvalFuncWrapper_37
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_39 = main_const_eval_39
    util_create_list_171 = [input_174]
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_39, util_create_list_171, CACHED_main_const_eval_39
    )
    CACHED_main_const_eval_39 = utils_constEvalFuncWrapper_38
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_40 = main_const_eval_40
    util_create_list_172 = [input_13]
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_40, util_create_list_172, CACHED_main_const_eval_40
    )
    CACHED_main_const_eval_40 = utils_constEvalFuncWrapper_39
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_41 = main_const_eval_41
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(
        const_41, CACHED_main_const_eval_41
    )
    CACHED_main_const_eval_41 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_42 = main_const_eval_42
    util_create_list_173 = [input_50]
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_42, util_create_list_173, CACHED_main_const_eval_42
    )
    CACHED_main_const_eval_42 = utils_constEvalFuncWrapper_40
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_43 = main_const_eval_43
    util_create_list_174 = [input_23]
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_43, util_create_list_174, CACHED_main_const_eval_43
    )
    CACHED_main_const_eval_43 = utils_constEvalFuncWrapper_41
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_44 = main_const_eval_44
    util_create_list_175 = [input_85]
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_44, util_create_list_175, CACHED_main_const_eval_44
    )
    CACHED_main_const_eval_44 = utils_constEvalFuncWrapper_42
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_45 = main_const_eval_45
    util_create_list_176 = [input_98]
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_45, util_create_list_176, CACHED_main_const_eval_45
    )
    CACHED_main_const_eval_45 = utils_constEvalFuncWrapper_43
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_46 = main_const_eval_46
    util_create_list_177 = [input_2]
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_46, util_create_list_177, CACHED_main_const_eval_46
    )
    CACHED_main_const_eval_46 = utils_constEvalFuncWrapper_44
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_47 = main_const_eval_47
    util_create_list_178 = [input_45]
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_47, util_create_list_178, CACHED_main_const_eval_47
    )
    CACHED_main_const_eval_47 = utils_constEvalFuncWrapper_45
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_48 = main_const_eval_48
    util_create_list_179 = [input_43]
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_48, util_create_list_179, CACHED_main_const_eval_48
    )
    CACHED_main_const_eval_48 = utils_constEvalFuncWrapper_46
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_49 = main_const_eval_49
    util_create_list_180 = [input_8]
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_49, util_create_list_180, CACHED_main_const_eval_49
    )
    CACHED_main_const_eval_49 = utils_constEvalFuncWrapper_47
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_50 = main_const_eval_50
    util_create_list_181 = [input_19]
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_50, util_create_list_181, CACHED_main_const_eval_50
    )
    CACHED_main_const_eval_50 = utils_constEvalFuncWrapper_48
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_51 = main_const_eval_51
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(
        const_51, CACHED_main_const_eval_51
    )
    CACHED_main_const_eval_51 = utils_constEvalFuncWrapperZeroArg_2
    utils_constEvalFuncWrapperZeroArg_2_0 = utils_constEvalFuncWrapperZeroArg_2[0]
    const_52 = main_const_eval_52
    util_create_list_182 = [input_83]
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_52, util_create_list_182, CACHED_main_const_eval_52
    )
    CACHED_main_const_eval_52 = utils_constEvalFuncWrapper_49
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_53 = main_const_eval_53
    util_create_list_183 = [input_1]
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_53, util_create_list_183, CACHED_main_const_eval_53
    )
    CACHED_main_const_eval_53 = utils_constEvalFuncWrapper_50
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_54 = main_const_eval_54
    util_create_list_184 = [input_17]
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_54, util_create_list_184, CACHED_main_const_eval_54
    )
    CACHED_main_const_eval_54 = utils_constEvalFuncWrapper_51
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_55 = main_const_eval_55
    util_create_list_185 = [input_14]
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_55, util_create_list_185, CACHED_main_const_eval_55
    )
    CACHED_main_const_eval_55 = utils_constEvalFuncWrapper_52
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_56 = main_const_eval_56
    util_create_list_186 = [input_198]
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_56, util_create_list_186, CACHED_main_const_eval_56
    )
    CACHED_main_const_eval_56 = utils_constEvalFuncWrapper_53
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_57 = main_const_eval_57
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(
        const_57, CACHED_main_const_eval_57
    )
    CACHED_main_const_eval_57 = utils_constEvalFuncWrapperZeroArg_3
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    utils_constEvalFuncWrapperZeroArg_3_1 = utils_constEvalFuncWrapperZeroArg_3[1]
    utils_constEvalFuncWrapperZeroArg_3_2 = utils_constEvalFuncWrapperZeroArg_3[2]
    utils_constEvalFuncWrapperZeroArg_3_3 = utils_constEvalFuncWrapperZeroArg_3[3]
    utils_constEvalFuncWrapperZeroArg_3_4 = utils_constEvalFuncWrapperZeroArg_3[4]
    utils_constEvalFuncWrapperZeroArg_3_5 = utils_constEvalFuncWrapperZeroArg_3[5]
    utils_constEvalFuncWrapperZeroArg_3_6 = utils_constEvalFuncWrapperZeroArg_3[6]
    utils_constEvalFuncWrapperZeroArg_3_7 = utils_constEvalFuncWrapperZeroArg_3[7]
    utils_constEvalFuncWrapperZeroArg_3_8 = utils_constEvalFuncWrapperZeroArg_3[8]
    utils_constEvalFuncWrapperZeroArg_3_9 = utils_constEvalFuncWrapperZeroArg_3[9]
    utils_constEvalFuncWrapperZeroArg_3_10 = utils_constEvalFuncWrapperZeroArg_3[10]
    utils_constEvalFuncWrapperZeroArg_3_11 = utils_constEvalFuncWrapperZeroArg_3[11]
    utils_constEvalFuncWrapperZeroArg_3_12 = utils_constEvalFuncWrapperZeroArg_3[12]
    utils_constEvalFuncWrapperZeroArg_3_13 = utils_constEvalFuncWrapperZeroArg_3[13]
    utils_constEvalFuncWrapperZeroArg_3_14 = utils_constEvalFuncWrapperZeroArg_3[14]
    utils_constEvalFuncWrapperZeroArg_3_15 = utils_constEvalFuncWrapperZeroArg_3[15]
    utils_constEvalFuncWrapperZeroArg_3_16 = utils_constEvalFuncWrapperZeroArg_3[16]
    utils_constEvalFuncWrapperZeroArg_3_17 = utils_constEvalFuncWrapperZeroArg_3[17]
    utils_constEvalFuncWrapperZeroArg_3_18 = utils_constEvalFuncWrapperZeroArg_3[18]
    utils_constEvalFuncWrapperZeroArg_3_19 = utils_constEvalFuncWrapperZeroArg_3[19]
    utils_constEvalFuncWrapperZeroArg_3_20 = utils_constEvalFuncWrapperZeroArg_3[20]
    utils_constEvalFuncWrapperZeroArg_3_21 = utils_constEvalFuncWrapperZeroArg_3[21]
    utils_constEvalFuncWrapperZeroArg_3_22 = utils_constEvalFuncWrapperZeroArg_3[22]
    utils_constEvalFuncWrapperZeroArg_3_23 = utils_constEvalFuncWrapperZeroArg_3[23]
    utils_constEvalFuncWrapperZeroArg_3_24 = utils_constEvalFuncWrapperZeroArg_3[24]
    const_58 = main_const_eval_58
    util_create_list_187 = [input_122]
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_58, util_create_list_187, CACHED_main_const_eval_58
    )
    CACHED_main_const_eval_58 = utils_constEvalFuncWrapper_54
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_59 = main_const_eval_59
    util_create_list_188 = [input_184]
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_59, util_create_list_188, CACHED_main_const_eval_59
    )
    CACHED_main_const_eval_59 = utils_constEvalFuncWrapper_55
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_60 = main_const_eval_60
    util_create_list_189 = [input_35]
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_60, util_create_list_189, CACHED_main_const_eval_60
    )
    CACHED_main_const_eval_60 = utils_constEvalFuncWrapper_56
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_61 = main_const_eval_61
    util_create_list_190 = [input_148]
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_61, util_create_list_190, CACHED_main_const_eval_61
    )
    CACHED_main_const_eval_61 = utils_constEvalFuncWrapper_57
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_62 = main_const_eval_62
    util_create_list_191 = [input_166]
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_62, util_create_list_191, CACHED_main_const_eval_62
    )
    CACHED_main_const_eval_62 = utils_constEvalFuncWrapper_58
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_63 = main_const_eval_63
    util_create_list_192 = [input_61]
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_63, util_create_list_192, CACHED_main_const_eval_63
    )
    CACHED_main_const_eval_63 = utils_constEvalFuncWrapper_59
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_64 = main_const_eval_64
    util_create_list_193 = [input_160]
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_64, util_create_list_193, CACHED_main_const_eval_64
    )
    CACHED_main_const_eval_64 = utils_constEvalFuncWrapper_60
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_65 = main_const_eval_65
    util_create_list_194 = [input_71]
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_65, util_create_list_194, CACHED_main_const_eval_65
    )
    CACHED_main_const_eval_65 = utils_constEvalFuncWrapper_61
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_66 = main_const_eval_66
    util_create_list_195 = [input_92]
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_66, util_create_list_195, CACHED_main_const_eval_66
    )
    CACHED_main_const_eval_66 = utils_constEvalFuncWrapper_62
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_67 = main_const_eval_67
    util_create_list_196 = [input_53]
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_67, util_create_list_196, CACHED_main_const_eval_67
    )
    CACHED_main_const_eval_67 = utils_constEvalFuncWrapper_63
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_68 = main_const_eval_68
    util_create_list_197 = [input_55]
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_68, util_create_list_197, CACHED_main_const_eval_68
    )
    CACHED_main_const_eval_68 = utils_constEvalFuncWrapper_64
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_69 = main_const_eval_69
    util_create_list_198 = [input_140]
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_69, util_create_list_198, CACHED_main_const_eval_69
    )
    CACHED_main_const_eval_69 = utils_constEvalFuncWrapper_65
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_70 = main_const_eval_70
    util_create_list_199 = [input_68]
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_70, util_create_list_199, CACHED_main_const_eval_70
    )
    CACHED_main_const_eval_70 = utils_constEvalFuncWrapper_66
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_71 = main_const_eval_71
    util_create_list_200 = [input_9]
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_71, util_create_list_200, CACHED_main_const_eval_71
    )
    CACHED_main_const_eval_71 = utils_constEvalFuncWrapper_67
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_72 = main_const_eval_72
    util_create_list_201 = [input_104]
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_72, util_create_list_201, CACHED_main_const_eval_72
    )
    CACHED_main_const_eval_72 = utils_constEvalFuncWrapper_68
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_73 = main_const_eval_73
    util_create_list_202 = [input_25]
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_73, util_create_list_202, CACHED_main_const_eval_73
    )
    CACHED_main_const_eval_73 = utils_constEvalFuncWrapper_69
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_74 = main_const_eval_74
    util_create_list_203 = [input_133]
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_74, util_create_list_203, CACHED_main_const_eval_74
    )
    CACHED_main_const_eval_74 = utils_constEvalFuncWrapper_70
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_75 = main_const_eval_75
    util_create_list_204 = [input_86]
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_75, util_create_list_204, CACHED_main_const_eval_75
    )
    CACHED_main_const_eval_75 = utils_constEvalFuncWrapper_71
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_76 = main_const_eval_76
    util_create_list_205 = [input_103]
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_76, util_create_list_205, CACHED_main_const_eval_76
    )
    CACHED_main_const_eval_76 = utils_constEvalFuncWrapper_72
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_77 = main_const_eval_77
    util_create_list_206 = [input_176]
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_77, util_create_list_206, CACHED_main_const_eval_77
    )
    CACHED_main_const_eval_77 = utils_constEvalFuncWrapper_73
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_78 = main_const_eval_78
    util_create_list_207 = [input_57]
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_78, util_create_list_207, CACHED_main_const_eval_78
    )
    CACHED_main_const_eval_78 = utils_constEvalFuncWrapper_74
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_79 = main_const_eval_79
    util_create_list_208 = [input_49]
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_79, util_create_list_208, CACHED_main_const_eval_79
    )
    CACHED_main_const_eval_79 = utils_constEvalFuncWrapper_75
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_80 = main_const_eval_80
    util_create_list_209 = [input_119]
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_80, util_create_list_209, CACHED_main_const_eval_80
    )
    CACHED_main_const_eval_80 = utils_constEvalFuncWrapper_76
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_81 = main_const_eval_81
    util_create_list_210 = [input_91]
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_81, util_create_list_210, CACHED_main_const_eval_81
    )
    CACHED_main_const_eval_81 = utils_constEvalFuncWrapper_77
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_82 = main_const_eval_82
    util_create_list_211 = [input_149, input_150]
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_82, util_create_list_211, CACHED_main_const_eval_82
    )
    CACHED_main_const_eval_82 = utils_constEvalFuncWrapper_78
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_83 = main_const_eval_83
    util_create_list_212 = [input_74]
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_83, util_create_list_212, CACHED_main_const_eval_83
    )
    CACHED_main_const_eval_83 = utils_constEvalFuncWrapper_79
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_84 = main_const_eval_84
    util_create_list_213 = [input_170]
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_84, util_create_list_213, CACHED_main_const_eval_84
    )
    CACHED_main_const_eval_84 = utils_constEvalFuncWrapper_80
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_85 = main_const_eval_85
    util_create_list_214 = [input_73]
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_85, util_create_list_214, CACHED_main_const_eval_85
    )
    CACHED_main_const_eval_85 = utils_constEvalFuncWrapper_81
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_86 = main_const_eval_86
    util_create_list_215 = [input_5]
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_86, util_create_list_215, CACHED_main_const_eval_86
    )
    CACHED_main_const_eval_86 = utils_constEvalFuncWrapper_82
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_87 = main_const_eval_87
    util_create_list_216 = [input_67]
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_87, util_create_list_216, CACHED_main_const_eval_87
    )
    CACHED_main_const_eval_87 = utils_constEvalFuncWrapper_83
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_88 = main_const_eval_88
    util_create_list_217 = [input_180]
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_88, util_create_list_217, CACHED_main_const_eval_88
    )
    CACHED_main_const_eval_88 = utils_constEvalFuncWrapper_84
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_89 = main_const_eval_89
    util_create_list_218 = [input_105]
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_89, util_create_list_218, CACHED_main_const_eval_89
    )
    CACHED_main_const_eval_89 = utils_constEvalFuncWrapper_85
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_90 = main_const_eval_90
    util_create_list_219 = [input_93]
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_90, util_create_list_219, CACHED_main_const_eval_90
    )
    CACHED_main_const_eval_90 = utils_constEvalFuncWrapper_86
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_91 = main_const_eval_91
    util_create_list_220 = [input_99]
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_91, util_create_list_220, CACHED_main_const_eval_91
    )
    CACHED_main_const_eval_91 = utils_constEvalFuncWrapper_87
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_92 = main_const_eval_92
    util_create_list_221 = [input_63]
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_92, util_create_list_221, CACHED_main_const_eval_92
    )
    CACHED_main_const_eval_92 = utils_constEvalFuncWrapper_88
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_93 = main_const_eval_93
    util_create_list_222 = [input_109]
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_93, util_create_list_222, CACHED_main_const_eval_93
    )
    CACHED_main_const_eval_93 = utils_constEvalFuncWrapper_89
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_94 = main_const_eval_94
    util_create_list_223 = [input_26]
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_94, util_create_list_223, CACHED_main_const_eval_94
    )
    CACHED_main_const_eval_94 = utils_constEvalFuncWrapper_90
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_95 = main_const_eval_95
    util_create_list_224 = [input_146]
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_95, util_create_list_224, CACHED_main_const_eval_95
    )
    CACHED_main_const_eval_95 = utils_constEvalFuncWrapper_91
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_96 = main_const_eval_96
    util_create_list_225 = [input_200]
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_96, util_create_list_225, CACHED_main_const_eval_96
    )
    CACHED_main_const_eval_96 = utils_constEvalFuncWrapper_92
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_97 = main_const_eval_97
    util_create_list_226 = [input_113]
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_97, util_create_list_226, CACHED_main_const_eval_97
    )
    CACHED_main_const_eval_97 = utils_constEvalFuncWrapper_93
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_98 = main_const_eval_98
    util_create_list_227 = [input_129]
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_98, util_create_list_227, CACHED_main_const_eval_98
    )
    CACHED_main_const_eval_98 = utils_constEvalFuncWrapper_94
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_99 = main_const_eval_99
    util_create_list_228 = [input_47]
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_99, util_create_list_228, CACHED_main_const_eval_99
    )
    CACHED_main_const_eval_99 = utils_constEvalFuncWrapper_95
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_100 = main_const_eval_100
    util_create_list_229 = [input_164]
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_100, util_create_list_229, CACHED_main_const_eval_100
    )
    CACHED_main_const_eval_100 = utils_constEvalFuncWrapper_96
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_101 = main_const_eval_101
    util_create_list_230 = [input_158]
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_101, util_create_list_230, CACHED_main_const_eval_101
    )
    CACHED_main_const_eval_101 = utils_constEvalFuncWrapper_97
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_102 = main_const_eval_102
    util_create_list_231 = [input_33]
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_102, util_create_list_231, CACHED_main_const_eval_102
    )
    CACHED_main_const_eval_102 = utils_constEvalFuncWrapper_98
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_103 = main_const_eval_103
    util_create_list_232 = [input_196]
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_103, util_create_list_232, CACHED_main_const_eval_103
    )
    CACHED_main_const_eval_103 = utils_constEvalFuncWrapper_99
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_104 = main_const_eval_104
    util_create_list_233 = [input_147]
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_104, util_create_list_233, CACHED_main_const_eval_104
    )
    CACHED_main_const_eval_104 = utils_constEvalFuncWrapper_100
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_105 = main_const_eval_105
    util_create_list_234 = [input_134]
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_105, util_create_list_234, CACHED_main_const_eval_105
    )
    CACHED_main_const_eval_105 = utils_constEvalFuncWrapper_101
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_106 = main_const_eval_106
    util_create_list_235 = [input_44]
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_106, util_create_list_235, CACHED_main_const_eval_106
    )
    CACHED_main_const_eval_106 = utils_constEvalFuncWrapper_102
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_107 = main_const_eval_107
    util_create_list_236 = [input_37]
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_107, util_create_list_236, CACHED_main_const_eval_107
    )
    CACHED_main_const_eval_107 = utils_constEvalFuncWrapper_103
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_108 = main_const_eval_108
    util_create_list_237 = [input_162]
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_108, util_create_list_237, CACHED_main_const_eval_108
    )
    CACHED_main_const_eval_108 = utils_constEvalFuncWrapper_104
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_109 = main_const_eval_109
    util_create_list_238 = [input_7]
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_109, util_create_list_238, CACHED_main_const_eval_109
    )
    CACHED_main_const_eval_109 = utils_constEvalFuncWrapper_105
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_110 = main_const_eval_110
    util_create_list_239 = [input_39]
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_110, util_create_list_239, CACHED_main_const_eval_110
    )
    CACHED_main_const_eval_110 = utils_constEvalFuncWrapper_106
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_111 = main_const_eval_111
    util_create_list_240 = [input_41]
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_111, util_create_list_240, CACHED_main_const_eval_111
    )
    CACHED_main_const_eval_111 = utils_constEvalFuncWrapper_107
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_112 = main_const_eval_112
    util_create_list_241 = [input_143]
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_112, util_create_list_241, CACHED_main_const_eval_112
    )
    CACHED_main_const_eval_112 = utils_constEvalFuncWrapper_108
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_113 = main_const_eval_113
    util_create_list_242 = [input_81]
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_113, util_create_list_242, CACHED_main_const_eval_113
    )
    CACHED_main_const_eval_113 = utils_constEvalFuncWrapper_109
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_114 = main_const_eval_114
    util_create_list_243 = [input_121]
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_114, util_create_list_243, CACHED_main_const_eval_114
    )
    CACHED_main_const_eval_114 = utils_constEvalFuncWrapper_110
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_115 = main_const_eval_115
    util_create_list_244 = [input_151]
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_115, util_create_list_244, CACHED_main_const_eval_115
    )
    CACHED_main_const_eval_115 = utils_constEvalFuncWrapper_111
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_116 = main_const_eval_116
    util_create_list_245 = [input_139]
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_116, util_create_list_245, CACHED_main_const_eval_116
    )
    CACHED_main_const_eval_116 = utils_constEvalFuncWrapper_112
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_117 = main_const_eval_117
    util_create_list_246 = [input_11]
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_117, util_create_list_246, CACHED_main_const_eval_117
    )
    CACHED_main_const_eval_117 = utils_constEvalFuncWrapper_113
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_118 = main_const_eval_118
    util_create_list_247 = [input_89]
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_118, util_create_list_247, CACHED_main_const_eval_118
    )
    CACHED_main_const_eval_118 = utils_constEvalFuncWrapper_114
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_119 = main_const_eval_119
    utils_constEvalFuncWrapperZeroArg_4 = utils.constEvalFuncWrapperZeroArg(
        const_119, CACHED_main_const_eval_119
    )
    CACHED_main_const_eval_119 = utils_constEvalFuncWrapperZeroArg_4
    utils_constEvalFuncWrapperZeroArg_4_0 = utils_constEvalFuncWrapperZeroArg_4[0]
    utils_constEvalFuncWrapperZeroArg_4_1 = utils_constEvalFuncWrapperZeroArg_4[1]
    utils_constEvalFuncWrapperZeroArg_4_2 = utils_constEvalFuncWrapperZeroArg_4[2]
    utils_constEvalFuncWrapperZeroArg_4_3 = utils_constEvalFuncWrapperZeroArg_4[3]
    utils_constEvalFuncWrapperZeroArg_4_4 = utils_constEvalFuncWrapperZeroArg_4[4]
    utils_constEvalFuncWrapperZeroArg_4_5 = utils_constEvalFuncWrapperZeroArg_4[5]
    utils_constEvalFuncWrapperZeroArg_4_6 = utils_constEvalFuncWrapperZeroArg_4[6]
    utils_constEvalFuncWrapperZeroArg_4_7 = utils_constEvalFuncWrapperZeroArg_4[7]
    utils_constEvalFuncWrapperZeroArg_4_8 = utils_constEvalFuncWrapperZeroArg_4[8]
    utils_constEvalFuncWrapperZeroArg_4_9 = utils_constEvalFuncWrapperZeroArg_4[9]
    utils_constEvalFuncWrapperZeroArg_4_10 = utils_constEvalFuncWrapperZeroArg_4[10]
    utils_constEvalFuncWrapperZeroArg_4_11 = utils_constEvalFuncWrapperZeroArg_4[11]
    const_120 = main_const_eval_120
    util_create_list_248 = [input_101]
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_120, util_create_list_248, CACHED_main_const_eval_120
    )
    CACHED_main_const_eval_120 = utils_constEvalFuncWrapper_115
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_121 = main_const_eval_121
    util_create_list_249 = [input_56]
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_121, util_create_list_249, CACHED_main_const_eval_121
    )
    CACHED_main_const_eval_121 = utils_constEvalFuncWrapper_116
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_122 = main_const_eval_122
    util_create_list_250 = [input_154]
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_122, util_create_list_250, CACHED_main_const_eval_122
    )
    CACHED_main_const_eval_122 = utils_constEvalFuncWrapper_117
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_123 = main_const_eval_123
    util_create_list_251 = [input_32]
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_123, util_create_list_251, CACHED_main_const_eval_123
    )
    CACHED_main_const_eval_123 = utils_constEvalFuncWrapper_118
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_124 = main_const_eval_124
    util_create_list_252 = [input_38]
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_124, util_create_list_252, CACHED_main_const_eval_124
    )
    CACHED_main_const_eval_124 = utils_constEvalFuncWrapper_119
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_125 = main_const_eval_125
    util_create_list_253 = [input_107]
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_125, util_create_list_253, CACHED_main_const_eval_125
    )
    CACHED_main_const_eval_125 = utils_constEvalFuncWrapper_120
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_126 = main_const_eval_126
    util_create_list_254 = [input_29]
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_126, util_create_list_254, CACHED_main_const_eval_126
    )
    CACHED_main_const_eval_126 = utils_constEvalFuncWrapper_121
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_127 = main_const_eval_127
    util_create_list_255 = [input_87]
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_127, util_create_list_255, CACHED_main_const_eval_127
    )
    CACHED_main_const_eval_127 = utils_constEvalFuncWrapper_122
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_128 = main_const_eval_128
    utils_constEvalFuncWrapperZeroArg_5 = utils.constEvalFuncWrapperZeroArg(
        const_128, CACHED_main_const_eval_128
    )
    CACHED_main_const_eval_128 = utils_constEvalFuncWrapperZeroArg_5
    utils_constEvalFuncWrapperZeroArg_5_0 = utils_constEvalFuncWrapperZeroArg_5[0]
    const_129 = main_const_eval_129
    util_create_list_256 = [input_95]
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_129, util_create_list_256, CACHED_main_const_eval_129
    )
    CACHED_main_const_eval_129 = utils_constEvalFuncWrapper_123
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_130 = main_const_eval_130
    util_create_list_257 = [input_182]
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_130, util_create_list_257, CACHED_main_const_eval_130
    )
    CACHED_main_const_eval_130 = utils_constEvalFuncWrapper_124
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_131 = main_const_eval_131
    util_create_list_258 = [input_69]
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_131, util_create_list_258, CACHED_main_const_eval_131
    )
    CACHED_main_const_eval_131 = utils_constEvalFuncWrapper_125
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    const_132 = main_const_eval_132
    util_create_list_259 = [input_51]
    utils_constEvalFuncWrapper_126 = utils.constEvalFuncWrapper(
        const_132, util_create_list_259, CACHED_main_const_eval_132
    )
    CACHED_main_const_eval_132 = utils_constEvalFuncWrapper_126
    utils_constEvalFuncWrapper_126_0 = utils_constEvalFuncWrapper_126[0]
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    CLIPVisionEmbeddings_0_0_0 = CLIPVisionEmbeddings_0_0(
        utils_constEvalFuncWrapper_78_0,
        utils_constEvalFuncWrapper_9_0,
        utils_DeviceGetter_get_device_7,
        input_152,
        utils_constEvalFuncWrapper_111_0,
    )
    LayerNorm_1_0_0 = LayerNorm_1_0(
        CLIPVisionEmbeddings_0_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_0,
        utils_constEvalFuncWrapper_100_0,
        utils_constEvalFuncWrapper_57_0,
    )
    v_370, v_371 = CLIPEncoderLayer_2_0(
        LayerNorm_1_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_1,
    )
    Linear_3_0_0 = Linear_3_0(v_370)
    v_372, v_373 = LayerNorm_4_0(
        utils_constEvalFuncWrapper_2_0,
        Linear_3_0_0,
        v_371,
        utils_constEvalFuncWrapper_91_0,
    )
    v_374, v_375, v_376 = Linear_5_0(
        v_372,
        input_155,
        utils_constEvalFuncWrapper_117_0,
        v_373,
        input_157,
        input_144,
        utils_constEvalFuncWrapper_108_0,
        utils_constEvalFuncWrapper_16_0,
    )
    CLIPAttention_6_0_0 = CLIPAttention_6_0(
        v_374, utils_constEvalFuncWrapperZeroArg_5_0, v_375, v_376
    )
    Linear_7_0_0 = Linear_7_0(
        input_142, CLIPAttention_6_0_0, utils_constEvalFuncWrapper_26_0
    )
    v_377, v_378, v_379 = CLIPEncoderLayer_8_0(
        utils_constEvalFuncWrapperZeroArg_3_2,
        Linear_7_0_0,
        LayerNorm_1_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_9_0_0 = Linear_9_0(v_377)
    LayerNorm_10_0_0 = LayerNorm_10_0(
        utils_constEvalFuncWrapper_112_0,
        utils_constEvalFuncWrapper_65_0,
        Linear_9_0_0,
        v_379,
    )
    Linear_11_0_0 = Linear_11_0(
        input_138, LayerNorm_10_0_0, utils_constEvalFuncWrapper_3_0
    )
    QuickGELUActivation_12_0_0 = QuickGELUActivation_12_0(
        Linear_11_0_0, utils_constEvalFuncWrapperZeroArg_4_0
    )
    Linear_13_0_0 = Linear_13_0(
        QuickGELUActivation_12_0_0, input_136, utils_constEvalFuncWrapper_34_0
    )
    v_380, v_381, v_382 = CLIPEncoderLayer_14_0(
        utils_constEvalFuncWrapperZeroArg_3_3,
        Linear_13_0_0,
        v_378,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_15_0_0 = Linear_15_0(v_380)
    v_383, v_384 = LayerNorm_16_0(
        utils_constEvalFuncWrapper_101_0,
        Linear_15_0_0,
        utils_constEvalFuncWrapper_70_0,
        v_382,
    )
    v_385, v_386, v_387 = Linear_17_0(
        v_383,
        input_161,
        input_159,
        v_384,
        utils_constEvalFuncWrapper_60_0,
        input_132,
        utils_constEvalFuncWrapper_97_0,
        utils_constEvalFuncWrapper_19_0,
    )
    CLIPAttention_18_0_0 = CLIPAttention_18_0(
        v_385, v_386, utils_constEvalFuncWrapperZeroArg_5_0, v_387
    )
    Linear_19_0_0 = Linear_19_0(
        input_130, utils_constEvalFuncWrapper_94_0, CLIPAttention_18_0_0
    )
    v_388, v_389, v_390 = CLIPEncoderLayer_20_0(
        utils_constEvalFuncWrapperZeroArg_1_0,
        v_381,
        utils_constEvalFuncWrapperZeroArg_3_4,
        Linear_19_0_0,
    )
    Linear_21_0_0 = Linear_21_0(v_389)
    LayerNorm_22_0_0 = LayerNorm_22_0(
        utils_constEvalFuncWrapper_30_0,
        v_390,
        utils_constEvalFuncWrapper_5_0,
        Linear_21_0_0,
    )
    Linear_23_0_0 = Linear_23_0(
        input_126, utils_constEvalFuncWrapper_14_0, LayerNorm_22_0_0
    )
    QuickGELUActivation_24_0_0 = QuickGELUActivation_24_0(
        utils_constEvalFuncWrapperZeroArg_4_1, Linear_23_0_0
    )
    Linear_25_0_0 = Linear_25_0(
        utils_constEvalFuncWrapper_8_0, QuickGELUActivation_24_0_0, input_124
    )
    v_391, v_392, v_393 = CLIPEncoderLayer_26_0(
        v_388,
        utils_constEvalFuncWrapperZeroArg_3_5,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_25_0_0,
    )
    Linear_27_0_0 = Linear_27_0(v_391)
    v_394, v_395 = LayerNorm_28_0(
        utils_constEvalFuncWrapper_54_0,
        utils_constEvalFuncWrapper_110_0,
        Linear_27_0_0,
        v_393,
    )
    v_396, v_397, v_398 = Linear_29_0(
        input_165,
        utils_constEvalFuncWrapper_104_0,
        input_163,
        utils_constEvalFuncWrapper_96_0,
        input_120,
        v_394,
        utils_constEvalFuncWrapper_76_0,
        v_395,
    )
    CLIPAttention_30_0_0 = CLIPAttention_30_0(
        utils_constEvalFuncWrapperZeroArg_5_0, v_396, v_397, v_398
    )
    Linear_31_0_0 = Linear_31_0(
        input_118, utils_constEvalFuncWrapper_32_0, CLIPAttention_30_0_0
    )
    v_399, v_400, v_401 = CLIPEncoderLayer_32_0(
        utils_constEvalFuncWrapperZeroArg_3_6,
        Linear_31_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        v_392,
    )
    Linear_33_0_0 = Linear_33_0(v_401)
    LayerNorm_34_0_0 = LayerNorm_34_0(
        v_399,
        utils_constEvalFuncWrapper_33_0,
        utils_constEvalFuncWrapper_7_0,
        Linear_33_0_0,
    )
    Linear_35_0_0 = Linear_35_0(
        input_114, LayerNorm_34_0_0, utils_constEvalFuncWrapper_93_0
    )
    QuickGELUActivation_36_0_0 = QuickGELUActivation_36_0(
        Linear_35_0_0, utils_constEvalFuncWrapperZeroArg_4_2
    )
    Linear_37_0_0 = Linear_37_0(
        input_112, utils_constEvalFuncWrapper_20_0, QuickGELUActivation_36_0_0
    )
    v_402, v_403, v_404 = CLIPEncoderLayer_38_0(
        utils_constEvalFuncWrapperZeroArg_3_7,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_37_0_0,
        v_400,
    )
    Linear_39_0_0 = Linear_39_0(v_403)
    v_405, v_406 = LayerNorm_40_0(
        utils_constEvalFuncWrapper_25_0,
        v_404,
        Linear_39_0_0,
        utils_constEvalFuncWrapper_89_0,
    )
    v_407, v_408, v_409 = Linear_41_0(
        v_405,
        utils_constEvalFuncWrapper_120_0,
        input_108,
        input_167,
        utils_constEvalFuncWrapper_58_0,
        v_406,
        utils_constEvalFuncWrapper_35_0,
        input_169,
    )
    CLIPAttention_42_0_0 = CLIPAttention_42_0(
        v_407, utils_constEvalFuncWrapperZeroArg_5_0, v_408, v_409
    )
    Linear_43_0_0 = Linear_43_0(
        input_106, utils_constEvalFuncWrapper_85_0, CLIPAttention_42_0_0
    )
    v_410, v_411, v_412 = CLIPEncoderLayer_44_0(
        Linear_43_0_0,
        v_402,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_8,
    )
    Linear_45_0_0 = Linear_45_0(v_411)
    LayerNorm_46_0_0 = LayerNorm_46_0(
        v_410,
        utils_constEvalFuncWrapper_72_0,
        Linear_45_0_0,
        utils_constEvalFuncWrapper_68_0,
    )
    Linear_47_0_0 = Linear_47_0(
        LayerNorm_46_0_0, input_102, utils_constEvalFuncWrapper_115_0
    )
    QuickGELUActivation_48_0_0 = QuickGELUActivation_48_0(
        Linear_47_0_0, utils_constEvalFuncWrapperZeroArg_4_3
    )
    Linear_49_0_0 = Linear_49_0(
        input_100, QuickGELUActivation_48_0_0, utils_constEvalFuncWrapper_87_0
    )
    v_413, v_414, v_415 = CLIPEncoderLayer_50_0(
        Linear_49_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_9,
        v_412,
    )
    Linear_51_0_0 = Linear_51_0(v_413)
    v_416, v_417 = LayerNorm_52_0(
        v_414,
        utils_constEvalFuncWrapper_43_0,
        utils_constEvalFuncWrapper_22_0,
        Linear_51_0_0,
    )
    v_418, v_419, v_420 = Linear_53_0(
        utils_constEvalFuncWrapper_80_0,
        utils_constEvalFuncWrapper_36_0,
        input_96,
        v_416,
        input_171,
        v_417,
        utils_constEvalFuncWrapper_123_0,
        input_173,
    )
    CLIPAttention_54_0_0 = CLIPAttention_54_0(
        utils_constEvalFuncWrapperZeroArg_5_0, v_418, v_419, v_420
    )
    Linear_55_0_0 = Linear_55_0(
        CLIPAttention_54_0_0, input_94, utils_constEvalFuncWrapper_86_0
    )
    v_421, v_422, v_423 = CLIPEncoderLayer_56_0(
        utils_constEvalFuncWrapperZeroArg_3_10,
        utils_constEvalFuncWrapperZeroArg_1_0,
        v_415,
        Linear_55_0_0,
    )
    Linear_57_0_0 = Linear_57_0(v_421)
    LayerNorm_58_0_0 = LayerNorm_58_0(
        utils_constEvalFuncWrapper_77_0,
        v_423,
        utils_constEvalFuncWrapper_62_0,
        Linear_57_0_0,
    )
    Linear_59_0_0 = Linear_59_0(
        LayerNorm_58_0_0, utils_constEvalFuncWrapper_114_0, input_90
    )
    QuickGELUActivation_60_0_0 = QuickGELUActivation_60_0(
        utils_constEvalFuncWrapperZeroArg_4_4, Linear_59_0_0
    )
    Linear_61_0_0 = Linear_61_0(
        input_88, utils_constEvalFuncWrapper_122_0, QuickGELUActivation_60_0_0
    )
    v_424, v_425, v_426 = CLIPEncoderLayer_62_0(
        utils_constEvalFuncWrapperZeroArg_3_11,
        Linear_61_0_0,
        v_422,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_63_0_0 = Linear_63_0(v_424)
    v_427, v_428 = LayerNorm_64_0(
        utils_constEvalFuncWrapper_71_0,
        v_425,
        utils_constEvalFuncWrapper_42_0,
        Linear_63_0_0,
    )
    v_429, v_430, v_431 = Linear_65_0(
        input_175,
        v_427,
        utils_constEvalFuncWrapper_49_0,
        utils_constEvalFuncWrapper_38_0,
        utils_constEvalFuncWrapper_73_0,
        input_177,
        v_428,
        input_84,
    )
    CLIPAttention_66_0_0 = CLIPAttention_66_0(
        utils_constEvalFuncWrapperZeroArg_5_0, v_429, v_430, v_431
    )
    Linear_67_0_0 = Linear_67_0(
        utils_constEvalFuncWrapper_109_0, CLIPAttention_66_0_0, input_82
    )
    v_432, v_433, v_434 = CLIPEncoderLayer_68_0(
        Linear_67_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_12,
        v_426,
    )
    Linear_69_0_0 = Linear_69_0(v_434)
    LayerNorm_70_0_0 = LayerNorm_70_0(
        utils_constEvalFuncWrapper_31_0,
        v_432,
        utils_constEvalFuncWrapper_28_0,
        Linear_69_0_0,
    )
    Linear_71_0_0 = Linear_71_0(
        utils_constEvalFuncWrapper_23_0, input_78, LayerNorm_70_0_0
    )
    QuickGELUActivation_72_0_0 = QuickGELUActivation_72_0(
        Linear_71_0_0, utils_constEvalFuncWrapperZeroArg_4_5
    )
    Linear_73_0_0 = Linear_73_0(
        QuickGELUActivation_72_0_0, utils_constEvalFuncWrapper_37_0, input_76
    )
    v_435, v_436, v_437 = CLIPEncoderLayer_74_0(
        utils_constEvalFuncWrapperZeroArg_3_13,
        utils_constEvalFuncWrapperZeroArg_1_0,
        v_433,
        Linear_73_0_0,
    )
    Linear_75_0_0 = Linear_75_0(v_437)
    v_438, v_439 = LayerNorm_76_0(
        utils_constEvalFuncWrapper_79_0,
        utils_constEvalFuncWrapper_81_0,
        Linear_75_0_0,
        v_436,
    )
    v_440, v_441, v_442 = Linear_77_0(
        utils_constEvalFuncWrapper_84_0,
        v_438,
        input_179,
        input_181,
        input_72,
        utils_constEvalFuncWrapper_0_0,
        v_439,
        utils_constEvalFuncWrapper_61_0,
    )
    CLIPAttention_78_0_0 = CLIPAttention_78_0(
        utils_constEvalFuncWrapperZeroArg_5_0, v_440, v_441, v_442
    )
    Linear_79_0_0 = Linear_79_0(
        CLIPAttention_78_0_0, input_70, utils_constEvalFuncWrapper_125_0
    )
    v_443, v_444, v_445 = CLIPEncoderLayer_80_0(
        v_435,
        Linear_79_0_0,
        utils_constEvalFuncWrapperZeroArg_3_14,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_81_0_0 = Linear_81_0(v_444)
    LayerNorm_82_0_0 = LayerNorm_82_0(
        utils_constEvalFuncWrapper_83_0,
        Linear_81_0_0,
        v_445,
        utils_constEvalFuncWrapper_66_0,
    )
    Linear_83_0_0 = Linear_83_0(
        LayerNorm_82_0_0, utils_constEvalFuncWrapper_27_0, input_66
    )
    QuickGELUActivation_84_0_0 = QuickGELUActivation_84_0(
        Linear_83_0_0, utils_constEvalFuncWrapperZeroArg_4_6
    )
    Linear_85_0_0 = Linear_85_0(
        utils_constEvalFuncWrapper_88_0, QuickGELUActivation_84_0_0, input_64
    )
    v_446, v_447, v_448 = CLIPEncoderLayer_86_0(
        v_443,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_85_0_0,
        utils_constEvalFuncWrapperZeroArg_3_15,
    )
    Linear_87_0_0 = Linear_87_0(v_446)
    v_449, v_450 = LayerNorm_88_0(
        utils_constEvalFuncWrapper_59_0,
        Linear_87_0_0,
        utils_constEvalFuncWrapper_24_0,
        v_448,
    )
    v_451, v_452, v_453 = Linear_89_0(
        input_60,
        utils_constEvalFuncWrapper_55_0,
        v_449,
        v_450,
        input_183,
        utils_constEvalFuncWrapper_124_0,
        input_185,
        utils_constEvalFuncWrapper_1_0,
    )
    CLIPAttention_90_0_0 = CLIPAttention_90_0(
        v_451, utils_constEvalFuncWrapperZeroArg_5_0, v_452, v_453
    )
    Linear_91_0_0 = Linear_91_0(
        utils_constEvalFuncWrapper_74_0, input_58, CLIPAttention_90_0_0
    )
    v_454, v_455, v_456 = CLIPEncoderLayer_92_0(
        utils_constEvalFuncWrapperZeroArg_1_0,
        v_447,
        Linear_91_0_0,
        utils_constEvalFuncWrapperZeroArg_3_16,
    )
    Linear_93_0_0 = Linear_93_0(v_456)
    LayerNorm_94_0_0 = LayerNorm_94_0(
        v_455,
        Linear_93_0_0,
        utils_constEvalFuncWrapper_64_0,
        utils_constEvalFuncWrapper_116_0,
    )
    Linear_95_0_0 = Linear_95_0(
        input_54, LayerNorm_94_0_0, utils_constEvalFuncWrapper_63_0
    )
    QuickGELUActivation_96_0_0 = QuickGELUActivation_96_0(
        utils_constEvalFuncWrapperZeroArg_4_7, Linear_95_0_0
    )
    Linear_97_0_0 = Linear_97_0(
        QuickGELUActivation_96_0_0, utils_constEvalFuncWrapper_126_0, input_52
    )
    v_457, v_458, v_459 = CLIPEncoderLayer_98_0(
        v_454,
        Linear_97_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_3_17,
    )
    Linear_99_0_0 = Linear_99_0(v_457)
    v_460, v_461 = LayerNorm_100_0(
        Linear_99_0_0,
        utils_constEvalFuncWrapper_40_0,
        v_459,
        utils_constEvalFuncWrapper_75_0,
    )
    v_462, v_463, v_464 = Linear_101_0(
        v_460,
        utils_constEvalFuncWrapper_12_0,
        v_461,
        input_189,
        input_48,
        input_187,
        utils_constEvalFuncWrapper_95_0,
        utils_constEvalFuncWrapper_10_0,
    )
    CLIPAttention_102_0_0 = CLIPAttention_102_0(
        v_462, v_463, utils_constEvalFuncWrapperZeroArg_5_0, v_464
    )
    Linear_103_0_0 = Linear_103_0(
        input_46, utils_constEvalFuncWrapper_45_0, CLIPAttention_102_0_0
    )
    v_465, v_466, v_467 = CLIPEncoderLayer_104_0(
        utils_constEvalFuncWrapperZeroArg_3_18,
        v_458,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_103_0_0,
    )
    Linear_105_0_0 = Linear_105_0(v_467)
    LayerNorm_106_0_0 = LayerNorm_106_0(
        Linear_105_0_0,
        v_465,
        utils_constEvalFuncWrapper_102_0,
        utils_constEvalFuncWrapper_46_0,
    )
    Linear_107_0_0 = Linear_107_0(
        utils_constEvalFuncWrapper_107_0, LayerNorm_106_0_0, input_42
    )
    QuickGELUActivation_108_0_0 = QuickGELUActivation_108_0(
        utils_constEvalFuncWrapperZeroArg_4_8, Linear_107_0_0
    )
    Linear_109_0_0 = Linear_109_0(
        utils_constEvalFuncWrapper_106_0, QuickGELUActivation_108_0_0, input_40
    )
    v_468, v_469, v_470 = CLIPEncoderLayer_110_0(
        utils_constEvalFuncWrapperZeroArg_3_19,
        v_466,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_109_0_0,
    )
    Linear_111_0_0 = Linear_111_0(v_470)
    v_471, v_472 = LayerNorm_112_0(
        Linear_111_0_0,
        utils_constEvalFuncWrapper_119_0,
        v_469,
        utils_constEvalFuncWrapper_103_0,
    )
    v_473, v_474, v_475 = Linear_113_0(
        input_193,
        v_471,
        v_472,
        utils_constEvalFuncWrapper_56_0,
        input_36,
        utils_constEvalFuncWrapper_11_0,
        utils_constEvalFuncWrapper_29_0,
        input_191,
    )
    CLIPAttention_114_0_0 = CLIPAttention_114_0(
        utils_constEvalFuncWrapperZeroArg_5_0, v_473, v_474, v_475
    )
    Linear_115_0_0 = Linear_115_0(
        utils_constEvalFuncWrapper_98_0, CLIPAttention_114_0_0, input_34
    )
    v_476, v_477, v_478 = CLIPEncoderLayer_116_0(
        v_468,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_115_0_0,
        utils_constEvalFuncWrapperZeroArg_3_20,
    )
    Linear_117_0_0 = Linear_117_0(v_477)
    LayerNorm_118_0_0 = LayerNorm_118_0(
        Linear_117_0_0,
        utils_constEvalFuncWrapper_118_0,
        utils_constEvalFuncWrapper_6_0,
        v_478,
    )
    Linear_119_0_0 = Linear_119_0(
        LayerNorm_118_0_0, utils_constEvalFuncWrapper_121_0, input_30
    )
    QuickGELUActivation_120_0_0 = QuickGELUActivation_120_0(
        Linear_119_0_0, utils_constEvalFuncWrapperZeroArg_4_9
    )
    Linear_121_0_0 = Linear_121_0(
        input_28, QuickGELUActivation_120_0_0, utils_constEvalFuncWrapper_21_0
    )
    v_479, v_480, v_481 = CLIPEncoderLayer_122_0(
        v_476,
        utils_constEvalFuncWrapperZeroArg_3_21,
        Linear_121_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_123_0_0 = Linear_123_0(v_481)
    v_482, v_483 = LayerNorm_124_0(
        Linear_123_0_0,
        v_480,
        utils_constEvalFuncWrapper_90_0,
        utils_constEvalFuncWrapper_69_0,
    )
    v_484, v_485, v_486 = Linear_125_0(
        input_195,
        utils_constEvalFuncWrapper_41_0,
        v_482,
        input_197,
        v_483,
        utils_constEvalFuncWrapper_99_0,
        input_24,
        utils_constEvalFuncWrapper_17_0,
    )
    CLIPAttention_126_0_0 = CLIPAttention_126_0(
        v_484, utils_constEvalFuncWrapperZeroArg_5_0, v_485, v_486
    )
    Linear_127_0_0 = Linear_127_0(
        CLIPAttention_126_0_0, input_22, utils_constEvalFuncWrapper_15_0
    )
    v_487, v_488, v_489 = CLIPEncoderLayer_128_0(
        v_479,
        utils_constEvalFuncWrapperZeroArg_3_22,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_127_0_0,
    )
    Linear_129_0_0 = Linear_129_0(v_488)
    LayerNorm_130_0_0 = LayerNorm_130_0(
        Linear_129_0_0,
        utils_constEvalFuncWrapper_48_0,
        utils_constEvalFuncWrapper_4_0,
        v_487,
    )
    Linear_131_0_0 = Linear_131_0(
        utils_constEvalFuncWrapper_51_0, input_18, LayerNorm_130_0_0
    )
    QuickGELUActivation_132_0_0 = QuickGELUActivation_132_0(
        Linear_131_0_0, utils_constEvalFuncWrapperZeroArg_4_10
    )
    Linear_133_0_0 = Linear_133_0(
        input_16, QuickGELUActivation_132_0_0, utils_constEvalFuncWrapper_18_0
    )
    v_490, v_491, v_492 = CLIPEncoderLayer_134_0(
        utils_constEvalFuncWrapperZeroArg_3_23,
        utils_constEvalFuncWrapperZeroArg_1_0,
        Linear_133_0_0,
        v_489,
    )
    Linear_135_0_0 = Linear_135_0(v_490)
    v_493, v_494 = LayerNorm_136_0(
        v_491,
        utils_constEvalFuncWrapper_39_0,
        Linear_135_0_0,
        utils_constEvalFuncWrapper_52_0,
    )
    v_495, v_496, v_497 = Linear_137_0(
        input_199,
        utils_constEvalFuncWrapper_53_0,
        utils_constEvalFuncWrapper_92_0,
        v_493,
        input_12,
        input_201,
        utils_constEvalFuncWrapper_113_0,
        v_494,
    )
    CLIPAttention_138_0_0 = CLIPAttention_138_0(
        v_495, utils_constEvalFuncWrapperZeroArg_5_0, v_496, v_497
    )
    Linear_139_0_0 = Linear_139_0(
        input_10, utils_constEvalFuncWrapper_67_0, CLIPAttention_138_0_0
    )
    v_498, v_499, v_500 = CLIPEncoderLayer_140_0(
        utils_constEvalFuncWrapperZeroArg_3_24,
        v_492,
        Linear_139_0_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    Linear_141_0_0 = Linear_141_0(v_499)
    LayerNorm_142_0_0 = LayerNorm_142_0(
        utils_constEvalFuncWrapper_105_0,
        Linear_141_0_0,
        v_498,
        utils_constEvalFuncWrapper_47_0,
    )
    Linear_143_0_0 = Linear_143_0(
        utils_constEvalFuncWrapper_82_0, input_6, LayerNorm_142_0_0
    )
    QuickGELUActivation_144_0_0 = QuickGELUActivation_144_0(
        Linear_143_0_0, utils_constEvalFuncWrapperZeroArg_4_11
    )
    Linear_145_0_0 = Linear_145_0(
        input_4, utils_constEvalFuncWrapper_13_0, QuickGELUActivation_144_0_0
    )
    CLIPEncoderLayer_146_0_0 = CLIPEncoderLayer_146_0(Linear_145_0_0, v_500)
    CLIPVisionTransformer_147_0_0 = CLIPVisionTransformer_147_0(
        utils_constEvalFuncWrapper_44_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        CLIPEncoderLayer_146_0_0,
        utils_constEvalFuncWrapperZeroArg_2_0,
        utils_constEvalFuncWrapper_50_0,
    )
    Linear_148_0_0 = Linear_148_0(CLIPVisionTransformer_147_0_0, input_0)
    ttnn.deallocate(input_152, False)
    util_create_list_260 = [Linear_148_0_0, CLIPEncoderLayer_146_0_0]
    return util_create_list_260


def Linear_141_0(input):
    ttnn_reshape_221 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_221


def LayerNorm_22_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_0 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_multiply_0,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_multiply_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn.deallocate(ttnn_multiply_1, False)
    return ttnn_add_0


def CLIPAttention_126_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_to_memory_config_0,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_matmul_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_to_memory_config_1,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_reshape_222,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_multiply_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_typecast_1,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_softmax_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_typecast_2,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_reshape_223,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_to_memory_config_2,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_matmul_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_to_memory_config_3,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_0, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_1, False)
    ttnn.deallocate(ttnn_reshape_222, False)
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn.deallocate(ttnn_softmax_0, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_reshape_223, False)
    ttnn.deallocate(ttnn_to_memory_config_2, False)
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn.deallocate(ttnn_to_memory_config_3, False)
    return ttnn_reshape_224


def CLIPEncoderLayer_56_0(input_0, input_1, input_2, input_3):
    ttnn_add_1 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_add_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_0 = ttnn.sum(
        ttnn_to_memory_config_4,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_sum_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_multiply_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_to_memory_config_5,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_0 = ttnn.neg(
        ttnn_reshape_225,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_2 = ttnn.add(
        ttnn_to_memory_config_4,
        ttnn_neg_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_6 = ttnn.to_memory_config(
        ttnn_add_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_to_memory_config_6,
        ttnn_to_memory_config_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_7 = ttnn.to_memory_config(
        ttnn_multiply_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_1 = ttnn.sum(
        ttnn_to_memory_config_7,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_sum_1,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_3 = ttnn.add(
        ttnn_multiply_5,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_8 = ttnn.to_memory_config(
        ttnn_add_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_to_memory_config_8,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_rsqrt_0,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_1, False)
    # ttnn.deallocate(ttnn_to_memory_config_4, False)
    ttnn.deallocate(ttnn_sum_0, False)
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn.deallocate(ttnn_to_memory_config_5, False)
    ttnn.deallocate(ttnn_reshape_225, False)
    ttnn.deallocate(ttnn_neg_0, False)
    ttnn.deallocate(ttnn_add_2, False)
    # ttnn.deallocate(ttnn_to_memory_config_6, False)
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn.deallocate(ttnn_to_memory_config_7, False)
    ttnn.deallocate(ttnn_sum_1, False)
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn.deallocate(ttnn_add_3, False)
    ttnn.deallocate(ttnn_to_memory_config_8, False)
    ttnn.deallocate(ttnn_rsqrt_0, False)
    # ttnn.deallocate(ttnn_reshape_226, False)
    return ttnn_to_memory_config_6, ttnn_to_memory_config_4, ttnn_reshape_226


def Linear_37_0(input_0, input_1, input_2):
    ttnn_matmul_2 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_9 = ttnn.to_memory_config(
        ttnn_add_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_to_memory_config_9,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn.deallocate(ttnn_add_4, False)
    ttnn.deallocate(ttnn_to_memory_config_9, False)
    return ttnn_reshape_227


def CLIPEncoderLayer_38_0(input_0, input_1, input_2, input_3):
    ttnn_add_5 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_10 = ttnn.to_memory_config(
        ttnn_add_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_2 = ttnn.sum(
        ttnn_to_memory_config_10,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_sum_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_11 = ttnn.to_memory_config(
        ttnn_multiply_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_to_memory_config_11,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_1 = ttnn.neg(
        ttnn_reshape_228,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_6 = ttnn.add(
        ttnn_to_memory_config_10,
        ttnn_neg_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_12 = ttnn.to_memory_config(
        ttnn_add_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_to_memory_config_12,
        ttnn_to_memory_config_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_13 = ttnn.to_memory_config(
        ttnn_multiply_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_3 = ttnn.sum(
        ttnn_to_memory_config_13,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_sum_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_7 = ttnn.add(
        ttnn_multiply_8,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_14 = ttnn.to_memory_config(
        ttnn_add_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_to_memory_config_14,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_rsqrt_1,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_5, False)
    # ttnn.deallocate(ttnn_to_memory_config_10, False)
    ttnn.deallocate(ttnn_sum_2, False)
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn.deallocate(ttnn_to_memory_config_11, False)
    ttnn.deallocate(ttnn_reshape_228, False)
    ttnn.deallocate(ttnn_neg_1, False)
    ttnn.deallocate(ttnn_add_6, False)
    # ttnn.deallocate(ttnn_to_memory_config_12, False)
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn.deallocate(ttnn_to_memory_config_13, False)
    ttnn.deallocate(ttnn_sum_3, False)
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn.deallocate(ttnn_add_7, False)
    ttnn.deallocate(ttnn_to_memory_config_14, False)
    ttnn.deallocate(ttnn_rsqrt_1, False)
    # ttnn.deallocate(ttnn_reshape_229, False)
    return ttnn_to_memory_config_10, ttnn_to_memory_config_12, ttnn_reshape_229


def QuickGELUActivation_60_0(input_0, input_1):
    ttnn_multiply_9 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_15 = ttnn.to_memory_config(
        ttnn_multiply_9,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_0 = ttnn.sigmoid(
        ttnn_to_memory_config_15,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_10 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn.deallocate(ttnn_to_memory_config_15, False)
    ttnn.deallocate(ttnn_sigmoid_0, False)
    return ttnn_multiply_10


def LayerNorm_70_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_11 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_12 = ttnn.multiply(
        ttnn_multiply_11,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_8 = ttnn.add(
        ttnn_multiply_12,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_11, False)
    ttnn.deallocate(ttnn_multiply_12, False)
    return ttnn_add_8


def LayerNorm_34_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_13 = ttnn.multiply(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_14 = ttnn.multiply(
        ttnn_multiply_13,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_9 = ttnn.add(
        ttnn_multiply_14,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_13, False)
    ttnn.deallocate(ttnn_multiply_14, False)
    return ttnn_add_9


def CLIPEncoderLayer_2_0(input_0, input_1, input_2):
    ttnn_sum_4 = ttnn.sum(
        input_0,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_sum_4,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_16 = ttnn.to_memory_config(
        ttnn_multiply_15,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_to_memory_config_16,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_2 = ttnn.neg(
        ttnn_reshape_230,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_10 = ttnn.add(
        input_0,
        ttnn_neg_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_17 = ttnn.to_memory_config(
        ttnn_add_10,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_16 = ttnn.multiply(
        ttnn_to_memory_config_17,
        ttnn_to_memory_config_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_18 = ttnn.to_memory_config(
        ttnn_multiply_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_5 = ttnn.sum(
        ttnn_to_memory_config_18,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_17 = ttnn.multiply(
        ttnn_sum_5,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_11 = ttnn.add(
        ttnn_multiply_17,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_19 = ttnn.to_memory_config(
        ttnn_add_11,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_to_memory_config_19,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_rsqrt_2,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    # ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_sum_4, False)
    ttnn.deallocate(ttnn_multiply_15, False)
    ttnn.deallocate(ttnn_to_memory_config_16, False)
    ttnn.deallocate(ttnn_reshape_230, False)
    ttnn.deallocate(ttnn_neg_2, False)
    ttnn.deallocate(ttnn_add_10, False)
    # ttnn.deallocate(ttnn_to_memory_config_17, False)
    ttnn.deallocate(ttnn_multiply_16, False)
    ttnn.deallocate(ttnn_to_memory_config_18, False)
    ttnn.deallocate(ttnn_sum_5, False)
    ttnn.deallocate(ttnn_multiply_17, False)
    ttnn.deallocate(ttnn_add_11, False)
    ttnn.deallocate(ttnn_to_memory_config_19, False)
    ttnn.deallocate(ttnn_rsqrt_2, False)
    # ttnn.deallocate(ttnn_reshape_231, False)
    return ttnn_to_memory_config_17, ttnn_reshape_231


def Linear_33_0(input):
    ttnn_reshape_232 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_232


def LayerNorm_28_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_18 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_19 = ttnn.multiply(
        ttnn_multiply_18,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_12 = ttnn.add(
        ttnn_multiply_19,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_20 = ttnn.to_memory_config(
        ttnn_add_12,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_18, False)
    ttnn.deallocate(ttnn_multiply_19, False)
    return ttnn_to_memory_config_20, ttnn_add_12


def Linear_119_0(input_0, input_1, input_2):
    ttnn_matmul_3 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_13 = ttnn.add(
        ttnn_matmul_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_21 = ttnn.to_memory_config(
        ttnn_add_13,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn.deallocate(ttnn_add_13, False)
    return ttnn_to_memory_config_21


def Linear_65_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_4 = ttnn.matmul(
        input_1,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_22 = ttnn.to_memory_config(
        input_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_4,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_23 = ttnn.to_memory_config(
        input_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_to_memory_config_22,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_24 = ttnn.to_memory_config(
        ttnn_add_14,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_to_memory_config_23,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_15 = ttnn.add(
        ttnn_matmul_5,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_16 = ttnn.add(
        ttnn_matmul_6,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_to_memory_config_24,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_25 = ttnn.to_memory_config(
        ttnn_add_15,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_26 = ttnn.to_memory_config(
        ttnn_add_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_50 = ttnn.permute(
        ttnn_reshape_233,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_to_memory_config_25,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_to_memory_config_26,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_permute_50,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_51 = ttnn.permute(
        ttnn_reshape_234,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_52 = ttnn.permute(
        ttnn_reshape_235,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_permute_51,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_permute_52,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(input_6, False)
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn.deallocate(ttnn_add_14, False)
    ttnn.deallocate(ttnn_to_memory_config_22, False)
    ttnn.deallocate(ttnn_to_memory_config_23, False)
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn.deallocate(ttnn_to_memory_config_24, False)
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn.deallocate(ttnn_reshape_233, False)
    ttnn.deallocate(ttnn_add_16, False)
    ttnn.deallocate(ttnn_add_15, False)
    ttnn.deallocate(ttnn_to_memory_config_25, False)
    ttnn.deallocate(ttnn_permute_50, False)
    ttnn.deallocate(ttnn_to_memory_config_26, False)
    ttnn.deallocate(ttnn_reshape_234, False)
    ttnn.deallocate(ttnn_reshape_235, False)
    ttnn.deallocate(ttnn_permute_51, False)
    ttnn.deallocate(ttnn_permute_52, False)
    return ttnn_reshape_238, ttnn_reshape_237, ttnn_reshape_236


def LayerNorm_4_0(input_0, input_1, input_2, input_3):
    input_1.is_allocated()
    input_2.is_allocated()
    ttnn_multiply_20 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_21 = ttnn.multiply(
        ttnn_multiply_20,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_17 = ttnn.add(
        ttnn_multiply_21,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_27 = ttnn.to_memory_config(
        ttnn_add_17,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_20, False)
    ttnn.deallocate(ttnn_multiply_21, False)
    return ttnn_to_memory_config_27, ttnn_add_17


def QuickGELUActivation_12_0(input_0, input_1):
    ttnn_multiply_22 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_28 = ttnn.to_memory_config(
        ttnn_multiply_22,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_1 = ttnn.sigmoid(
        ttnn_to_memory_config_28,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_23 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_22, False)
    ttnn.deallocate(ttnn_to_memory_config_28, False)
    ttnn.deallocate(ttnn_sigmoid_1, False)
    return ttnn_multiply_23


def Linear_137_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_7 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_29 = ttnn.to_memory_config(
        input_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_18 = ttnn.add(
        ttnn_matmul_7,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_30 = ttnn.to_memory_config(
        input_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_to_memory_config_29,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_31 = ttnn.to_memory_config(
        ttnn_add_18,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_to_memory_config_30,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_19 = ttnn.add(
        ttnn_matmul_8,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_20 = ttnn.add(
        ttnn_matmul_9,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_239 = ttnn.reshape(
        ttnn_to_memory_config_31,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_32 = ttnn.to_memory_config(
        ttnn_add_19,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_33 = ttnn.to_memory_config(
        ttnn_add_20,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_53 = ttnn.permute(
        ttnn_reshape_239,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_to_memory_config_32,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_to_memory_config_33,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_permute_53,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_54 = ttnn.permute(
        ttnn_reshape_240,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_55 = ttnn.permute(
        ttnn_reshape_241,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_permute_54,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_permute_55,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_7, False)
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn.deallocate(ttnn_add_18, False)
    ttnn.deallocate(ttnn_to_memory_config_29, False)
    ttnn.deallocate(ttnn_to_memory_config_30, False)
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn.deallocate(ttnn_to_memory_config_31, False)
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn.deallocate(ttnn_reshape_239, False)
    ttnn.deallocate(ttnn_add_20, False)
    ttnn.deallocate(ttnn_add_19, False)
    ttnn.deallocate(ttnn_to_memory_config_32, False)
    ttnn.deallocate(ttnn_permute_53, False)
    ttnn.deallocate(ttnn_to_memory_config_33, False)
    ttnn.deallocate(ttnn_reshape_240, False)
    ttnn.deallocate(ttnn_reshape_241, False)
    ttnn.deallocate(ttnn_permute_54, False)
    ttnn.deallocate(ttnn_permute_55, False)
    return ttnn_reshape_244, ttnn_reshape_242, ttnn_reshape_243


def Linear_139_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_24 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_245 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_24,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_34 = ttnn.to_memory_config(
        ttnn_reshape_245,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_to_memory_config_34,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_21 = ttnn.add(
        ttnn_matmul_10,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_35 = ttnn.to_memory_config(
        ttnn_add_21,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_246 = ttnn.reshape(
        ttnn_to_memory_config_35,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_24, False)
    ttnn.deallocate(ttnn_reshape_245, False)
    ttnn.deallocate(ttnn_to_memory_config_34, False)
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn.deallocate(ttnn_add_21, False)
    ttnn.deallocate(ttnn_to_memory_config_35, False)
    return ttnn_reshape_246


def CLIPEncoderLayer_122_0(input_0, input_1, input_2, input_3):
    ttnn_add_22 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_36 = ttnn.to_memory_config(
        ttnn_add_22,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_6 = ttnn.sum(
        ttnn_to_memory_config_36,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_24 = ttnn.multiply(
        ttnn_sum_6,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_37 = ttnn.to_memory_config(
        ttnn_multiply_24,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_247 = ttnn.reshape(
        ttnn_to_memory_config_37,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_3 = ttnn.neg(
        ttnn_reshape_247,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_23 = ttnn.add(
        ttnn_to_memory_config_36,
        ttnn_neg_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_38 = ttnn.to_memory_config(
        ttnn_add_23,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_25 = ttnn.multiply(
        ttnn_to_memory_config_38,
        ttnn_to_memory_config_38,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_39 = ttnn.to_memory_config(
        ttnn_multiply_25,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_7 = ttnn.sum(
        ttnn_to_memory_config_39,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_26 = ttnn.multiply(
        ttnn_sum_7,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_24 = ttnn.add(
        ttnn_multiply_26,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_40 = ttnn.to_memory_config(
        ttnn_add_24,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_to_memory_config_40,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_248 = ttnn.reshape(
        ttnn_rsqrt_3,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_22, False)
    # ttnn.deallocate(ttnn_to_memory_config_36, False)
    ttnn.deallocate(ttnn_sum_6, False)
    ttnn.deallocate(ttnn_multiply_24, False)
    ttnn.deallocate(ttnn_to_memory_config_37, False)
    ttnn.deallocate(ttnn_reshape_247, False)
    ttnn.deallocate(ttnn_neg_3, False)
    ttnn.deallocate(ttnn_add_23, False)
    # ttnn.deallocate(ttnn_to_memory_config_38, False)
    ttnn.deallocate(ttnn_multiply_25, False)
    ttnn.deallocate(ttnn_to_memory_config_39, False)
    ttnn.deallocate(ttnn_sum_7, False)
    ttnn.deallocate(ttnn_multiply_26, False)
    ttnn.deallocate(ttnn_add_24, False)
    ttnn.deallocate(ttnn_to_memory_config_40, False)
    ttnn.deallocate(ttnn_rsqrt_3, False)
    # ttnn.deallocate(ttnn_reshape_248, False)
    return ttnn_to_memory_config_36, ttnn_reshape_248, ttnn_to_memory_config_38


def LayerNorm_118_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_27 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_28 = ttnn.multiply(
        ttnn_multiply_27,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_25 = ttnn.add(
        ttnn_multiply_28,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_27, False)
    ttnn.deallocate(ttnn_multiply_28, False)
    return ttnn_add_25


def CLIPEncoderLayer_14_0(input_0, input_1, input_2, input_3):
    ttnn_add_26 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_41 = ttnn.to_memory_config(
        ttnn_add_26,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_8 = ttnn.sum(
        ttnn_to_memory_config_41,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_29 = ttnn.multiply(
        ttnn_sum_8,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_42 = ttnn.to_memory_config(
        ttnn_multiply_29,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_249 = ttnn.reshape(
        ttnn_to_memory_config_42,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_4 = ttnn.neg(
        ttnn_reshape_249,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_27 = ttnn.add(
        ttnn_to_memory_config_41,
        ttnn_neg_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_43 = ttnn.to_memory_config(
        ttnn_add_27,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_to_memory_config_43,
        ttnn_to_memory_config_43,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_44 = ttnn.to_memory_config(
        ttnn_multiply_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_9 = ttnn.sum(
        ttnn_to_memory_config_44,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_31 = ttnn.multiply(
        ttnn_sum_9,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_28 = ttnn.add(
        ttnn_multiply_31,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_45 = ttnn.to_memory_config(
        ttnn_add_28,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_to_memory_config_45,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_250 = ttnn.reshape(
        ttnn_rsqrt_4,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_26, False)
    # ttnn.deallocate(ttnn_to_memory_config_41, False)
    ttnn.deallocate(ttnn_sum_8, False)
    ttnn.deallocate(ttnn_multiply_29, False)
    ttnn.deallocate(ttnn_to_memory_config_42, False)
    ttnn.deallocate(ttnn_reshape_249, False)
    ttnn.deallocate(ttnn_neg_4, False)
    ttnn.deallocate(ttnn_add_27, False)
    # ttnn.deallocate(ttnn_to_memory_config_43, False)
    ttnn.deallocate(ttnn_multiply_30, False)
    ttnn.deallocate(ttnn_to_memory_config_44, False)
    ttnn.deallocate(ttnn_sum_9, False)
    ttnn.deallocate(ttnn_multiply_31, False)
    ttnn.deallocate(ttnn_add_28, False)
    ttnn.deallocate(ttnn_to_memory_config_45, False)
    ttnn.deallocate(ttnn_rsqrt_4, False)
    # ttnn.deallocate(ttnn_reshape_250, False)
    return ttnn_to_memory_config_43, ttnn_to_memory_config_41, ttnn_reshape_250


def Linear_107_0(input_0, input_1, input_2):
    ttnn_matmul_11 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_29 = ttnn.add(
        ttnn_matmul_11,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_46 = ttnn.to_memory_config(
        ttnn_add_29,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn.deallocate(ttnn_add_29, False)
    return ttnn_to_memory_config_46


def Linear_83_0(input_0, input_1, input_2):
    ttnn_matmul_12 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_30 = ttnn.add(
        ttnn_matmul_12,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_47 = ttnn.to_memory_config(
        ttnn_add_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn.deallocate(ttnn_add_30, False)
    return ttnn_to_memory_config_47


def Linear_17_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_13 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_48 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_31 = ttnn.add(
        ttnn_matmul_13,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_49 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_to_memory_config_48,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_50 = ttnn.to_memory_config(
        ttnn_add_31,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_to_memory_config_49,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_32 = ttnn.add(
        ttnn_matmul_14,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_33 = ttnn.add(
        ttnn_matmul_15,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_251 = ttnn.reshape(
        ttnn_to_memory_config_50,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_51 = ttnn.to_memory_config(
        ttnn_add_32,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_52 = ttnn.to_memory_config(
        ttnn_add_33,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_56 = ttnn.permute(
        ttnn_reshape_251,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_252 = ttnn.reshape(
        ttnn_to_memory_config_51,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_253 = ttnn.reshape(
        ttnn_to_memory_config_52,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_254 = ttnn.reshape(
        ttnn_permute_56,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_57 = ttnn.permute(
        ttnn_reshape_252,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_58 = ttnn.permute(
        ttnn_reshape_253,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_255 = ttnn.reshape(
        ttnn_permute_57,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_256 = ttnn.reshape(
        ttnn_permute_58,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn.deallocate(ttnn_add_31, False)
    ttnn.deallocate(ttnn_to_memory_config_48, False)
    ttnn.deallocate(ttnn_to_memory_config_49, False)
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn.deallocate(ttnn_to_memory_config_50, False)
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn.deallocate(ttnn_reshape_251, False)
    ttnn.deallocate(ttnn_add_33, False)
    ttnn.deallocate(ttnn_add_32, False)
    ttnn.deallocate(ttnn_to_memory_config_51, False)
    ttnn.deallocate(ttnn_permute_56, False)
    ttnn.deallocate(ttnn_to_memory_config_52, False)
    ttnn.deallocate(ttnn_reshape_252, False)
    ttnn.deallocate(ttnn_reshape_253, False)
    ttnn.deallocate(ttnn_permute_57, False)
    ttnn.deallocate(ttnn_permute_58, False)
    return ttnn_reshape_255, ttnn_reshape_254, ttnn_reshape_256


def Linear_101_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_16 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_53 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_34 = ttnn.add(
        ttnn_matmul_16,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_54 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_to_memory_config_53,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_55 = ttnn.to_memory_config(
        ttnn_add_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_18 = ttnn.matmul(
        ttnn_to_memory_config_54,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_35 = ttnn.add(
        ttnn_matmul_17,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_36 = ttnn.add(
        ttnn_matmul_18,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_257 = ttnn.reshape(
        ttnn_to_memory_config_55,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_56 = ttnn.to_memory_config(
        ttnn_add_35,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_57 = ttnn.to_memory_config(
        ttnn_add_36,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_59 = ttnn.permute(
        ttnn_reshape_257,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_258 = ttnn.reshape(
        ttnn_to_memory_config_56,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_259 = ttnn.reshape(
        ttnn_to_memory_config_57,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_260 = ttnn.reshape(
        ttnn_permute_59,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_60 = ttnn.permute(
        ttnn_reshape_258,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_61 = ttnn.permute(
        ttnn_reshape_259,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_261 = ttnn.reshape(
        ttnn_permute_60,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_262 = ttnn.reshape(
        ttnn_permute_61,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_16, False)
    ttnn.deallocate(ttnn_add_34, False)
    ttnn.deallocate(ttnn_to_memory_config_53, False)
    ttnn.deallocate(ttnn_to_memory_config_54, False)
    ttnn.deallocate(ttnn_matmul_18, False)
    ttnn.deallocate(ttnn_to_memory_config_55, False)
    ttnn.deallocate(ttnn_matmul_17, False)
    ttnn.deallocate(ttnn_reshape_257, False)
    ttnn.deallocate(ttnn_add_36, False)
    ttnn.deallocate(ttnn_add_35, False)
    ttnn.deallocate(ttnn_to_memory_config_56, False)
    ttnn.deallocate(ttnn_permute_59, False)
    ttnn.deallocate(ttnn_to_memory_config_57, False)
    ttnn.deallocate(ttnn_reshape_258, False)
    ttnn.deallocate(ttnn_reshape_259, False)
    ttnn.deallocate(ttnn_permute_60, False)
    ttnn.deallocate(ttnn_permute_61, False)
    return ttnn_reshape_261, ttnn_reshape_260, ttnn_reshape_262


def CLIPVisionEmbeddings_0_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_permute_62 = ttnn.permute(
        input_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_263 = ttnn.reshape(
        ttnn_permute_62,
        [1, 1, 100352, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_reshape_263,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_0,
        weight_tensor=input_4,
        device=input_2,
        in_channels=3,
        out_channels=768,
        batch_size=2,
        input_height=224,
        input_width=224,
        kernel_size=[32, 32],
        stride=[32, 32],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=32,
            act_block_w_div=1,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            transpose_shards=False,
            output_layout=ttnn.Layout.TILE,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            in_place=False,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_264 = ttnn.reshape(
        ttnn_conv2d_0,
        [2, 7, 7, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_63 = ttnn.permute(
        ttnn_reshape_264,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_265 = ttnn.reshape(
        ttnn_permute_63,
        [2, 768, 49],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_261 = [input_1, ttnn_reshape_265]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_261,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_37 = ttnn.add(
        ttnn_concat_0,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_58 = ttnn.to_memory_config(
        ttnn_add_37,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_64 = ttnn.permute(
        ttnn_to_memory_config_58,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_permute_62, False)
    ttnn.deallocate(ttnn_reshape_263, False)
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn.deallocate(ttnn_reshape_264, False)
    ttnn.deallocate(ttnn_permute_63, False)
    ttnn.deallocate(ttnn_reshape_265, False)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn.deallocate(ttnn_add_37, False)
    ttnn.deallocate(ttnn_to_memory_config_58, False)
    return ttnn_permute_64


def LayerNorm_106_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_32 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_33 = ttnn.multiply(
        ttnn_multiply_32,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_38 = ttnn.add(
        ttnn_multiply_33,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_32, False)
    ttnn.deallocate(ttnn_multiply_33, False)
    return ttnn_add_38


def Linear_73_0(input_0, input_1, input_2):
    ttnn_matmul_19 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_39 = ttnn.add(
        ttnn_matmul_19,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_59 = ttnn.to_memory_config(
        ttnn_add_39,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_266 = ttnn.reshape(
        ttnn_to_memory_config_59,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_19, False)
    ttnn.deallocate(ttnn_add_39, False)
    ttnn.deallocate(ttnn_to_memory_config_59, False)
    return ttnn_reshape_266


def Linear_121_0(input_0, input_1, input_2):
    ttnn_matmul_20 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_40 = ttnn.add(
        ttnn_matmul_20,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_60 = ttnn.to_memory_config(
        ttnn_add_40,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_267 = ttnn.reshape(
        ttnn_to_memory_config_60,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_20, False)
    ttnn.deallocate(ttnn_add_40, False)
    ttnn.deallocate(ttnn_to_memory_config_60, False)
    return ttnn_reshape_267


def Linear_105_0(input):
    ttnn_reshape_268 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_268


def Linear_133_0(input_0, input_1, input_2):
    ttnn_matmul_21 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_41 = ttnn.add(
        ttnn_matmul_21,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_61 = ttnn.to_memory_config(
        ttnn_add_41,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_269 = ttnn.reshape(
        ttnn_to_memory_config_61,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_21, False)
    ttnn.deallocate(ttnn_add_41, False)
    ttnn.deallocate(ttnn_to_memory_config_61, False)
    return ttnn_reshape_269


def Linear_131_0(input_0, input_1, input_2):
    ttnn_matmul_22 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_42 = ttnn.add(
        ttnn_matmul_22,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_62 = ttnn.to_memory_config(
        ttnn_add_42,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_22, False)
    ttnn.deallocate(ttnn_add_42, False)
    return ttnn_to_memory_config_62


def Linear_7_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_25 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_270 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_25,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_63 = ttnn.to_memory_config(
        ttnn_reshape_270,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_23 = ttnn.matmul(
        ttnn_to_memory_config_63,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_43 = ttnn.add(
        ttnn_matmul_23,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_64 = ttnn.to_memory_config(
        ttnn_add_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_271 = ttnn.reshape(
        ttnn_to_memory_config_64,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_25, False)
    ttnn.deallocate(ttnn_reshape_270, False)
    ttnn.deallocate(ttnn_to_memory_config_63, False)
    ttnn.deallocate(ttnn_matmul_23, False)
    ttnn.deallocate(ttnn_add_43, False)
    ttnn.deallocate(ttnn_to_memory_config_64, False)
    return ttnn_reshape_271


def QuickGELUActivation_96_0(input_0, input_1):
    ttnn_multiply_34 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_65 = ttnn.to_memory_config(
        ttnn_multiply_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_2 = ttnn.sigmoid(
        ttnn_to_memory_config_65,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_35 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_34, False)
    ttnn.deallocate(ttnn_to_memory_config_65, False)
    ttnn.deallocate(ttnn_sigmoid_2, False)
    return ttnn_multiply_35


def QuickGELUActivation_132_0(input_0, input_1):
    ttnn_multiply_36 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_66 = ttnn.to_memory_config(
        ttnn_multiply_36,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_3 = ttnn.sigmoid(
        ttnn_to_memory_config_66,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_37 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_36, False)
    ttnn.deallocate(ttnn_to_memory_config_66, False)
    ttnn.deallocate(ttnn_sigmoid_3, False)
    return ttnn_multiply_37


def QuickGELUActivation_48_0(input_0, input_1):
    ttnn_multiply_38 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_67 = ttnn.to_memory_config(
        ttnn_multiply_38,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_4 = ttnn.sigmoid(
        ttnn_to_memory_config_67,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_39 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_38, False)
    ttnn.deallocate(ttnn_to_memory_config_67, False)
    ttnn.deallocate(ttnn_sigmoid_4, False)
    return ttnn_multiply_39


def CLIPEncoderLayer_80_0(input_0, input_1, input_2, input_3):
    ttnn_add_44 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_68 = ttnn.to_memory_config(
        ttnn_add_44,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_10 = ttnn.sum(
        ttnn_to_memory_config_68,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_40 = ttnn.multiply(
        ttnn_sum_10,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_69 = ttnn.to_memory_config(
        ttnn_multiply_40,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_272 = ttnn.reshape(
        ttnn_to_memory_config_69,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_5 = ttnn.neg(
        ttnn_reshape_272,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_45 = ttnn.add(
        ttnn_to_memory_config_68,
        ttnn_neg_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_70 = ttnn.to_memory_config(
        ttnn_add_45,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_41 = ttnn.multiply(
        ttnn_to_memory_config_70,
        ttnn_to_memory_config_70,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_71 = ttnn.to_memory_config(
        ttnn_multiply_41,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_11 = ttnn.sum(
        ttnn_to_memory_config_71,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_42 = ttnn.multiply(
        ttnn_sum_11,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_46 = ttnn.add(
        ttnn_multiply_42,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_72 = ttnn.to_memory_config(
        ttnn_add_46,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_5 = ttnn.rsqrt(
        ttnn_to_memory_config_72,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_273 = ttnn.reshape(
        ttnn_rsqrt_5,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_44, False)
    # ttnn.deallocate(ttnn_to_memory_config_68, False)
    ttnn.deallocate(ttnn_sum_10, False)
    ttnn.deallocate(ttnn_multiply_40, False)
    ttnn.deallocate(ttnn_to_memory_config_69, False)
    ttnn.deallocate(ttnn_reshape_272, False)
    ttnn.deallocate(ttnn_neg_5, False)
    ttnn.deallocate(ttnn_add_45, False)
    # ttnn.deallocate(ttnn_to_memory_config_70, False)
    ttnn.deallocate(ttnn_multiply_41, False)
    ttnn.deallocate(ttnn_to_memory_config_71, False)
    ttnn.deallocate(ttnn_sum_11, False)
    ttnn.deallocate(ttnn_multiply_42, False)
    ttnn.deallocate(ttnn_add_46, False)
    ttnn.deallocate(ttnn_to_memory_config_72, False)
    ttnn.deallocate(ttnn_rsqrt_5, False)
    # ttnn.deallocate(ttnn_reshape_273, False)
    return ttnn_to_memory_config_68, ttnn_to_memory_config_70, ttnn_reshape_273


def QuickGELUActivation_84_0(input_0, input_1):
    ttnn_multiply_43 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_73 = ttnn.to_memory_config(
        ttnn_multiply_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_5 = ttnn.sigmoid(
        ttnn_to_memory_config_73,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_44 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_43, False)
    ttnn.deallocate(ttnn_to_memory_config_73, False)
    ttnn.deallocate(ttnn_sigmoid_5, False)
    return ttnn_multiply_44


def LayerNorm_58_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_45 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_46 = ttnn.multiply(
        ttnn_multiply_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_47 = ttnn.add(
        ttnn_multiply_46,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_45, False)
    ttnn.deallocate(ttnn_multiply_46, False)
    return ttnn_add_47


def CLIPEncoderLayer_26_0(input_0, input_1, input_2, input_3):
    ttnn_add_48 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_74 = ttnn.to_memory_config(
        ttnn_add_48,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_12 = ttnn.sum(
        ttnn_to_memory_config_74,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_47 = ttnn.multiply(
        ttnn_sum_12,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_75 = ttnn.to_memory_config(
        ttnn_multiply_47,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_274 = ttnn.reshape(
        ttnn_to_memory_config_75,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_6 = ttnn.neg(
        ttnn_reshape_274,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_49 = ttnn.add(
        ttnn_to_memory_config_74,
        ttnn_neg_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_76 = ttnn.to_memory_config(
        ttnn_add_49,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_48 = ttnn.multiply(
        ttnn_to_memory_config_76,
        ttnn_to_memory_config_76,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_77 = ttnn.to_memory_config(
        ttnn_multiply_48,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_13 = ttnn.sum(
        ttnn_to_memory_config_77,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_49 = ttnn.multiply(
        ttnn_sum_13,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_50 = ttnn.add(
        ttnn_multiply_49,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_78 = ttnn.to_memory_config(
        ttnn_add_50,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_6 = ttnn.rsqrt(
        ttnn_to_memory_config_78,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_275 = ttnn.reshape(
        ttnn_rsqrt_6,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_48, False)
    # ttnn.deallocate(ttnn_to_memory_config_74, False)
    ttnn.deallocate(ttnn_sum_12, False)
    ttnn.deallocate(ttnn_multiply_47, False)
    ttnn.deallocate(ttnn_to_memory_config_75, False)
    ttnn.deallocate(ttnn_reshape_274, False)
    ttnn.deallocate(ttnn_neg_6, False)
    ttnn.deallocate(ttnn_add_49, False)
    # ttnn.deallocate(ttnn_to_memory_config_76, False)
    ttnn.deallocate(ttnn_multiply_48, False)
    ttnn.deallocate(ttnn_to_memory_config_77, False)
    ttnn.deallocate(ttnn_sum_13, False)
    ttnn.deallocate(ttnn_multiply_49, False)
    ttnn.deallocate(ttnn_add_50, False)
    ttnn.deallocate(ttnn_to_memory_config_78, False)
    ttnn.deallocate(ttnn_rsqrt_6, False)
    # ttnn.deallocate(ttnn_reshape_275, False)
    return ttnn_to_memory_config_76, ttnn_to_memory_config_74, ttnn_reshape_275


def Linear_67_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_26 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_276 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_26,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_79 = ttnn.to_memory_config(
        ttnn_reshape_276,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_24 = ttnn.matmul(
        ttnn_to_memory_config_79,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_51 = ttnn.add(
        ttnn_matmul_24,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_80 = ttnn.to_memory_config(
        ttnn_add_51,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_277 = ttnn.reshape(
        ttnn_to_memory_config_80,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_26, False)
    ttnn.deallocate(ttnn_reshape_276, False)
    ttnn.deallocate(ttnn_to_memory_config_79, False)
    ttnn.deallocate(ttnn_matmul_24, False)
    ttnn.deallocate(ttnn_add_51, False)
    ttnn.deallocate(ttnn_to_memory_config_80, False)
    return ttnn_reshape_277


def LayerNorm_40_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_50 = ttnn.multiply(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_51 = ttnn.multiply(
        ttnn_multiply_50,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_52 = ttnn.add(
        ttnn_multiply_51,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_81 = ttnn.to_memory_config(
        ttnn_add_52,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_50, False)
    ttnn.deallocate(ttnn_multiply_51, False)
    return ttnn_add_52, ttnn_to_memory_config_81


def CLIPEncoderLayer_32_0(input_0, input_1, input_2, input_3):
    ttnn_add_53 = ttnn.add(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_82 = ttnn.to_memory_config(
        ttnn_add_53,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_14 = ttnn.sum(
        ttnn_to_memory_config_82,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_52 = ttnn.multiply(
        ttnn_sum_14,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_83 = ttnn.to_memory_config(
        ttnn_multiply_52,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_278 = ttnn.reshape(
        ttnn_to_memory_config_83,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_7 = ttnn.neg(
        ttnn_reshape_278,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_54 = ttnn.add(
        ttnn_to_memory_config_82,
        ttnn_neg_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_84 = ttnn.to_memory_config(
        ttnn_add_54,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_53 = ttnn.multiply(
        ttnn_to_memory_config_84,
        ttnn_to_memory_config_84,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_85 = ttnn.to_memory_config(
        ttnn_multiply_53,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_15 = ttnn.sum(
        ttnn_to_memory_config_85,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_sum_15,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_55 = ttnn.add(
        ttnn_multiply_54,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_86 = ttnn.to_memory_config(
        ttnn_add_55,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_7 = ttnn.rsqrt(
        ttnn_to_memory_config_86,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_279 = ttnn.reshape(
        ttnn_rsqrt_7,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_53, False)
    # ttnn.deallocate(ttnn_to_memory_config_82, False)
    ttnn.deallocate(ttnn_sum_14, False)
    ttnn.deallocate(ttnn_multiply_52, False)
    ttnn.deallocate(ttnn_to_memory_config_83, False)
    ttnn.deallocate(ttnn_reshape_278, False)
    ttnn.deallocate(ttnn_neg_7, False)
    ttnn.deallocate(ttnn_add_54, False)
    # ttnn.deallocate(ttnn_to_memory_config_84, False)
    ttnn.deallocate(ttnn_multiply_53, False)
    ttnn.deallocate(ttnn_to_memory_config_85, False)
    ttnn.deallocate(ttnn_sum_15, False)
    ttnn.deallocate(ttnn_multiply_54, False)
    ttnn.deallocate(ttnn_add_55, False)
    ttnn.deallocate(ttnn_to_memory_config_86, False)
    ttnn.deallocate(ttnn_rsqrt_7, False)
    # ttnn.deallocate(ttnn_reshape_279, False)
    return ttnn_reshape_279, ttnn_to_memory_config_82, ttnn_to_memory_config_84


def Linear_19_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_27 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_280 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_27,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_87 = ttnn.to_memory_config(
        ttnn_reshape_280,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_to_memory_config_87,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_56 = ttnn.add(
        ttnn_matmul_25,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_88 = ttnn.to_memory_config(
        ttnn_add_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_281 = ttnn.reshape(
        ttnn_to_memory_config_88,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_27, False)
    ttnn.deallocate(ttnn_reshape_280, False)
    ttnn.deallocate(ttnn_to_memory_config_87, False)
    ttnn.deallocate(ttnn_matmul_25, False)
    ttnn.deallocate(ttnn_add_56, False)
    ttnn.deallocate(ttnn_to_memory_config_88, False)
    return ttnn_reshape_281


def CLIPEncoderLayer_50_0(input_0, input_1, input_2, input_3):
    ttnn_add_57 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_89 = ttnn.to_memory_config(
        ttnn_add_57,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_16 = ttnn.sum(
        ttnn_to_memory_config_89,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_sum_16,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_90 = ttnn.to_memory_config(
        ttnn_multiply_55,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_282 = ttnn.reshape(
        ttnn_to_memory_config_90,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_8 = ttnn.neg(
        ttnn_reshape_282,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_58 = ttnn.add(
        ttnn_to_memory_config_89,
        ttnn_neg_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_91 = ttnn.to_memory_config(
        ttnn_add_58,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_56 = ttnn.multiply(
        ttnn_to_memory_config_91,
        ttnn_to_memory_config_91,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_92 = ttnn.to_memory_config(
        ttnn_multiply_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_17 = ttnn.sum(
        ttnn_to_memory_config_92,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_57 = ttnn.multiply(
        ttnn_sum_17,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_59 = ttnn.add(
        ttnn_multiply_57,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_93 = ttnn.to_memory_config(
        ttnn_add_59,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_8 = ttnn.rsqrt(
        ttnn_to_memory_config_93,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_283 = ttnn.reshape(
        ttnn_rsqrt_8,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_57, False)
    # ttnn.deallocate(ttnn_to_memory_config_89, False)
    ttnn.deallocate(ttnn_sum_16, False)
    ttnn.deallocate(ttnn_multiply_55, False)
    ttnn.deallocate(ttnn_to_memory_config_90, False)
    ttnn.deallocate(ttnn_reshape_282, False)
    ttnn.deallocate(ttnn_neg_8, False)
    ttnn.deallocate(ttnn_add_58, False)
    # ttnn.deallocate(ttnn_to_memory_config_91, False)
    ttnn.deallocate(ttnn_multiply_56, False)
    ttnn.deallocate(ttnn_to_memory_config_92, False)
    ttnn.deallocate(ttnn_sum_17, False)
    ttnn.deallocate(ttnn_multiply_57, False)
    ttnn.deallocate(ttnn_add_59, False)
    ttnn.deallocate(ttnn_to_memory_config_93, False)
    ttnn.deallocate(ttnn_rsqrt_8, False)
    # ttnn.deallocate(ttnn_reshape_283, False)
    return ttnn_to_memory_config_91, ttnn_reshape_283, ttnn_to_memory_config_89


def Linear_29_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_26 = ttnn.matmul(
        input_7,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_94 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_60 = ttnn.add(
        ttnn_matmul_26,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_95 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_to_memory_config_94,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_96 = ttnn.to_memory_config(
        ttnn_add_60,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_28 = ttnn.matmul(
        ttnn_to_memory_config_95,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_61 = ttnn.add(
        ttnn_matmul_27,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_62 = ttnn.add(
        ttnn_matmul_28,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_284 = ttnn.reshape(
        ttnn_to_memory_config_96,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_97 = ttnn.to_memory_config(
        ttnn_add_61,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_98 = ttnn.to_memory_config(
        ttnn_add_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_65 = ttnn.permute(
        ttnn_reshape_284,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_285 = ttnn.reshape(
        ttnn_to_memory_config_97,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_286 = ttnn.reshape(
        ttnn_to_memory_config_98,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_287 = ttnn.reshape(
        ttnn_permute_65,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_66 = ttnn.permute(
        ttnn_reshape_285,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_67 = ttnn.permute(
        ttnn_reshape_286,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_288 = ttnn.reshape(
        ttnn_permute_66,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_289 = ttnn.reshape(
        ttnn_permute_67,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_7, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_26, False)
    ttnn.deallocate(ttnn_add_60, False)
    ttnn.deallocate(ttnn_to_memory_config_94, False)
    ttnn.deallocate(ttnn_to_memory_config_95, False)
    ttnn.deallocate(ttnn_matmul_28, False)
    ttnn.deallocate(ttnn_to_memory_config_96, False)
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn.deallocate(ttnn_reshape_284, False)
    ttnn.deallocate(ttnn_add_62, False)
    ttnn.deallocate(ttnn_add_61, False)
    ttnn.deallocate(ttnn_to_memory_config_97, False)
    ttnn.deallocate(ttnn_permute_65, False)
    ttnn.deallocate(ttnn_to_memory_config_98, False)
    ttnn.deallocate(ttnn_reshape_285, False)
    ttnn.deallocate(ttnn_reshape_286, False)
    ttnn.deallocate(ttnn_permute_66, False)
    ttnn.deallocate(ttnn_permute_67, False)
    return ttnn_reshape_288, ttnn_reshape_289, ttnn_reshape_287


def Linear_61_0(input_0, input_1, input_2):
    ttnn_matmul_29 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_63 = ttnn.add(
        ttnn_matmul_29,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_99 = ttnn.to_memory_config(
        ttnn_add_63,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_290 = ttnn.reshape(
        ttnn_to_memory_config_99,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_29, False)
    ttnn.deallocate(ttnn_add_63, False)
    ttnn.deallocate(ttnn_to_memory_config_99, False)
    return ttnn_reshape_290


def CLIPAttention_30_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_100 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_to_memory_config_100,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_101 = ttnn.to_memory_config(
        ttnn_matmul_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_291 = ttnn.reshape(
        ttnn_to_memory_config_101,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_58 = ttnn.multiply(
        ttnn_reshape_291,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_multiply_58,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_1 = ttnn.softmax(
        ttnn_typecast_3,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_softmax_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_292 = ttnn.reshape(
        ttnn_typecast_4,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_102 = ttnn.to_memory_config(
        ttnn_reshape_292,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_to_memory_config_102,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_103 = ttnn.to_memory_config(
        ttnn_matmul_31,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_293 = ttnn.reshape(
        ttnn_to_memory_config_103,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_100, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_101, False)
    ttnn.deallocate(ttnn_reshape_291, False)
    ttnn.deallocate(ttnn_multiply_58, False)
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn.deallocate(ttnn_softmax_1, False)
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn.deallocate(ttnn_reshape_292, False)
    ttnn.deallocate(ttnn_to_memory_config_102, False)
    ttnn.deallocate(ttnn_matmul_31, False)
    ttnn.deallocate(ttnn_to_memory_config_103, False)
    return ttnn_reshape_293


def Linear_41_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_32 = ttnn.matmul(
        input_0,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_104 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_64 = ttnn.add(
        ttnn_matmul_32,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_105 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_to_memory_config_104,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_106 = ttnn.to_memory_config(
        ttnn_add_64,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_34 = ttnn.matmul(
        ttnn_to_memory_config_105,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_65 = ttnn.add(
        ttnn_matmul_33,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_66 = ttnn.add(
        ttnn_matmul_34,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_294 = ttnn.reshape(
        ttnn_to_memory_config_106,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_107 = ttnn.to_memory_config(
        ttnn_add_65,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_108 = ttnn.to_memory_config(
        ttnn_add_66,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_68 = ttnn.permute(
        ttnn_reshape_294,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_295 = ttnn.reshape(
        ttnn_to_memory_config_107,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_296 = ttnn.reshape(
        ttnn_to_memory_config_108,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_297 = ttnn.reshape(
        ttnn_permute_68,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_69 = ttnn.permute(
        ttnn_reshape_295,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_70 = ttnn.permute(
        ttnn_reshape_296,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_298 = ttnn.reshape(
        ttnn_permute_69,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_299 = ttnn.reshape(
        ttnn_permute_70,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn.deallocate(ttnn_add_64, False)
    ttnn.deallocate(ttnn_to_memory_config_104, False)
    ttnn.deallocate(ttnn_to_memory_config_105, False)
    ttnn.deallocate(ttnn_matmul_34, False)
    ttnn.deallocate(ttnn_to_memory_config_106, False)
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn.deallocate(ttnn_reshape_294, False)
    ttnn.deallocate(ttnn_add_66, False)
    ttnn.deallocate(ttnn_add_65, False)
    ttnn.deallocate(ttnn_to_memory_config_107, False)
    ttnn.deallocate(ttnn_permute_68, False)
    ttnn.deallocate(ttnn_to_memory_config_108, False)
    ttnn.deallocate(ttnn_reshape_295, False)
    ttnn.deallocate(ttnn_reshape_296, False)
    ttnn.deallocate(ttnn_permute_69, False)
    ttnn.deallocate(ttnn_permute_70, False)
    return ttnn_reshape_297, ttnn_reshape_298, ttnn_reshape_299


def Linear_97_0(input_0, input_1, input_2):
    ttnn_matmul_35 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_67 = ttnn.add(
        ttnn_matmul_35,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_109 = ttnn.to_memory_config(
        ttnn_add_67,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_300 = ttnn.reshape(
        ttnn_to_memory_config_109,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_35, False)
    ttnn.deallocate(ttnn_add_67, False)
    ttnn.deallocate(ttnn_to_memory_config_109, False)
    return ttnn_reshape_300


def CLIPEncoderLayer_44_0(input_0, input_1, input_2, input_3):
    ttnn_add_68 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_110 = ttnn.to_memory_config(
        ttnn_add_68,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_18 = ttnn.sum(
        ttnn_to_memory_config_110,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_sum_18,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_111 = ttnn.to_memory_config(
        ttnn_multiply_59,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_301 = ttnn.reshape(
        ttnn_to_memory_config_111,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_9 = ttnn.neg(
        ttnn_reshape_301,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_69 = ttnn.add(
        ttnn_to_memory_config_110,
        ttnn_neg_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_112 = ttnn.to_memory_config(
        ttnn_add_69,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_60 = ttnn.multiply(
        ttnn_to_memory_config_112,
        ttnn_to_memory_config_112,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_113 = ttnn.to_memory_config(
        ttnn_multiply_60,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_19 = ttnn.sum(
        ttnn_to_memory_config_113,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_61 = ttnn.multiply(
        ttnn_sum_19,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_70 = ttnn.add(
        ttnn_multiply_61,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_114 = ttnn.to_memory_config(
        ttnn_add_70,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_9 = ttnn.rsqrt(
        ttnn_to_memory_config_114,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_302 = ttnn.reshape(
        ttnn_rsqrt_9,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_68, False)
    # ttnn.deallocate(ttnn_to_memory_config_110, False)
    ttnn.deallocate(ttnn_sum_18, False)
    ttnn.deallocate(ttnn_multiply_59, False)
    ttnn.deallocate(ttnn_to_memory_config_111, False)
    ttnn.deallocate(ttnn_reshape_301, False)
    ttnn.deallocate(ttnn_neg_9, False)
    ttnn.deallocate(ttnn_add_69, False)
    # ttnn.deallocate(ttnn_to_memory_config_112, False)
    ttnn.deallocate(ttnn_multiply_60, False)
    ttnn.deallocate(ttnn_to_memory_config_113, False)
    ttnn.deallocate(ttnn_sum_19, False)
    ttnn.deallocate(ttnn_multiply_61, False)
    ttnn.deallocate(ttnn_add_70, False)
    ttnn.deallocate(ttnn_to_memory_config_114, False)
    ttnn.deallocate(ttnn_rsqrt_9, False)
    # ttnn.deallocate(ttnn_reshape_302, False)
    return ttnn_reshape_302, ttnn_to_memory_config_112, ttnn_to_memory_config_110


def CLIPEncoderLayer_68_0(input_0, input_1, input_2, input_3):
    ttnn_add_71 = ttnn.add(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_115 = ttnn.to_memory_config(
        ttnn_add_71,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_20 = ttnn.sum(
        ttnn_to_memory_config_115,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_62 = ttnn.multiply(
        ttnn_sum_20,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_116 = ttnn.to_memory_config(
        ttnn_multiply_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_303 = ttnn.reshape(
        ttnn_to_memory_config_116,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_10 = ttnn.neg(
        ttnn_reshape_303,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_72 = ttnn.add(
        ttnn_to_memory_config_115,
        ttnn_neg_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_117 = ttnn.to_memory_config(
        ttnn_add_72,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_63 = ttnn.multiply(
        ttnn_to_memory_config_117,
        ttnn_to_memory_config_117,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_118 = ttnn.to_memory_config(
        ttnn_multiply_63,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_21 = ttnn.sum(
        ttnn_to_memory_config_118,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_64 = ttnn.multiply(
        ttnn_sum_21,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_73 = ttnn.add(
        ttnn_multiply_64,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_119 = ttnn.to_memory_config(
        ttnn_add_73,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_10 = ttnn.rsqrt(
        ttnn_to_memory_config_119,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_304 = ttnn.reshape(
        ttnn_rsqrt_10,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_71, False)
    # ttnn.deallocate(ttnn_to_memory_config_115, False)
    ttnn.deallocate(ttnn_sum_20, False)
    ttnn.deallocate(ttnn_multiply_62, False)
    ttnn.deallocate(ttnn_to_memory_config_116, False)
    ttnn.deallocate(ttnn_reshape_303, False)
    ttnn.deallocate(ttnn_neg_10, False)
    ttnn.deallocate(ttnn_add_72, False)
    # ttnn.deallocate(ttnn_to_memory_config_117, False)
    ttnn.deallocate(ttnn_multiply_63, False)
    ttnn.deallocate(ttnn_to_memory_config_118, False)
    ttnn.deallocate(ttnn_sum_21, False)
    ttnn.deallocate(ttnn_multiply_64, False)
    ttnn.deallocate(ttnn_add_73, False)
    ttnn.deallocate(ttnn_to_memory_config_119, False)
    ttnn.deallocate(ttnn_rsqrt_10, False)
    # ttnn.deallocate(ttnn_reshape_304, False)
    return ttnn_reshape_304, ttnn_to_memory_config_115, ttnn_to_memory_config_117


def CLIPEncoderLayer_116_0(input_0, input_1, input_2, input_3):
    ttnn_add_74 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_120 = ttnn.to_memory_config(
        ttnn_add_74,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_22 = ttnn.sum(
        ttnn_to_memory_config_120,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_65 = ttnn.multiply(
        ttnn_sum_22,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_121 = ttnn.to_memory_config(
        ttnn_multiply_65,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_305 = ttnn.reshape(
        ttnn_to_memory_config_121,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_11 = ttnn.neg(
        ttnn_reshape_305,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_75 = ttnn.add(
        ttnn_to_memory_config_120,
        ttnn_neg_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_122 = ttnn.to_memory_config(
        ttnn_add_75,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_66 = ttnn.multiply(
        ttnn_to_memory_config_122,
        ttnn_to_memory_config_122,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_123 = ttnn.to_memory_config(
        ttnn_multiply_66,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_23 = ttnn.sum(
        ttnn_to_memory_config_123,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_67 = ttnn.multiply(
        ttnn_sum_23,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_76 = ttnn.add(
        ttnn_multiply_67,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_124 = ttnn.to_memory_config(
        ttnn_add_76,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_11 = ttnn.rsqrt(
        ttnn_to_memory_config_124,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_306 = ttnn.reshape(
        ttnn_rsqrt_11,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_74, False)
    # ttnn.deallocate(ttnn_to_memory_config_120, False)
    ttnn.deallocate(ttnn_sum_22, False)
    ttnn.deallocate(ttnn_multiply_65, False)
    ttnn.deallocate(ttnn_to_memory_config_121, False)
    ttnn.deallocate(ttnn_reshape_305, False)
    ttnn.deallocate(ttnn_neg_11, False)
    ttnn.deallocate(ttnn_add_75, False)
    # ttnn.deallocate(ttnn_to_memory_config_122, False)
    ttnn.deallocate(ttnn_multiply_66, False)
    ttnn.deallocate(ttnn_to_memory_config_123, False)
    ttnn.deallocate(ttnn_sum_23, False)
    ttnn.deallocate(ttnn_multiply_67, False)
    ttnn.deallocate(ttnn_add_76, False)
    ttnn.deallocate(ttnn_to_memory_config_124, False)
    ttnn.deallocate(ttnn_rsqrt_11, False)
    # ttnn.deallocate(ttnn_reshape_306, False)
    return ttnn_to_memory_config_120, ttnn_to_memory_config_122, ttnn_reshape_306


def Linear_5_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_36 = ttnn.matmul(
        input_3,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_125 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_77 = ttnn.add(
        ttnn_matmul_36,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_126 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_to_memory_config_125,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_127 = ttnn.to_memory_config(
        ttnn_add_77,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_38 = ttnn.matmul(
        ttnn_to_memory_config_126,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_78 = ttnn.add(
        ttnn_matmul_37,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_79 = ttnn.add(
        ttnn_matmul_38,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_307 = ttnn.reshape(
        ttnn_to_memory_config_127,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_128 = ttnn.to_memory_config(
        ttnn_add_78,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_129 = ttnn.to_memory_config(
        ttnn_add_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_71 = ttnn.permute(
        ttnn_reshape_307,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_308 = ttnn.reshape(
        ttnn_to_memory_config_128,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_309 = ttnn.reshape(
        ttnn_to_memory_config_129,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_310 = ttnn.reshape(
        ttnn_permute_71,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_72 = ttnn.permute(
        ttnn_reshape_308,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_73 = ttnn.permute(
        ttnn_reshape_309,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_311 = ttnn.reshape(
        ttnn_permute_72,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_312 = ttnn.reshape(
        ttnn_permute_73,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_36, False)
    ttnn.deallocate(ttnn_add_77, False)
    ttnn.deallocate(ttnn_to_memory_config_125, False)
    ttnn.deallocate(ttnn_to_memory_config_126, False)
    ttnn.deallocate(ttnn_matmul_38, False)
    ttnn.deallocate(ttnn_to_memory_config_127, False)
    ttnn.deallocate(ttnn_matmul_37, False)
    ttnn.deallocate(ttnn_reshape_307, False)
    ttnn.deallocate(ttnn_add_79, False)
    ttnn.deallocate(ttnn_add_78, False)
    ttnn.deallocate(ttnn_to_memory_config_128, False)
    ttnn.deallocate(ttnn_permute_71, False)
    ttnn.deallocate(ttnn_to_memory_config_129, False)
    ttnn.deallocate(ttnn_reshape_308, False)
    ttnn.deallocate(ttnn_reshape_309, False)
    ttnn.deallocate(ttnn_permute_72, False)
    ttnn.deallocate(ttnn_permute_73, False)
    return ttnn_reshape_310, ttnn_reshape_312, ttnn_reshape_311


def LayerNorm_10_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_68 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_69 = ttnn.multiply(
        ttnn_multiply_68,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_80 = ttnn.add(
        ttnn_multiply_69,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_68, False)
    ttnn.deallocate(ttnn_multiply_69, False)
    return ttnn_add_80


def Linear_93_0(input):
    ttnn_reshape_313 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_313


def Linear_115_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_28 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_314 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_28,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_130 = ttnn.to_memory_config(
        ttnn_reshape_314,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_39 = ttnn.matmul(
        ttnn_to_memory_config_130,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_81 = ttnn.add(
        ttnn_matmul_39,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_131 = ttnn.to_memory_config(
        ttnn_add_81,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_315 = ttnn.reshape(
        ttnn_to_memory_config_131,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_28, False)
    ttnn.deallocate(ttnn_reshape_314, False)
    ttnn.deallocate(ttnn_to_memory_config_130, False)
    ttnn.deallocate(ttnn_matmul_39, False)
    ttnn.deallocate(ttnn_add_81, False)
    ttnn.deallocate(ttnn_to_memory_config_131, False)
    return ttnn_reshape_315


def CLIPEncoderLayer_110_0(input_0, input_1, input_2, input_3):
    ttnn_add_82 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_132 = ttnn.to_memory_config(
        ttnn_add_82,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_24 = ttnn.sum(
        ttnn_to_memory_config_132,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_70 = ttnn.multiply(
        ttnn_sum_24,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_133 = ttnn.to_memory_config(
        ttnn_multiply_70,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_316 = ttnn.reshape(
        ttnn_to_memory_config_133,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_12 = ttnn.neg(
        ttnn_reshape_316,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_83 = ttnn.add(
        ttnn_to_memory_config_132,
        ttnn_neg_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_134 = ttnn.to_memory_config(
        ttnn_add_83,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_71 = ttnn.multiply(
        ttnn_to_memory_config_134,
        ttnn_to_memory_config_134,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_135 = ttnn.to_memory_config(
        ttnn_multiply_71,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_25 = ttnn.sum(
        ttnn_to_memory_config_135,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_72 = ttnn.multiply(
        ttnn_sum_25,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_84 = ttnn.add(
        ttnn_multiply_72,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_136 = ttnn.to_memory_config(
        ttnn_add_84,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_12 = ttnn.rsqrt(
        ttnn_to_memory_config_136,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_317 = ttnn.reshape(
        ttnn_rsqrt_12,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_82, False)
    # ttnn.deallocate(ttnn_to_memory_config_132, False)
    ttnn.deallocate(ttnn_sum_24, False)
    ttnn.deallocate(ttnn_multiply_70, False)
    ttnn.deallocate(ttnn_to_memory_config_133, False)
    ttnn.deallocate(ttnn_reshape_316, False)
    ttnn.deallocate(ttnn_neg_12, False)
    ttnn.deallocate(ttnn_add_83, False)
    # ttnn.deallocate(ttnn_to_memory_config_134, False)
    ttnn.deallocate(ttnn_multiply_71, False)
    ttnn.deallocate(ttnn_to_memory_config_135, False)
    ttnn.deallocate(ttnn_sum_25, False)
    ttnn.deallocate(ttnn_multiply_72, False)
    ttnn.deallocate(ttnn_add_84, False)
    ttnn.deallocate(ttnn_to_memory_config_136, False)
    ttnn.deallocate(ttnn_rsqrt_12, False)
    # ttnn.deallocate(ttnn_reshape_317, False)
    return ttnn_to_memory_config_132, ttnn_reshape_317, ttnn_to_memory_config_134


def Linear_71_0(input_0, input_1, input_2):
    ttnn_matmul_40 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_85 = ttnn.add(
        ttnn_matmul_40,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_137 = ttnn.to_memory_config(
        ttnn_add_85,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_40, False)
    ttnn.deallocate(ttnn_add_85, False)
    return ttnn_to_memory_config_137


def QuickGELUActivation_120_0(input_0, input_1):
    ttnn_multiply_73 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_138 = ttnn.to_memory_config(
        ttnn_multiply_73,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_6 = ttnn.sigmoid(
        ttnn_to_memory_config_138,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_74 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_73, False)
    ttnn.deallocate(ttnn_to_memory_config_138, False)
    ttnn.deallocate(ttnn_sigmoid_6, False)
    return ttnn_multiply_74


def Linear_59_0(input_0, input_1, input_2):
    ttnn_matmul_41 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_86 = ttnn.add(
        ttnn_matmul_41,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_139 = ttnn.to_memory_config(
        ttnn_add_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_41, False)
    ttnn.deallocate(ttnn_add_86, False)
    return ttnn_to_memory_config_139


def Linear_129_0(input):
    ttnn_reshape_318 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_318


def CLIPAttention_102_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_140 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_42 = ttnn.matmul(
        ttnn_to_memory_config_140,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_141 = ttnn.to_memory_config(
        ttnn_matmul_42,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_319 = ttnn.reshape(
        ttnn_to_memory_config_141,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_75 = ttnn.multiply(
        ttnn_reshape_319,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_multiply_75,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_2 = ttnn.softmax(
        ttnn_typecast_5,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_softmax_2,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_320 = ttnn.reshape(
        ttnn_typecast_6,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_142 = ttnn.to_memory_config(
        ttnn_reshape_320,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_43 = ttnn.matmul(
        ttnn_to_memory_config_142,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_143 = ttnn.to_memory_config(
        ttnn_matmul_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_321 = ttnn.reshape(
        ttnn_to_memory_config_143,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_140, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_42, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_141, False)
    ttnn.deallocate(ttnn_reshape_319, False)
    ttnn.deallocate(ttnn_multiply_75, False)
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn.deallocate(ttnn_softmax_2, False)
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn.deallocate(ttnn_reshape_320, False)
    ttnn.deallocate(ttnn_to_memory_config_142, False)
    ttnn.deallocate(ttnn_matmul_43, False)
    ttnn.deallocate(ttnn_to_memory_config_143, False)
    return ttnn_reshape_321


def Linear_13_0(input_0, input_1, input_2):
    ttnn_matmul_44 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_87 = ttnn.add(
        ttnn_matmul_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_144 = ttnn.to_memory_config(
        ttnn_add_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_322 = ttnn.reshape(
        ttnn_to_memory_config_144,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_44, False)
    ttnn.deallocate(ttnn_add_87, False)
    ttnn.deallocate(ttnn_to_memory_config_144, False)
    return ttnn_reshape_322


def Linear_11_0(input_0, input_1, input_2):
    ttnn_matmul_45 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_88 = ttnn.add(
        ttnn_matmul_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_145 = ttnn.to_memory_config(
        ttnn_add_88,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_45, False)
    ttnn.deallocate(ttnn_add_88, False)
    return ttnn_to_memory_config_145


def Linear_125_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_46 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_146 = ttnn.to_memory_config(
        input_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_89 = ttnn.add(
        ttnn_matmul_46,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_147 = ttnn.to_memory_config(
        input_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_47 = ttnn.matmul(
        ttnn_to_memory_config_146,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_148 = ttnn.to_memory_config(
        ttnn_add_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_48 = ttnn.matmul(
        ttnn_to_memory_config_147,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_90 = ttnn.add(
        ttnn_matmul_47,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_91 = ttnn.add(
        ttnn_matmul_48,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_323 = ttnn.reshape(
        ttnn_to_memory_config_148,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_149 = ttnn.to_memory_config(
        ttnn_add_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_150 = ttnn.to_memory_config(
        ttnn_add_91,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_74 = ttnn.permute(
        ttnn_reshape_323,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_324 = ttnn.reshape(
        ttnn_to_memory_config_149,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_325 = ttnn.reshape(
        ttnn_to_memory_config_150,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_326 = ttnn.reshape(
        ttnn_permute_74,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_75 = ttnn.permute(
        ttnn_reshape_324,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_76 = ttnn.permute(
        ttnn_reshape_325,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_327 = ttnn.reshape(
        ttnn_permute_75,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_328 = ttnn.reshape(
        ttnn_permute_76,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_4, False)
    ttnn.deallocate(ttnn_matmul_46, False)
    ttnn.deallocate(ttnn_add_89, False)
    ttnn.deallocate(ttnn_to_memory_config_146, False)
    ttnn.deallocate(ttnn_to_memory_config_147, False)
    ttnn.deallocate(ttnn_matmul_48, False)
    ttnn.deallocate(ttnn_to_memory_config_148, False)
    ttnn.deallocate(ttnn_matmul_47, False)
    ttnn.deallocate(ttnn_reshape_323, False)
    ttnn.deallocate(ttnn_add_91, False)
    ttnn.deallocate(ttnn_add_90, False)
    ttnn.deallocate(ttnn_to_memory_config_149, False)
    ttnn.deallocate(ttnn_permute_74, False)
    ttnn.deallocate(ttnn_to_memory_config_150, False)
    ttnn.deallocate(ttnn_reshape_324, False)
    ttnn.deallocate(ttnn_reshape_325, False)
    ttnn.deallocate(ttnn_permute_75, False)
    ttnn.deallocate(ttnn_permute_76, False)
    return ttnn_reshape_328, ttnn_reshape_327, ttnn_reshape_326


def Linear_87_0(input):
    ttnn_reshape_329 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_329


def CLIPAttention_78_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_151 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_49 = ttnn.matmul(
        ttnn_to_memory_config_151,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_152 = ttnn.to_memory_config(
        ttnn_matmul_49,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_330 = ttnn.reshape(
        ttnn_to_memory_config_152,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_76 = ttnn.multiply(
        ttnn_reshape_330,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_multiply_76,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_3 = ttnn.softmax(
        ttnn_typecast_7,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_softmax_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_331 = ttnn.reshape(
        ttnn_typecast_8,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_153 = ttnn.to_memory_config(
        ttnn_reshape_331,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_50 = ttnn.matmul(
        ttnn_to_memory_config_153,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_154 = ttnn.to_memory_config(
        ttnn_matmul_50,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_332 = ttnn.reshape(
        ttnn_to_memory_config_154,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_151, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_49, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_152, False)
    ttnn.deallocate(ttnn_reshape_330, False)
    ttnn.deallocate(ttnn_multiply_76, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn.deallocate(ttnn_softmax_3, False)
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn.deallocate(ttnn_reshape_331, False)
    ttnn.deallocate(ttnn_to_memory_config_153, False)
    ttnn.deallocate(ttnn_matmul_50, False)
    ttnn.deallocate(ttnn_to_memory_config_154, False)
    return ttnn_reshape_332


def Linear_63_0(input):
    ttnn_reshape_333 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_333


def Linear_75_0(input):
    ttnn_reshape_334 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_334


def LayerNorm_88_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_77 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_78 = ttnn.multiply(
        ttnn_multiply_77,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_92 = ttnn.add(
        ttnn_multiply_78,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_155 = ttnn.to_memory_config(
        ttnn_add_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_77, False)
    ttnn.deallocate(ttnn_multiply_78, False)
    return ttnn_to_memory_config_155, ttnn_add_92


def LayerNorm_1_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_sum_26 = ttnn.sum(
        input_0,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_79 = ttnn.multiply(
        ttnn_sum_26,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_156 = ttnn.to_memory_config(
        ttnn_multiply_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_335 = ttnn.reshape(
        ttnn_to_memory_config_156,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_13 = ttnn.neg(
        ttnn_reshape_335,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_93 = ttnn.add(
        input_0,
        ttnn_neg_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_157 = ttnn.to_memory_config(
        ttnn_add_93,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_80 = ttnn.multiply(
        ttnn_to_memory_config_157,
        ttnn_to_memory_config_157,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_158 = ttnn.to_memory_config(
        ttnn_multiply_80,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_27 = ttnn.sum(
        ttnn_to_memory_config_158,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_81 = ttnn.multiply(
        ttnn_sum_27,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_159 = ttnn.to_memory_config(
        ttnn_multiply_81,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_336 = ttnn.reshape(
        ttnn_to_memory_config_159,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_94 = ttnn.add(
        ttnn_reshape_336,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_160 = ttnn.to_memory_config(
        ttnn_add_94,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_13 = ttnn.rsqrt(
        ttnn_to_memory_config_160,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_82 = ttnn.multiply(
        ttnn_to_memory_config_157,
        ttnn_rsqrt_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_83 = ttnn.multiply(
        ttnn_multiply_82,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_95 = ttnn.add(
        ttnn_multiply_83,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_161 = ttnn.to_memory_config(
        ttnn_add_95,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_sum_26, False)
    ttnn.deallocate(ttnn_multiply_79, False)
    ttnn.deallocate(ttnn_to_memory_config_156, False)
    ttnn.deallocate(ttnn_reshape_335, False)
    ttnn.deallocate(ttnn_neg_13, False)
    ttnn.deallocate(ttnn_add_93, False)
    ttnn.deallocate(ttnn_to_memory_config_157, False)
    ttnn.deallocate(ttnn_multiply_80, False)
    ttnn.deallocate(ttnn_to_memory_config_158, False)
    ttnn.deallocate(ttnn_sum_27, False)
    ttnn.deallocate(ttnn_multiply_81, False)
    ttnn.deallocate(ttnn_to_memory_config_159, False)
    ttnn.deallocate(ttnn_reshape_336, False)
    ttnn.deallocate(ttnn_add_94, False)
    ttnn.deallocate(ttnn_to_memory_config_160, False)
    ttnn.deallocate(ttnn_rsqrt_13, False)
    ttnn.deallocate(ttnn_multiply_82, False)
    ttnn.deallocate(ttnn_multiply_83, False)
    ttnn.deallocate(ttnn_add_95, False)
    return ttnn_to_memory_config_161


def Linear_3_0(input):
    ttnn_reshape_337 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_337


def Linear_117_0(input):
    ttnn_reshape_338 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_338


def Linear_103_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_29 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_339 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_29,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_162 = ttnn.to_memory_config(
        ttnn_reshape_339,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_51 = ttnn.matmul(
        ttnn_to_memory_config_162,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_96 = ttnn.add(
        ttnn_matmul_51,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_163 = ttnn.to_memory_config(
        ttnn_add_96,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_340 = ttnn.reshape(
        ttnn_to_memory_config_163,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_29, False)
    ttnn.deallocate(ttnn_reshape_339, False)
    ttnn.deallocate(ttnn_to_memory_config_162, False)
    ttnn.deallocate(ttnn_matmul_51, False)
    ttnn.deallocate(ttnn_add_96, False)
    ttnn.deallocate(ttnn_to_memory_config_163, False)
    return ttnn_reshape_340


def Linear_123_0(input):
    ttnn_reshape_341 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_341


def LayerNorm_64_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_84 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_85 = ttnn.multiply(
        ttnn_multiply_84,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_97 = ttnn.add(
        ttnn_multiply_85,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_164 = ttnn.to_memory_config(
        ttnn_add_97,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_84, False)
    ttnn.deallocate(ttnn_multiply_85, False)
    return ttnn_add_97, ttnn_to_memory_config_164


def CLIPEncoderLayer_62_0(input_0, input_1, input_2, input_3):
    ttnn_add_98 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_165 = ttnn.to_memory_config(
        ttnn_add_98,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_28 = ttnn.sum(
        ttnn_to_memory_config_165,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_86 = ttnn.multiply(
        ttnn_sum_28,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_166 = ttnn.to_memory_config(
        ttnn_multiply_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_342 = ttnn.reshape(
        ttnn_to_memory_config_166,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_14 = ttnn.neg(
        ttnn_reshape_342,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_99 = ttnn.add(
        ttnn_to_memory_config_165,
        ttnn_neg_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_167 = ttnn.to_memory_config(
        ttnn_add_99,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_87 = ttnn.multiply(
        ttnn_to_memory_config_167,
        ttnn_to_memory_config_167,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_168 = ttnn.to_memory_config(
        ttnn_multiply_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_29 = ttnn.sum(
        ttnn_to_memory_config_168,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_88 = ttnn.multiply(
        ttnn_sum_29,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_100 = ttnn.add(
        ttnn_multiply_88,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_169 = ttnn.to_memory_config(
        ttnn_add_100,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_14 = ttnn.rsqrt(
        ttnn_to_memory_config_169,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_343 = ttnn.reshape(
        ttnn_rsqrt_14,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_98, False)
    # ttnn.deallocate(ttnn_to_memory_config_165, False)
    ttnn.deallocate(ttnn_sum_28, False)
    ttnn.deallocate(ttnn_multiply_86, False)
    ttnn.deallocate(ttnn_to_memory_config_166, False)
    ttnn.deallocate(ttnn_reshape_342, False)
    ttnn.deallocate(ttnn_neg_14, False)
    ttnn.deallocate(ttnn_add_99, False)
    # ttnn.deallocate(ttnn_to_memory_config_167, False)
    ttnn.deallocate(ttnn_multiply_87, False)
    ttnn.deallocate(ttnn_to_memory_config_168, False)
    ttnn.deallocate(ttnn_sum_29, False)
    ttnn.deallocate(ttnn_multiply_88, False)
    ttnn.deallocate(ttnn_add_100, False)
    ttnn.deallocate(ttnn_to_memory_config_169, False)
    ttnn.deallocate(ttnn_rsqrt_14, False)
    # ttnn.deallocate(ttnn_reshape_343, False)
    return ttnn_to_memory_config_167, ttnn_reshape_343, ttnn_to_memory_config_165


def CLIPEncoderLayer_86_0(input_0, input_1, input_2, input_3):
    ttnn_add_101 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_170 = ttnn.to_memory_config(
        ttnn_add_101,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_30 = ttnn.sum(
        ttnn_to_memory_config_170,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_89 = ttnn.multiply(
        ttnn_sum_30,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_171 = ttnn.to_memory_config(
        ttnn_multiply_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_344 = ttnn.reshape(
        ttnn_to_memory_config_171,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_15 = ttnn.neg(
        ttnn_reshape_344,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_102 = ttnn.add(
        ttnn_to_memory_config_170,
        ttnn_neg_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_172 = ttnn.to_memory_config(
        ttnn_add_102,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_90 = ttnn.multiply(
        ttnn_to_memory_config_172,
        ttnn_to_memory_config_172,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_173 = ttnn.to_memory_config(
        ttnn_multiply_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_31 = ttnn.sum(
        ttnn_to_memory_config_173,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_91 = ttnn.multiply(
        ttnn_sum_31,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_103 = ttnn.add(
        ttnn_multiply_91,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_174 = ttnn.to_memory_config(
        ttnn_add_103,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_15 = ttnn.rsqrt(
        ttnn_to_memory_config_174,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_345 = ttnn.reshape(
        ttnn_rsqrt_15,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_101, False)
    # ttnn.deallocate(ttnn_to_memory_config_170, False)
    ttnn.deallocate(ttnn_sum_30, False)
    ttnn.deallocate(ttnn_multiply_89, False)
    ttnn.deallocate(ttnn_to_memory_config_171, False)
    ttnn.deallocate(ttnn_reshape_344, False)
    ttnn.deallocate(ttnn_neg_15, False)
    ttnn.deallocate(ttnn_add_102, False)
    # ttnn.deallocate(ttnn_to_memory_config_172, False)
    ttnn.deallocate(ttnn_multiply_90, False)
    ttnn.deallocate(ttnn_to_memory_config_173, False)
    ttnn.deallocate(ttnn_sum_31, False)
    ttnn.deallocate(ttnn_multiply_91, False)
    ttnn.deallocate(ttnn_add_103, False)
    ttnn.deallocate(ttnn_to_memory_config_174, False)
    ttnn.deallocate(ttnn_rsqrt_15, False)
    # ttnn.deallocate(ttnn_reshape_345, False)
    return ttnn_to_memory_config_172, ttnn_to_memory_config_170, ttnn_reshape_345


def QuickGELUActivation_108_0(input_0, input_1):
    ttnn_multiply_92 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_175 = ttnn.to_memory_config(
        ttnn_multiply_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_7 = ttnn.sigmoid(
        ttnn_to_memory_config_175,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_93 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_92, False)
    ttnn.deallocate(ttnn_to_memory_config_175, False)
    ttnn.deallocate(ttnn_sigmoid_7, False)
    return ttnn_multiply_93


def Linear_21_0(input):
    ttnn_reshape_346 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_346


def QuickGELUActivation_144_0(input_0, input_1):
    ttnn_multiply_94 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_176 = ttnn.to_memory_config(
        ttnn_multiply_94,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_8 = ttnn.sigmoid(
        ttnn_to_memory_config_176,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_95 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_94, False)
    ttnn.deallocate(ttnn_to_memory_config_176, False)
    ttnn.deallocate(ttnn_sigmoid_8, False)
    return ttnn_multiply_95


def CLIPEncoderLayer_104_0(input_0, input_1, input_2, input_3):
    ttnn_add_104 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_177 = ttnn.to_memory_config(
        ttnn_add_104,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_32 = ttnn.sum(
        ttnn_to_memory_config_177,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_96 = ttnn.multiply(
        ttnn_sum_32,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_178 = ttnn.to_memory_config(
        ttnn_multiply_96,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_347 = ttnn.reshape(
        ttnn_to_memory_config_178,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_16 = ttnn.neg(
        ttnn_reshape_347,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_105 = ttnn.add(
        ttnn_to_memory_config_177,
        ttnn_neg_16,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_179 = ttnn.to_memory_config(
        ttnn_add_105,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_97 = ttnn.multiply(
        ttnn_to_memory_config_179,
        ttnn_to_memory_config_179,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_180 = ttnn.to_memory_config(
        ttnn_multiply_97,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_33 = ttnn.sum(
        ttnn_to_memory_config_180,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_98 = ttnn.multiply(
        ttnn_sum_33,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_106 = ttnn.add(
        ttnn_multiply_98,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_181 = ttnn.to_memory_config(
        ttnn_add_106,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_16 = ttnn.rsqrt(
        ttnn_to_memory_config_181,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_348 = ttnn.reshape(
        ttnn_rsqrt_16,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_104, False)
    # ttnn.deallocate(ttnn_to_memory_config_177, False)
    ttnn.deallocate(ttnn_sum_32, False)
    ttnn.deallocate(ttnn_multiply_96, False)
    ttnn.deallocate(ttnn_to_memory_config_178, False)
    ttnn.deallocate(ttnn_reshape_347, False)
    ttnn.deallocate(ttnn_neg_16, False)
    ttnn.deallocate(ttnn_add_105, False)
    # ttnn.deallocate(ttnn_to_memory_config_179, False)
    ttnn.deallocate(ttnn_multiply_97, False)
    ttnn.deallocate(ttnn_to_memory_config_180, False)
    ttnn.deallocate(ttnn_sum_33, False)
    ttnn.deallocate(ttnn_multiply_98, False)
    ttnn.deallocate(ttnn_add_106, False)
    ttnn.deallocate(ttnn_to_memory_config_181, False)
    ttnn.deallocate(ttnn_rsqrt_16, False)
    # ttnn.deallocate(ttnn_reshape_348, False)
    return ttnn_reshape_348, ttnn_to_memory_config_177, ttnn_to_memory_config_179


def Linear_25_0(input_0, input_1, input_2):
    ttnn_matmul_52 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_107 = ttnn.add(
        ttnn_matmul_52,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_182 = ttnn.to_memory_config(
        ttnn_add_107,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_349 = ttnn.reshape(
        ttnn_to_memory_config_182,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_52, False)
    ttnn.deallocate(ttnn_add_107, False)
    ttnn.deallocate(ttnn_to_memory_config_182, False)
    return ttnn_reshape_349


def LayerNorm_94_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_99 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_100 = ttnn.multiply(
        ttnn_multiply_99,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_108 = ttnn.add(
        ttnn_multiply_100,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_99, False)
    ttnn.deallocate(ttnn_multiply_100, False)
    return ttnn_add_108


def Linear_57_0(input):
    ttnn_reshape_350 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_350


def Linear_35_0(input_0, input_1, input_2):
    ttnn_matmul_53 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_109 = ttnn.add(
        ttnn_matmul_53,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_183 = ttnn.to_memory_config(
        ttnn_add_109,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_53, False)
    ttnn.deallocate(ttnn_add_109, False)
    return ttnn_to_memory_config_183


def Linear_148_0(input_0, input_1):
    ttnn_matmul_54 = ttnn.matmul(
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
    ttnn.deallocate(input_0, False)
    return ttnn_matmul_54


def Linear_45_0(input):
    ttnn_reshape_351 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_351


def LayerNorm_112_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_101 = ttnn.multiply(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_102 = ttnn.multiply(
        ttnn_multiply_101,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_110 = ttnn.add(
        ttnn_multiply_102,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_184 = ttnn.to_memory_config(
        ttnn_add_110,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_101, False)
    ttnn.deallocate(ttnn_multiply_102, False)
    return ttnn_to_memory_config_184, ttnn_add_110


def CLIPEncoderLayer_92_0(input_0, input_1, input_2, input_3):
    ttnn_add_111 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_185 = ttnn.to_memory_config(
        ttnn_add_111,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_34 = ttnn.sum(
        ttnn_to_memory_config_185,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_103 = ttnn.multiply(
        ttnn_sum_34,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_186 = ttnn.to_memory_config(
        ttnn_multiply_103,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_352 = ttnn.reshape(
        ttnn_to_memory_config_186,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_17 = ttnn.neg(
        ttnn_reshape_352,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_112 = ttnn.add(
        ttnn_to_memory_config_185,
        ttnn_neg_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_187 = ttnn.to_memory_config(
        ttnn_add_112,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_104 = ttnn.multiply(
        ttnn_to_memory_config_187,
        ttnn_to_memory_config_187,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_188 = ttnn.to_memory_config(
        ttnn_multiply_104,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_35 = ttnn.sum(
        ttnn_to_memory_config_188,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_105 = ttnn.multiply(
        ttnn_sum_35,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_113 = ttnn.add(
        ttnn_multiply_105,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_189 = ttnn.to_memory_config(
        ttnn_add_113,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_17 = ttnn.rsqrt(
        ttnn_to_memory_config_189,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_353 = ttnn.reshape(
        ttnn_rsqrt_17,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_111, False)
    # ttnn.deallocate(ttnn_to_memory_config_185, False)
    ttnn.deallocate(ttnn_sum_34, False)
    ttnn.deallocate(ttnn_multiply_103, False)
    ttnn.deallocate(ttnn_to_memory_config_186, False)
    ttnn.deallocate(ttnn_reshape_352, False)
    ttnn.deallocate(ttnn_neg_17, False)
    ttnn.deallocate(ttnn_add_112, False)
    # ttnn.deallocate(ttnn_to_memory_config_187, False)
    ttnn.deallocate(ttnn_multiply_104, False)
    ttnn.deallocate(ttnn_to_memory_config_188, False)
    ttnn.deallocate(ttnn_sum_35, False)
    ttnn.deallocate(ttnn_multiply_105, False)
    ttnn.deallocate(ttnn_add_113, False)
    ttnn.deallocate(ttnn_to_memory_config_189, False)
    ttnn.deallocate(ttnn_rsqrt_17, False)
    # ttnn.deallocate(ttnn_reshape_353, False)
    return ttnn_to_memory_config_185, ttnn_reshape_353, ttnn_to_memory_config_187


def Linear_27_0(input):
    ttnn_reshape_354 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_354


def LayerNorm_124_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_106 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_107 = ttnn.multiply(
        ttnn_multiply_106,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_114 = ttnn.add(
        ttnn_multiply_107,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_190 = ttnn.to_memory_config(
        ttnn_add_114,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_106, False)
    ttnn.deallocate(ttnn_multiply_107, False)
    return ttnn_add_114, ttnn_to_memory_config_190


def Linear_143_0(input_0, input_1, input_2):
    ttnn_matmul_55 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_115 = ttnn.add(
        ttnn_matmul_55,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_191 = ttnn.to_memory_config(
        ttnn_add_115,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_55, False)
    ttnn.deallocate(ttnn_add_115, False)
    return ttnn_to_memory_config_191


def CLIPAttention_18_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_192 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_56 = ttnn.matmul(
        ttnn_to_memory_config_192,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_193 = ttnn.to_memory_config(
        ttnn_matmul_56,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_355 = ttnn.reshape(
        ttnn_to_memory_config_193,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_108 = ttnn.multiply(
        ttnn_reshape_355,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_multiply_108,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_4 = ttnn.softmax(
        ttnn_typecast_9,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_softmax_4,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_356 = ttnn.reshape(
        ttnn_typecast_10,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_194 = ttnn.to_memory_config(
        ttnn_reshape_356,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_57 = ttnn.matmul(
        ttnn_to_memory_config_194,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_195 = ttnn.to_memory_config(
        ttnn_matmul_57,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_357 = ttnn.reshape(
        ttnn_to_memory_config_195,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_192, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_56, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_193, False)
    ttnn.deallocate(ttnn_reshape_355, False)
    ttnn.deallocate(ttnn_multiply_108, False)
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_softmax_4, False)
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn.deallocate(ttnn_reshape_356, False)
    ttnn.deallocate(ttnn_to_memory_config_194, False)
    ttnn.deallocate(ttnn_matmul_57, False)
    ttnn.deallocate(ttnn_to_memory_config_195, False)
    return ttnn_reshape_357


def Linear_145_0(input_0, input_1, input_2):
    ttnn_matmul_58 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_116 = ttnn.add(
        ttnn_matmul_58,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_196 = ttnn.to_memory_config(
        ttnn_add_116,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_358 = ttnn.reshape(
        ttnn_to_memory_config_196,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_58, False)
    ttnn.deallocate(ttnn_add_116, False)
    ttnn.deallocate(ttnn_to_memory_config_196, False)
    return ttnn_reshape_358


def Linear_49_0(input_0, input_1, input_2):
    ttnn_matmul_59 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_117 = ttnn.add(
        ttnn_matmul_59,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_197 = ttnn.to_memory_config(
        ttnn_add_117,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_359 = ttnn.reshape(
        ttnn_to_memory_config_197,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_59, False)
    ttnn.deallocate(ttnn_add_117, False)
    ttnn.deallocate(ttnn_to_memory_config_197, False)
    return ttnn_reshape_359


def LayerNorm_100_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_109 = ttnn.multiply(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_110 = ttnn.multiply(
        ttnn_multiply_109,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_118 = ttnn.add(
        ttnn_multiply_110,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_198 = ttnn.to_memory_config(
        ttnn_add_118,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_109, False)
    ttnn.deallocate(ttnn_multiply_110, False)
    return ttnn_to_memory_config_198, ttnn_add_118


def Linear_91_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_30 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_360 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_30,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_199 = ttnn.to_memory_config(
        ttnn_reshape_360,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_60 = ttnn.matmul(
        ttnn_to_memory_config_199,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_119 = ttnn.add(
        ttnn_matmul_60,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_200 = ttnn.to_memory_config(
        ttnn_add_119,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_361 = ttnn.reshape(
        ttnn_to_memory_config_200,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_30, False)
    ttnn.deallocate(ttnn_reshape_360, False)
    ttnn.deallocate(ttnn_to_memory_config_199, False)
    ttnn.deallocate(ttnn_matmul_60, False)
    ttnn.deallocate(ttnn_add_119, False)
    ttnn.deallocate(ttnn_to_memory_config_200, False)
    return ttnn_reshape_361


def Linear_39_0(input):
    ttnn_reshape_362 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_362


def CLIPAttention_54_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_201 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_61 = ttnn.matmul(
        ttnn_to_memory_config_201,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_202 = ttnn.to_memory_config(
        ttnn_matmul_61,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_363 = ttnn.reshape(
        ttnn_to_memory_config_202,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_111 = ttnn.multiply(
        ttnn_reshape_363,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_multiply_111,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_5 = ttnn.softmax(
        ttnn_typecast_11,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_softmax_5,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_364 = ttnn.reshape(
        ttnn_typecast_12,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_203 = ttnn.to_memory_config(
        ttnn_reshape_364,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_62 = ttnn.matmul(
        ttnn_to_memory_config_203,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_204 = ttnn.to_memory_config(
        ttnn_matmul_62,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_365 = ttnn.reshape(
        ttnn_to_memory_config_204,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_201, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_61, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_202, False)
    ttnn.deallocate(ttnn_reshape_363, False)
    ttnn.deallocate(ttnn_multiply_111, False)
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn.deallocate(ttnn_softmax_5, False)
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn.deallocate(ttnn_reshape_364, False)
    ttnn.deallocate(ttnn_to_memory_config_203, False)
    ttnn.deallocate(ttnn_matmul_62, False)
    ttnn.deallocate(ttnn_to_memory_config_204, False)
    return ttnn_reshape_365


def QuickGELUActivation_72_0(input_0, input_1):
    ttnn_multiply_112 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_205 = ttnn.to_memory_config(
        ttnn_multiply_112,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_9 = ttnn.sigmoid(
        ttnn_to_memory_config_205,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_113 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_112, False)
    ttnn.deallocate(ttnn_to_memory_config_205, False)
    ttnn.deallocate(ttnn_sigmoid_9, False)
    return ttnn_multiply_113


def Linear_77_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_63 = ttnn.matmul(
        input_6,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_206 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_120 = ttnn.add(
        ttnn_matmul_63,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_207 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_64 = ttnn.matmul(
        ttnn_to_memory_config_206,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_208 = ttnn.to_memory_config(
        ttnn_add_120,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_65 = ttnn.matmul(
        ttnn_to_memory_config_207,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_121 = ttnn.add(
        ttnn_matmul_64,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_122 = ttnn.add(
        ttnn_matmul_65,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_366 = ttnn.reshape(
        ttnn_to_memory_config_208,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_209 = ttnn.to_memory_config(
        ttnn_add_121,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_210 = ttnn.to_memory_config(
        ttnn_add_122,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_77 = ttnn.permute(
        ttnn_reshape_366,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_367 = ttnn.reshape(
        ttnn_to_memory_config_209,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_368 = ttnn.reshape(
        ttnn_to_memory_config_210,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_369 = ttnn.reshape(
        ttnn_permute_77,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_78 = ttnn.permute(
        ttnn_reshape_367,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_79 = ttnn.permute(
        ttnn_reshape_368,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_370 = ttnn.reshape(
        ttnn_permute_78,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_371 = ttnn.reshape(
        ttnn_permute_79,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_6, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_63, False)
    ttnn.deallocate(ttnn_add_120, False)
    ttnn.deallocate(ttnn_to_memory_config_206, False)
    ttnn.deallocate(ttnn_to_memory_config_207, False)
    ttnn.deallocate(ttnn_matmul_65, False)
    ttnn.deallocate(ttnn_to_memory_config_208, False)
    ttnn.deallocate(ttnn_matmul_64, False)
    ttnn.deallocate(ttnn_reshape_366, False)
    ttnn.deallocate(ttnn_add_122, False)
    ttnn.deallocate(ttnn_add_121, False)
    ttnn.deallocate(ttnn_to_memory_config_209, False)
    ttnn.deallocate(ttnn_permute_77, False)
    ttnn.deallocate(ttnn_to_memory_config_210, False)
    ttnn.deallocate(ttnn_reshape_367, False)
    ttnn.deallocate(ttnn_reshape_368, False)
    ttnn.deallocate(ttnn_permute_78, False)
    ttnn.deallocate(ttnn_permute_79, False)
    return ttnn_reshape_371, ttnn_reshape_369, ttnn_reshape_370


def Linear_47_0(input_0, input_1, input_2):
    ttnn_matmul_66 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_123 = ttnn.add(
        ttnn_matmul_66,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_211 = ttnn.to_memory_config(
        ttnn_add_123,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_66, False)
    ttnn.deallocate(ttnn_add_123, False)
    return ttnn_to_memory_config_211


def CLIPAttention_90_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_212 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_67 = ttnn.matmul(
        ttnn_to_memory_config_212,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_213 = ttnn.to_memory_config(
        ttnn_matmul_67,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_372 = ttnn.reshape(
        ttnn_to_memory_config_213,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_114 = ttnn.multiply(
        ttnn_reshape_372,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_multiply_114,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_6 = ttnn.softmax(
        ttnn_typecast_13,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_softmax_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_373 = ttnn.reshape(
        ttnn_typecast_14,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_214 = ttnn.to_memory_config(
        ttnn_reshape_373,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_68 = ttnn.matmul(
        ttnn_to_memory_config_214,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_215 = ttnn.to_memory_config(
        ttnn_matmul_68,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_374 = ttnn.reshape(
        ttnn_to_memory_config_215,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_212, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_matmul_67, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_213, False)
    ttnn.deallocate(ttnn_reshape_372, False)
    ttnn.deallocate(ttnn_multiply_114, False)
    ttnn.deallocate(ttnn_typecast_13, False)
    ttnn.deallocate(ttnn_softmax_6, False)
    ttnn.deallocate(ttnn_typecast_14, False)
    ttnn.deallocate(ttnn_reshape_373, False)
    ttnn.deallocate(ttnn_to_memory_config_214, False)
    ttnn.deallocate(ttnn_matmul_68, False)
    ttnn.deallocate(ttnn_to_memory_config_215, False)
    return ttnn_reshape_374


def LayerNorm_82_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_115 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_116 = ttnn.multiply(
        ttnn_multiply_115,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_124 = ttnn.add(
        ttnn_multiply_116,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_115, False)
    ttnn.deallocate(ttnn_multiply_116, False)
    return ttnn_add_124


def LayerNorm_136_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_117 = ttnn.multiply(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_118 = ttnn.multiply(
        ttnn_multiply_117,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_125 = ttnn.add(
        ttnn_multiply_118,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_216 = ttnn.to_memory_config(
        ttnn_add_125,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_117, False)
    ttnn.deallocate(ttnn_multiply_118, False)
    return ttnn_add_125, ttnn_to_memory_config_216


def LayerNorm_76_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_119 = ttnn.multiply(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_120 = ttnn.multiply(
        ttnn_multiply_119,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_126 = ttnn.add(
        ttnn_multiply_120,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_217 = ttnn.to_memory_config(
        ttnn_add_126,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_119, False)
    ttnn.deallocate(ttnn_multiply_120, False)
    return ttnn_to_memory_config_217, ttnn_add_126


def Linear_89_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_69 = ttnn.matmul(
        input_3,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_218 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_127 = ttnn.add(
        ttnn_matmul_69,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_219 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_70 = ttnn.matmul(
        ttnn_to_memory_config_218,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_220 = ttnn.to_memory_config(
        ttnn_add_127,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_71 = ttnn.matmul(
        ttnn_to_memory_config_219,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_128 = ttnn.add(
        ttnn_matmul_70,
        input_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_129 = ttnn.add(
        ttnn_matmul_71,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_375 = ttnn.reshape(
        ttnn_to_memory_config_220,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_221 = ttnn.to_memory_config(
        ttnn_add_128,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_222 = ttnn.to_memory_config(
        ttnn_add_129,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_80 = ttnn.permute(
        ttnn_reshape_375,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_376 = ttnn.reshape(
        ttnn_to_memory_config_221,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_377 = ttnn.reshape(
        ttnn_to_memory_config_222,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_378 = ttnn.reshape(
        ttnn_permute_80,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_81 = ttnn.permute(
        ttnn_reshape_376,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_82 = ttnn.permute(
        ttnn_reshape_377,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_379 = ttnn.reshape(
        ttnn_permute_81,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_380 = ttnn.reshape(
        ttnn_permute_82,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_69, False)
    ttnn.deallocate(ttnn_add_127, False)
    ttnn.deallocate(ttnn_to_memory_config_218, False)
    ttnn.deallocate(ttnn_to_memory_config_219, False)
    ttnn.deallocate(ttnn_matmul_71, False)
    ttnn.deallocate(ttnn_to_memory_config_220, False)
    ttnn.deallocate(ttnn_matmul_70, False)
    ttnn.deallocate(ttnn_reshape_375, False)
    ttnn.deallocate(ttnn_add_129, False)
    ttnn.deallocate(ttnn_add_128, False)
    ttnn.deallocate(ttnn_to_memory_config_221, False)
    ttnn.deallocate(ttnn_permute_80, False)
    ttnn.deallocate(ttnn_to_memory_config_222, False)
    ttnn.deallocate(ttnn_reshape_376, False)
    ttnn.deallocate(ttnn_reshape_377, False)
    ttnn.deallocate(ttnn_permute_81, False)
    ttnn.deallocate(ttnn_permute_82, False)
    return ttnn_reshape_379, ttnn_reshape_380, ttnn_reshape_378


def LayerNorm_52_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_121 = ttnn.multiply(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_122 = ttnn.multiply(
        ttnn_multiply_121,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_130 = ttnn.add(
        ttnn_multiply_122,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_223 = ttnn.to_memory_config(
        ttnn_add_130,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_multiply_121, False)
    ttnn.deallocate(ttnn_multiply_122, False)
    return ttnn_add_130, ttnn_to_memory_config_223


def Linear_69_0(input):
    ttnn_reshape_381 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_381


def CLIPEncoderLayer_140_0(input_0, input_1, input_2, input_3):
    ttnn_add_131 = ttnn.add(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_224 = ttnn.to_memory_config(
        ttnn_add_131,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_36 = ttnn.sum(
        ttnn_to_memory_config_224,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_123 = ttnn.multiply(
        ttnn_sum_36,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_225 = ttnn.to_memory_config(
        ttnn_multiply_123,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_382 = ttnn.reshape(
        ttnn_to_memory_config_225,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_18 = ttnn.neg(
        ttnn_reshape_382,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_132 = ttnn.add(
        ttnn_to_memory_config_224,
        ttnn_neg_18,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_226 = ttnn.to_memory_config(
        ttnn_add_132,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_124 = ttnn.multiply(
        ttnn_to_memory_config_226,
        ttnn_to_memory_config_226,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_227 = ttnn.to_memory_config(
        ttnn_multiply_124,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_37 = ttnn.sum(
        ttnn_to_memory_config_227,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_125 = ttnn.multiply(
        ttnn_sum_37,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_133 = ttnn.add(
        ttnn_multiply_125,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_228 = ttnn.to_memory_config(
        ttnn_add_133,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_18 = ttnn.rsqrt(
        ttnn_to_memory_config_228,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_383 = ttnn.reshape(
        ttnn_rsqrt_18,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_131, False)
    # ttnn.deallocate(ttnn_to_memory_config_224, False)
    ttnn.deallocate(ttnn_sum_36, False)
    ttnn.deallocate(ttnn_multiply_123, False)
    ttnn.deallocate(ttnn_to_memory_config_225, False)
    ttnn.deallocate(ttnn_reshape_382, False)
    ttnn.deallocate(ttnn_neg_18, False)
    ttnn.deallocate(ttnn_add_132, False)
    # ttnn.deallocate(ttnn_to_memory_config_226, False)
    ttnn.deallocate(ttnn_multiply_124, False)
    ttnn.deallocate(ttnn_to_memory_config_227, False)
    ttnn.deallocate(ttnn_sum_37, False)
    ttnn.deallocate(ttnn_multiply_125, False)
    ttnn.deallocate(ttnn_add_133, False)
    ttnn.deallocate(ttnn_to_memory_config_228, False)
    ttnn.deallocate(ttnn_rsqrt_18, False)
    # ttnn.deallocate(ttnn_reshape_383, False)
    return ttnn_reshape_383, ttnn_to_memory_config_226, ttnn_to_memory_config_224


def CLIPEncoderLayer_146_0(input_0, input_1):
    ttnn_add_134 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_229 = ttnn.to_memory_config(
        ttnn_add_134,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_add_134, False)
    return ttnn_to_memory_config_229


def Linear_55_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_31 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_384 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_31,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_230 = ttnn.to_memory_config(
        ttnn_reshape_384,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_72 = ttnn.matmul(
        ttnn_to_memory_config_230,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_135 = ttnn.add(
        ttnn_matmul_72,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_231 = ttnn.to_memory_config(
        ttnn_add_135,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_385 = ttnn.reshape(
        ttnn_to_memory_config_231,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_31, False)
    ttnn.deallocate(ttnn_reshape_384, False)
    ttnn.deallocate(ttnn_to_memory_config_230, False)
    ttnn.deallocate(ttnn_matmul_72, False)
    ttnn.deallocate(ttnn_add_135, False)
    ttnn.deallocate(ttnn_to_memory_config_231, False)
    return ttnn_reshape_385


def Linear_127_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_32 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_386 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_32,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_232 = ttnn.to_memory_config(
        ttnn_reshape_386,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_73 = ttnn.matmul(
        ttnn_to_memory_config_232,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_136 = ttnn.add(
        ttnn_matmul_73,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_233 = ttnn.to_memory_config(
        ttnn_add_136,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_387 = ttnn.reshape(
        ttnn_to_memory_config_233,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_32, False)
    ttnn.deallocate(ttnn_reshape_386, False)
    ttnn.deallocate(ttnn_to_memory_config_232, False)
    ttnn.deallocate(ttnn_matmul_73, False)
    ttnn.deallocate(ttnn_add_136, False)
    ttnn.deallocate(ttnn_to_memory_config_233, False)
    return ttnn_reshape_387


def Linear_95_0(input_0, input_1, input_2):
    ttnn_matmul_74 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_137 = ttnn.add(
        ttnn_matmul_74,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_234 = ttnn.to_memory_config(
        ttnn_add_137,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_74, False)
    ttnn.deallocate(ttnn_add_137, False)
    return ttnn_to_memory_config_234


def QuickGELUActivation_24_0(input_0, input_1):
    ttnn_multiply_126 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_235 = ttnn.to_memory_config(
        ttnn_multiply_126,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_10 = ttnn.sigmoid(
        ttnn_to_memory_config_235,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_127 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_126, False)
    ttnn.deallocate(ttnn_to_memory_config_235, False)
    ttnn.deallocate(ttnn_sigmoid_10, False)
    return ttnn_multiply_127


def Linear_113_0(
    input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7
):
    ttnn_matmul_75 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_236 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_138 = ttnn.add(
        ttnn_matmul_75,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_237 = ttnn.to_memory_config(
        input_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_76 = ttnn.matmul(
        ttnn_to_memory_config_236,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_238 = ttnn.to_memory_config(
        ttnn_add_138,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_77 = ttnn.matmul(
        ttnn_to_memory_config_237,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_139 = ttnn.add(
        ttnn_matmul_76,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_140 = ttnn.add(
        ttnn_matmul_77,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_388 = ttnn.reshape(
        ttnn_to_memory_config_238,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_239 = ttnn.to_memory_config(
        ttnn_add_139,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_240 = ttnn.to_memory_config(
        ttnn_add_140,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_83 = ttnn.permute(
        ttnn_reshape_388,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_389 = ttnn.reshape(
        ttnn_to_memory_config_239,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_390 = ttnn.reshape(
        ttnn_to_memory_config_240,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_391 = ttnn.reshape(
        ttnn_permute_83,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_84 = ttnn.permute(
        ttnn_reshape_389,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_85 = ttnn.permute(
        ttnn_reshape_390,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_392 = ttnn.reshape(
        ttnn_permute_84,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_393 = ttnn.reshape(
        ttnn_permute_85,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_75, False)
    ttnn.deallocate(ttnn_add_138, False)
    ttnn.deallocate(ttnn_to_memory_config_236, False)
    ttnn.deallocate(ttnn_to_memory_config_237, False)
    ttnn.deallocate(ttnn_matmul_77, False)
    ttnn.deallocate(ttnn_to_memory_config_238, False)
    ttnn.deallocate(ttnn_matmul_76, False)
    ttnn.deallocate(ttnn_reshape_388, False)
    ttnn.deallocate(ttnn_add_140, False)
    ttnn.deallocate(ttnn_add_139, False)
    ttnn.deallocate(ttnn_to_memory_config_239, False)
    ttnn.deallocate(ttnn_permute_83, False)
    ttnn.deallocate(ttnn_to_memory_config_240, False)
    ttnn.deallocate(ttnn_reshape_389, False)
    ttnn.deallocate(ttnn_reshape_390, False)
    ttnn.deallocate(ttnn_permute_84, False)
    ttnn.deallocate(ttnn_permute_85, False)
    return ttnn_reshape_393, ttnn_reshape_392, ttnn_reshape_391


def LayerNorm_130_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_128 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_129 = ttnn.multiply(
        ttnn_multiply_128,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_141 = ttnn.add(
        ttnn_multiply_129,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_128, False)
    ttnn.deallocate(ttnn_multiply_129, False)
    return ttnn_add_141


def CLIPAttention_42_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_241 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_78 = ttnn.matmul(
        ttnn_to_memory_config_241,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_242 = ttnn.to_memory_config(
        ttnn_matmul_78,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_394 = ttnn.reshape(
        ttnn_to_memory_config_242,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_130 = ttnn.multiply(
        ttnn_reshape_394,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_multiply_130,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_7 = ttnn.softmax(
        ttnn_typecast_15,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_softmax_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_395 = ttnn.reshape(
        ttnn_typecast_16,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_243 = ttnn.to_memory_config(
        ttnn_reshape_395,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_79 = ttnn.matmul(
        ttnn_to_memory_config_243,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_244 = ttnn.to_memory_config(
        ttnn_matmul_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_396 = ttnn.reshape(
        ttnn_to_memory_config_244,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_241, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_78, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_242, False)
    ttnn.deallocate(ttnn_reshape_394, False)
    ttnn.deallocate(ttnn_multiply_130, False)
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn.deallocate(ttnn_softmax_7, False)
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn.deallocate(ttnn_reshape_395, False)
    ttnn.deallocate(ttnn_to_memory_config_243, False)
    ttnn.deallocate(ttnn_matmul_79, False)
    ttnn.deallocate(ttnn_to_memory_config_244, False)
    return ttnn_reshape_396


def CLIPEncoderLayer_8_0(input_0, input_1, input_2, input_3):
    ttnn_add_142 = ttnn.add(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_245 = ttnn.to_memory_config(
        ttnn_add_142,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_38 = ttnn.sum(
        ttnn_to_memory_config_245,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_131 = ttnn.multiply(
        ttnn_sum_38,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_246 = ttnn.to_memory_config(
        ttnn_multiply_131,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_397 = ttnn.reshape(
        ttnn_to_memory_config_246,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_19 = ttnn.neg(
        ttnn_reshape_397,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_143 = ttnn.add(
        ttnn_to_memory_config_245,
        ttnn_neg_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_247 = ttnn.to_memory_config(
        ttnn_add_143,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_132 = ttnn.multiply(
        ttnn_to_memory_config_247,
        ttnn_to_memory_config_247,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_248 = ttnn.to_memory_config(
        ttnn_multiply_132,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_39 = ttnn.sum(
        ttnn_to_memory_config_248,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_133 = ttnn.multiply(
        ttnn_sum_39,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_144 = ttnn.add(
        ttnn_multiply_133,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_249 = ttnn.to_memory_config(
        ttnn_add_144,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_19 = ttnn.rsqrt(
        ttnn_to_memory_config_249,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_398 = ttnn.reshape(
        ttnn_rsqrt_19,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_142, False)
    # ttnn.deallocate(ttnn_to_memory_config_245, False)
    ttnn.deallocate(ttnn_sum_38, False)
    ttnn.deallocate(ttnn_multiply_131, False)
    ttnn.deallocate(ttnn_to_memory_config_246, False)
    ttnn.deallocate(ttnn_reshape_397, False)
    ttnn.deallocate(ttnn_neg_19, False)
    ttnn.deallocate(ttnn_add_143, False)
    # ttnn.deallocate(ttnn_to_memory_config_247, False)
    ttnn.deallocate(ttnn_multiply_132, False)
    ttnn.deallocate(ttnn_to_memory_config_248, False)
    ttnn.deallocate(ttnn_sum_39, False)
    ttnn.deallocate(ttnn_multiply_133, False)
    ttnn.deallocate(ttnn_add_144, False)
    ttnn.deallocate(ttnn_to_memory_config_249, False)
    ttnn.deallocate(ttnn_rsqrt_19, False)
    # ttnn.deallocate(ttnn_reshape_398, False)
    return ttnn_to_memory_config_247, ttnn_to_memory_config_245, ttnn_reshape_398


def Linear_23_0(input_0, input_1, input_2):
    ttnn_matmul_80 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_145 = ttnn.add(
        ttnn_matmul_80,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 384],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_250 = ttnn.to_memory_config(
        ttnn_add_145,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_80, False)
    ttnn.deallocate(ttnn_add_145, False)
    return ttnn_to_memory_config_250


def Linear_99_0(input):
    ttnn_reshape_399 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_399


def Linear_79_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_33 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_400 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_33,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_251 = ttnn.to_memory_config(
        ttnn_reshape_400,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_81 = ttnn.matmul(
        ttnn_to_memory_config_251,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_146 = ttnn.add(
        ttnn_matmul_81,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_252 = ttnn.to_memory_config(
        ttnn_add_146,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_401 = ttnn.reshape(
        ttnn_to_memory_config_252,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_33, False)
    ttnn.deallocate(ttnn_reshape_400, False)
    ttnn.deallocate(ttnn_to_memory_config_251, False)
    ttnn.deallocate(ttnn_matmul_81, False)
    ttnn.deallocate(ttnn_add_146, False)
    ttnn.deallocate(ttnn_to_memory_config_252, False)
    return ttnn_reshape_401


def CLIPAttention_114_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_253 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_82 = ttnn.matmul(
        ttnn_to_memory_config_253,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_254 = ttnn.to_memory_config(
        ttnn_matmul_82,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_402 = ttnn.reshape(
        ttnn_to_memory_config_254,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_134 = ttnn.multiply(
        ttnn_reshape_402,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_multiply_134,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_8 = ttnn.softmax(
        ttnn_typecast_17,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_softmax_8,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_403 = ttnn.reshape(
        ttnn_typecast_18,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_255 = ttnn.to_memory_config(
        ttnn_reshape_403,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_83 = ttnn.matmul(
        ttnn_to_memory_config_255,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_256 = ttnn.to_memory_config(
        ttnn_matmul_83,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_404 = ttnn.reshape(
        ttnn_to_memory_config_256,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_253, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_82, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_254, False)
    ttnn.deallocate(ttnn_reshape_402, False)
    ttnn.deallocate(ttnn_multiply_134, False)
    ttnn.deallocate(ttnn_typecast_17, False)
    ttnn.deallocate(ttnn_softmax_8, False)
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn.deallocate(ttnn_reshape_403, False)
    ttnn.deallocate(ttnn_to_memory_config_255, False)
    ttnn.deallocate(ttnn_matmul_83, False)
    ttnn.deallocate(ttnn_to_memory_config_256, False)
    return ttnn_reshape_404


def LayerNorm_16_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_135 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_136 = ttnn.multiply(
        ttnn_multiply_135,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_147 = ttnn.add(
        ttnn_multiply_136,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_257 = ttnn.to_memory_config(
        ttnn_add_147,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_135, False)
    ttnn.deallocate(ttnn_multiply_136, False)
    return ttnn_to_memory_config_257, ttnn_add_147


def Linear_9_0(input):
    ttnn_reshape_405 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_405


def Linear_81_0(input):
    ttnn_reshape_406 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_406


def LayerNorm_46_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_137 = ttnn.multiply(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_138 = ttnn.multiply(
        ttnn_multiply_137,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_148 = ttnn.add(
        ttnn_multiply_138,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_multiply_137, False)
    ttnn.deallocate(ttnn_multiply_138, False)
    return ttnn_add_148


def CLIPEncoderLayer_98_0(input_0, input_1, input_2, input_3):
    ttnn_add_149 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_258 = ttnn.to_memory_config(
        ttnn_add_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_40 = ttnn.sum(
        ttnn_to_memory_config_258,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_139 = ttnn.multiply(
        ttnn_sum_40,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_259 = ttnn.to_memory_config(
        ttnn_multiply_139,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_407 = ttnn.reshape(
        ttnn_to_memory_config_259,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_20 = ttnn.neg(
        ttnn_reshape_407,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_150 = ttnn.add(
        ttnn_to_memory_config_258,
        ttnn_neg_20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_260 = ttnn.to_memory_config(
        ttnn_add_150,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_140 = ttnn.multiply(
        ttnn_to_memory_config_260,
        ttnn_to_memory_config_260,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_261 = ttnn.to_memory_config(
        ttnn_multiply_140,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_41 = ttnn.sum(
        ttnn_to_memory_config_261,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_141 = ttnn.multiply(
        ttnn_sum_41,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_151 = ttnn.add(
        ttnn_multiply_141,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_262 = ttnn.to_memory_config(
        ttnn_add_151,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_20 = ttnn.rsqrt(
        ttnn_to_memory_config_262,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_408 = ttnn.reshape(
        ttnn_rsqrt_20,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_add_149, False)
    # ttnn.deallocate(ttnn_to_memory_config_258, False)
    ttnn.deallocate(ttnn_sum_40, False)
    ttnn.deallocate(ttnn_multiply_139, False)
    ttnn.deallocate(ttnn_to_memory_config_259, False)
    ttnn.deallocate(ttnn_reshape_407, False)
    ttnn.deallocate(ttnn_neg_20, False)
    ttnn.deallocate(ttnn_add_150, False)
    # ttnn.deallocate(ttnn_to_memory_config_260, False)
    ttnn.deallocate(ttnn_multiply_140, False)
    ttnn.deallocate(ttnn_to_memory_config_261, False)
    ttnn.deallocate(ttnn_sum_41, False)
    ttnn.deallocate(ttnn_multiply_141, False)
    ttnn.deallocate(ttnn_add_151, False)
    ttnn.deallocate(ttnn_to_memory_config_262, False)
    ttnn.deallocate(ttnn_rsqrt_20, False)
    # ttnn.deallocate(ttnn_reshape_408, False)
    return ttnn_to_memory_config_260, ttnn_to_memory_config_258, ttnn_reshape_408


def Linear_109_0(input_0, input_1, input_2):
    ttnn_matmul_84 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_152 = ttnn.add(
        ttnn_matmul_84,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_263 = ttnn.to_memory_config(
        ttnn_add_152,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_409 = ttnn.reshape(
        ttnn_to_memory_config_263,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_84, False)
    ttnn.deallocate(ttnn_add_152, False)
    ttnn.deallocate(ttnn_to_memory_config_263, False)
    return ttnn_reshape_409


def CLIPEncoderLayer_74_0(input_0, input_1, input_2, input_3):
    ttnn_add_153 = ttnn.add(
        input_2,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_264 = ttnn.to_memory_config(
        ttnn_add_153,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_42 = ttnn.sum(
        ttnn_to_memory_config_264,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_142 = ttnn.multiply(
        ttnn_sum_42,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_265 = ttnn.to_memory_config(
        ttnn_multiply_142,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_410 = ttnn.reshape(
        ttnn_to_memory_config_265,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_21 = ttnn.neg(
        ttnn_reshape_410,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_154 = ttnn.add(
        ttnn_to_memory_config_264,
        ttnn_neg_21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_266 = ttnn.to_memory_config(
        ttnn_add_154,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_143 = ttnn.multiply(
        ttnn_to_memory_config_266,
        ttnn_to_memory_config_266,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_267 = ttnn.to_memory_config(
        ttnn_multiply_143,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_43 = ttnn.sum(
        ttnn_to_memory_config_267,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_144 = ttnn.multiply(
        ttnn_sum_43,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_155 = ttnn.add(
        ttnn_multiply_144,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_268 = ttnn.to_memory_config(
        ttnn_add_155,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_21 = ttnn.rsqrt(
        ttnn_to_memory_config_268,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_411 = ttnn.reshape(
        ttnn_rsqrt_21,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_153, False)
    # ttnn.deallocate(ttnn_to_memory_config_264, False)
    ttnn.deallocate(ttnn_sum_42, False)
    ttnn.deallocate(ttnn_multiply_142, False)
    ttnn.deallocate(ttnn_to_memory_config_265, False)
    ttnn.deallocate(ttnn_reshape_410, False)
    ttnn.deallocate(ttnn_neg_21, False)
    ttnn.deallocate(ttnn_add_154, False)
    # ttnn.deallocate(ttnn_to_memory_config_266, False)
    ttnn.deallocate(ttnn_multiply_143, False)
    ttnn.deallocate(ttnn_to_memory_config_267, False)
    ttnn.deallocate(ttnn_sum_43, False)
    ttnn.deallocate(ttnn_multiply_144, False)
    ttnn.deallocate(ttnn_add_155, False)
    ttnn.deallocate(ttnn_to_memory_config_268, False)
    ttnn.deallocate(ttnn_rsqrt_21, False)
    # ttnn.deallocate(ttnn_reshape_411, False)
    return ttnn_to_memory_config_264, ttnn_reshape_411, ttnn_to_memory_config_266


def CLIPEncoderLayer_128_0(input_0, input_1, input_2, input_3):
    ttnn_add_156 = ttnn.add(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_269 = ttnn.to_memory_config(
        ttnn_add_156,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_44 = ttnn.sum(
        ttnn_to_memory_config_269,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_145 = ttnn.multiply(
        ttnn_sum_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_270 = ttnn.to_memory_config(
        ttnn_multiply_145,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_412 = ttnn.reshape(
        ttnn_to_memory_config_270,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_22 = ttnn.neg(
        ttnn_reshape_412,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_157 = ttnn.add(
        ttnn_to_memory_config_269,
        ttnn_neg_22,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_271 = ttnn.to_memory_config(
        ttnn_add_157,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_146 = ttnn.multiply(
        ttnn_to_memory_config_271,
        ttnn_to_memory_config_271,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_272 = ttnn.to_memory_config(
        ttnn_multiply_146,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_45 = ttnn.sum(
        ttnn_to_memory_config_272,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_147 = ttnn.multiply(
        ttnn_sum_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_158 = ttnn.add(
        ttnn_multiply_147,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_273 = ttnn.to_memory_config(
        ttnn_add_158,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_22 = ttnn.rsqrt(
        ttnn_to_memory_config_273,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_413 = ttnn.reshape(
        ttnn_rsqrt_22,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_156, False)
    # ttnn.deallocate(ttnn_to_memory_config_269, False)
    ttnn.deallocate(ttnn_sum_44, False)
    ttnn.deallocate(ttnn_multiply_145, False)
    ttnn.deallocate(ttnn_to_memory_config_270, False)
    ttnn.deallocate(ttnn_reshape_412, False)
    ttnn.deallocate(ttnn_neg_22, False)
    ttnn.deallocate(ttnn_add_157, False)
    # ttnn.deallocate(ttnn_to_memory_config_271, False)
    ttnn.deallocate(ttnn_multiply_146, False)
    ttnn.deallocate(ttnn_to_memory_config_272, False)
    ttnn.deallocate(ttnn_sum_45, False)
    ttnn.deallocate(ttnn_multiply_147, False)
    ttnn.deallocate(ttnn_add_158, False)
    ttnn.deallocate(ttnn_to_memory_config_273, False)
    ttnn.deallocate(ttnn_rsqrt_22, False)
    # ttnn.deallocate(ttnn_reshape_413, False)
    return ttnn_reshape_413, ttnn_to_memory_config_271, ttnn_to_memory_config_269


def Linear_43_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_34 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_414 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_34,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_274 = ttnn.to_memory_config(
        ttnn_reshape_414,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_85 = ttnn.matmul(
        ttnn_to_memory_config_274,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_159 = ttnn.add(
        ttnn_matmul_85,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_275 = ttnn.to_memory_config(
        ttnn_add_159,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_415 = ttnn.reshape(
        ttnn_to_memory_config_275,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_34, False)
    ttnn.deallocate(ttnn_reshape_414, False)
    ttnn.deallocate(ttnn_to_memory_config_274, False)
    ttnn.deallocate(ttnn_matmul_85, False)
    ttnn.deallocate(ttnn_add_159, False)
    ttnn.deallocate(ttnn_to_memory_config_275, False)
    return ttnn_reshape_415


def CLIPAttention_138_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_276 = ttnn.to_memory_config(
        input_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_86 = ttnn.matmul(
        ttnn_to_memory_config_276,
        input_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_277 = ttnn.to_memory_config(
        ttnn_matmul_86,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_416 = ttnn.reshape(
        ttnn_to_memory_config_277,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_148 = ttnn.multiply(
        ttnn_reshape_416,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_multiply_148,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_9 = ttnn.softmax(
        ttnn_typecast_19,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_softmax_9,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_417 = ttnn.reshape(
        ttnn_typecast_20,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_278 = ttnn.to_memory_config(
        ttnn_reshape_417,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_87 = ttnn.matmul(
        ttnn_to_memory_config_278,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_279 = ttnn.to_memory_config(
        ttnn_matmul_87,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_418 = ttnn.reshape(
        ttnn_to_memory_config_279,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_276, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_86, False)
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_277, False)
    ttnn.deallocate(ttnn_reshape_416, False)
    ttnn.deallocate(ttnn_multiply_148, False)
    ttnn.deallocate(ttnn_typecast_19, False)
    ttnn.deallocate(ttnn_softmax_9, False)
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn.deallocate(ttnn_reshape_417, False)
    ttnn.deallocate(ttnn_to_memory_config_278, False)
    ttnn.deallocate(ttnn_matmul_87, False)
    ttnn.deallocate(ttnn_to_memory_config_279, False)
    return ttnn_reshape_418


def Linear_85_0(input_0, input_1, input_2):
    ttnn_matmul_88 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_160 = ttnn.add(
        ttnn_matmul_88,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_280 = ttnn.to_memory_config(
        ttnn_add_160,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_419 = ttnn.reshape(
        ttnn_to_memory_config_280,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_matmul_88, False)
    ttnn.deallocate(ttnn_add_160, False)
    ttnn.deallocate(ttnn_to_memory_config_280, False)
    return ttnn_reshape_419


def CLIPAttention_6_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_281 = ttnn.to_memory_config(
        input_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_89 = ttnn.matmul(
        ttnn_to_memory_config_281,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_282 = ttnn.to_memory_config(
        ttnn_matmul_89,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_420 = ttnn.reshape(
        ttnn_to_memory_config_282,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_149 = ttnn.multiply(
        ttnn_reshape_420,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_multiply_149,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_10 = ttnn.softmax(
        ttnn_typecast_21,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_softmax_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_421 = ttnn.reshape(
        ttnn_typecast_22,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_283 = ttnn.to_memory_config(
        ttnn_reshape_421,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_90 = ttnn.matmul(
        ttnn_to_memory_config_283,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_284 = ttnn.to_memory_config(
        ttnn_matmul_90,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_422 = ttnn.reshape(
        ttnn_to_memory_config_284,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_to_memory_config_281, False)
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_matmul_89, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_to_memory_config_282, False)
    ttnn.deallocate(ttnn_reshape_420, False)
    ttnn.deallocate(ttnn_multiply_149, False)
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn.deallocate(ttnn_softmax_10, False)
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn.deallocate(ttnn_reshape_421, False)
    ttnn.deallocate(ttnn_to_memory_config_283, False)
    ttnn.deallocate(ttnn_matmul_90, False)
    ttnn.deallocate(ttnn_to_memory_config_284, False)
    return ttnn_reshape_422


def Linear_51_0(input):
    ttnn_reshape_423 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_423


def Linear_31_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_35 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_424 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_35,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_285 = ttnn.to_memory_config(
        ttnn_reshape_424,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_91 = ttnn.matmul(
        ttnn_to_memory_config_285,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_161 = ttnn.add(
        ttnn_matmul_91,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_286 = ttnn.to_memory_config(
        ttnn_add_161,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_425 = ttnn.reshape(
        ttnn_to_memory_config_286,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_transformer_concatenate_heads_35, False)
    ttnn.deallocate(ttnn_reshape_424, False)
    ttnn.deallocate(ttnn_to_memory_config_285, False)
    ttnn.deallocate(ttnn_matmul_91, False)
    ttnn.deallocate(ttnn_add_161, False)
    ttnn.deallocate(ttnn_to_memory_config_286, False)
    return ttnn_reshape_425


def Linear_15_0(input):
    ttnn_reshape_426 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_426


def CLIPAttention_66_0(input_0, input_1, input_2, input_3):
    ttnn_to_memory_config_287 = ttnn.to_memory_config(
        input_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_92 = ttnn.matmul(
        ttnn_to_memory_config_287,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_288 = ttnn.to_memory_config(
        ttnn_matmul_92,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_427 = ttnn.reshape(
        ttnn_to_memory_config_288,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_150 = ttnn.multiply(
        ttnn_reshape_427,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_multiply_150,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_softmax_11 = ttnn.softmax(
        ttnn_typecast_23,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_softmax_11,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_428 = ttnn.reshape(
        ttnn_typecast_24,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_289 = ttnn.to_memory_config(
        ttnn_reshape_428,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_93 = ttnn.matmul(
        ttnn_to_memory_config_289,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_290 = ttnn.to_memory_config(
        ttnn_matmul_93,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_429 = ttnn.reshape(
        ttnn_to_memory_config_290,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_to_memory_config_287, False)
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_matmul_92, False)
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_to_memory_config_288, False)
    ttnn.deallocate(ttnn_reshape_427, False)
    ttnn.deallocate(ttnn_multiply_150, False)
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn.deallocate(ttnn_softmax_11, False)
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn.deallocate(ttnn_reshape_428, False)
    ttnn.deallocate(ttnn_to_memory_config_289, False)
    ttnn.deallocate(ttnn_matmul_93, False)
    ttnn.deallocate(ttnn_to_memory_config_290, False)
    return ttnn_reshape_429


def Linear_111_0(input):
    ttnn_reshape_430 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_430


def LayerNorm_142_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_151 = ttnn.multiply(
        input_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_152 = ttnn.multiply(
        ttnn_multiply_151,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_162 = ttnn.add(
        ttnn_multiply_152,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_1, False)
    ttnn.deallocate(ttnn_multiply_151, False)
    ttnn.deallocate(ttnn_multiply_152, False)
    return ttnn_add_162


def CLIPEncoderLayer_134_0(input_0, input_1, input_2, input_3):
    ttnn_add_163 = ttnn.add(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_291 = ttnn.to_memory_config(
        ttnn_add_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_46 = ttnn.sum(
        ttnn_to_memory_config_291,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_153 = ttnn.multiply(
        ttnn_sum_46,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_292 = ttnn.to_memory_config(
        ttnn_multiply_153,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_431 = ttnn.reshape(
        ttnn_to_memory_config_292,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_23 = ttnn.neg(
        ttnn_reshape_431,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_164 = ttnn.add(
        ttnn_to_memory_config_291,
        ttnn_neg_23,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_293 = ttnn.to_memory_config(
        ttnn_add_164,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_154 = ttnn.multiply(
        ttnn_to_memory_config_293,
        ttnn_to_memory_config_293,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_294 = ttnn.to_memory_config(
        ttnn_multiply_154,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_47 = ttnn.sum(
        ttnn_to_memory_config_294,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_155 = ttnn.multiply(
        ttnn_sum_47,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_165 = ttnn.add(
        ttnn_multiply_155,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_295 = ttnn.to_memory_config(
        ttnn_add_165,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_23 = ttnn.rsqrt(
        ttnn_to_memory_config_295,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_432 = ttnn.reshape(
        ttnn_rsqrt_23,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn.deallocate(ttnn_add_163, False)
    # ttnn.deallocate(ttnn_to_memory_config_291, False)
    ttnn.deallocate(ttnn_sum_46, False)
    ttnn.deallocate(ttnn_multiply_153, False)
    ttnn.deallocate(ttnn_to_memory_config_292, False)
    ttnn.deallocate(ttnn_reshape_431, False)
    ttnn.deallocate(ttnn_neg_23, False)
    ttnn.deallocate(ttnn_add_164, False)
    # ttnn.deallocate(ttnn_to_memory_config_293, False)
    ttnn.deallocate(ttnn_multiply_154, False)
    ttnn.deallocate(ttnn_to_memory_config_294, False)
    ttnn.deallocate(ttnn_sum_47, False)
    ttnn.deallocate(ttnn_multiply_155, False)
    ttnn.deallocate(ttnn_add_165, False)
    ttnn.deallocate(ttnn_to_memory_config_295, False)
    ttnn.deallocate(ttnn_rsqrt_23, False)
    # ttnn.deallocate(ttnn_reshape_432, False)
    return ttnn_to_memory_config_293, ttnn_reshape_432, ttnn_to_memory_config_291


def CLIPEncoderLayer_20_0(input_0, input_1, input_2, input_3):
    ttnn_add_166 = ttnn.add(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_296 = ttnn.to_memory_config(
        ttnn_add_166,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_48 = ttnn.sum(
        ttnn_to_memory_config_296,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_156 = ttnn.multiply(
        ttnn_sum_48,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_297 = ttnn.to_memory_config(
        ttnn_multiply_156,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_433 = ttnn.reshape(
        ttnn_to_memory_config_297,
        [2, 50, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_24 = ttnn.neg(
        ttnn_reshape_433,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_167 = ttnn.add(
        ttnn_to_memory_config_296,
        ttnn_neg_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_298 = ttnn.to_memory_config(
        ttnn_add_167,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_157 = ttnn.multiply(
        ttnn_to_memory_config_298,
        ttnn_to_memory_config_298,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_299 = ttnn.to_memory_config(
        ttnn_multiply_157,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_49 = ttnn.sum(
        ttnn_to_memory_config_299,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_158 = ttnn.multiply(
        ttnn_sum_49,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_168 = ttnn.add(
        ttnn_multiply_158,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_300 = ttnn.to_memory_config(
        ttnn_add_168,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_24 = ttnn.rsqrt(
        ttnn_to_memory_config_300,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_434 = ttnn.reshape(
        ttnn_rsqrt_24,
        [100, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(ttnn_add_166, False)
    # ttnn.deallocate(ttnn_to_memory_config_296, False)
    ttnn.deallocate(ttnn_sum_48, False)
    ttnn.deallocate(ttnn_multiply_156, False)
    ttnn.deallocate(ttnn_to_memory_config_297, False)
    ttnn.deallocate(ttnn_reshape_433, False)
    ttnn.deallocate(ttnn_neg_24, False)
    ttnn.deallocate(ttnn_add_167, False)
    # ttnn.deallocate(ttnn_to_memory_config_298, False)
    ttnn.deallocate(ttnn_multiply_157, False)
    ttnn.deallocate(ttnn_to_memory_config_299, False)
    ttnn.deallocate(ttnn_sum_49, False)
    ttnn.deallocate(ttnn_multiply_158, False)
    ttnn.deallocate(ttnn_add_168, False)
    ttnn.deallocate(ttnn_to_memory_config_300, False)
    ttnn.deallocate(ttnn_rsqrt_24, False)
    # ttnn.deallocate(ttnn_reshape_434, False)
    return ttnn_to_memory_config_296, ttnn_to_memory_config_298, ttnn_reshape_434


def Linear_53_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7):
    ttnn_matmul_94 = ttnn.matmul(
        input_3,
        input_7,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_301 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_169 = ttnn.add(
        ttnn_matmul_94,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_302 = ttnn.to_memory_config(
        input_5,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_matmul_95 = ttnn.matmul(
        ttnn_to_memory_config_301,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_to_memory_config_303 = ttnn.to_memory_config(
        ttnn_add_169,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_96 = ttnn.matmul(
        ttnn_to_memory_config_302,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_170 = ttnn.add(
        ttnn_matmul_95,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_171 = ttnn.add(
        ttnn_matmul_96,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_reshape_435 = ttnn.reshape(
        ttnn_to_memory_config_303,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_304 = ttnn.to_memory_config(
        ttnn_add_170,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_305 = ttnn.to_memory_config(
        ttnn_add_171,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_86 = ttnn.permute(
        ttnn_reshape_435,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_436 = ttnn.reshape(
        ttnn_to_memory_config_304,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_437 = ttnn.reshape(
        ttnn_to_memory_config_305,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_438 = ttnn.reshape(
        ttnn_permute_86,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_87 = ttnn.permute(
        ttnn_reshape_436,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_permute_88 = ttnn.permute(
        ttnn_reshape_437,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_439 = ttnn.reshape(
        ttnn_permute_87,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_440 = ttnn.reshape(
        ttnn_permute_88,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_3, False)
    ttnn.deallocate(input_5, False)
    ttnn.deallocate(ttnn_matmul_94, False)
    ttnn.deallocate(ttnn_add_169, False)
    ttnn.deallocate(ttnn_to_memory_config_301, False)
    ttnn.deallocate(ttnn_to_memory_config_302, False)
    ttnn.deallocate(ttnn_matmul_96, False)
    ttnn.deallocate(ttnn_to_memory_config_303, False)
    ttnn.deallocate(ttnn_matmul_95, False)
    ttnn.deallocate(ttnn_reshape_435, False)
    ttnn.deallocate(ttnn_add_171, False)
    ttnn.deallocate(ttnn_add_170, False)
    ttnn.deallocate(ttnn_to_memory_config_304, False)
    ttnn.deallocate(ttnn_permute_86, False)
    ttnn.deallocate(ttnn_to_memory_config_305, False)
    ttnn.deallocate(ttnn_reshape_436, False)
    ttnn.deallocate(ttnn_reshape_437, False)
    ttnn.deallocate(ttnn_permute_87, False)
    ttnn.deallocate(ttnn_permute_88, False)
    return ttnn_reshape_440, ttnn_reshape_439, ttnn_reshape_438


def QuickGELUActivation_36_0(input_0, input_1):
    ttnn_multiply_159 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_306 = ttnn.to_memory_config(
        ttnn_multiply_159,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_11 = ttnn.sigmoid(
        ttnn_to_memory_config_306,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_160 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn.deallocate(ttnn_multiply_159, False)
    ttnn.deallocate(ttnn_to_memory_config_306, False)
    ttnn.deallocate(ttnn_sigmoid_11, False)
    return ttnn_multiply_160


def CLIPVisionTransformer_147_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_slice_0 = ttnn.slice(
        input_2,
        [0, 0, 0],
        [2, 1, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_441 = ttnn.reshape(
        ttnn_slice_0,
        [2, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_50 = ttnn.sum(
        ttnn_reshape_441,
        [1],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_161 = ttnn.multiply(
        ttnn_sum_50,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_307 = ttnn.to_memory_config(
        ttnn_multiply_161,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_442 = ttnn.reshape(
        ttnn_to_memory_config_307,
        [2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_25 = ttnn.neg(
        ttnn_reshape_442,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_172 = ttnn.add(
        ttnn_reshape_441,
        ttnn_neg_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_308 = ttnn.to_memory_config(
        ttnn_add_172,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_162 = ttnn.multiply(
        ttnn_to_memory_config_308,
        ttnn_to_memory_config_308,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_309 = ttnn.to_memory_config(
        ttnn_multiply_162,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_51 = ttnn.sum(
        ttnn_to_memory_config_309,
        [1],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_163 = ttnn.multiply(
        ttnn_sum_51,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_310 = ttnn.to_memory_config(
        ttnn_multiply_163,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_443 = ttnn.reshape(
        ttnn_to_memory_config_310,
        [2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_173 = ttnn.add(
        ttnn_reshape_443,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_311 = ttnn.to_memory_config(
        ttnn_add_173,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rsqrt_25 = ttnn.rsqrt(
        ttnn_to_memory_config_311,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_164 = ttnn.multiply(
        ttnn_to_memory_config_308,
        ttnn_rsqrt_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_multiply_165 = ttnn.multiply(
        ttnn_multiply_164,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_174 = ttnn.add(
        ttnn_multiply_165,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_312 = ttnn.to_memory_config(
        ttnn_add_174,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn.deallocate(ttnn_reshape_441, False)
    ttnn.deallocate(ttnn_sum_50, False)
    ttnn.deallocate(ttnn_multiply_161, False)
    ttnn.deallocate(ttnn_to_memory_config_307, False)
    ttnn.deallocate(ttnn_reshape_442, False)
    ttnn.deallocate(ttnn_neg_25, False)
    ttnn.deallocate(ttnn_add_172, False)
    ttnn.deallocate(ttnn_to_memory_config_308, False)
    ttnn.deallocate(ttnn_multiply_162, False)
    ttnn.deallocate(ttnn_to_memory_config_309, False)
    ttnn.deallocate(ttnn_sum_51, False)
    ttnn.deallocate(ttnn_multiply_163, False)
    ttnn.deallocate(ttnn_to_memory_config_310, False)
    ttnn.deallocate(ttnn_reshape_443, False)
    ttnn.deallocate(ttnn_add_173, False)
    ttnn.deallocate(ttnn_to_memory_config_311, False)
    ttnn.deallocate(ttnn_rsqrt_25, False)
    ttnn.deallocate(ttnn_multiply_164, False)
    ttnn.deallocate(ttnn_multiply_165, False)
    ttnn.deallocate(ttnn_add_174, False)
    return ttnn_to_memory_config_312


def Linear_135_0(input):
    ttnn_reshape_444 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return ttnn_reshape_444


def load_inputs_for__main():
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_139 = utils.load_tensor(
        "./tensors/arg139.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_140 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_141 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_142 = utils.load_tensor(
        "./tensors/arg142.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_143 = utils.load_tensor(
        "./tensors/arg143.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_144 = utils.load_tensor(
        "./tensors/arg144.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_145 = utils.load_tensor(
        "./tensors/arg145.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_146 = utils.load_tensor(
        "./tensors/arg146.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_147 = utils.load_tensor(
        "./tensors/arg147.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_148 = utils.load_tensor(
        "./tensors/arg148.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_149 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_150 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_151 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    # Don't load activation tensor, it will be loaded in main() function
    utils_load_tensor_152 = None
    # utils_load_tensor_152 = utils.load_tensor(
    #     "./tensors/arg152.tensorbin",
    #     ttnn.Layout.TILE,
    #     ttnn.DataType.BFLOAT16,
    #     utils_DeviceGetter_get_device_8,
    #     ttnn.MemoryConfig(
    #         ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    #     ),
    # )
    utils_load_tensor_153 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_154 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_155 = utils.load_tensor(
        "./tensors/arg155.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_156 = utils.load_tensor(
        "./tensors/arg156.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_157 = utils.load_tensor(
        "./tensors/arg157.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_158 = utils.load_tensor(
        "./tensors/arg158.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_159 = utils.load_tensor(
        "./tensors/arg159.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_160 = utils.load_tensor(
        "./tensors/arg160.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_161 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_162 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_163 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_164 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_165 = utils.load_tensor(
        "./tensors/arg165.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_166 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_167 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_168 = utils.load_tensor(
        "./tensors/arg168.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_169 = utils.load_tensor(
        "./tensors/arg169.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_170 = utils.load_tensor(
        "./tensors/arg170.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_171 = utils.load_tensor(
        "./tensors/arg171.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_172 = utils.load_tensor(
        "./tensors/arg172.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_173 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_174 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_175 = utils.load_tensor(
        "./tensors/arg175.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_176 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_177 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_178 = utils.load_tensor(
        "./tensors/arg178.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_179 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_180 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_181 = utils.load_tensor(
        "./tensors/arg181.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_182 = utils.load_tensor(
        "./tensors/arg182.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_183 = utils.load_tensor(
        "./tensors/arg183.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_184 = utils.load_tensor(
        "./tensors/arg184.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_185 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_186 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_187 = utils.load_tensor(
        "./tensors/arg187.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_188 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_189 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_190 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_191 = utils.load_tensor(
        "./tensors/arg191.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_192 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_193 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_194 = utils.load_tensor(
        "./tensors/arg194.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_195 = utils.load_tensor(
        "./tensors/arg195.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_196 = utils.load_tensor(
        "./tensors/arg196.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_197 = utils.load_tensor(
        "./tensors/arg197.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_198 = utils.load_tensor(
        "./tensors/arg198.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_199 = utils.load_tensor(
        "./tensors/arg199.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_200 = utils.load_tensor(
        "./tensors/arg200.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_201 = utils.load_tensor(
        "./tensors/arg201.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_8,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_262 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
        utils_load_tensor_126,
        utils_load_tensor_127,
        utils_load_tensor_128,
        utils_load_tensor_129,
        utils_load_tensor_130,
        utils_load_tensor_131,
        utils_load_tensor_132,
        utils_load_tensor_133,
        utils_load_tensor_134,
        utils_load_tensor_135,
        utils_load_tensor_136,
        utils_load_tensor_137,
        utils_load_tensor_138,
        utils_load_tensor_139,
        utils_load_tensor_140,
        utils_load_tensor_141,
        utils_load_tensor_142,
        utils_load_tensor_143,
        utils_load_tensor_144,
        utils_load_tensor_145,
        utils_load_tensor_146,
        utils_load_tensor_147,
        utils_load_tensor_148,
        utils_load_tensor_149,
        utils_load_tensor_150,
        utils_load_tensor_151,
        utils_load_tensor_152,
        utils_load_tensor_153,
        utils_load_tensor_154,
        utils_load_tensor_155,
        utils_load_tensor_156,
        utils_load_tensor_157,
        utils_load_tensor_158,
        utils_load_tensor_159,
        utils_load_tensor_160,
        utils_load_tensor_161,
        utils_load_tensor_162,
        utils_load_tensor_163,
        utils_load_tensor_164,
        utils_load_tensor_165,
        utils_load_tensor_166,
        utils_load_tensor_167,
        utils_load_tensor_168,
        utils_load_tensor_169,
        utils_load_tensor_170,
        utils_load_tensor_171,
        utils_load_tensor_172,
        utils_load_tensor_173,
        utils_load_tensor_174,
        utils_load_tensor_175,
        utils_load_tensor_176,
        utils_load_tensor_177,
        utils_load_tensor_178,
        utils_load_tensor_179,
        utils_load_tensor_180,
        utils_load_tensor_181,
        utils_load_tensor_182,
        utils_load_tensor_183,
        utils_load_tensor_184,
        utils_load_tensor_185,
        utils_load_tensor_186,
        utils_load_tensor_187,
        utils_load_tensor_188,
        utils_load_tensor_189,
        utils_load_tensor_190,
        utils_load_tensor_191,
        utils_load_tensor_192,
        utils_load_tensor_193,
        utils_load_tensor_194,
        utils_load_tensor_195,
        utils_load_tensor_196,
        utils_load_tensor_197,
        utils_load_tensor_198,
        utils_load_tensor_199,
        utils_load_tensor_200,
        utils_load_tensor_201,
    ]
    return util_create_list_262


def run_pytorch_model(input):
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model(pixel_values=input)

    return outputs


def get_input():
    # This function fetches the example input image from the COCO dataset
    #
    # Batch size is 2
    # Dtype is bfloat16

    # Fetch image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs["pixel_values"]

    # Batch 2
    inputs = inputs.repeat(2, 1, 1, 1)

    # Convert to bfloat16
    inputs = inputs.to(torch.bfloat16)

    return inputs


def main():
    # Load input tensor
    input_torch = get_input()

    # Calculate torch output
    output_torch = run_pytorch_model(input_torch)

    # Convert torch input to host TTNN tensor
    input_ttnn_host = ttnn.from_torch(input_torch)
    input_ttnn_host = ttnn.to_layout(input_ttnn_host, ttnn.Layout.TILE)
    input_ttnn_host = ttnn.to_dtype(input_ttnn_host, ttnn.DataType.BFLOAT16)

    # Load params
    load_inputs_for__main_0 = load_inputs_for__main()

    # Get device
    device = utils.DeviceGetter.get_device((1, 1))

    # Run ttnn model
    for i in range(10):
        start_time = time.time()

        # Move input to device and override the input (activation) tensor in tensor list
        input_ttnn_device = ttnn.to_device(input_ttnn_host, device, ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ))
        load_inputs_for__main_0[152] = input_ttnn_device

        # Run ttnn model
        out = _main(load_inputs_for__main_0)

        # Get outputs
        out_img_embeddings = ttnn.from_device(out[0], blocking=True)
        out_last_hidden_state = ttnn.from_device(out[1], blocking=True)

        end_time = time.time()

        # Calculate FPS and PCC
        fps = (2.0 / (end_time - start_time))  # batch size is 2
        pcc = utils.calculate_pcc(output_torch['image_embeds'], ttnn.to_torch(out_img_embeddings))

        # Print results
        print(f"FPS for iteration {i}: {fps:.2f}")
        print(f"PCC for iteration {i}: {pcc:.6f}")

    return 0


if __name__ == "__main__":
    main()
