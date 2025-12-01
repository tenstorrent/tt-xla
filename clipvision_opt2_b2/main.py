import ttnn
import utils
import time
import torch
import numpy as np
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from transformers.image_utils import load_image

from loader import load_inputs_for_clipvision_ttnn
from model import *
from consteval import *


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
    inputs_for_clipvision_ttnn = load_inputs_for_clipvision_ttnn()

    # Get device
    device = utils.DeviceGetter.get_device((1, 1))

    # Run ttnn model
    for i in range(10):
        start_time = time.time()

        # Move input to device and override the input (activation) tensor in tensor list
        input_ttnn_device = ttnn.to_device(input_ttnn_host, device, ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ))
        inputs_for_clipvision_ttnn[152] = input_ttnn_device

        # Run ttnn model
        out = run_ttnn_model(inputs_for_clipvision_ttnn)

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


def run_pytorch_model(input):
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.to(torch.bfloat16)

    with torch.inference_mode():
        outputs = model(pixel_values=input)

    return outputs


def run_ttnn_model(input):
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


if __name__ == "__main__":
    main()
