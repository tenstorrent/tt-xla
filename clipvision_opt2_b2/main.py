from . import utils
import ttnn
import time
import torch
import numpy as np
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from transformers.image_utils import load_image

from .loader import load_inputs_for_clipvision_ttnn
from .model import *
from .consteval import *

from tracy import signpost

import pytest

@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32*1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)  
def test_main(mesh_device):
    # Load input tensor
    input_torch = get_input()

    # Calculate torch output
    output_torch = run_pytorch_model(input_torch)

    # Convert torch input to host TTNN tensor
    input_ttnn_host = ttnn.from_torch(input_torch)
    input_ttnn_host = ttnn.to_layout(input_ttnn_host, ttnn.Layout.TILE)
    input_ttnn_host = ttnn.to_dtype(input_ttnn_host, ttnn.DataType.BFLOAT16)

    # Load params
    inputs_for_clipvision_ttnn = load_inputs_for_clipvision_ttnn(mesh_device=mesh_device)

    # Run ttnn model
    for i in range(10):
        start_time = time.time()

        signpost(f"ttnn_model_start_{i}")
        # Move input to device and override the input (activation) tensor in tensor list
        input_ttnn_device = ttnn.to_device(input_ttnn_host, mesh_device, ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ))
        inputs_for_clipvision_ttnn[152] = input_ttnn_device

        # Run ttnn model
        out = run_ttnn_model(mesh_device, inputs_for_clipvision_ttnn)

        # Get outputs
        out_img_embeddings = ttnn.from_device(out[0], blocking=True)
        out_last_hidden_state = ttnn.from_device(out[1], blocking=True)
        signpost(f"ttnn_model_end_{i}")

        # ttnn.synchronize_device(device) ## <==

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


def run_ttnn_model(mesh_device, input):
    # Unpack all inputs
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
    
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost("const_eval_start")
    # Execute all const_eval functions and get results
    ce = execute_all_const_evals(mesh_device, input)
    signpost("const_eval_end")

    ttnn.ReadDeviceProfiler(mesh_device)

    # Map const_eval results to their variable names (for backward compatibility)
    utils_constEvalFuncWrapperZeroArg_0_0 = ce[0][0]
    utils_constEvalFuncWrapper_0_0 = ce[1][0]
    utils_constEvalFuncWrapper_1_0 = ce[2][0]
    utils_constEvalFuncWrapper_2_0 = ce[3][0]
    utils_constEvalFuncWrapper_3_0 = ce[4][0]
    utils_constEvalFuncWrapper_4_0 = ce[5][0]
    utils_constEvalFuncWrapper_5_0 = ce[6][0]
    utils_constEvalFuncWrapper_6_0 = ce[7][0]
    utils_constEvalFuncWrapper_7_0 = ce[8][0]
    utils_constEvalFuncWrapper_8_0 = ce[9][0]
    utils_constEvalFuncWrapper_9_0 = ce[10][0]
    utils_constEvalFuncWrapper_10_0 = ce[11][0]
    utils_constEvalFuncWrapper_11_0 = ce[12][0]
    utils_constEvalFuncWrapper_12_0 = ce[13][0]
    utils_constEvalFuncWrapper_13_0 = ce[14][0]
    utils_constEvalFuncWrapper_14_0 = ce[15][0]
    utils_constEvalFuncWrapper_15_0 = ce[16][0]
    utils_constEvalFuncWrapper_16_0 = ce[17][0]
    utils_constEvalFuncWrapper_17_0 = ce[18][0]
    utils_constEvalFuncWrapper_18_0 = ce[19][0]
    utils_constEvalFuncWrapper_19_0 = ce[20][0]
    utils_constEvalFuncWrapper_20_0 = ce[21][0]
    utils_constEvalFuncWrapper_21_0 = ce[22][0]
    utils_constEvalFuncWrapper_22_0 = ce[23][0]
    utils_constEvalFuncWrapper_23_0 = ce[24][0]
    utils_constEvalFuncWrapper_24_0 = ce[25][0]
    utils_constEvalFuncWrapper_25_0 = ce[26][0]
    utils_constEvalFuncWrapper_26_0 = ce[27][0]
    utils_constEvalFuncWrapper_27_0 = ce[28][0]
    utils_constEvalFuncWrapper_28_0 = ce[29][0]
    utils_constEvalFuncWrapper_29_0 = ce[30][0]
    utils_constEvalFuncWrapper_30_0 = ce[31][0]
    utils_constEvalFuncWrapper_31_0 = ce[32][0]
    utils_constEvalFuncWrapper_32_0 = ce[33][0]
    utils_constEvalFuncWrapper_33_0 = ce[34][0]
    utils_constEvalFuncWrapper_34_0 = ce[35][0]
    utils_constEvalFuncWrapper_35_0 = ce[36][0]
    utils_constEvalFuncWrapper_36_0 = ce[37][0]
    utils_constEvalFuncWrapper_37_0 = ce[38][0]
    utils_constEvalFuncWrapper_38_0 = ce[39][0]
    utils_constEvalFuncWrapper_39_0 = ce[40][0]
    utils_constEvalFuncWrapperZeroArg_1_0 = ce[41][0]
    utils_constEvalFuncWrapper_40_0 = ce[42][0]
    utils_constEvalFuncWrapper_41_0 = ce[43][0]
    utils_constEvalFuncWrapper_42_0 = ce[44][0]
    utils_constEvalFuncWrapper_43_0 = ce[45][0]
    utils_constEvalFuncWrapper_44_0 = ce[46][0]
    utils_constEvalFuncWrapper_45_0 = ce[47][0]
    utils_constEvalFuncWrapper_46_0 = ce[48][0]
    utils_constEvalFuncWrapper_47_0 = ce[49][0]
    utils_constEvalFuncWrapper_48_0 = ce[50][0]
    utils_constEvalFuncWrapperZeroArg_2_0 = ce[51][0]
    utils_constEvalFuncWrapper_49_0 = ce[52][0]
    utils_constEvalFuncWrapper_50_0 = ce[53][0]
    utils_constEvalFuncWrapper_51_0 = ce[54][0]
    utils_constEvalFuncWrapper_52_0 = ce[55][0]
    utils_constEvalFuncWrapper_53_0 = ce[56][0]
    utils_constEvalFuncWrapperZeroArg_3_0 = ce[57][0]
    utils_constEvalFuncWrapper_54_0 = ce[58][0]
    utils_constEvalFuncWrapper_55_0 = ce[59][0]
    utils_constEvalFuncWrapper_56_0 = ce[60][0]
    utils_constEvalFuncWrapper_57_0 = ce[61][0]
    utils_constEvalFuncWrapper_58_0 = ce[62][0]
    utils_constEvalFuncWrapper_59_0 = ce[63][0]
    utils_constEvalFuncWrapper_60_0 = ce[64][0]
    utils_constEvalFuncWrapper_61_0 = ce[65][0]
    utils_constEvalFuncWrapper_62_0 = ce[66][0]
    utils_constEvalFuncWrapper_63_0 = ce[67][0]
    utils_constEvalFuncWrapper_64_0 = ce[68][0]
    utils_constEvalFuncWrapper_65_0 = ce[69][0]
    utils_constEvalFuncWrapper_66_0 = ce[70][0]
    utils_constEvalFuncWrapper_67_0 = ce[71][0]
    utils_constEvalFuncWrapper_68_0 = ce[72][0]
    utils_constEvalFuncWrapper_69_0 = ce[73][0]
    utils_constEvalFuncWrapper_70_0 = ce[74][0]
    utils_constEvalFuncWrapper_71_0 = ce[75][0]
    utils_constEvalFuncWrapper_72_0 = ce[76][0]
    utils_constEvalFuncWrapper_73_0 = ce[77][0]
    utils_constEvalFuncWrapper_74_0 = ce[78][0]
    utils_constEvalFuncWrapper_75_0 = ce[79][0]
    utils_constEvalFuncWrapper_76_0 = ce[80][0]
    utils_constEvalFuncWrapper_77_0 = ce[81][0]
    utils_constEvalFuncWrapper_78_0 = ce[82][0]
    utils_constEvalFuncWrapper_79_0 = ce[83][0]
    utils_constEvalFuncWrapper_80_0 = ce[84][0]
    utils_constEvalFuncWrapper_81_0 = ce[85][0]
    utils_constEvalFuncWrapper_82_0 = ce[86][0]
    utils_constEvalFuncWrapper_83_0 = ce[87][0]
    utils_constEvalFuncWrapper_84_0 = ce[88][0]
    utils_constEvalFuncWrapper_85_0 = ce[89][0]
    utils_constEvalFuncWrapper_86_0 = ce[90][0]
    utils_constEvalFuncWrapper_87_0 = ce[91][0]
    utils_constEvalFuncWrapper_88_0 = ce[92][0]
    utils_constEvalFuncWrapper_89_0 = ce[93][0]
    utils_constEvalFuncWrapper_90_0 = ce[94][0]
    utils_constEvalFuncWrapper_91_0 = ce[95][0]
    utils_constEvalFuncWrapper_92_0 = ce[96][0]
    utils_constEvalFuncWrapper_93_0 = ce[97][0]
    utils_constEvalFuncWrapper_94_0 = ce[98][0]
    utils_constEvalFuncWrapper_95_0 = ce[99][0]
    utils_constEvalFuncWrapper_96_0 = ce[100][0]
    utils_constEvalFuncWrapper_97_0 = ce[101][0]
    utils_constEvalFuncWrapper_98_0 = ce[102][0]
    utils_constEvalFuncWrapper_99_0 = ce[103][0]
    utils_constEvalFuncWrapper_100_0 = ce[104][0]
    utils_constEvalFuncWrapper_101_0 = ce[105][0]
    utils_constEvalFuncWrapper_102_0 = ce[106][0]
    utils_constEvalFuncWrapper_103_0 = ce[107][0]
    utils_constEvalFuncWrapper_104_0 = ce[108][0]
    utils_constEvalFuncWrapper_105_0 = ce[109][0]
    utils_constEvalFuncWrapper_106_0 = ce[110][0]
    utils_constEvalFuncWrapper_107_0 = ce[111][0]
    utils_constEvalFuncWrapper_108_0 = ce[112][0]
    utils_constEvalFuncWrapper_109_0 = ce[113][0]
    utils_constEvalFuncWrapper_110_0 = ce[114][0]
    utils_constEvalFuncWrapper_111_0 = ce[115][0]
    utils_constEvalFuncWrapper_112_0 = ce[116][0]
    utils_constEvalFuncWrapper_113_0 = ce[117][0]
    utils_constEvalFuncWrapper_114_0 = ce[118][0]
    utils_constEvalFuncWrapperZeroArg_4_0 = ce[119][0]
    utils_constEvalFuncWrapper_115_0 = ce[120][0]
    utils_constEvalFuncWrapper_116_0 = ce[121][0]
    utils_constEvalFuncWrapper_117_0 = ce[122][0]
    utils_constEvalFuncWrapper_118_0 = ce[123][0]
    utils_constEvalFuncWrapper_119_0 = ce[124][0]
    utils_constEvalFuncWrapper_120_0 = ce[125][0]
    utils_constEvalFuncWrapper_121_0 = ce[126][0]
    utils_constEvalFuncWrapper_122_0 = ce[127][0]
    utils_constEvalFuncWrapperZeroArg_5_0 = ce[128][0]
    utils_constEvalFuncWrapper_123_0 = ce[129][0]
    utils_constEvalFuncWrapper_124_0 = ce[130][0]
    utils_constEvalFuncWrapper_125_0 = ce[131][0]
    utils_constEvalFuncWrapper_126_0 = ce[132][0]
    # Additional indices for multi-element results
    utils_constEvalFuncWrapperZeroArg_3_1 = ce[57][1]
    utils_constEvalFuncWrapperZeroArg_3_2 = ce[57][2]
    utils_constEvalFuncWrapperZeroArg_3_3 = ce[57][3]
    utils_constEvalFuncWrapperZeroArg_3_4 = ce[57][4]
    utils_constEvalFuncWrapperZeroArg_3_5 = ce[57][5]
    utils_constEvalFuncWrapperZeroArg_3_6 = ce[57][6]
    utils_constEvalFuncWrapperZeroArg_3_7 = ce[57][7]
    utils_constEvalFuncWrapperZeroArg_3_8 = ce[57][8]
    utils_constEvalFuncWrapperZeroArg_3_9 = ce[57][9]
    utils_constEvalFuncWrapperZeroArg_3_10 = ce[57][10]
    utils_constEvalFuncWrapperZeroArg_3_11 = ce[57][11]
    utils_constEvalFuncWrapperZeroArg_3_12 = ce[57][12]
    utils_constEvalFuncWrapperZeroArg_3_13 = ce[57][13]
    utils_constEvalFuncWrapperZeroArg_3_14 = ce[57][14]
    utils_constEvalFuncWrapperZeroArg_3_15 = ce[57][15]
    utils_constEvalFuncWrapperZeroArg_3_16 = ce[57][16]
    utils_constEvalFuncWrapperZeroArg_3_17 = ce[57][17]
    utils_constEvalFuncWrapperZeroArg_3_18 = ce[57][18]
    utils_constEvalFuncWrapperZeroArg_3_19 = ce[57][19]
    utils_constEvalFuncWrapperZeroArg_3_20 = ce[57][20]
    utils_constEvalFuncWrapperZeroArg_3_21 = ce[57][21]
    utils_constEvalFuncWrapperZeroArg_3_22 = ce[57][22]
    utils_constEvalFuncWrapperZeroArg_3_23 = ce[57][23]
    utils_constEvalFuncWrapperZeroArg_3_24 = ce[57][24]
    utils_constEvalFuncWrapperZeroArg_4_1 = ce[119][1]
    utils_constEvalFuncWrapperZeroArg_4_2 = ce[119][2]
    utils_constEvalFuncWrapperZeroArg_4_3 = ce[119][3]
    utils_constEvalFuncWrapperZeroArg_4_4 = ce[119][4]
    utils_constEvalFuncWrapperZeroArg_4_5 = ce[119][5]
    utils_constEvalFuncWrapperZeroArg_4_6 = ce[119][6]
    utils_constEvalFuncWrapperZeroArg_4_7 = ce[119][7]
    utils_constEvalFuncWrapperZeroArg_4_8 = ce[119][8]
    utils_constEvalFuncWrapperZeroArg_4_9 = ce[119][9]
    utils_constEvalFuncWrapperZeroArg_4_10 = ce[119][10]
    utils_constEvalFuncWrapperZeroArg_4_11 = ce[119][11]

    ttnn.ReadDeviceProfiler(mesh_device)
    
    # Model code
    CLIPVisionEmbeddings_0_0_0 = CLIPVisionEmbeddings_0_0(
        utils_constEvalFuncWrapper_78_0,
        utils_constEvalFuncWrapper_9_0,
        mesh_device,
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
    signpost("mid")
    ttnn.ReadDeviceProfiler(mesh_device)

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
