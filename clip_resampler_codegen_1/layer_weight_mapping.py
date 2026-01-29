# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight and CER mappings for each CLIP encoder layer."""

# Weight indices per layer (extracted from model_ttnn.py forward method)
# Each layer uses these weight indices from self.weights[]
LAYER_WEIGHT_INDICES = {
    # Layer 0 is special - has extra layer_norm1 weights
    0: {
        "layer_norm1_bias": 383,
        "layer_norm1_weight": 384,
        "out_proj_weight": 380,
        "layer_norm2_bias": 377,
        "layer_norm2_weight": 378,
        "fc1_weight": 376,
        "fc2_weight": 374,
        "layer_norm1_next_weight": 372,
        "layer_norm1_next_bias": 371,
    },
    1: {
        "out_proj_weight": 368,
        "layer_norm2_weight": 366,
        "layer_norm2_bias": 365,
        "fc1_weight": 364,
        "fc2_weight": 362,
        "layer_norm1_next_weight": 360,
        "layer_norm1_next_bias": 359,
    },
    2: {
        "out_proj_weight": 356,
        "layer_norm2_weight": 354,
        "layer_norm2_bias": 353,
        "fc1_weight": 352,
        "fc2_weight": 350,
        "layer_norm1_next_weight": 348,
        "layer_norm1_next_bias": 347,
    },
}

# CER keys per layer (from const eval results)
# These are fused/precomputed weights stored in self.cer[]
LAYER_CER_KEYS = {
    0: {
        "qkv_bias": "utils_constEvalFuncWrapper_47_0",
        "out_proj_bias": "utils_constEvalFuncWrapper_124_0",
        "qkv_weight": "utils_constEvalFuncWrapper_70_0",
        "fc1_bias": "utils_constEvalFuncWrapper_73_0",
        "fc2_bias": "utils_constEvalFuncWrapper_42_0",
    },
    1: {
        "qkv_bias": "utils_constEvalFuncWrapper_62_0",
        "out_proj_bias": "utils_constEvalFuncWrapper_55_0",
        "qkv_weight": "utils_constEvalFuncWrapper_157_0",
        "fc1_bias": "utils_constEvalFuncWrapper_13_0",
        "fc2_bias": "utils_constEvalFuncWrapper_21_0",
    },
    2: {
        "qkv_bias": "utils_constEvalFuncWrapper_80_0",
        "out_proj_bias": "utils_constEvalFuncWrapper_25_0",
        "qkv_weight": "utils_constEvalFuncWrapper_122_0",
        "fc1_bias": "utils_constEvalFuncWrapper_81_0",
        "fc2_bias": "utils_constEvalFuncWrapper_146_0",
    },
}


def get_layer_weights(weights, layer_idx):
    """
    Get weight dictionary for a specific encoder layer.

    Args:
        weights: The full weights list
        layer_idx: Layer index (0-30)

    Returns:
        dict: Mapping from semantic name to weight tensor
    """
    if layer_idx == 0:
        # Layer 0 has extra layer_norm1
        base = 384
        return {
            "layer_norm1_bias": weights[383],
            "layer_norm1_weight": weights[384],
            "out_proj_weight": weights[380],
            "layer_norm2_bias": weights[377],
            "layer_norm2_weight": weights[378],
            "fc1_weight": weights[376],
            "fc2_weight": weights[374],
            "layer_norm1_next_weight": weights[372],
            "layer_norm1_next_bias": weights[371],
        }
    elif layer_idx == 30:
        # Layer 30 is last - no layer_norm1_next
        # Method indices: 123, 124, 125, 126
        # Weights: 20, 18, 17, 14, 16
        return {
            "out_proj_weight": weights[20],
            "layer_norm2_weight": weights[18],
            "layer_norm2_bias": weights[17],
            "fc1_weight": weights[16],
            "fc2_weight": weights[14],
            # No layer_norm1_next for last layer
        }
    else:
        # Layers 1-29: standard pattern
        # Weight indices descend by 12 per layer starting from layer 1
        # Layer 1: 368, 366, 365, 364, 362, 360, 359
        # Layer 2: 356, 354, 353, 352, 350, 348, 347
        # Pattern: base = 368 - (layer_idx - 1) * 12
        base = 368 - (layer_idx - 1) * 12
        return {
            "out_proj_weight": weights[base],
            "layer_norm2_weight": weights[base - 2],
            "layer_norm2_bias": weights[base - 3],
            "fc1_weight": weights[base - 4],
            "fc2_weight": weights[base - 6],
            "layer_norm1_next_weight": weights[base - 8],
            "layer_norm1_next_bias": weights[base - 9],
        }


def get_layer_cer(cer, layer_idx):
    """
    Get CER dictionary for a specific encoder layer.

    This is more complex because CER keys don't follow a simple numeric pattern.
    We need to extract from the forward() method calls.

    Args:
        cer: The full CER dictionary
        layer_idx: Layer index (0-30)

    Returns:
        dict: Mapping from semantic name to CER tensor
    """
    # These mappings were extracted from analyzing the forward() method
    # Each layer has 5 CER values: qkv_bias, out_proj_bias, qkv_weight, fc1_bias, fc2_bias

    cer_mappings = {
        0: {
            "qkv_bias": "utils_constEvalFuncWrapper_47_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_124_0",
            "qkv_weight": "utils_constEvalFuncWrapper_70_0",
            "fc1_bias": "utils_constEvalFuncWrapper_73_0",
            "fc2_bias": "utils_constEvalFuncWrapper_42_0",
        },
        1: {
            "qkv_bias": "utils_constEvalFuncWrapper_62_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_55_0",
            "qkv_weight": "utils_constEvalFuncWrapper_157_0",
            "fc1_bias": "utils_constEvalFuncWrapper_13_0",
            "fc2_bias": "utils_constEvalFuncWrapper_21_0",
        },
        2: {
            "qkv_bias": "utils_constEvalFuncWrapper_80_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_25_0",
            "qkv_weight": "utils_constEvalFuncWrapper_122_0",
            "fc1_bias": "utils_constEvalFuncWrapper_81_0",
            "fc2_bias": "utils_constEvalFuncWrapper_146_0",
        },
        3: {
            "qkv_bias": "utils_constEvalFuncWrapper_90_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_132_0",
            "qkv_weight": "utils_constEvalFuncWrapper_26_0",
            "fc1_bias": "utils_constEvalFuncWrapper_145_0",
            "fc2_bias": "utils_constEvalFuncWrapper_10_0",
        },
        4: {
            "qkv_bias": "utils_constEvalFuncWrapper_97_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_43_0",
            "qkv_weight": "utils_constEvalFuncWrapper_127_0",
            "fc1_bias": "utils_constEvalFuncWrapper_150_0",
            "fc2_bias": "utils_constEvalFuncWrapper_149_0",
        },
        5: {
            "qkv_bias": "utils_constEvalFuncWrapper_158_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_69_0",
            "qkv_weight": "utils_constEvalFuncWrapper_96_0",
            "fc1_bias": "utils_constEvalFuncWrapper_91_0",
            "fc2_bias": "utils_constEvalFuncWrapper_106_0",
        },
        6: {
            "qkv_bias": "utils_constEvalFuncWrapper_99_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_46_0",
            "qkv_weight": "utils_constEvalFuncWrapper_128_0",
            "fc1_bias": "utils_constEvalFuncWrapper_53_0",
            "fc2_bias": "utils_constEvalFuncWrapper_103_0",
        },
        7: {
            "qkv_bias": "utils_constEvalFuncWrapper_120_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_84_0",
            "qkv_weight": "utils_constEvalFuncWrapper_49_0",
            "fc1_bias": "utils_constEvalFuncWrapper_153_0",
            "fc2_bias": "utils_constEvalFuncWrapper_27_0",
        },
        8: {
            "qkv_bias": "utils_constEvalFuncWrapper_40_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_74_0",
            "qkv_weight": "utils_constEvalFuncWrapper_29_0",
            "fc1_bias": "utils_constEvalFuncWrapper_24_0",
            "fc2_bias": "utils_constEvalFuncWrapper_93_0",
        },
        9: {
            "qkv_bias": "utils_constEvalFuncWrapper_119_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_133_0",
            "qkv_weight": "utils_constEvalFuncWrapper_113_0",
            "fc1_bias": "utils_constEvalFuncWrapper_2_0",
            "fc2_bias": "utils_constEvalFuncWrapper_155_0",
        },
        10: {
            "qkv_bias": "utils_constEvalFuncWrapper_152_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_64_0",
            "qkv_weight": "utils_constEvalFuncWrapper_71_0",
            "fc1_bias": "utils_constEvalFuncWrapper_95_0",
            "fc2_bias": "utils_constEvalFuncWrapper_85_0",
        },
        11: {
            "qkv_bias": "utils_constEvalFuncWrapper_67_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_140_0",
            "qkv_weight": "utils_constEvalFuncWrapper_116_0",
            "fc1_bias": "utils_constEvalFuncWrapper_156_0",
            "fc2_bias": "utils_constEvalFuncWrapper_151_0",
        },
        12: {
            "qkv_bias": "utils_constEvalFuncWrapper_87_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_136_0",
            "qkv_weight": "utils_constEvalFuncWrapper_68_0",
            "fc1_bias": "utils_constEvalFuncWrapper_5_0",
            "fc2_bias": "utils_constEvalFuncWrapper_15_0",
        },
        13: {
            "qkv_bias": "utils_constEvalFuncWrapper_1_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_126_0",
            "qkv_weight": "utils_constEvalFuncWrapper_102_0",
            "fc1_bias": "utils_constEvalFuncWrapper_92_0",
            "fc2_bias": "utils_constEvalFuncWrapper_109_0",
        },
        14: {
            "qkv_bias": "utils_constEvalFuncWrapper_86_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_101_0",
            "qkv_weight": "utils_constEvalFuncWrapper_11_0",
            "fc1_bias": "utils_constEvalFuncWrapper_18_0",
            "fc2_bias": "utils_constEvalFuncWrapper_141_0",
        },
        15: {
            "qkv_bias": "utils_constEvalFuncWrapper_72_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_114_0",
            "qkv_weight": "utils_constEvalFuncWrapper_23_0",
            "fc1_bias": "utils_constEvalFuncWrapper_83_0",
            "fc2_bias": "utils_constEvalFuncWrapper_154_0",
        },
        16: {
            "qkv_bias": "utils_constEvalFuncWrapper_118_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_89_0",
            "qkv_weight": "utils_constEvalFuncWrapper_63_0",
            "fc1_bias": "utils_constEvalFuncWrapper_130_0",
            "fc2_bias": "utils_constEvalFuncWrapper_104_0",
        },
        17: {
            "qkv_bias": "utils_constEvalFuncWrapper_34_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_7_0",
            "qkv_weight": "utils_constEvalFuncWrapper_17_0",
            "fc1_bias": "utils_constEvalFuncWrapper_108_0",
            "fc2_bias": "utils_constEvalFuncWrapper_19_0",
        },
        18: {
            "qkv_bias": "utils_constEvalFuncWrapper_134_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_112_0",
            "qkv_weight": "utils_constEvalFuncWrapper_100_0",
            "fc1_bias": "utils_constEvalFuncWrapper_94_0",
            "fc2_bias": "utils_constEvalFuncWrapper_147_0",
        },
        19: {
            "qkv_bias": "utils_constEvalFuncWrapper_12_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_50_0",
            "qkv_weight": "utils_constEvalFuncWrapper_52_0",
            "fc1_bias": "utils_constEvalFuncWrapper_44_0",
            "fc2_bias": "utils_constEvalFuncWrapper_28_0",
        },
        20: {
            "qkv_bias": "utils_constEvalFuncWrapper_78_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_60_0",
            "qkv_weight": "utils_constEvalFuncWrapper_65_0",
            "fc1_bias": "utils_constEvalFuncWrapper_82_0",
            "fc2_bias": "utils_constEvalFuncWrapper_107_0",
        },
        21: {
            "qkv_bias": "utils_constEvalFuncWrapper_37_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_20_0",
            "qkv_weight": "utils_constEvalFuncWrapper_111_0",
            "fc1_bias": "utils_constEvalFuncWrapper_110_0",
            "fc2_bias": "utils_constEvalFuncWrapper_160_0",
        },
        22: {
            "qkv_bias": "utils_constEvalFuncWrapper_148_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_57_0",
            "qkv_weight": "utils_constEvalFuncWrapper_33_0",
            "fc1_bias": "utils_constEvalFuncWrapper_125_0",
            "fc2_bias": "utils_constEvalFuncWrapper_4_0",
        },
        23: {
            "qkv_bias": "utils_constEvalFuncWrapper_36_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_32_0",
            "qkv_weight": "utils_constEvalFuncWrapper_51_0",
            "fc1_bias": "utils_constEvalFuncWrapper_0_0",
            "fc2_bias": "utils_constEvalFuncWrapper_22_0",
        },
        24: {
            "qkv_bias": "utils_constEvalFuncWrapper_76_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_143_0",
            "qkv_weight": "utils_constEvalFuncWrapper_59_0",
            "fc1_bias": "utils_constEvalFuncWrapper_144_0",
            "fc2_bias": "utils_constEvalFuncWrapper_139_0",
        },
        25: {
            "qkv_bias": "utils_constEvalFuncWrapper_61_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_31_0",
            "qkv_weight": "utils_constEvalFuncWrapper_58_0",
            "fc1_bias": "utils_constEvalFuncWrapper_117_0",
            "fc2_bias": "utils_constEvalFuncWrapper_39_0",
        },
        26: {
            "qkv_bias": "utils_constEvalFuncWrapper_77_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_105_0",
            "qkv_weight": "utils_constEvalFuncWrapper_9_0",
            "fc1_bias": "utils_constEvalFuncWrapper_123_0",
            "fc2_bias": "utils_constEvalFuncWrapper_98_0",
        },
        27: {
            "qkv_bias": "utils_constEvalFuncWrapper_159_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_8_0",
            "qkv_weight": "utils_constEvalFuncWrapper_41_0",
            "fc1_bias": "utils_constEvalFuncWrapper_129_0",
            "fc2_bias": "utils_constEvalFuncWrapper_115_0",
        },
        28: {
            "qkv_bias": "utils_constEvalFuncWrapper_16_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_3_0",
            "qkv_weight": "utils_constEvalFuncWrapper_121_0",
            "fc1_bias": "utils_constEvalFuncWrapper_56_0",
            "fc2_bias": "utils_constEvalFuncWrapper_14_0",
        },
        29: {
            "qkv_bias": "utils_constEvalFuncWrapper_45_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_79_0",
            "qkv_weight": "utils_constEvalFuncWrapper_75_0",
            "fc1_bias": "utils_constEvalFuncWrapper_38_0",
            "fc2_bias": "utils_constEvalFuncWrapper_35_0",
        },
        30: {
            "qkv_bias": "utils_constEvalFuncWrapper_48_0",
            "out_proj_bias": "utils_constEvalFuncWrapper_138_0",
            "qkv_weight": "utils_constEvalFuncWrapper_131_0",
            "fc1_bias": "utils_constEvalFuncWrapper_135_0",
            "fc2_bias": "utils_constEvalFuncWrapper_54_0",
        },
    }

    mapping = cer_mappings.get(layer_idx, {})
    return {key: cer[cer_key] for key, cer_key in mapping.items()}
