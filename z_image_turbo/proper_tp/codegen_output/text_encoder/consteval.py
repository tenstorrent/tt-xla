# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Consteval functions for TextEncoder (Qwen3) TTNN graph.

Replaces 258 auto-generated main_const_eval_N() functions with 2 transformation
patterns applied via a data-driven lookup table, plus 4 constant generators,
plus 1 ROPE precomputation.

Transformation types:
  A (basic_rom, 1):    to_device only — keeps ROW_MAJOR (embedding table for ttnn.embedding)
  B (basic_tile, 252): to_device + to_layout(TILE) — standard weight loading

Special entries (not standard weight loading):
  ce_100 (ROPE): loads inv_freq (arg220), does to_device+to_layout then computes
                 cos/sin tables for 7 tokens → returns [cos [1,1,7,128] BF16, sin [1,1,7,128] BF16]
  ce_115:  Causal mask — complex [1,1,7,7] BF16 (lower triangular attention mask via embedding)
  ce_167:  -inf scalar [1,1,1,1] BF16 TILE — attention fill value for masked positions
  ce_197:  Lower triangular mask [1,1,7,7] BF16 — causal mask matrix via ge comparison
  ce_254:  Zero scalar [1,1,1,1] BF16 TILE — attention fill value for valid positions
"""

import ttnn

DRAM_MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

# ---------------------------------------------------------------------------
# Consteval lookup table: ce_idx -> (arg_idx, transform_type)
# ---------------------------------------------------------------------------
# 254 entries total: 1 type "A" (ce_0), 252 type "B", 1 type "ROPE" (ce_100)
# ce_115, ce_167, ce_197, ce_254 are CONST_GEN (no input args) — handled separately
CONSTEVAL_MAP = {
    0: (218, "A"),
    1: (109, "B"),
    2: (202, "B"),
    3: (182, "B"),
    4: (70, "B"),
    5: (269, "B"),
    6: (327, "B"),
    7: (17, "B"),
    8: (143, "B"),
    9: (188, "B"),
    10: (264, "B"),
    11: (149, "B"),
    12: (103, "B"),
    13: (400, "B"),
    14: (295, "B"),
    15: (300, "B"),
    16: (369, "B"),
    17: (64, "B"),
    18: (155, "B"),
    19: (274, "B"),
    20: (214, "B"),
    21: (115, "B"),
    22: (76, "B"),
    23: (121, "B"),
    24: (208, "B"),
    25: (190, "B"),
    26: (374, "B"),
    27: (82, "B"),
    28: (384, "B"),
    29: (279, "B"),
    30: (196, "B"),
    31: (379, "B"),
    32: (389, "B"),
    33: (88, "B"),
    34: (91, "B"),
    35: (184, "B"),
    36: (249, "B"),
    37: (97, "B"),
    38: (29, "B"),
    39: (390, "B"),
    40: (206, "B"),
    41: (254, "B"),
    42: (259, "B"),
    43: (200, "B"),
    44: (35, "B"),
    45: (176, "B"),
    46: (385, "B"),
    47: (337, "B"),
    48: (342, "B"),
    49: (395, "B"),
    50: (307, "B"),
    51: (194, "B"),
    52: (332, "B"),
    53: (244, "B"),
    54: (23, "B"),
    55: (158, "B"),
    56: (85, "B"),
    57: (322, "B"),
    58: (11, "B"),
    59: (164, "B"),
    60: (79, "B"),
    61: (170, "B"),
    62: (317, "B"),
    63: (312, "B"),
    64: (5, "B"),
    65: (20, "B"),
    66: (86, "B"),
    67: (152, "B"),
    68: (113, "B"),
    69: (205, "B"),
    70: (352, "B"),
    71: (47, "B"),
    72: (380, "B"),
    73: (145, "B"),
    74: (166, "B"),
    75: (211, "B"),
    76: (80, "B"),
    77: (172, "B"),
    78: (146, "B"),
    79: (26, "B"),
    80: (178, "B"),
    81: (133, "B"),
    82: (151, "B"),
    83: (347, "B"),
    84: (173, "B"),
    85: (92, "B"),
    86: (41, "B"),
    87: (14, "B"),
    88: (357, "B"),
    89: (53, "B"),
    90: (191, "B"),
    91: (185, "B"),
    92: (139, "B"),
    93: (59, "B"),
    94: (362, "B"),
    95: (127, "B"),
    96: (179, "B"),
    97: (65, "B"),
    98: (131, "B"),
    99: (367, "B"),
    100: (220, "ROPE"),
    101: (8, "B"),
    102: (161, "B"),
    103: (104, "B"),
    104: (38, "B"),
    105: (397, "B"),
    106: (68, "B"),
    107: (167, "B"),
    108: (98, "B"),
    109: (74, "B"),
    110: (163, "B"),
    111: (101, "B"),
    112: (392, "B"),
    113: (32, "B"),
    114: (157, "B"),
    116: (95, "B"),
    117: (199, "B"),
    118: (44, "B"),
    119: (181, "B"),
    120: (62, "B"),
    121: (169, "B"),
    122: (107, "B"),
    123: (137, "B"),
    124: (187, "B"),
    125: (119, "B"),
    126: (50, "B"),
    127: (193, "B"),
    128: (125, "B"),
    129: (56, "B"),
    130: (350, "B"),
    131: (382, "B"),
    132: (272, "B"),
    133: (83, "B"),
    134: (240, "B"),
    135: (4, "B"),
    136: (175, "B"),
    137: (319, "B"),
    138: (245, "B"),
    139: (377, "B"),
    140: (49, "B"),
    141: (215, "B"),
    142: (37, "B"),
    143: (209, "B"),
    144: (345, "B"),
    145: (387, "B"),
    146: (43, "B"),
    147: (110, "B"),
    148: (89, "B"),
    149: (203, "B"),
    150: (372, "B"),
    151: (235, "B"),
    152: (277, "B"),
    153: (128, "B"),
    154: (77, "B"),
    155: (262, "B"),
    156: (230, "B"),
    157: (122, "B"),
    158: (71, "B"),
    159: (2, "B"),
    160: (116, "B"),
    161: (267, "B"),
    162: (154, "B"),
    163: (365, "B"),
    164: (134, "B"),
    165: (25, "B"),
    166: (335, "B"),
    168: (61, "B"),
    169: (197, "B"),
    170: (124, "B"),
    171: (31, "B"),
    172: (282, "B"),
    173: (118, "B"),
    174: (340, "B"),
    175: (287, "B"),
    176: (140, "B"),
    177: (292, "B"),
    178: (224, "B"),
    179: (67, "B"),
    180: (130, "B"),
    181: (160, "B"),
    182: (148, "B"),
    183: (355, "B"),
    184: (55, "B"),
    185: (360, "B"),
    186: (229, "B"),
    187: (142, "B"),
    188: (375, "B"),
    189: (234, "B"),
    190: (136, "B"),
    191: (73, "B"),
    192: (370, "B"),
    193: (40, "B"),
    194: (239, "B"),
    195: (349, "B"),
    196: (297, "B"),
    198: (294, "B"),
    199: (302, "B"),
    200: (100, "B"),
    201: (212, "B"),
    202: (399, "B"),
    203: (247, "B"),
    204: (106, "B"),
    205: (112, "B"),
    206: (242, "B"),
    207: (354, "B"),
    208: (330, "B"),
    209: (19, "B"),
    210: (222, "B"),
    211: (315, "B"),
    212: (299, "B"),
    213: (394, "B"),
    214: (252, "B"),
    215: (1, "B"),
    216: (94, "B"),
    217: (320, "B"),
    218: (284, "B"),
    219: (289, "B"),
    220: (7, "B"),
    221: (325, "B"),
    222: (225, "B"),
    223: (13, "B"),
    224: (257, "B"),
    225: (334, "B"),
    226: (255, "B"),
    227: (232, "B"),
    228: (364, "B"),
    229: (310, "B"),
    230: (285, "B"),
    231: (304, "B"),
    232: (359, "B"),
    233: (305, "B"),
    234: (58, "B"),
    235: (250, "B"),
    236: (309, "B"),
    237: (52, "B"),
    238: (314, "B"),
    239: (237, "B"),
    240: (275, "B"),
    241: (290, "B"),
    242: (46, "B"),
    243: (16, "B"),
    244: (260, "B"),
    245: (34, "B"),
    246: (344, "B"),
    247: (265, "B"),
    248: (227, "B"),
    249: (10, "B"),
    250: (339, "B"),
    251: (28, "B"),
    252: (280, "B"),
    253: (324, "B"),
    255: (270, "B"),
    256: (329, "B"),
    257: (22, "B"),
}


def _apply_transform(tensor, transform: str):
    """Apply a single consteval transformation to a host TTNN tensor."""
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MC)

    if transform == "A":
        return [t]  # ROW_MAJOR stays — used as embedding table

    if transform == "B":
        t2 = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(t, False)
        return [t2]

    raise ValueError(f"Unknown transform type: {transform!r}")


def _ce_100_rope(inv_freq_tensor):
    """
    RoPE frequency precomputation (ce_100).

    Takes inv_freq (arg220, shape [64]) on host, computes cos/sin tables for
    sequence length 7, returns [cos [1,1,7,128] BF16, sin [1,1,7,128] BF16].
    """
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    ttnn_to_device_100 = ttnn.to_device(
        inv_freq_tensor,
        device=device,
        memory_config=DRAM_MC,
    )
    ttnn_to_layout_99 = ttnn.to_layout(
        ttnn_to_device_100, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_to_device_100, False)
    ttnn_Tensor_0 = ttnn.Tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 7],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MC,
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_99,
        [1, 64, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_to_layout_99, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_0,
        ttnn_Tensor_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=DRAM_MC,
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn.deallocate(ttnn_Tensor_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_matmul_0,
        [0, 2, 1],
        memory_config=DRAM_MC,
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 7, 64],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_reshape_1, ttnn_reshape_1],
        3,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_cos_0 = ttnn.cos(
        ttnn_concat_0,
        memory_config=DRAM_MC,
    )
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_cos_0,
        ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_cos_0, False)
    ttnn_sin_0 = ttnn.sin(
        ttnn_concat_0,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_sin_0,
        ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_sin_0, False)
    return [ttnn_typecast_0, ttnn_typecast_1]


def _ce_115_causal_mask():
    """
    Causal mask generator (ce_115) — no input args.

    Builds a [1,1,7,7] BF16 lower-triangular attention mask via cumsum+embedding.
    Returns [mask_tensor].
    """
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    ttnn_Tensor_1 = ttnn.Tensor(
        [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
        [1, 1, 1, 7, 2],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MC,
    )
    ttnn_Tensor_2 = ttnn.Tensor(
        [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
        [1, 1, 7, 1, 2],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MC,
    )
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 7]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    ttnn_cumsum_0 = ttnn.cumsum(
        ttnn_full_0,
        1,
        dtype=ttnn.DataType.INT32,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_full_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_cumsum_0,
        [1, 0],
        memory_config=DRAM_MC,
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_cumsum_0, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_Tensor_2,
        [0, 0, 0, 0, 1],
        [1, 1, 7, 1, 2],
        [1, 1, 1, 1, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_Tensor_2, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_slice_0,
        [1, 7],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_reshape_2,
        ttnn.DataType.UINT32,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_to_layout_114 = ttnn.to_layout(
        ttnn_typecast_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_permute_1,
        ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_to_layout_115 = ttnn.to_layout(
        ttnn_typecast_3, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_114,
        ttnn_to_layout_115,
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_to_layout_114, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_embedding_0,
        ttnn.DataType.INT32,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_typecast_4,
        [1, 1, 7, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn_slice_1 = ttnn.slice(
        ttnn_Tensor_1,
        [0, 0, 0, 0, 1],
        [1, 1, 1, 7, 2],
        [1, 1, 1, 1, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_Tensor_1, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_slice_1,
        [1, 7],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_reshape_4,
        ttnn.DataType.UINT32,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_to_layout_116 = ttnn.to_layout(
        ttnn_typecast_5, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn_embedding_1 = ttnn.embedding(
        ttnn_to_layout_116,
        ttnn_to_layout_115,
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_to_layout_116, False)
    ttnn.deallocate(ttnn_to_layout_115, False)
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_embedding_1,
        ttnn.DataType.INT32,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_embedding_1, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_typecast_6,
        [1, 1, 1, 7],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn_eq_0 = ttnn.eq(
        ttnn_reshape_3,
        ttnn_reshape_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn.deallocate(ttnn_reshape_3, False)
    return [ttnn_eq_0]


def _ce_167_neg_inf_scalar():
    """
    Negative-infinity scalar generator (ce_167) — no input args.

    Produces a [1,1,1,1] BF16 TILE tensor filled with -inf.
    Used as attention fill value for masked positions.
    Returns [scalar_tensor].
    """
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_full_1,
        [1, 1, 1, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_full_1, False)
    return [ttnn_reshape_6]


def _ce_197_lower_tri_mask():
    """
    Lower-triangular causal mask generator (ce_197) — no input args.

    Builds a [1,1,7,7] BF16 mask where entry (i,j)=1.0 iff j<=i (lower triangle).
    Returns [mask_tensor].
    """
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    ttnn_Tensor_3 = ttnn.Tensor(
        [0, 1, 2, 3, 4, 5, 6],
        [1, 1, 7],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MC,
    )
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_Tensor_3,
        [1, 1, 1, 7],
        memory_config=DRAM_MC,
    )
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_Tensor_3,
        [1, 1, 7, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_Tensor_3, False)
    ttnn_ge_0 = ttnn.ge(
        ttnn_reshape_8,
        ttnn_reshape_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn.deallocate(ttnn_reshape_7, False)
    return [ttnn_ge_0]


def _ce_254_zero_scalar():
    """
    Zero scalar generator (ce_254) — no input args.

    Produces a [1,1,1,1] BF16 TILE tensor filled with 0.0.
    Used as attention fill value for valid positions.
    Returns [scalar_tensor].
    """
    import utils

    device = utils.DeviceGetter.get_device((1, 4))

    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_full_2,
        [1, 1, 1, 1],
        memory_config=DRAM_MC,
    )
    ttnn.deallocate(ttnn_full_2, False)
    return [ttnn_reshape_9]


def run_const_evals(inputs: list) -> dict:
    """
    Run all consteval functions and return the populated cache dict.

    Args:
        inputs: list of host TTNN tensors (model weights + constants), indexed
                by the arg_idx values in CONSTEVAL_MAP.

    Returns:
        dict mapping "main_const_eval_N" -> list of TTNN tensors on device.
    """
    cache = {}

    # --- 253 standard weight-loading entries (A and B transforms) ---
    for ce_idx, (arg_idx, transform) in CONSTEVAL_MAP.items():
        if transform == "ROPE":
            # handled separately below
            continue
        result = _apply_transform(inputs[arg_idx], transform)
        cache[f"main_const_eval_{ce_idx}"] = result

    # --- ce_100: RoPE precomputation (returns 2 tensors) ---
    rope_arg_idx = CONSTEVAL_MAP[100][0]  # 220
    rope_result = _ce_100_rope(inputs[rope_arg_idx])
    cache["main_const_eval_100"] = rope_result  # [cos, sin]

    # --- ce_115: causal mask (no input args) ---
    cache["main_const_eval_115"] = _ce_115_causal_mask()

    # --- ce_167: -inf scalar (no input args) ---
    cache["main_const_eval_167"] = _ce_167_neg_inf_scalar()

    # --- ce_197: lower-triangular mask (no input args) ---
    cache["main_const_eval_197"] = _ce_197_lower_tri_mask()

    # --- ce_254: zero scalar (no input args) ---
    cache["main_const_eval_254"] = _ce_254_zero_scalar()

    return cache
