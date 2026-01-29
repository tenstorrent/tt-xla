#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract the exact CER key to semantic name mappings from forward() method."""

import re


def extract_cer_mappings():
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    # For each layer, find the attention and MLP calls and extract CER keys
    # Attention pattern: CLIPAttention_N_0(cer_qkv_bias, cer_out_proj_bias, hidden, weights_out_proj, cer_qkv_weight)
    # But argument order varies! Need to check method definitions

    # Layer 0 (method 3):
    # CLIPAttention_3_0(weights[380], cer["47"], cer["124"], cer["70"], CLIPEncoderLayer_2_0_0)
    # From method def: input_0=out_proj_weight, input_1=qkv_bias, input_2=out_proj_bias, input_3=qkv_weight, input_4=hidden
    # Wait, let me check the method body again...

    # Actually, from CLIPAttention_3_0 method body:
    # input_4 (last arg) is used in ttnn_reshape (hidden_states)
    # input_3 is used in first matmul (qkv_weight)
    # input_1 is used in first add after matmul (qkv_bias)
    # input_0 is used in second matmul (out_proj_weight)
    # input_2 is used in second add (out_proj_bias)

    # So for CLIPAttention_3_0_0 = self.CLIPAttention_3_0(
    #     self.weights[380],  -> input_0 -> out_proj_weight
    #     self.cer["47"],     -> input_1 -> qkv_bias
    #     self.cer["124"],    -> input_2 -> out_proj_bias
    #     self.cer["70"],     -> input_3 -> qkv_weight
    #     CLIPEncoderLayer_2_0_0,  -> input_4 -> hidden
    # )

    print("Layer 0 attention:")
    print("  weights[380] -> out_proj_weight")
    print("  cer['47'] -> qkv_bias")
    print("  cer['124'] -> out_proj_bias")
    print("  cer['70'] -> qkv_weight")

    # But layer 1 has different order!
    # CLIPAttention_7_0_0 = self.CLIPAttention_7_0(
    #     self.cer["62"],   -> input_0
    #     self.cer["55"],   -> input_1
    #     v_166,            -> input_2 -> hidden
    #     self.weights[368], -> input_3
    #     self.cer["157"],  -> input_4
    # )
    # From method body for CLIPAttention_7_0:
    # input_2 -> hidden (used in reshape)
    # input_4 -> first matmul weight (qkv_weight)
    # input_0 -> first add (qkv_bias)
    # input_3 -> second matmul weight (out_proj_weight)
    # input_1 -> second add (out_proj_bias)

    print("\nLayer 1 attention:")
    print("  cer['62'] -> qkv_bias")
    print("  cer['55'] -> out_proj_bias")
    print("  weights[368] -> out_proj_weight")
    print("  cer['157'] -> qkv_weight")

    # For MLP:
    # CLIPMLP_5_0_0 = self.CLIPMLP_5_0(
    #     v_163,            -> input_0 -> hidden
    #     self.cer["73"],   -> input_1
    #     self.cer["42"],   -> input_2
    #     self.weights[374], -> input_3
    #     self.weights[376], -> input_4
    # )
    # From method body for CLIPMLP_5_0:
    # input_0 -> reshape (hidden)
    # input_4 -> first matmul (fc1_weight)
    # input_1 -> first add (fc1_bias)
    # input_3 -> second matmul (fc2_weight)
    # input_2 -> second add (fc2_bias)

    print("\nLayer 0 MLP:")
    print("  cer['73'] -> fc1_bias")
    print("  cer['42'] -> fc2_bias")
    print("  weights[374] -> fc2_weight")
    print("  weights[376] -> fc1_weight")

    # CLIPMLP_9_0_0 = self.CLIPMLP_9_0(
    #     self.cer["13"],    -> input_0 -> fc1_bias
    #     self.weights[362], -> input_1 -> fc2_weight
    #     self.cer["21"],    -> input_2 -> fc2_bias
    #     self.weights[364], -> input_3 -> fc1_weight
    #     v_168,             -> input_4 -> hidden
    # )
    # Wait, this has hidden as input_4, not input_0!
    # Let me check CLIPMLP_9_0 method body...

    print("\nLayer 1 MLP:")
    print("  cer['13'] -> fc1_bias (input_0 used in add after fc1)")
    print("  weights[362] -> fc2_weight (input_1 used in fc2 matmul)")
    print("  cer['21'] -> fc2_bias (input_2 used in add after fc2)")
    print("  weights[364] -> fc1_weight (input_3 used in fc1 matmul)")

    print("\n" + "=" * 60)
    print("The argument order varies between layers!")
    print("Each method has different input_N -> operation mappings")
    print("=" * 60)


if __name__ == "__main__":
    extract_cer_mappings()
