# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch reference wrapper for ZImageTransformer2DModel.

Loads the model from HuggingFace, applies RoPE patching required for
TT-MLIR compatibility, and provides a clean forward() interface.
"""

import os
import sys

import torch
import torch.nn as nn

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
PATCH_SIZE = 2
F_PATCH_SIZE = 1

HEAD_DIM = 128
ORIGINAL_HEADS = 30
PADDED_HEADS = 32
EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256


def load_model():
    """Load ZImageTransformer2DModel from HuggingFace in bfloat16.

    Also patches the RoPE embedder to use real-valued operations instead of
    torch.polar / view_as_complex (not supported by TT-MLIR).

    Returns:
        transformer: ZImageTransformer2DModel in eval mode.
    """
    # Patch RoPE before loading so precompute_freqs_cis returns real tensors.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _patch_dir = os.path.join(_HERE, "..", "..", "..")
    sys.path.insert(0, os.path.abspath(_patch_dir))
    from proper_tp.common import patch_rope_for_tt
    patch_rope_for_tt()

    from diffusers import ZImageTransformer2DModel
    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer.eval()
    print(
        f"  Loaded transformer ({sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params)"
    )
    return transformer


def pad_heads(transformer):
    """Pad attention heads from 30 → 32 with zero-weight dummy heads (in place).

    Zero-weight dummy heads are mathematically transparent — they contribute
    zero to all attention outputs.  Padding is necessary so that 32 / 4 = 8
    heads per device divides evenly in the 4-way TP setup.
    """
    def _pad_layer(layer):
        attn = layer.attention
        in_dim = attn.to_q.weight.shape[1]
        for proj in (attn.to_q, attn.to_k, attn.to_v):
            w = proj.weight.data
            proj.weight = nn.Parameter(
                torch.cat([w, torch.zeros(EXTRA_DIM, in_dim, dtype=w.dtype)], dim=0),
                requires_grad=False,
            )
            if proj.bias is not None:
                b = proj.bias.data
                proj.bias = nn.Parameter(
                    torch.cat([b, torch.zeros(EXTRA_DIM, dtype=b.dtype)]),
                    requires_grad=False,
                )
        proj = attn.to_out[0]
        w = proj.weight.data
        proj.weight = nn.Parameter(
            torch.cat([w, torch.zeros(w.shape[0], EXTRA_DIM, dtype=w.dtype)], dim=1),
            requires_grad=False,
        )
        attn.heads = PADDED_HEADS

    all_layers = list(transformer.layers) + list(transformer.noise_refiner) + list(transformer.context_refiner)
    for layer in all_layers:
        _pad_layer(layer)

    print(f"  Head padding: {ORIGINAL_HEADS} → {PADDED_HEADS} heads")


def forward(transformer, latents, timestep, cap_feats, patch_size=PATCH_SIZE, f_patch_size=F_PATCH_SIZE):
    """Run a CPU forward pass through the transformer.

    Args:
        transformer: ZImageTransformer2DModel (must have patch_rope_for_tt applied).
        latents: List of [C, F, H, W] bfloat16 tensors (one per batch item).
        timestep: [1] float tensor (e.g. torch.tensor([0.5])).
        cap_feats: [seq_len, 2560] bfloat16 caption embeddings.
        patch_size: spatial patch size (default 2).
        f_patch_size: temporal patch size (default 1).

    Returns:
        List of output tensors, one per batch item.
    """
    with torch.no_grad():
        result = transformer(
            x=latents,
            t=timestep,
            cap_feats=cap_feats if isinstance(cap_feats, list) else [cap_feats],
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            return_dict=False,
        )
    outputs = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(outputs, list):
        outputs = [outputs]
    return outputs
