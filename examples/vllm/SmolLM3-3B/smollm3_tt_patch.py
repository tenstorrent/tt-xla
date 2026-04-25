# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM3 TT hardware compatibility patch.

This module patches SmolLM3's NoPE (no positional embedding) layers to use RoPE
on all layers. This produces uniform graph shapes that TTNN can handle.

Import this module BEFORE loading the SmolLM3 model to apply patches.
"""

import importlib


def patch_smollm3_attention():
    """Patch SmolLM3Attention to force RoPE on all layers for TT compatibility."""
    try:
        mod = importlib.import_module("transformers.models.smollm3.modeling_smollm3")
    except ImportError:
        return  # SmolLM3 not available in this transformers version

    SmolLM3Attention = mod.SmolLM3Attention
    if hasattr(SmolLM3Attention, "_tt_patched"):
        return  # Already patched

    original_forward = SmolLM3Attention.forward

    def patched_forward(
        self, hidden_states, position_embeddings, attention_mask=None,
        past_key_values=None, cache_position=None, **kwargs
    ):
        orig = self.use_rope
        self.use_rope = True
        result = original_forward(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values, cache_position, **kwargs
        )
        self.use_rope = orig
        return result

    SmolLM3Attention.forward = patched_forward
    SmolLM3Attention._tt_patched = True


# Auto-apply when imported
patch_smollm3_attention()
