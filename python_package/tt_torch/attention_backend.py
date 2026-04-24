# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent attention backend for HuggingFace transformers.

HuggingFace exposes `AttentionInterface` (`transformers/modeling_utils.py`)
as a pluggable registry of attention forward functions. Built-in keys are
"sdpa", "flex_attention", "flash_attention_{2,3,4}", "eager", plus their
"paged|..." variants. Any model that dispatches via
`ALL_ATTENTION_FUNCTIONS.get_interface(...)` will pick up whichever backend
is selected through `attn_implementation="..."` at `from_pretrained` time.

This module adds a backend named "tt_sdpa" whose forward calls
`torch.ops.tt.scaled_dot_product_attention` — a stablehlo.custom_call
lowering declared in `tt_torch/custom_ops.py`. We prefer the custom_call
form over a composite decomposition because tt-mlir has a native fused
attention kernel; emitting a custom_call keeps the op opaque all the way
through the compiler so the fused kernel is preserved.

Usage:

    from tt_torch.attention_backend import register_tt_attention_backend, TT_ATTENTION_BACKEND_NAME

    register_tt_attention_backend()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=TT_ATTENTION_BACKEND_NAME
    )

Shape constraint: `tt::scaled_dot_product_attention` requires the query
sequence length to be a multiple of 32. For short prompts, pad to a
tile-aligned length (the MoE demos already do this for `sparse_matmul`).
"""

from __future__ import annotations

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

# Ensure torch.ops.tt.* are registered.
from . import custom_ops  # noqa: F401

logger = logging.get_logger(__name__)

TT_ATTENTION_BACKEND_NAME = "tt_sdpa"


def tt_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention forward implemented with `tt::scaled_dot_product_attention`.

    Signature mirrors `sdpa_attention_forward` in
    `transformers/integrations/sdpa_attention.py`:

        query, key, value : [B, num_heads_or_kv, seq_len, head_dim]
        attention_mask    : optional pre-expanded 4D mask (or None for causal)
        returns           : (attn_output [B, seq_len, num_heads, head_dim], None)
    """
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`tt_sdpa` attention does not support `output_attentions=True`. "
            "Set your attention to `eager` if you need it."
        )
    if dropout != 0.0:
        # The custom op has no dropout; this is only meaningful in training.
        logger.warning_once(
            "`tt_sdpa` ignores `dropout` — inference path, no dropout applied."
        )

    # Mirror the is_causal resolution sdpa_attention_forward does.
    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", True)
    )
    is_causal = bool(query.shape[2] > 1 and attention_mask is None and is_causal)
    # The custom op requires is_causal=False whenever an explicit mask is
    # provided; the mask already encodes causality if needed.
    if attention_mask is not None:
        is_causal = False

    kwargs_tt = {}
    if scaling is not None:
        kwargs_tt["scale"] = scaling

    attn_output = torch.ops.tt.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=is_causal,
        **kwargs_tt,
    )
    # Transformers expects [B, seq_len, num_heads, head_dim]; the op returns
    # [B, num_heads, seq_len, head_dim] like every other SDPA backend.
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def register_tt_attention_backend() -> None:
    """Register the `tt_sdpa` attention backend globally. Idempotent."""
    ALL_ATTENTION_FUNCTIONS.register(
        TT_ATTENTION_BACKEND_NAME, tt_sdpa_attention_forward
    )
    assert (
        TT_ATTENTION_BACKEND_NAME in ALL_ATTENTION_FUNCTIONS
    ), "registration did not stick"
