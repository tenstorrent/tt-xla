# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HF ``AttentionInterface`` backend wrapping ``torch.ops.tt.scaled_dot_product_attention``.

Register via ``register_tt_attention_backend()`` then pass
``attn_implementation="tt_sdpa"`` to ``from_pretrained``.
Query seq_len must be a multiple of 32.
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
    """SDPA forward via ``tt::scaled_dot_product_attention``.
    CPU tensors fall back to HF `sdpa`."""
    if query.device.type == "cpu":
        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

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

    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", True)
    )
    is_causal = bool(query.shape[2] > 1 and attention_mask is None and is_causal)

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
    if TT_ATTENTION_BACKEND_NAME not in ALL_ATTENTION_FUNCTIONS:
        raise RuntimeError(f"{TT_ATTENTION_BACKEND_NAME} registration failed")
