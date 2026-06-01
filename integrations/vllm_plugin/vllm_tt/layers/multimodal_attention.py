# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..logger import tt_init_logger

logger = tt_init_logger(__name__)


def tt_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Fused TT SDPA attention for an HF vision tower.

    vLLM has no API to route a HF-instantiated vision tower's attention to a
    device backend, so we register this into HF's ALL_ATTENTION_FUNCTIONS and
    point the tower's _attn_implementation at it.

    Q/K/V: [batch, heads, seq, head_dim]; returns (attn_output, None) with
    attn_output [batch, seq, heads, head_dim] (the sdpa_attention_forward
    contract).
    """
    n_rep = getattr(module, "num_key_value_groups", 1) or 1
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", False)
    )
    if scaling is None:
        scaling = float(query.shape[-1]) ** -0.5

    # The TT SDPA op asserts query/key seq length is tile-aligned (multiple of
    # 32). Vision encoders typically produce non-aligned patch counts. Pad
    # Q/K/V along the seq dim with zeros and add an additive mask to zero out
    # softmax contributions from padded K positions. Trim padded Q rows from
    # the output afterwards.
    _TILE = 32
    seq_q = query.shape[2]
    seq_k = key.shape[2]
    pad_q = (-seq_q) % _TILE
    pad_k = (-seq_k) % _TILE

    if pad_q > 0:
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_q))
    if pad_k > 0:
        key = torch.nn.functional.pad(key, (0, 0, 0, pad_k))
        value = torch.nn.functional.pad(value, (0, 0, 0, pad_k))
        masked_value = torch.finfo(query.dtype).min
        # TTIR SDPA op requires the mask's dim 2 to equal the (padded) query
        # sequence length — broadcast-shaped [B, 1, 1, K] is rejected. Use the
        # explicit [B, 1, Q_pad, K_pad] layout. Since masking only depends on
        # the key position, every query row is identical.
        if attention_mask is None:
            attention_mask = torch.zeros(
                (query.shape[0], 1, seq_q + pad_q, seq_k + pad_k),
                dtype=query.dtype,
                device=query.device,
            )
            attention_mask[..., seq_k:] = masked_value
        else:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_k), value=masked_value
            )
        # The TT SDPA op disallows is_causal=True when attn_mask is set.
        is_causal = False

    attn_output = torch.ops.tt.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=is_causal,
        attn_mask=attention_mask,
        scale=scaling,
    )

    if pad_q > 0:
        attn_output = attn_output[:, :, :seq_q, :]

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# Idempotently register so reloads (e.g. pytest reruns) don't raise.
if "tt" not in ALL_ATTENTION_FUNCTIONS:
    ALL_ATTENTION_FUNCTIONS.register("tt", tt_sdpa_attention_forward)


def override_vision_attention(model: torch.nn.Module) -> None:
    """Point the HF vision/audio tower's _attn_implementation at the registered
    "tt" attention so its forward dispatches to tt_sdpa_attention_forward.
    No-op on towers without an HF config (getattr-guarded)."""
    for tower_attr in ("vision_tower", "audio_tower"):
        tower = getattr(model, tower_attr, None)
        if tower is None:
            continue
        cfg = getattr(tower, "config", None)
        if cfg is None:
            continue
        prev = getattr(cfg, "_attn_implementation", None)
        cfg._attn_implementation = "tt"
        logger.info("Routed %s attention through TT SDPA (was %r)", tower_attr, prev)
