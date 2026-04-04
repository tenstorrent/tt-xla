# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import torch
from transformers.cache_utils import StaticCache, StaticLayer, StaticSlidingWindowLayer
from transformers.masking_utils import create_sliding_window_causal_mask


def override_gpt_oss_sliding_window_causal_mask():
    """
    Override gpt_oss's modeling so that its
    create_sliding_window_causal_mask points to the TT-friendly version.

    Call this before torch.compile so that dynamo traces through the
    patched function.
    """
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod

    gpt_oss_mod.create_sliding_window_causal_mask = tt_create_sliding_window_causal_mask


def override_cache_sliding_window_layers(
    cache: StaticCache,
    max_cache_len: int,
    sliding_window: int,
) -> None:
    """
    Replace each StaticSlidingWindowLayer in cache.layers with TTStaticSlidingWindowLayer.
    """
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, StaticSlidingWindowLayer):
            tt_layer = TTStaticSlidingWindowLayer(
                max_cache_len=max_cache_len, sliding_window=sliding_window
            )
            tt_layer.keys = layer.keys
            tt_layer.values = layer.values
            tt_layer.is_initialized = True
            tt_layer.device = layer.device
            tt_layer.dtype = layer.dtype
            tt_layer.max_batch_size = layer.max_batch_size
            tt_layer.num_heads = layer.num_heads
            tt_layer.head_dim = layer.head_dim
            cache.layers[i] = tt_layer


def tt_create_sliding_window_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    """
    Compile-friendly sliding-window causal mask for the always-roll cache layout.

    Builds a 4D additive mask (batch, 1, query_length, sliding_window) using
    only broadcasting tensor operations — no get_mask_sizes(),
    and no mutable Python state.  Designed to run with torch.compile on TT
    hardware as part of the model forward graph.
    """
    if past_key_values is None:
        return create_sliding_window_causal_mask(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            **kwargs,
        )

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        return attention_mask

    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0
    sliding_window = past_key_values.layers[layer_idx].max_cache_len

    batch_size = input_embeds.shape[0]
    query_length = cache_position.shape[0]
    dtype = input_embeds.dtype
    device = cache_position.device
    min_val = torch.finfo(dtype).min

    buffer_pos = torch.arange(sliding_window, device=device)

    if query_length == 1:
        kv_pos = (cache_position[0] + 1) - sliding_window + buffer_pos
    else:
        kv_pos = torch.cat(
            (
                cache_position[0] - sliding_window + buffer_pos,
                cache_position,
            )
        )

    valid = kv_pos >= 0
    causal = kv_pos.unsqueeze(0) <= cache_position.unsqueeze(1)
    in_window = kv_pos.unsqueeze(0) > (cache_position.unsqueeze(1) - sliding_window)

    mask = valid.unsqueeze(0) & causal & in_window

    if attention_mask is not None and attention_mask.ndim == 2:
        padding_mask = attention_mask.to(device=device, dtype=torch.bool)
        kv_pos_idx = kv_pos.clamp(min=0).long()
        padding = padding_mask[:, kv_pos_idx]
        mask = mask.unsqueeze(0).unsqueeze(0) & padding.unsqueeze(1).unsqueeze(1)
    else:
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    return torch.where(
        mask,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_val, dtype=dtype, device=device),
    )


class TTStaticSlidingWindowLayer(StaticLayer):
    """
    Sliding-window cache layer using the always-roll strategy.

    Every call to update() rolls the buffer left by n positions
    (where n = key_states.shape[-2], a static shape) and writes the new
    tokens into the rightmost ``n`` slots.  This produces a right-aligned,
    chronologically-ordered buffer with no branching and no mutable Python
    state, so torch.compile traces a single straight-line graph per
    distinct n and avoids recompilations.

    get_mask_sizes() returns constants so it does not trigger
    torch.compile recompilations.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        n = key_states.shape[-2]

        # Cat old buffer + new tokens before mutating (n > 1 only).
        # Shape is always (max_cache_len + n) — both static.
        if n > 1:
            full_key_states = torch.cat((self.keys, key_states), dim=-2)
            full_value_states = torch.cat((self.values, value_states), dim=-2)

        if n >= self.max_cache_len:
            self.keys.copy_(key_states[:, :, -self.max_cache_len :, :])
            self.values.copy_(value_states[:, :, -self.max_cache_len :, :])
        else:
            # Roll left by n and write new tokens into the rightmost n slots.
            new_keys = self.keys.roll(-n, dims=-2)
            new_values = self.values.roll(-n, dims=-2)
            new_keys[:, :, -n:] = key_states
            new_values[:, :, -n:] = value_states
            self.keys.copy_(new_keys)
            self.values.copy_(new_values)

        if n == 1:
            return self.keys, self.values
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return constant (kv_length, kv_offset) — used by create_causal_mask."""
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return (self.keys[0, 0].any(dim=-1)).sum().item()
