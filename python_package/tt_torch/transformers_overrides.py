# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

import torch
from transformers.cache_utils import StaticCache, StaticLayer, StaticSlidingWindowLayer
from transformers.masking_utils import (
    create_causal_mask,
    create_chunked_causal_mask,
    create_sliding_window_causal_mask,
)


def override_model_sliding_window_causal_mask(model):
    """Apply TT-friendly in-place rewrites to a HuggingFace model's modeling module.

    Generic replacement for the per-model ``override_<name>_sliding_window_causal_mask``
    helpers: the modeling module is resolved from ``type(model).__module__``
    and its ``create_sliding_window_causal_mask`` is rebound to the
    compile-friendly TT variant.

    Args:
        model: A HuggingFace ``PreTrainedModel`` instance.
    """
    print("override_model_sliding_window_causal_mask")
    import importlib

    mod = importlib.import_module(type(model).__module__)
    mod.create_sliding_window_causal_mask = tt_create_sliding_window_causal_mask

    # Multimodal models (e.g. Gemma3) do not call `create_sliding_window_causal_mask`
    # directly; instead they build the whole mask mapping up front via
    # `create_causal_mask_mapping` -> `create_masks_for_generate` and pass it down
    # as a dict. Patch that path too so the sliding-window entry uses the TT mask
    # whose key/value length matches the always-roll cache (`sliding_window +
    # query_length`). Text-only models (e.g. Olmo3) do not import this symbol, so
    # the `hasattr` guard keeps them on the direct-call path above.
    if hasattr(mod, "create_masks_for_generate"):
        mod.create_masks_for_generate = tt_create_masks_for_generate


def override_cache_sliding_window_layers(
    cache: StaticCache,
    max_cache_len: int,
    sliding_window: int,
) -> None:
    """
    Replace each StaticSlidingWindowLayer in cache.layers with TTStaticSlidingWindowLayer.
    """
    for i, layer in enumerate(cache.layers):
        if not isinstance(layer, StaticSlidingWindowLayer):
            continue

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
        tt_layer.v_head_dim = layer.v_head_dim
        tt_layer.k_head_dim = layer.k_head_dim
        cache.layers[i] = tt_layer


def tt_create_sliding_window_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function=None,
    and_mask_function=None,
    **kwargs,
) -> torch.Tensor:
    """
    Compile-friendly sliding-window causal mask for the always-roll cache layout.

    Builds a 4D additive mask (batch, 1, query_length, sliding_window +
    query_length) using only broadcasting tensor operations — no
    get_mask_sizes(), and no mutable Python state. Designed to run with
    torch.compile on TT hardware as part of the model forward graph.

    `cache_position` is optional: callers that only provide `position_ids`
    (e.g. olmo3's decoder forward, or gemma3's `create_masks_for_generate`)
    have it derived here. `or_mask_function` / `and_mask_function` allow models
    like Gemma3 to overlay their image (token-type) bidirectional mask.
    """
    if past_key_values is None:
        return create_sliding_window_causal_mask(
            config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            **kwargs,
        )

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        return attention_mask

    device = inputs_embeds.device

    # Derive cache_position when not supplied (only position_ids was passed).
    if cache_position is None:
        if position_ids is not None:
            cache_position = position_ids[0].to(device=device)
        else:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=device
            )

    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0
    _layer = past_key_values.layers[layer_idx]
    # DynamicSlidingWindowLayer exposes .sliding_window; static layers use .max_cache_len
    sliding_window = getattr(_layer, "sliding_window", None) or _layer.max_cache_len

    batch_size = inputs_embeds.shape[0]
    query_length = cache_position.shape[0]
    dtype = inputs_embeds.dtype
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

    # (query_length, kv_length) boolean "allowed" matrix.
    mask = (valid.unsqueeze(0) & causal & in_window).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, 1, query_length, kv_pos.shape[0])

    if attention_mask is not None and attention_mask.ndim == 2:
        padding_mask = attention_mask.to(device=device, dtype=torch.bool)
        kv_pos_idx = kv_pos.clamp(min=0).long()
        padding = padding_mask[:, kv_pos_idx]
        mask = mask & padding.unsqueeze(1).unsqueeze(1)

    # Overlay model-specific mask functions (e.g. Gemma3 image bidirectional
    # attention via token_type_ids). These operate on absolute token positions,
    # so index queries by `cache_position` and keys by `kv_pos`. Never re-enable
    # out-of-range cache slots (kv_pos < 0).
    if or_mask_function is not None or and_mask_function is not None:
        # Mask functions index into per-token tensors (e.g. group_ids) by these
        # indices, so clamp out-of-range cache slots to 0 to stay in bounds;
        # they are removed afterwards via `valid_kv`.
        q_idx = cache_position.clamp(min=0).view(query_length, 1)
        kv_idx = kv_pos.clamp(min=0).view(1, kv_pos.shape[0])
        valid_kv = (kv_pos >= 0).view(1, 1, 1, kv_pos.shape[0])
        if or_mask_function is not None:
            extra = torch.stack(
                [
                    or_mask_function(
                        torch.tensor(b, device=device), None, q_idx, kv_idx
                    ).to(torch.bool)
                    for b in range(batch_size)
                ],
                dim=0,
            ).unsqueeze(
                1
            )  # (batch, 1, query_length, kv_length)
            mask = mask | (extra & valid_kv)
        if and_mask_function is not None:
            restrict = torch.stack(
                [
                    and_mask_function(
                        torch.tensor(b, device=device), None, q_idx, kv_idx
                    ).to(torch.bool)
                    for b in range(batch_size)
                ],
                dim=0,
            ).unsqueeze(1)
            mask = mask & restrict

    return torch.where(
        mask,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_val, dtype=dtype, device=device),
    )


def tt_create_masks_for_generate(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function=None,
    and_mask_function=None,
    **kwargs,
):
    """TT-compatible replacement for `create_masks_for_generate`.

    Mirrors the stock implementation but routes `sliding_attention` layers to
    the always-roll TT sliding-window mask so the mask key/value length matches
    `TTStaticSlidingWindowLayer` (sliding_window + query_length). Used by models
    that build their mask mapping up front (e.g. Gemma3 multimodal).
    """
    effective_config = config.get_text_config()
    mask_kwargs = {
        "config": effective_config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    def _build(pattern):
        if pattern == "sliding_attention":
            return tt_create_sliding_window_causal_mask(**mask_kwargs)
        if pattern == "chunked_attention":
            return create_chunked_causal_mask(**mask_kwargs)
        return create_causal_mask(**mask_kwargs)

    if hasattr(effective_config, "layer_types"):
        return {
            pattern: _build(pattern) for pattern in set(effective_config.layer_types)
        }
    if getattr(effective_config, "sliding_window", None) is not None:
        return tt_create_sliding_window_causal_mask(**mask_kwargs)
    if getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    return create_causal_mask(**mask_kwargs)


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
            self.lazy_initialization(key_states, value_states)

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

    @torch.compiler.disable
    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return (self.keys[0, 0].cpu().any(dim=-1)).sum().item()
