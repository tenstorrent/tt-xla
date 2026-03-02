# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-XLA compatible transformers cache layer.

TTStaticSlidingWindowLayer is a drop-in replacement for
transformers.cache_utils.StaticSlidingWindowLayer that keeps cumulative_length
as a 1-element tensor (shape (1,)) instead of a Python int. get_mask_sizes() returns (kv_length, kv_offset) as ints for compatibility with
callers (e.g. masking_utils). update() uses a scalar for conditionals (may graph-break).
"""

from typing import Any, Optional

import torch

from transformers.cache_utils import StaticCache, StaticLayer, StaticSlidingWindowLayer


def replace_static_sliding_window_layers_with_tt(
    cache: StaticCache,
    max_cache_len: int,
    sliding_window: int,
) -> None:
    """
    Replace each StaticSlidingWindowLayer in cache.layers with TTStaticSlidingWindowLayer.
    Call after cache.early_initialization() so the new layers get the same keys/values/state.
    """
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, StaticSlidingWindowLayer):
            tt_layer = TTStaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=sliding_window)
            tt_layer.keys = layer.keys
            tt_layer.values = layer.values
            tt_layer.is_initialized = True
            tt_layer.device = layer.device
            tt_layer.dtype = layer.dtype
            tt_layer.max_batch_size = layer.max_batch_size
            tt_layer.num_heads = layer.num_heads
            tt_layer.head_dim = layer.head_dim
            tt_layer.cumulative_length = torch.tensor([0], dtype=torch.long, device=layer.keys.device)
            cache.layers[i] = tt_layer


class TTStaticSlidingWindowLayer(StaticLayer):
    """
    Static sliding-window cache layer: cumulative_length is a 1-element tensor.
    get_mask_sizes returns (int, int); update() is simple if/elif/else (scalar read may graph-break).
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        super().lazy_initialization(key_states)
        self.cumulative_length = torch.tensor([0], dtype=torch.long, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.is_initialized:
            self.lazy_initialization(key_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        cumulative_length = self.cumulative_length.clone()
        self.cumulative_length += key_states.shape[-2]

        is_full = cumulative_length >= self.max_cache_len
        overflow = cumulative_length + key_states.shape[-2] > self.max_cache_len

        if is_full:
            if key_states.shape[-2] == 1:
                new_keys = self.keys.roll(-1, dims=-2)
                new_values = self.values.roll(-1, dims=-2)

                index = torch.tensor([-1], dtype=int, device=self.device)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states

                self.keys.copy_(new_keys)
                self.values.copy_(new_values)
                return self.keys, self.values
            else:
                full_key_states = torch.cat((self.keys[:, :, 1:, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, 1:, :], value_states), dim=-2)
        elif overflow:
            if cumulative_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                full_key_states = torch.cat((self.keys[:, :, :cumulative_length, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, :cumulative_length, :], value_states), dim=-2)
        else:
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            return self.keys, self.values

        self.keys.copy_(full_key_states[:, :, -self.max_cache_len :, :])
        self.values.copy_(full_value_states[:, :, -self.max_cache_len :, :])
        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        sliding_window = self.max_cache_len
        is_full = self.cumulative_length >= self.max_cache_len

        kv_offset = max(self.cumulative_length - sliding_window + 1, 0)
        if isinstance(kv_offset, torch.Tensor) and kv_offset.dim() == 1:
            kv_offset = kv_offset[0]

        if is_full:
            kv_length = sliding_window + query_length - 1
        elif self.cumulative_length + query_length > sliding_window:
            kv_length = self.cumulative_length + query_length
        else:
            kv_length = sliding_window

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        cl = self.cumulative_length
        return int(cl[0]) if isinstance(cl, torch.Tensor) else cl
