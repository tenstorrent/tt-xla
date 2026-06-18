# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

import torch
import torch.nn as nn
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


class TTRotaryEmbedding(nn.Module):
    """TT-compatible RotaryEmbedding that computes cos/sin on-the-fly.

    vLLM's RotaryEmbedding pre-builds a cos_sin_cache and uses index_select
    (gather) with position_ids at runtime. This lowers to ttir.embedding which
    requires indices on host via from_device, breaking metal trace mode.

    This replacement computes cos/sin from inv_freq and positions using math
    ops (outer product + cos/sin) that stay entirely on device.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        assert isinstance(layer, RotaryEmbedding)
        self.head_size = layer.head_size
        self.rotary_dim = layer.rotary_dim
        self.is_neox_style = layer.is_neox_style
        # Delegates to the subclass implementation for correct frequency scaling
        # (e.g. Llama3RotaryEmbedding applies frequency-dependent scaling)
        inv_freq = layer._compute_inv_freq(layer.base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(num_tokens, -1, self.head_size)
        x_rot = ApplyRotaryEmb.forward_static(
            x[..., : self.rotary_dim], cos, sin, self.is_neox_style
        )
        if self.rotary_dim == self.head_size:
            return x_rot.reshape(orig_shape)
        return torch.cat((x_rot, x[..., self.rotary_dim :]), dim=-1).reshape(orig_shape)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions_flat = positions.flatten().to(torch.float32)
        num_tokens = positions_flat.shape[0]

        freqs = torch.outer(positions_flat, self.inv_freq)
        cos = freqs.cos().to(query.dtype)
        sin = freqs.sin().to(query.dtype)

        query = self._apply_rotary(query, cos, sin, num_tokens)
        if key is not None:
            key = self._apply_rotary(key, cos, sin, num_tokens)
        return query, key


def override_rotary_embedding_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RotaryEmbedding)
    return TTRotaryEmbedding(layer)
