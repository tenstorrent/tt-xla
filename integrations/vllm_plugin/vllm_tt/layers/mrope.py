# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from types import MethodType

import torch
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


def _apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Reorganize chunked [T...H...W...] frequencies into interleaved layout."""
    out = x[0].clone()
    out[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    out[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return out


def _tt_mrope_forward(
    self: MRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """TT-safe MRoPE forward path based on vLLM's native implementation.

    Unlike vLLM's own (fully flattened, batch-less) forward_native, this
    plugin's model_runner keeps an explicit request-batch dimension on
    positions:
        positions: [num_reqs, num_tokens] (text only) or
                   [3, num_reqs, num_tokens] (T/H/W, multimodal)
    while query/key arrive already flattened by vLLM's own model code
    (query/key no longer carry a separate batch axis, vLLM >=0.20):
        query: [num_reqs * num_tokens, num_heads * head_size]
        key:   [num_reqs * num_tokens, num_kv_heads * head_size]
    cos/sin are computed in positions' batched layout, then flattened to
    match query/key's flattened leading dimension.
    """

    del offsets
    assert positions.ndim in (2, 3)
    assert key is not None

    is_mrope_positions = positions.ndim == 3 and positions.shape[0] == 3

    total_tokens = query.shape[0]
    assert key.shape[0] == total_tokens

    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if is_mrope_positions:
        assert self.mrope_section
        if self.mrope_interleaved:
            cos = _apply_interleaved_rope(cos, self.mrope_section)
            sin = _apply_interleaved_rope(sin, self.mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )

    # cos/sin are [num_reqs, num_tokens, rotary_dim // 2] here (mrope, after the
    # T/H/W section select above, or plain, straight from indexing
    # cos_sin_cache). Flatten to match query/key's already-flattened
    # [num_reqs * num_tokens, ...] convention.
    cos = cos.reshape(total_tokens, -1)
    sin = sin.reshape(total_tokens, -1)

    query_shape = query.shape
    query = query.view(total_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = self.apply_rotary_emb.forward_native(query_rot, cos, sin)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(total_tokens, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_pass = key[..., self.rotary_dim :]
    key_rot = self.apply_rotary_emb.forward_native(key_rot, cos, sin)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def override_mrope_module(layer: torch.nn.Module) -> torch.nn.Module:
    """Override MRoPE forward with TT local implementation.

    This keeps the original module instance (and all its parameters/buffers)
    while decoupling behavior from installed base vLLM.
    """

    assert isinstance(layer, MRotaryEmbedding)
    layer.forward = MethodType(_tt_mrope_forward, layer)
    return layer
