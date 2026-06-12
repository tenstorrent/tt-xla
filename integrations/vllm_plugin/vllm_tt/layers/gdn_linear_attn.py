# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from types import MethodType

import torch
from einops import rearrange
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention


def _tt_gdn_core_fallback(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    num_v_heads: int,
    tp_size: int,
    key_dim: int,
    value_dim: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Pure PyTorch GDN core approximation without Triton/custom ops."""

    _, _, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    value = value.reshape(-1, num_v_heads // tp_size, head_v_dim)

    gate = torch.sigmoid(a).unsqueeze(-1)
    beta = torch.sigmoid(b).unsqueeze(-1)

    # Compute a stable vectorized recurrence for
    # s_t = gate_t * s_{t-1} + beta_t * value_t.
    eps = 1e-6
    gate_prefix = torch.cumprod(gate.clamp_min(eps), dim=0)
    scaled_value = beta * value / gate_prefix
    return gate_prefix * torch.cumsum(scaled_value, dim=0)


def _tt_gdn_forward(
    self: GatedDeltaNetAttention,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """TT-safe GDN forward that accepts packed or batched hidden states.

    vLLM's upstream GDN implementation assumes token-major ``[num_tokens, hidden]``
    inputs. The TT model runner can reach this layer with batched tensors such as
    ``[batch, tokens, hidden]`` during warmup and prefill. Flatten the leading
    dimensions into the token axis before running the original math, then restore
    the original output shape.
    """

    hidden_shape = hidden_states.shape
    if hidden_states.ndim < 2:
        raise ValueError(
            "Expected hidden_states to have at least 2 dimensions, "
            f"got shape {hidden_shape}"
        )

    hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
    num_tokens = hidden_states.size(0)

    if hasattr(self, "in_proj_qkv"):
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()
    else:
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()

    core_attn_out = _tt_gdn_core_fallback(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        num_v_heads=self.num_v_heads,
        tp_size=self.tp_size,
        key_dim=self.key_dim,
        value_dim=self.value_dim,
        head_v_dim=self.head_v_dim,
    )

    z_shape = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    projected_output, _ = self.out_proj(core_attn_out)
    output.copy_(projected_output.reshape_as(output))


def override_gdn_linear_attn_module(layer: torch.nn.Module) -> torch.nn.Module:
    """Override vLLM GDN forward with a TT-compatible shape-normalizing path."""

    assert isinstance(layer, GatedDeltaNetAttention)
    layer.forward = MethodType(_tt_gdn_forward, layer)
    return layer
