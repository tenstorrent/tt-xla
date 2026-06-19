# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Pure-PyTorch depthwise causal conv1d for GDN.

Mirrors the semantics of vLLM's ``causal_conv1d_fn`` (prefill) and
``causal_conv1d_update`` (decode) used in ``GatedDeltaNetAttention._forward_core``,
but written in plain PyTorch so it traces through the TT backend.

Conventions
-----------
* ``weight``    : ``[conv_dim, K]`` depthwise kernel (one filter per channel).
* ``conv_state``: ``[N, conv_dim, K-1]`` cache, channel-major — the same layout
  upstream feeds these ops after ``self_kv_cache[0].transpose(-1, -2)``. It holds
  the last ``K-1`` *input* tokens per request slot. Mutated in place.
"""

import torch
import torch.nn.functional as F


def _apply_activation(y: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return y
    if activation in ("silu", "swish"):
        return F.silu(y)
    raise NotImplementedError(f"Unsupported conv activation '{activation}'")


def tt_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state: torch.Tensor,
    has_initial_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Prefill depthwise causal conv1d over packed variable-length sequences.

    Args:
        x: ``[conv_dim, total_tokens]`` channel-major activations (transposed).
        weight: ``[conv_dim, K]`` depthwise kernel.
        bias: ``[conv_dim]`` or ``None``.
        activation: ``"silu"``/``"swish"`` or ``None``.
        conv_state: ``[N, conv_dim, K-1]`` cache; trailing context written back
            in place at ``cache_indices``.
        has_initial_state: ``[num_seqs]`` bool — whether to seed left context
            from ``conv_state`` (continuation) vs zeros (fresh prefill).
        cache_indices: ``[num_seqs]`` long — state slot per sequence.
        query_start_loc: ``[num_seqs+1]`` int — cumulative token offsets.

    Returns:
        ``[conv_dim, total_tokens]`` post-conv (+bias, +activation).
    """
    conv_dim, K = weight.shape
    out = torch.empty_like(x)
    num_seqs = cache_indices.shape[0]

    for n in range(num_seqs):
        start = int(query_start_loc[n])
        end = int(query_start_loc[n + 1])
        if end <= start:
            continue
        slot = int(cache_indices[n])
        L = end - start
        seq = x[:, start:end]  # [conv_dim, L]

        if bool(has_initial_state[n]):
            left = conv_state[slot].to(x.dtype)  # [conv_dim, K-1]
        else:
            left = torch.zeros(
                (conv_dim, K - 1), dtype=x.dtype, device=x.device
            )
        padded = torch.cat([left, seq], dim=-1)  # [conv_dim, K-1+L]

        # Depthwise causal conv as K slice-multiply-adds (avoids a grouped
        # conv2d lowering): y[:, t] = sum_j w[:, j] * padded[:, t+j].
        y = weight[:, 0:1] * padded[:, 0:L]
        for j in range(1, K):
            y = y + weight[:, j:j + 1] * padded[:, j:j + L]
        if bias is not None:
            y = y + bias.unsqueeze(-1)
        out[:, start:end] = _apply_activation(y, activation)

        # New state: the last K-1 input tokens (zero-padded on the left if the
        # sequence is shorter than K-1).
        new_state = padded[:, -(K - 1):] if K > 1 else padded[:, 0:0]
        conv_state[slot] = new_state.to(conv_state.dtype)

    return out


def tt_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Single-token decode conv1d update.

    Args:
        x: ``[num_tokens, conv_dim]`` one new token per decoding sequence.
        conv_state: ``[N, conv_dim, K-1]`` cache; shifted + appended in place.
        weight: ``[conv_dim, K]`` depthwise kernel.
        bias: ``[conv_dim]`` or ``None``.
        activation: ``"silu"``/``"swish"`` or ``None``.
        conv_state_indices: ``[num_tokens]`` long — state slot per token.

    Returns:
        ``[num_tokens, conv_dim]`` post-conv (+bias, +activation).
    """
    conv_dim, K = weight.shape
    num_tokens = x.shape[0]
    out = torch.empty_like(x)

    for t in range(num_tokens):
        slot = int(conv_state_indices[t])
        state = conv_state[slot].to(x.dtype)  # [conv_dim, K-1]
        new_tok = x[t].unsqueeze(-1)  # [conv_dim, 1]
        window = torch.cat([state, new_tok], dim=-1)  # [conv_dim, K]
        y = (weight * window).sum(dim=-1)  # [conv_dim]
        if bias is not None:
            y = y + bias
        out[t] = _apply_activation(y, activation)
        # Shift the window left by one: drop the oldest, keep the newest K-1.
        conv_state[slot] = window[:, 1:].to(conv_state.dtype)

    return out
