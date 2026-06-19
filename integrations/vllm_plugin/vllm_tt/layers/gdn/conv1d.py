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

# When True, use the *proper* depthwise conv (torch F.conv1d with groups=conv_dim,
# which lowers to a single ttnn.conv2d). This is the correct/perf path and should
# be the default once tt-mlir/tt-metal can run it — today it FAILS on Wormhole
# (large channel counts: "DRAM Auto slice could not find valid slice
# configuration"; small: hangs on readback). See gdn_depthwise_conv1d_bug.md.
#
# When False (default), use a memory-light, device-correct workaround: the same
# depthwise causal conv expressed as K shifted multiply-adds, entirely in the
# activation dtype (no fp32), with no grouped conv2d. Perf is not a concern for
# this branch.
_USE_NATIVE_CONV1D = False


def _apply_activation(y: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return y
    if activation in ("silu", "swish"):
        return F.silu(y)
    raise NotImplementedError(f"Unsupported conv activation '{activation}'")


def _depthwise_causal_conv_1seq(
    seq: torch.Tensor,
    left: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv for one sequence.

    ``seq`` is ``[conv_dim, L]``, ``left`` is the ``[conv_dim, K-1]`` left context
    (already masked: zeros for a fresh prefill). Returns ``(out [conv_dim, L],
    new_state [conv_dim, K-1])``. Selected behavior follows ``_USE_NATIVE_CONV1D``.
    """
    conv_dim, K = weight.shape
    L = seq.shape[1]
    # One small concat to form the K-1 left context + the sequence.
    padded = torch.cat([left, seq], dim=-1)  # [conv_dim, K-1+L]

    if _USE_NATIVE_CONV1D:
        # Proper depthwise conv. Lowers to ttnn.conv2d(groups=conv_dim) — correct
        # but currently broken on device (see module note / bug report).
        y = F.conv1d(
            padded.unsqueeze(0), weight.unsqueeze(1), bias=bias, groups=conv_dim
        ).squeeze(0)
    else:
        # Memory-light workaround: y[:, t] = sum_j w[:, j] * padded[:, t+j].
        # Stays in seq.dtype (no fp32 upcast) and emits only elementwise
        # mul/add — no grouped conv2d, no large intermediates.
        y = weight[:, 0:1] * padded[:, 0:L]
        for j in range(1, K):
            y = y + weight[:, j:j + 1] * padded[:, j:j + L]
        if bias is not None:
            y = y + bias.unsqueeze(-1)

    out = _apply_activation(y, activation)
    new_state = padded[:, -(K - 1):] if K > 1 else padded[:, 0:0]
    return out, new_state


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
    num_seqs = cache_indices.shape[0]

    if num_seqs == 1:
        # Branchless single-sequence path (max_num_seqs=1). Reading tensor
        # *values* with int()/bool() (the loop below) creates dynamo graph
        # breaks that crash the torch.compile(backend="tt") partitioner, so this
        # path uses only tensor index ops and a mask-multiply. ``num_seqs`` is a
        # static shape, so this branch is chosen at trace time. Left context =
        # conv_state masked by has_initial_state (zeros for a fresh prefill).
        left = conv_state.index_select(0, cache_indices)[0].to(x.dtype)
        left = left * has_initial_state[0].to(x.dtype)
        out, new_state = _depthwise_causal_conv_1seq(
            x, left, weight, bias, activation
        )
        conv_state.index_copy_(
            0, cache_indices, new_state.unsqueeze(0).to(conv_state.dtype)
        )
        return out

    out = torch.empty_like(x)
    for n in range(num_seqs):
        start = int(query_start_loc[n])
        end = int(query_start_loc[n + 1])
        if end <= start:
            continue
        slot = int(cache_indices[n])
        seq = x[:, start:end]  # [conv_dim, L]
        if bool(has_initial_state[n]):
            left = conv_state[slot].to(x.dtype)  # [conv_dim, K-1]
        else:
            left = torch.zeros((conv_dim, K - 1), dtype=x.dtype, device=x.device)

        out_seq, new_state = _depthwise_causal_conv_1seq(
            seq, left, weight, bias, activation
        )
        out[:, start:end] = out_seq
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
