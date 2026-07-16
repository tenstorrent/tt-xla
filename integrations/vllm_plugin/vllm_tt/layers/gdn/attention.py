# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""TT-compatible override of vLLM's ``GatedDeltaNetAttention``.

Keeps upstream ``forward``'s Part 1 (input projection) and Part 3 (gated RMSNorm
+ out_proj) intact, and replaces Part 2 (the ``torch.ops.vllm.gdn_attention_core``
custom op, which reads global forward context and dispatches to Triton/FLA
kernels) with ``_tt_gdn_forward_core`` — a torch.compile-traceable
re-implementation that threads conv/ssm state through ``self.kv_cache`` exactly
like upstream ``_forward_core``, but using the pure-PyTorch TT ops in this package.

Out of scope (asserted off): speculative decoding (``spec_sequence_masks``).
"""

from types import MethodType

import torch
from einops import rearrange
from tt_torch.sharding import sharding_constraint_tensor
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention

from . import (
    tt_causal_conv1d_fn,
    tt_causal_conv1d_update,
    tt_chunk_gated_delta_rule,
    tt_fused_gdn_gating,
    tt_fused_recurrent_gated_delta_rule,
)

# EXPERIMENT: run prefill through the sequential recurrence instead of the
# chunk-parallel form. The chunk form materializes a dense [HV, C, C]
# unit-lower-triangular inverse (via _inv_unit_lower_tri) plus C x C score/decay
# matrices, which is a suspected source of device-DRAM / tile-padding blowup.
# The recurrence avoids all C x C tensors; for short sequences the token loop is
# cheap. Flip back to False to restore the chunk-parallel path.
_PREFILL_USE_RECURRENT = False


def _prefill_delta_rule(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    """Prefill core: sequential recurrence (no C x C inverse) or chunk-parallel,
    selected by ``_PREFILL_USE_RECURRENT``. Both return ``(o, final_state)``."""
    if _PREFILL_USE_RECURRENT:
        return tt_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=False,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=None,
            use_qk_l2norm_in_kernel=True,
        )
    return tt_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )


def _grouped_qkv_perm(self: GatedDeltaNetAttention) -> torch.Tensor:
    """Permutation from the merged ``[q | k | v]`` channel order to the model's
    native per-head-group order ``[q_g k_g v_gx3]`` for g in range(num_k_heads).

    Sharding the grouped order contiguously on "model" gives every device a whole
    set of head groups (aligned head-parallel), so the qkv split needs no gather.
    Returns a LongTensor of length ``conv_dim`` mapping new position -> old channel.
    Assumes the SPMD invariant ``tp_size == 1`` (whole tensor on the controller).
    """
    hk, hv = self.head_k_dim, self.head_v_dim
    gpv = self.num_v_heads // self.num_k_heads  # value heads per group (GQA ratio)
    key_dim, ng = self.key_dim, self.num_k_heads
    idx: list[int] = []
    for g in range(ng):
        idx += range(g * hk, g * hk + hk)  # q head g
        idx += range(key_dim + g * hk, key_dim + g * hk + hk)  # k head g
        vbase = 2 * key_dim + gpv * g * hv
        idx += range(vbase, vbase + gpv * hv)  # v heads for group g
    return torch.tensor(idx, dtype=torch.long)


def _rearrange_mixed_qkv_grouped(self: GatedDeltaNetAttention, mixed_qkv):
    """De-interleave the per-head-group channel layout into (q, k, v) head tensors.

    Inverse of the grouped packing in ``_input_projection``. Because "model" shards
    the group axis, the reshape stays sharding-aligned (no cross-device movement).
    """
    if mixed_qkv is None:
        return None, None, None
    length = mixed_qkv.size(0)
    hk, hv = self.head_k_dim, self.head_v_dim
    gpv = self.num_v_heads // self.num_k_heads
    ng = self.num_k_heads
    per_group = hk + hk + gpv * hv
    x = mixed_qkv.reshape(length, ng, per_group)
    q = x[:, :, :hk].reshape(1, length, ng, hk)
    k = x[:, :, hk : 2 * hk].reshape(1, length, ng, hk)
    v = x[:, :, 2 * hk :].reshape(1, length, self.num_v_heads, hv)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _input_projection(self: GatedDeltaNetAttention, hidden_states: torch.Tensor):
    """Upstream forward Part 1 — returns ``(mixed_qkv, z, b, a)``.

    Handles the LoRA path (``in_proj_qkv``), the Qwen3-Next interleaved layout,
    and the Qwen3.5 split layout, matching ``GatedDeltaNetAttention.forward``.
    """
    if hasattr(self, "in_proj_qkv"):
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
        return mixed_qkv, z, b.contiguous(), a.contiguous()

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)
    if self.gqa_interleaved_layout:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
        # Pack the channels in the native per-head-group order
        # ([q_g, k_g, v_g x gpv] for each group g) rather than de-interleaving to
        # a merged [q | k | v]. A contiguous "model" shard then covers whole head
        # groups, so the downstream conv/split are aligned head-parallel and need
        # no cross-device gather. _rearrange_mixed_qkv_grouped inverts this, and
        # override_gdn_linear_attn_module reorders the conv weight to match.
        # query, key: (l, ng, head_k_dim); value: (l, num_v_heads, head_v_dim).
        ng = self.num_k_heads // self.tp_size
        value_g = value.reshape(value.size(0), ng, -1)  # (l, ng, gpv*head_v_dim)
        mixed_qkv = torch.cat((query, key, value_g), dim=2).reshape(query.size(0), -1)
        return mixed_qkv, z, b.contiguous(), a.contiguous()

    qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
    z_size = self.value_dim // self.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, self.head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


def _conv_weights(self: GatedDeltaNetAttention) -> torch.Tensor:
    w = self.conv1d.weight
    return w.view(w.size(0), w.size(2))  # [conv_dim, K]


def _core_prefill(
    self: GatedDeltaNetAttention,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor | None,
    ssm_state: torch.Tensor | None,
    indices: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
) -> None:
    num_actual = mixed_qkv.size(0)
    conv_w = _conv_weights(self)
    conv_dim, K = conv_w.shape

    # Build the conv-state plumbing. When no real cache is bound (pure profiling)
    # fall back to a temporary zero state for a single packed sequence, so the
    # traced graph still runs the real conv op.
    if conv_state is None:
        conv_state = torch.zeros(
            (1, conv_dim, K - 1), dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
        indices = torch.zeros(1, dtype=torch.long, device=mixed_qkv.device)
        query_start_loc = torch.tensor(
            [0, num_actual], dtype=torch.int32, device=mixed_qkv.device
        )
    if has_initial_state is None:
        has_initial_state = torch.zeros(
            indices.shape[0], dtype=torch.bool, device=mixed_qkv.device
        )

    x = tt_causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_w,
        self.conv1d.bias,
        self.activation,
        conv_state=conv_state,
        has_initial_state=has_initial_state,
        cache_indices=indices,
        query_start_loc=query_start_loc,
    ).transpose(0, 1)

    if self._tt_mesh is not None and not self._tt_grouped_qkv:
        # Merged [q|k|v] layout: `x` arrives sharded contiguously on "model", and a
        # contiguous cut splits v into an UNEVEN per-shard head layout (e.g.
        # 0/8/20/20 of 48 heads on a 4-way axis), incompatible with g/beta's even
        # head sharding (12/12/12/12) -> mismatched heads -> garbage. A constraint
        # AFTER the head reshape can't recover it (v is already scrambled). So gather
        # the channel dim before the split; the per-head constraints below then
        # re-shard cleanly. The grouped layout avoids this entirely (no gather).
        x = sharding_constraint_tensor(x, self._tt_mesh, (None, None))

    q, k, v = self.rearrange_mixed_qkv(x)  # [1,L,H,K], [1,L,H,K], [1,L,HV,V]
    g, beta = tt_fused_gdn_gating(self.A_log, a, b, self.dt_bias)  # [L,HV]
    g, beta = g.unsqueeze(0), beta.unsqueeze(0)  # [1,L,HV]
    if self._tt_mesh is not None:
        # Re-shard the split tensors to a consistent head-parallel layout so the
        # delta rule pairs matching q/k/v heads with their g/beta gates (see the
        # channel-gather note above rearrange_mixed_qkv).
        q = sharding_constraint_tensor(
            q,
            self._tt_mesh,
            (None, None, "model", None),
        )
        k = sharding_constraint_tensor(
            k,
            self._tt_mesh,
            (None, None, "model", None),
        )
        v = sharding_constraint_tensor(
            v,
            self._tt_mesh,
            (None, None, "model", None),
        )
        g = sharding_constraint_tensor(
            g,
            self._tt_mesh,
            (None, None, "model"),
        )
        beta = sharding_constraint_tensor(
            beta,
            self._tt_mesh,
            (None, None, "model"),
        )
    scale = self.head_k_dim**-0.5

    if ssm_state is not None:
        # Gather/zero/scatter the recurrent state with tensor index ops and a
        # mask-multiply (no boolean-mask assignment), so the graph traces
        # cleanly through backend="tt".
        init = ssm_state.index_select(0, indices).to(torch.float32)
        init = init * has_initial_state.to(init.dtype).view(-1, 1, 1, 1)
        o, final_state = _prefill_delta_rule(
            q, k, v, g, beta, scale, init, query_start_loc
        )
        ssm_state.index_copy_(0, indices, final_state.to(ssm_state.dtype))
    else:
        o, _ = _prefill_delta_rule(q, k, v, g, beta, scale, None, query_start_loc)

    core_attn_out[:num_actual] = o.squeeze(0).to(core_attn_out.dtype)


def _core_decode(
    self: GatedDeltaNetAttention,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_decodes: int,
) -> None:
    num_actual = mixed_qkv.size(0)
    conv_w = _conv_weights(self)

    x = tt_causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_w,
        self.conv1d.bias,
        self.activation,
        conv_state_indices=indices[:num_actual],
    )
    if self._tt_mesh is not None and not self._tt_grouped_qkv:
        # Merged [q|k|v] path only: gather the channel dim before the split (see the
        # note in _core_prefill). The grouped layout is already aligned, no gather.
        x = sharding_constraint_tensor(x, self._tt_mesh, (None, None))

    q, k, v = self.rearrange_mixed_qkv(x)  # [1, num_decodes, H/HV, d]
    g, beta = tt_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    g, beta = g.unsqueeze(0), beta.unsqueeze(0)

    if self._tt_mesh is not None:
        q = sharding_constraint_tensor(q, self._tt_mesh, (None, None, "model", None))
        k = sharding_constraint_tensor(k, self._tt_mesh, (None, None, "model", None))
        v = sharding_constraint_tensor(v, self._tt_mesh, (None, None, "model", None))
        g = sharding_constraint_tensor(g, self._tt_mesh, (None, None, "model"))
        beta = sharding_constraint_tensor(beta, self._tt_mesh, (None, None, "model"))

    o, _ = tt_fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=self.head_k_dim**-0.5,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=query_start_loc[: num_decodes + 1],
        ssm_state_indices=indices[:num_actual],
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out[:num_actual] = o.squeeze(0).to(core_attn_out.dtype)


def _tt_gdn_forward_core(
    self: GatedDeltaNetAttention,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = md.get(self.prefix)

    kv = getattr(self, "kv_cache", None)
    conv_state = kv[0].transpose(-1, -2) if kv else None
    ssm_state = kv[1] if kv else None

    if md is None:
        # No metadata (profiling): stateless fresh prefill so the traced graph
        # still exercises the real ops.
        _core_prefill(
            self,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state=None,
            ssm_state=None,
            indices=None,
            query_start_loc=None,
            has_initial_state=None,
        )
        return

    assert (
        getattr(md, "spec_sequence_masks", None) is None
    ), "Speculative decoding is not supported by the TT GDN path."

    num_actual = md.num_actual_tokens
    indices = md.non_spec_state_indices_tensor
    qsl = md.non_spec_query_start_loc
    mixed_qkv = mixed_qkv[:num_actual]
    b = b[:num_actual]
    a = a[:num_actual]

    if md.num_decodes > 0 and md.num_prefills == 0:
        _core_decode(
            self,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state,
            ssm_state,
            indices,
            qsl,
            md.num_decodes,
        )
    elif md.num_prefills > 0 and md.num_decodes == 0:
        _core_prefill(
            self,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_state,
            ssm_state,
            indices,
            qsl,
            md.has_initial_state,
        )
    else:
        raise NotImplementedError(
            "Mixed prefill+decode batches are not supported by the TT GDN path "
            f"(num_prefills={md.num_prefills}, num_decodes={md.num_decodes}). "
            "Run with one request per step (max_num_seqs=1) for now."
        )


def _tt_gdn_forward(
    self: GatedDeltaNetAttention,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """TT GDN forward. Accepts packed ``[num_tokens, hidden]`` or batched
    ``[batch, tokens, hidden]`` inputs; flattens to token-major, runs upstream
    Parts 1/3 around the TT core, and writes back through ``output``."""
    hidden_shape = hidden_states.shape
    if hidden_states.ndim < 2:
        raise ValueError(f"Expected hidden_states with >=2 dims, got {hidden_shape}")
    hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
    num_tokens = hidden_states.size(0)

    # Part 1: input projection.
    mixed_qkv, z, b, a = _input_projection(self, hidden_states)

    # Part 2: core (conv + gating + delta rule, state-threaded).
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _tt_gdn_forward_core(self, mixed_qkv, b, a, core_attn_out)

    # Part 3: gated RMSNorm + output projection.
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    projected_output, _ = self.out_proj(core_attn_out)
    output.copy_(projected_output.reshape_as(output))


def override_gdn_linear_attn_module(layer: torch.nn.Module) -> torch.nn.Module:
    """Override vLLM GDN forward with the TT-compatible, state-threaded path."""
    assert isinstance(layer, GatedDeltaNetAttention)
    layer._tt_mesh = None
    # Grouped (gather-free) qkv layout: only for the interleaved-GQA projection,
    # which stores weights per head group. Reorder the conv weight/bias to that
    # same grouped channel order (runs post weight-load, pre-shard) and swap in the
    # de-interleaving rearrange. Other projection layouts keep the merged [q|k|v]
    # path (with the pre-split gather in _core_prefill/_core_decode).
    layer._tt_grouped_qkv = bool(getattr(layer, "gqa_interleaved_layout", False))
    if layer._tt_grouped_qkv:
        perm = _grouped_qkv_perm(layer).to(layer.conv1d.weight.device)
        with torch.no_grad():
            layer.conv1d.weight.data = layer.conv1d.weight.data[perm].contiguous()
            if layer.conv1d.bias is not None:
                layer.conv1d.bias.data = layer.conv1d.bias.data[perm].contiguous()
        layer.rearrange_mixed_qkv = MethodType(_rearrange_mixed_qkv_grouped, layer)
    layer.forward = MethodType(_tt_gdn_forward, layer)
    return layer
