# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Pure-PyTorch recurrent gated delta rule (decode path).

Mirrors ``fla...ops.fused_recurrent.fused_recurrent_gated_delta_rule``:

Per token ``t`` and value-head ``hv`` (state ``S`` is ``[V, K]``)::

    S      = exp(g_t) * S                  # gated decay
    u      = beta_t * (v_t - S @ k_t)      # delta correction (uses decayed S)
    S      = S + outer(u, k_t)
    o_t    = S @ (scale * q_t)

``q``/``k`` are L2-normalized first when ``use_qk_l2norm_in_kernel`` is set.
GQA is handled by repeating each key head ``HV // H`` times.
"""

import torch

from .l2norm import tt_l2norm_fwd


def tt_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent gated delta rule.

    Args:
        q, k: ``[B, T, H, K]``.
        v: ``[B, T, HV, V]`` (``HV >= H`` for GQA).
        g: ``[B, T, HV]`` log-decay.
        beta: ``[B, T, HV]`` write strength (defaults to ones).
        scale: q scaling; defaults to ``K ** -0.5``.
        initial_state: either ``[N, HV, V, K]`` (indexed by sequence) or, when
            ``ssm_state_indices`` is given, a full cache ``[num_blocks, HV, V, K]``.
        inplace_final_state: write the updated state back into ``initial_state``.
        cu_seqlens: ``[N+1]`` token offsets (requires ``B == 1``).
        ssm_state_indices: ``[N]`` slot per sequence into ``initial_state``.
        use_qk_l2norm_in_kernel: L2-normalize q and k before the recurrence.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, HV, V]`` and
        ``final_state`` the (updated) state tensor.
    """
    B, T, H, Kdim = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = Kdim**-0.5
    if cu_seqlens is not None and B != 1:
        raise ValueError("cu_seqlens requires batch size 1 (flatten inputs).")

    if use_qk_l2norm_in_kernel:
        q = tt_l2norm_fwd(q)
        k = tt_l2norm_fwd(k)

    if beta is None:
        beta = torch.ones_like(g)

    gqa = HV // H
    qf = q.to(torch.float32).repeat_interleave(gqa, dim=2)  # [B,T,HV,K]
    kf = k.to(torch.float32).repeat_interleave(gqa, dim=2)  # [B,T,HV,K]
    vf = v.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)

    # Build the per-sequence (start, end, slot) schedule.
    if cu_seqlens is not None:
        bounds = [
            (int(cu_seqlens[n]), int(cu_seqlens[n + 1])) for n in range(len(cu_seqlens) - 1)
        ]
        seqs = [(0, s, e, n) for n, (s, e) in enumerate(bounds)]
    else:
        seqs = [(b, 0, T, b) for b in range(B)]

    final_state = (
        initial_state
        if (initial_state is not None and inplace_final_state)
        else (initial_state.clone() if initial_state is not None else None)
    )
    if final_state is None:
        n_states = len(seqs)
        final_state = torch.zeros(
            (n_states, HV, V, Kdim), dtype=torch.float32, device=q.device
        )

    o = torch.zeros((B, T, HV, V), dtype=torch.float32, device=q.device)

    for (bi, start, end, seq_idx) in seqs:
        slot = int(ssm_state_indices[seq_idx]) if ssm_state_indices is not None else seq_idx
        S = final_state[slot].to(torch.float32)  # [HV, V, K]
        for t in range(start, end):
            decay = torch.exp(gf[bi, t]).view(HV, 1, 1)  # [HV,1,1]
            S = S * decay
            k_t = kf[bi, t]  # [HV, K]
            Sk = torch.einsum("hvd,hd->hv", S, k_t)  # [HV, V]
            u = (vf[bi, t] - Sk) * bf[bi, t].unsqueeze(-1)  # [HV, V]
            S = S + torch.einsum("hv,hd->hvd", u, k_t)  # [HV, V, K]
            q_t = qf[bi, t] * scale  # [HV, K]
            o[bi, t] = torch.einsum("hvd,hd->hv", S, q_t)  # [HV, V]
        final_state[slot] = S.to(final_state.dtype)

    return o.to(v.dtype), final_state
