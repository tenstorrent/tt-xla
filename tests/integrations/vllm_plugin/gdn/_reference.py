# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Ground-truth, framework-agnostic references for the GDN ops.

These are the mathematical definitions the TT ops must match. They are used to
(a) cross-check the FLA golden generator on the GPU box (catching state-layout
mismatches), and (b) provide a fallback golden when FLA is unavailable so the
consumer test can still exercise the TT ops on CPU.
"""

import torch
import torch.nn.functional as F


def ref_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.to(torch.float32)
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def ref_gating(A_log, a, b, dt_bias, beta_sp: float = 1.0, threshold: float = 20.0):
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus_x = F.softplus(x, beta=beta_sp, threshold=threshold)
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    return g, torch.sigmoid(b.to(torch.float32))


def ref_delta_rule(
    q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, l2norm=False
):
    """Naive token-by-token gated delta rule.

    q,k: ``[B,T,H,K]``  v: ``[B,T,HV,V]``  g,beta: ``[B,T,HV]``
    initial_state: ``[N,HV,V,K]``. Returns ``(o [B,T,HV,V], final_state [N,HV,V,K])``.
    """
    B, T, H, Kdim = q.shape
    HV, V = v.shape[2], v.shape[3]
    if l2norm:
        q, k = ref_l2norm(q), ref_l2norm(k)
    gqa = HV // H
    q = q.to(torch.float32).repeat_interleave(gqa, 2)
    k = k.to(torch.float32).repeat_interleave(gqa, 2)
    v, g, beta = v.float(), g.float(), beta.float()

    if cu_seqlens is not None:
        seqs = [
            (0, int(cu_seqlens[n]), int(cu_seqlens[n + 1]), n)
            for n in range(len(cu_seqlens) - 1)
        ]
    else:
        seqs = [(b, 0, T, b) for b in range(B)]
    n_states = len(seqs)

    if initial_state is None:
        state = torch.zeros(n_states, HV, V, Kdim)
    else:
        state = initial_state.clone().float()
    o = torch.zeros(B, T, HV, V)
    for (bi, start, end, sidx) in seqs:
        S = state[sidx].clone()
        for t in range(start, end):
            S = S * torch.exp(g[bi, t]).view(HV, 1, 1)
            Sk = torch.einsum("hvd,hd->hv", S, k[bi, t])
            u = (v[bi, t] - Sk) * beta[bi, t].unsqueeze(-1)
            S = S + torch.einsum("hv,hd->hvd", u, k[bi, t])
            o[bi, t] = torch.einsum("hvd,hd->hv", S, q[bi, t] * scale)
        state[sidx] = S
    return o, state
