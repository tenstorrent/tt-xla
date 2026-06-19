# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Pure-PyTorch chunk-sequential gated delta rule (prefill path).

Mathematically equivalent to ``fla...ops.chunk.chunk_gated_delta_rule`` and to
``tt_fused_recurrent_gated_delta_rule``, but processes the sequence in fixed-size
chunks (``C = 64``, matching FLA) so the traced graph scales with ``T / C``
sequential steps instead of ``T``. Each chunk is solved with batched matmuls plus
one unit-lower-triangular solve per head.

Derivation (per value-head, state ``S`` is ``[V, K]``, ``S_0`` enters the chunk):

Token recurrence ``S_i = exp(g_i) S_{i-1} + u_i k_i^T`` with
``u_i = beta_i (v_i - exp(g_i) S_{i-1} k_i)`` unrolls, using the within-chunk
inclusive cumulative log-decay ``c_i = sum_{j<=i} g_j``, to a unit-lower-triangular
system in the ``u`` vectors::

    (I + A) U = B
    A[m,p] = beta_m * exp(c_m - c_p) * (k_p . k_m)      (strictly lower, p < m)
    B[m]   = beta_m * (v_m - exp(c_m) * (S_0 k_m))

Then, with scale applied to q at readout::

    o_i        = scale * exp(c_i) * (S_0 q_i)
                 + scale * sum_{m<=i} exp(c_i - c_m) (q_i . k_m) u_m
    S_chunk    = exp(c_C) * S_0 + sum_m exp(c_C - c_m) u_m k_m^T

All decay ratios use ``exp(c_a - c_b)`` with ``c_a <= c_b`` (``g <= 0``), so they
lie in ``(0, 1]`` and never overflow.
"""

import torch

from .l2norm import tt_l2norm_fwd

CHUNK_SIZE = 64


def _inv_unit_lower_tri(L: torch.Tensor) -> torch.Tensor:
    """Exact inverse of a unit-lower-triangular matrix, matmul-only.

    ``L`` is ``[..., C, C]`` with unit diagonal. Uses the block identity
    ``[[A,0],[M,D]]^{-1} = [[A^-1,0],[-D^-1 M A^-1, D^-1]]`` recursively, so it
    is pure cat/matmul (no ``torch.linalg.solve_triangular``, which is a CPU
    fallback that breaks the torch.compile(backend="tt") graph partitioner).
    ``C`` is static under ``dynamic=False``, so the recursion unrolls at trace
    time (depth ~log2(C)).
    """
    C = L.shape[-1]
    if C == 1:
        return L  # 1x1 unit block is [[1]]; its inverse is itself.
    h = C // 2
    A = L[..., :h, :h]
    D = L[..., h:, h:]
    M = L[..., h:, :h]
    Ainv = _inv_unit_lower_tri(A)
    Dinv = _inv_unit_lower_tri(D)
    Binv = -Dinv @ M @ Ainv
    zero = torch.zeros(
        L.shape[:-2] + (h, C - h), dtype=L.dtype, device=L.device
    )
    top = torch.cat([Ainv, zero], dim=-1)
    bot = torch.cat([Binv, Dinv], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _process_chunk(qc, kc, vc, gc, betac, S0, scale):
    """One chunk. Tensors are head-major: q/k ``[HV,Lc,K]``, v ``[HV,Lc,V]``,
    g/beta ``[HV,Lc]``, ``S0`` ``[HV,V,K]``. Returns ``(o [HV,Lc,V], S_new)``."""
    Lc = qc.shape[1]
    device = qc.device

    c = torch.cumsum(gc, dim=1)  # [HV, Lc] inclusive cumulative log-decay
    exp_c = torch.exp(c)  # a_i = exp(c_i) in (0, 1]

    # Pairwise decay exponents, masked so only the valid (non-positive) entries
    # survive; invalid entries -> -inf -> exp 0.
    diff = c.unsqueeze(2) - c.unsqueeze(1)  # [HV, Lc, Lc], [.,a,b] = c_a - c_b
    strict_lower = torch.tril(
        torch.ones(Lc, Lc, device=device, dtype=torch.bool), diagonal=-1
    )
    lower_incl = torch.tril(
        torch.ones(Lc, Lc, device=device, dtype=torch.bool), diagonal=0
    )

    KK = torch.bmm(kc, kc.transpose(1, 2))  # [HV,Lc,Lc], [.,m,p] = k_m . k_p
    QK = torch.bmm(qc, kc.transpose(1, 2))  # [HV,Lc,Lc], [.,i,m] = q_i . k_m
    KS0 = torch.bmm(kc, S0.transpose(1, 2))  # [HV,Lc,V], [.,m] = S_0 k_m
    QS0 = torch.bmm(qc, S0.transpose(1, 2))  # [HV,Lc,V], [.,i] = S_0 q_i

    # A (strictly lower): beta_m * exp(c_m - c_p) * (k_m . k_p)
    expo_A = diff.masked_fill(~strict_lower, float("-inf"))
    A = betac.unsqueeze(2) * torch.exp(expo_A) * KK  # [HV,Lc,Lc]
    eye = torch.eye(Lc, device=device, dtype=A.dtype).unsqueeze(0)
    LA = eye + A  # unit lower triangular

    # B = beta_m * (v_m - exp(c_m) * S_0 k_m)
    B = betac.unsqueeze(2) * (vc - exp_c.unsqueeze(2) * KS0)  # [HV,Lc,V]
    # Solve (I + A) U = B via an explicit matmul-only inverse (solve_triangular
    # is a CPU fallback that breaks the TT compile graph).
    U = _inv_unit_lower_tri(LA) @ B

    # Readout.
    expo_M = diff.masked_fill(~lower_incl, float("-inf"))
    M = torch.exp(expo_M) * QK  # [HV,Lc,Lc]
    intra = scale * torch.bmm(M, U)  # [HV,Lc,V]
    inter = scale * exp_c.unsqueeze(2) * QS0  # [HV,Lc,V]
    o = intra + inter

    # New chunk state.
    cC = c[:, -1:]  # [HV,1]
    scaled_U = torch.exp(cC - c).unsqueeze(2) * U  # [HV,Lc,V]
    deltaS = torch.bmm(scaled_U.transpose(1, 2), kc)  # [HV,V,K]
    S_new = torch.exp(cC).unsqueeze(2) * S0 + deltaS  # [HV,V,K]
    return o, S_new


def tt_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk-sequential gated delta rule.

    Args:
        q, k: ``[B, T, H, K]``.
        v: ``[B, T, HV, V]`` (``HV >= H`` for GQA).
        g: ``[B, T, HV]`` log-decay.
        beta: ``[B, T, HV]`` write strength (defaults to ones).
        scale: q scaling; defaults to ``K ** -0.5``.
        initial_state: ``[N, HV, V, K]`` entering state per sequence.
        output_final_state: also return the per-sequence final state.
        cu_seqlens: ``[N+1]`` token offsets (requires ``B == 1``).
        use_qk_l2norm_in_kernel: L2-normalize q and k first.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, HV, V]`` and
        ``final_state`` ``[N, HV, V, K]`` (``None`` if not requested).
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
    kf = k.to(torch.float32).repeat_interleave(gqa, dim=2)
    vf = v.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)

    if cu_seqlens is not None:
        # cu_seqlens.shape is static under dynamic=False, but its *values* are
        # not — reading them with int() makes the chunk loop trip count
        # data-dependent, which dynamo cannot unroll and which crashes the
        # torch.compile(backend="tt") graph partitioner. For a single packed
        # sequence (the common max_num_seqs=1 case) the bounds are simply
        # [0, T] from the static shape, so we avoid int() entirely. Multiple
        # packed sequences still read the values (eager-only path).
        n_seq = cu_seqlens.shape[0] - 1
        if n_seq == 1:
            bounds = [(0, 0, T)]
        else:
            bounds = [
                (0, int(cu_seqlens[n]), int(cu_seqlens[n + 1]))
                for n in range(n_seq)
            ]
    else:
        bounds = [(b, 0, T) for b in range(B)]
    n_states = len(bounds)

    out = torch.zeros((B, T, HV, V), dtype=torch.float32, device=q.device)
    final_state = torch.zeros(
        (n_states, HV, V, Kdim), dtype=torch.float32, device=q.device
    )

    for seq_idx, (bi, start, end) in enumerate(bounds):
        if initial_state is not None:
            S0 = initial_state[seq_idx].to(torch.float32)  # [HV,V,K]
        else:
            S0 = torch.zeros((HV, V, Kdim), dtype=torch.float32, device=q.device)

        for cs in range(start, end, CHUNK_SIZE):
            ce = min(cs + CHUNK_SIZE, end)
            qc = qf[bi, cs:ce].transpose(0, 1)  # [HV,Lc,K]
            kc = kf[bi, cs:ce].transpose(0, 1)
            vc = vf[bi, cs:ce].transpose(0, 1)  # [HV,Lc,V]
            gc = gf[bi, cs:ce].transpose(0, 1)  # [HV,Lc]
            betac = bf[bi, cs:ce].transpose(0, 1)
            o_chunk, S0 = _process_chunk(qc, kc, vc, gc, betac, S0, scale)
            out[bi, cs:ce] = o_chunk.transpose(0, 1)  # back to [Lc,HV,V]

        final_state[seq_idx] = S0

    return out.to(v.dtype), (final_state if output_final_state else None)
