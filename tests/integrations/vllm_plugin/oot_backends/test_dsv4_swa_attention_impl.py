# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isolated correctness tests for the DeepSeek-V4 sliding-window (SWA-only)
attention impl on the ``tt`` platform.

Each test targets a *distinct* attention mechanism so a failure localises the
broken component:

  * ``test_dsv4_swa_prefill``            SWA prefill (window + sink), CPU vs TT
  * ``test_dsv4_sliding_window_boundary`` window boundary is respected
                                          (precision-independent, structural)
  * ``test_dsv4_attention_sink_math``    sink fold matches analytic (fp32) +
                                          strong-sink limit drives output -> 0
  * ``test_dsv4_attention_sink_decode``  sink fold on decode, CPU vs TT (native)
  * ``test_dsv4_swa_decode_window``      windowed decode vs analytic (CPU ref;
                                          HW needs the tt-mlir sliding_window diff)
  * ``test_dsv4_prefill_then_decode``    prefill fills SWA cache, decode reads it

The reference is either an independent pure-torch analytic golden (the "math"
checks, run in fp32 for a tight tolerance) or the impl's own CPU device path
(the "lowering" checks, run in bf16 vs the TT device) — the pattern the existing
``test_mla_attention_impl.py`` uses. The upstream vLLM DSV4 layer is CUDA-only
(FlashMLA + fp8), so it cannot serve as an on-device golden here.

Why the isolated-impl form (vs a full-model forward): vLLM's
``DeepseekV4MultiHeadLatentAttentionWrapper`` asserts a CUDA device in
``__init__`` and cannot be constructed on ``tt``. The numerically-meaningful bar
for this milestone is single-layer attention correctness, exercised here.
"""
import math

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr

from tests.utils import parametrize_arch

REQUIRED_PCC = 0.99  # bf16 lowering tolerance (CPU-impl vs TT-device)
MATH_PCC = 0.999  # fp32 math tolerance (impl vs analytic golden)

# Small, self-consistent SWA dims. Real DSV4: head_dim=512, nope=448, rope=64,
# kv_lora_rank=512 (== head_dim), v_head_dim=512. The mechanisms (window, sink)
# are dimension-independent, so we use tile-aligned small dims for fast compile.
_CFG = dict(
    num_heads=8,
    qk_nope_head_dim=32,  # P
    qk_rope_head_dim=32,  # R
    kv_lora_rank=64,  # L
    v_head_dim=64,  # V (== L, mirroring DSV4 where v_head_dim == kv_lora_rank)
)
_BLOCK = 64  # DSV4 SWA cache block_size (sparse_swa.py:74)


def _dims():
    N = _CFG["num_heads"]
    P = _CFG["qk_nope_head_dim"]
    R = _CFG["qk_rope_head_dim"]
    L = _CFG["kv_lora_rank"]
    V = _CFG["v_head_dim"]
    Hd = L + R
    scale = Hd**-0.5
    return N, P, R, L, V, Hd, scale


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().float(), b.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


def _make_impl(sliding_window, dtype=torch.bfloat16):
    from vllm_tt.attention_impls.attention_dsv4 import TTDeepseekV4AttentionBackendImpl

    N, P, R, L, V, Hd, scale = _dims()
    return TTDeepseekV4AttentionBackendImpl(
        num_heads=N,
        head_size=Hd,
        scale=scale,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=L,
        qk_nope_head_dim=P,
        qk_rope_head_dim=R,
        qk_head_dim=P + R,
        v_head_dim=V,
        compress_ratio=1,
    )


def _weights(dtype):
    N, P, R, L, V, Hd, scale = _dims()
    torch.manual_seed(0)
    W_UK_T = (torch.randn(N, P, L, dtype=torch.float32) / math.sqrt(L)).to(dtype)
    W_UV = (torch.randn(N, L, V, dtype=torch.float32) / math.sqrt(L)).to(dtype)
    return W_UK_T, W_UV


def _layer(device, W_UK_T, W_UV, sink):
    from types import SimpleNamespace

    return SimpleNamespace(
        W_UK_T=W_UK_T.to(device),
        W_UV=W_UV.to(device),
        attn_sink=None if sink is None else sink.to(device),
    )


def _windowed_causal_keep(S, window, device):
    i = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    keep = j <= i
    if window:
        keep = keep & (j > i - window)
    return keep


# --------------------------------------------------------------------------- #
# Analytic goldens (pure torch; run in fp32 for the "math" checks)
# --------------------------------------------------------------------------- #
def _golden_prefill(q_nope, q_pe, kv_c, k_pe, W_UK_T, W_UV, sink, window):
    """Windowed-causal MLA attention with per-head sink fold, one user."""
    N, P, R, L, V, Hd, scale = _dims()
    S = q_nope.shape[0]
    q_lat = torch.cat(
        [torch.einsum("snp,npl->snl", q_nope.float(), W_UK_T.float()), q_pe.float()],
        dim=-1,
    )  # [S, N, Hd]
    k_lat = torch.cat([kv_c.float(), k_pe.squeeze(1).float()], dim=-1)  # [S, Hd]
    logits = torch.einsum("snh,th->nst", q_lat, k_lat) * scale  # [N, S_q, S_k]
    keep = _windowed_causal_keep(S, window, logits.device)
    logits = logits.masked_fill(~keep[None], float("-inf"))
    if sink is not None:
        sink_col = sink.float()[:, None, None].expand(N, S, 1)
        z = torch.cat([logits, sink_col], dim=-1).softmax(-1)[..., :S]
    else:
        z = logits.softmax(-1)
    ctx = torch.einsum("nst,tl->nsl", z, k_lat[:, :L])  # [N, S_q, L]
    out = torch.einsum("nsl,nlv->nsv", ctx, W_UV.float())  # [N, S_q, V]
    return out.transpose(0, 1).reshape(S, N * V)


def _golden_decode(gathered_k, q_nope, q_pe, W_UK_T, W_UV, sink, window, cur_pos):
    """One decode token per user against a gathered latent-K history.

    gathered_k: [users, cur_pos+1, Hd]; q_nope/q_pe: [users, N, P|R].
    """
    N, P, R, L, V, Hd, scale = _dims()
    users = q_nope.shape[0]
    q_lat = torch.cat(
        [torch.einsum("unp,npl->unl", q_nope.float(), W_UK_T.float()), q_pe.float()],
        dim=-1,
    )  # [users, N, Hd]
    out = torch.empty(users, N, V)
    for u in range(users):
        lo = max(cur_pos - window + 1, 0) if window else 0
        kw = gathered_k[u, lo : cur_pos + 1].float()  # [win, Hd]
        logits = torch.einsum("nh,jh->nj", q_lat[u], kw) * scale  # [N, win]
        if sink is not None:
            z = torch.cat([logits, sink.float()[:, None]], dim=-1).softmax(-1)[:, :-1]
        else:
            z = logits.softmax(-1)
        ctx = torch.einsum("nj,jl->nl", z, kw[:, :L])  # [N, L]
        out[u] = torch.einsum("nl,nlv->nv", ctx, W_UV.float())  # [N, V]
    return out.reshape(users, N * V)


# --------------------------------------------------------------------------- #
# Impl runners
# --------------------------------------------------------------------------- #
def _run_prefill(device, sliding_window, q_nope, q_pe, kv_c, k_pe, sink, dtype):
    from vllm_tt.attention_impls.attention import TTMetadata

    N, P, R, L, V, Hd, scale = _dims()
    S = q_nope.shape[0]
    nb = (S + _BLOCK - 1) // _BLOCK
    kv_cache = torch.zeros(nb, 1, _BLOCK, Hd, dtype=dtype, device=device)
    page_table = torch.arange(nb, dtype=torch.int32, device=device).view(1, nb)
    cache_position = torch.full((1,), S - 1, dtype=torch.int32, device=device)

    W_UK_T, W_UV = _weights(dtype)
    layer = _layer(device, W_UK_T, W_UV, sink)
    md = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )
    out = torch.empty(S, N * V, dtype=dtype, device=device)
    _make_impl(sliding_window, dtype).forward(
        q=(q_nope.to(device), q_pe.to(device)),
        kv_c_normed=kv_c.to(device),
        k_pe=k_pe.to(device),
        kv_cache=kv_cache,
        attn_metadata=md,
        layer=layer,
        output=out,
    )
    return out, kv_cache


def _run_decode(
    device, sliding_window, q_nope, q_pe, kv_c, k_pe, seeded_cache, cur_pos, sink, dtype
):
    from vllm_tt.attention_impls.attention import TTMetadata

    N, P, R, L, V, Hd, scale = _dims()
    users = q_nope.shape[0]
    bpu = cur_pos // _BLOCK + 1
    page_table = torch.arange(users * bpu, dtype=torch.int32, device=device).view(
        users, bpu
    )
    cache_position = torch.full((users,), cur_pos, dtype=torch.int32, device=device)

    W_UK_T, W_UV = _weights(dtype)
    layer = _layer(device, W_UK_T, W_UV, sink)
    md = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )
    kv_cache = seeded_cache.clone().to(device)
    out = torch.empty(users, N * V, dtype=dtype, device=device)
    _make_impl(sliding_window, dtype).forward(
        q=(q_nope.to(device), q_pe.to(device)),
        kv_c_normed=kv_c.to(device),
        k_pe=k_pe.to(device),
        kv_cache=kv_cache,
        attn_metadata=md,
        layer=layer,
        output=out,
    )
    return out, kv_cache


def _gather(cache, page_table, npos):
    """Read logical positions [0, npos) out of the paged latent cache per user."""
    users = page_table.shape[0]
    Hd = cache.shape[-1]
    out = torch.empty(users, npos, Hd, dtype=cache.dtype)
    for u in range(users):
        for p in range(npos):
            blk = int(page_table[u, p // _BLOCK])
            out[u, p] = cache[blk, 0, p % _BLOCK, :]
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.push
@parametrize_arch(["single_device"])
@pytest.mark.parametrize("seq_len, window", [(64, 16), (64, 32), (96, 24)])
def test_dsv4_swa_prefill(seq_len, window, arch):
    """SWA prefill (window + sink): TT device matches the CPU reference path."""
    xr.set_device_type("TT")
    torch.manual_seed(0)
    N, P, R, L, V, Hd, scale = _dims()
    dtype = torch.bfloat16
    q_nope = torch.randn(seq_len, N, P, dtype=dtype)
    q_pe = torch.randn(seq_len, N, R, dtype=dtype)
    kv_c = torch.randn(seq_len, L, dtype=dtype)
    k_pe = torch.randn(seq_len, 1, R, dtype=dtype)
    sink = torch.linspace(-1.0, 2.0, N)

    golden, _ = _run_prefill(
        torch.device("cpu"), window, q_nope, q_pe, kv_c, k_pe, sink, dtype
    )
    dev_out, _ = _run_prefill(
        torch_xla.device(), window, q_nope, q_pe, kv_c, k_pe, sink, dtype
    )
    torch_xla.sync()
    dev_out = dev_out.cpu()

    assert dev_out.shape == golden.shape == (seq_len, N * V)
    pcc = _pcc(dev_out, golden)
    assert pcc >= REQUIRED_PCC, f"SWA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_attention_sink_math(arch):
    """The sink fold matches the analytic golden in fp32 (tight tolerance), and
    a strong sink drives the output toward zero (correct fold direction)."""
    xr.set_device_type("TT")
    torch.manual_seed(1)
    N, P, R, L, V, Hd, scale = _dims()
    seq_len, window = 64, 16
    q_nope = torch.randn(seq_len, N, P)
    q_pe = torch.randn(seq_len, N, R)
    kv_c = torch.randn(seq_len, L)
    k_pe = torch.randn(seq_len, 1, R)
    W_UK_T, W_UV = _weights(torch.float32)

    for sink in (None, torch.linspace(-1.0, 2.0, N)):
        golden = _golden_prefill(q_nope, q_pe, kv_c, k_pe, W_UK_T, W_UV, sink, window)
        cpu_out, _ = _run_prefill(
            torch.device("cpu"), window, q_nope, q_pe, kv_c, k_pe, sink, torch.float32
        )
        pcc = _pcc(cpu_out, golden)
        tag = "no-sink" if sink is None else "sink"
        assert pcc >= MATH_PCC, f"prefill[{tag}] math PCC {pcc:.6f} < {MATH_PCC}"

    # Strong per-head sink (>> logits) -> exp(sink) dominates -> output -> 0.
    strong = torch.full((N,), 30.0)
    out_strong, _ = _run_prefill(
        torch.device("cpu"), window, q_nope, q_pe, kv_c, k_pe, strong, torch.float32
    )
    assert out_strong.abs().max() < 1e-2, (
        f"strong sink should zero the output, got max |out| "
        f"{float(out_strong.abs().max()):.4g}"
    )


@pytest.mark.push
@parametrize_arch(["single_device"])
@pytest.mark.parametrize("device_kind", ["cpu", "tt"])
def test_dsv4_sliding_window_boundary(device_kind, arch):
    """Structural, precision-independent: a query attends ONLY to the last
    ``window`` keys. Perturbing a key strictly before a query's window must not
    change that query's output; perturbing one inside it must."""
    xr.set_device_type("TT")
    torch.manual_seed(2)
    N, P, R, L, V, Hd, scale = _dims()
    seq_len, window = 64, 16
    dtype = torch.bfloat16
    device = torch.device("cpu") if device_kind == "cpu" else torch_xla.device()

    q_nope = torch.randn(seq_len, N, P, dtype=dtype)
    q_pe = torch.randn(seq_len, N, R, dtype=dtype)
    kv_c = torch.randn(seq_len, L, dtype=dtype)
    k_pe = torch.randn(seq_len, 1, R, dtype=dtype)
    sink = None

    i0 = seq_len - 1  # last query; window = [i0-window+1, i0]
    outside = i0 - window - 2  # strictly before the window (>= 0)
    inside = i0 - 1  # inside the window
    assert 0 <= outside < i0 - window + 1

    def run(kv_c_, k_pe_):
        out, _ = _run_prefill(device, window, q_nope, q_pe, kv_c_, k_pe_, sink, dtype)
        torch_xla.sync()
        return out.cpu()

    base = run(kv_c, k_pe)

    # Perturb a key OUTSIDE query i0's window.
    kv_c_o = kv_c.clone()
    k_pe_o = k_pe.clone()
    kv_c_o[outside] = torch.randn(L, dtype=dtype)
    k_pe_o[outside] = torch.randn(1, R, dtype=dtype)
    out_o = run(kv_c_o, k_pe_o)

    base_i0 = base[i0]
    assert _pcc(out_o[i0], base_i0) >= 0.999, (
        "query outside-window perturbation changed the output -> window not "
        "respected (attends beyond the last `window` keys)"
    )

    # Perturb a key INSIDE query i0's window -> output MUST change.
    kv_c_i = kv_c.clone()
    k_pe_i = k_pe.clone()
    kv_c_i[inside] = torch.randn(L, dtype=dtype)
    k_pe_i[inside] = torch.randn(1, R, dtype=dtype)
    out_i = run(kv_c_i, k_pe_i)
    assert _pcc(out_i[i0], base_i0) < 0.999, (
        "in-window perturbation did not change the output -> window may be "
        "masking too aggressively"
    )


@pytest.mark.push
@parametrize_arch(["single_device"])
@pytest.mark.parametrize("users, cur_pos", [(1, 31), (2, 48)])
def test_dsv4_attention_sink_decode(users, cur_pos, arch):
    """Attention-sink fold on the paged MLA decode path (native ttnn sink):
    TT device matches the CPU reference. Runs full causal history (no window) so
    it exercises the sink on hardware today (the decode window needs the tt-mlir
    sliding_window plumbing; see test_dsv4_swa_decode_window)."""
    xr.set_device_type("TT")
    torch.manual_seed(3)
    N, P, R, L, V, Hd, scale = _dims()
    dtype = torch.bfloat16
    bpu = cur_pos // _BLOCK + 1
    seeded = torch.randn(users * bpu, 1, _BLOCK, Hd, dtype=dtype)
    q_nope = torch.randn(users, N, P, dtype=dtype)
    q_pe = torch.randn(users, N, R, dtype=dtype)
    kv_c = torch.randn(users, L, dtype=dtype)
    k_pe = torch.randn(users, 1, R, dtype=dtype)
    sink = torch.linspace(-1.0, 3.0, N)

    golden, _ = _run_decode(
        torch.device("cpu"),
        None,
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        seeded,
        cur_pos,
        sink,
        dtype,
    )
    dev_out, _ = _run_decode(
        torch_xla.device(), None, q_nope, q_pe, kv_c, k_pe, seeded, cur_pos, sink, dtype
    )
    torch_xla.sync()
    dev_out = dev_out.cpu()

    assert dev_out.shape == golden.shape == (users, N * V)
    pcc = _pcc(dev_out, golden)
    assert pcc >= REQUIRED_PCC, f"decode sink PCC {pcc:.5f} < {REQUIRED_PCC}"


@pytest.mark.push
@parametrize_arch(["single_device"])
# cur_pos=80 -> 2 cache blocks (block_size=64), a shape no *no-window* decode
# test uses. The tt plugin's compiled-program cache does not key on the
# sliding_window_size frontend attribute, so a same-shape no-window decode
# compiled earlier in the session would otherwise be reused here (window
# dropped). Real models are unaffected — all SWA layers share one window — but
# this test must not collide with test_dsv4_attention_sink_decode (window=None).
@pytest.mark.parametrize("users, cur_pos, window", [(1, 80, 16), (2, 80, 32)])
def test_dsv4_swa_decode_window(users, cur_pos, window, arch):
    """Windowed paged decode: matches the analytic windowed golden (fp32 CPU
    reference) AND applies the window on the tt device (bf16 tt-vs-cpu). The
    decode sliding_window is now plumbed through tt-mlir to the ttnn kernel
    (see tt_mlir_changes.md / DSV4_TT_Next_Steps.md §2); the window is active
    here (cur_pos + 1 > window)."""
    xr.set_device_type("TT")
    torch.manual_seed(4)
    N, P, R, L, V, Hd, scale = _dims()
    bpu = cur_pos // _BLOCK + 1

    # -- math (fp32, cpu): windowed decode vs analytic windowed golden --
    dtype = torch.float32
    seeded = torch.randn(users * bpu, 1, _BLOCK, Hd, dtype=dtype)
    q_nope = torch.randn(users, N, P, dtype=dtype)
    q_pe = torch.randn(users, N, R, dtype=dtype)
    kv_c = torch.randn(users, L, dtype=dtype)
    k_pe = torch.randn(users, 1, R, dtype=dtype)
    sink = torch.linspace(-1.0, 2.0, N)

    cpu_out, cache = _run_decode(
        torch.device("cpu"),
        window,
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        seeded,
        cur_pos,
        sink,
        dtype,
    )
    page_table = torch.arange(users * bpu, dtype=torch.int32).view(users, bpu)
    gathered = _gather(cache.cpu(), page_table, cur_pos + 1)
    W_UK_T, W_UV = _weights(dtype)
    golden = _golden_decode(gathered, q_nope, q_pe, W_UK_T, W_UV, sink, window, cur_pos)
    pcc = _pcc(cpu_out, golden)
    assert pcc >= MATH_PCC, f"windowed decode math PCC {pcc:.6f} < {MATH_PCC}"

    # -- lowering (bf16): the window is applied on the tt device (tt == cpu) --
    dtb = torch.bfloat16
    seeded_b = seeded.to(dtb)
    qn, qp, kc, kp = (t.to(dtb) for t in (q_nope, q_pe, kv_c, k_pe))
    cpu_b, _ = _run_decode(
        torch.device("cpu"), window, qn, qp, kc, kp, seeded_b, cur_pos, sink, dtb
    )
    tt_b, _ = _run_decode(
        torch_xla.device(), window, qn, qp, kc, kp, seeded_b, cur_pos, sink, dtb
    )
    torch_xla.sync()
    pcc_low = _pcc(tt_b.cpu(), cpu_b)
    assert pcc_low >= REQUIRED_PCC, (
        f"windowed decode tt-vs-cpu PCC {pcc_low:.5f} < {REQUIRED_PCC} — the "
        "sliding_window is not being applied on the tt device (check the "
        "tt-mlir sliding_window_size plumbing)"
    )


@pytest.mark.push
@parametrize_arch(["single_device"])
@pytest.mark.parametrize("seq_len, window", [(64, 16)])
def test_dsv4_prefill_then_decode(seq_len, window, arch):
    """Prefill fills the SWA latent cache; a following decode reads it. Verifies
    the attention output (CPU vs TT) and that the decode token is written at
    ``seq_len`` without clobbering the prefill context."""
    xr.set_device_type("TT")
    torch.manual_seed(5)
    N, P, R, L, V, Hd, scale = _dims()
    dtype = torch.bfloat16

    # Prefill.
    q_nope = torch.randn(seq_len, N, P, dtype=dtype)
    q_pe = torch.randn(seq_len, N, R, dtype=dtype)
    kv_c = torch.randn(seq_len, L, dtype=dtype)
    k_pe = torch.randn(seq_len, 1, R, dtype=dtype)
    sink = torch.linspace(-1.0, 2.0, N)

    def latent_k(kv_c_, k_pe_):
        return torch.cat([kv_c_, k_pe_.squeeze(1)], dim=-1)

    # We need an extra block for the decode token at position seq_len.
    from vllm_tt.attention_impls.attention import TTMetadata

    def run_pf(device):
        nb = seq_len // _BLOCK + 1
        kv_cache = torch.zeros(nb, 1, _BLOCK, Hd, dtype=dtype, device=device)
        page_table = torch.arange(nb, dtype=torch.int32, device=device).view(1, nb)
        cp = torch.full((1,), seq_len - 1, dtype=torch.int32, device=device)
        W_UK_T, W_UV = _weights(dtype)
        layer = _layer(device, W_UK_T, W_UV, sink)
        md = TTMetadata(
            cache_position=cp,
            attn_mask=None,
            page_table=page_table,
            is_causal=True,
            fill_page_table=page_table,
        )
        out = torch.empty(seq_len, N * V, dtype=dtype, device=device)
        _make_impl(window, dtype).forward(
            q=(q_nope.to(device), q_pe.to(device)),
            kv_c_normed=kv_c.to(device),
            k_pe=k_pe.to(device),
            kv_cache=kv_cache,
            attn_metadata=md,
            layer=layer,
            output=out,
        )
        return out, kv_cache, page_table

    cpu_pf, cpu_cache, page_table = run_pf(torch.device("cpu"))
    dev_pf, dev_cache, _ = run_pf(torch_xla.device())
    torch_xla.sync()
    dev_pf = dev_pf.cpu()

    assert _pcc(dev_pf, cpu_pf) >= REQUIRED_PCC, "prefill output CPU/TT mismatch"

    # The prefill tokens are in the cache; the decode block is still zero.
    filled = _gather(cpu_cache.cpu(), page_table.cpu(), seq_len)
    assert torch.allclose(filled[0], latent_k(kv_c, k_pe)), "prefill cache filled wrong"

    # Decode one token at position seq_len against the prefilled cache.
    q_nope_d = torch.randn(1, N, P, dtype=dtype)
    q_pe_d = torch.randn(1, N, R, dtype=dtype)
    kv_c_d = torch.randn(1, L, dtype=dtype)
    k_pe_d = torch.randn(1, 1, R, dtype=dtype)

    def run_dec(device, cache_after_prefill):
        page_table_l = torch.arange(
            cache_after_prefill.shape[0], dtype=torch.int32, device=device
        ).view(1, -1)
        cp = torch.full((1,), seq_len, dtype=torch.int32, device=device)
        W_UK_T, W_UV = _weights(dtype)
        layer = _layer(device, W_UK_T, W_UV, sink)
        md = TTMetadata(
            cache_position=cp,
            attn_mask=None,
            page_table=page_table_l,
            is_causal=True,
            fill_page_table=page_table_l,
        )
        out = torch.empty(1, N * V, dtype=dtype, device=device)
        kv_cache = cache_after_prefill.clone().to(device)
        # Decode uses full causal history (window on decode needs the tt-mlir
        # diff); this test checks cache plumbing + decode output parity.
        _make_impl(None, dtype).forward(
            q=(q_nope_d.to(device), q_pe_d.to(device)),
            kv_c_normed=kv_c_d.to(device),
            k_pe=k_pe_d.to(device),
            kv_cache=kv_cache,
            attn_metadata=md,
            layer=layer,
            output=out,
        )
        return out, kv_cache

    cpu_dec, cpu_dec_cache = run_dec(torch.device("cpu"), cpu_cache.cpu())
    dev_dec, dev_dec_cache = run_dec(torch_xla.device(), dev_cache.cpu())
    torch_xla.sync()
    dev_dec = dev_dec.cpu()

    assert _pcc(dev_dec, cpu_dec) >= REQUIRED_PCC, "decode output CPU/TT mismatch"

    # The decode token landed at position seq_len; prefix untouched.
    page_table_full = torch.arange(cpu_dec_cache.shape[0], dtype=torch.int32).view(
        1, -1
    )
    updated = _gather(cpu_dec_cache.cpu(), page_table_full, seq_len + 1)
    assert torch.allclose(
        updated[0, seq_len], latent_k(kv_c_d, k_pe_d)[0]
    ), "decode token not written at seq_len"
    assert torch.allclose(
        updated[0, :seq_len], latent_k(kv_c, k_pe)
    ), "decode clobbered prefill context"
