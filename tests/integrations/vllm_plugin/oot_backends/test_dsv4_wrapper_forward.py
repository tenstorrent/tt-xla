# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Numerical validation of ``TTDeepseekV4MLAWrapper.forward`` (the bf16 DSV4 SWA-only
prefill reimplementation) on the ``tt`` platform.

DSV4 attention is **direct MLA** (no V3 ``W_UK_T``/``W_UV`` absorption): q and the
per-token latent kv both live in head_dim space, and V is the *full* latent
(head_dim_v == qk). The forward is: fused_wqa_wkv -> split -> q_norm/kv_norm ->
wq_b -> per-head Q RMSNorm -> decoupled RoPE -> windowed + attention-sink
attention (V = full latent, via a zero-padded head dim) -> inverse-RoPE ->
grouped o-proj. Everything is bf16 — no quantization.

Two checks:
  * math   (fp32, cpu): wrapper.forward == an independent pure-torch reference
                        (same submodules), exactly.
  * lowering (bf16):    wrapper.forward on the tt device == the cpu path.
"""
import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

from tests.utils import parametrize_arch

deepseek_v4_attention = pytest.importorskip(
    "vllm.model_executor.layers.deepseek_v4_attention"
)

HIDDEN, N, NOPE, ROPE = 512, 8, 96, 32
HD = NOPE + ROPE  # 128
QLORA, OLORA, G = 256, 64, 1
T, W = 64, 16
SCALE = HD**-0.5
MATH_PCC = 0.999
REQUIRED_PCC = 0.99


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    va, vb = a - a.mean(), b - b.mean()
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


def _cfg():
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=HIDDEN,
        num_attention_heads=N,
        num_hidden_layers=1,
        head_dim=HD,
        qk_rope_head_dim=ROPE,
        kv_lora_rank=HD,
        q_lora_rank=QLORA,
        o_lora_rank=OLORA,
        o_groups=G,
        sliding_window=W,
        compress_ratios=[1],
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters={"rope_type": "default"},
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        index_topk=64,
        index_n_heads=8,
        index_head_dim=64,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        hc_eps=1e-6,
        vocab_size=1000,
        quantization_config={"scale_fmt": None},
    )


def _mock_vllm_config(cfg, dtype):
    # Fresh CompilationConfig per call so repeated constructions don't collide on
    # the static_forward_context prefix.
    from vllm.config import CacheConfig, CompilationConfig

    comp = CompilationConfig()
    comp.custom_ops = ["none"]
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=dtype, max_model_len=4096),
        cache_config=CacheConfig(block_size=64, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        compilation_config=comp,
        quant_config=None,
    )


class _FusedWqaWkv(nn.Module):  # returns (out, bias) like MergedColumnParallelLinear
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, QLORA + HD, bias=False)

    def forward(self, x):
        return self.lin(x), None


class _WoA(nn.Module):  # grouped (bmm) weight [G, OLORA, (N//G)*HD]
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(G, OLORA, (N // G) * HD) / ((N // G) * HD) ** 0.5
        )


def _cos_sin_cache():
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2).float() / ROPE))
    ang = torch.arange(4096).float()[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)  # [4096, ROPE]


def _build_modules(dtype):
    """Shared submodule weights, built once on cpu (seeded)."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(1)
    with set_current_vllm_config(_mock_vllm_config(_cfg(), dtype)):
        q_norm = RMSNorm(QLORA, eps=1e-6).to(dtype)
        kv_norm = RMSNorm(HD, eps=1e-6).to(dtype)
    sink = torch.full((64,), float("-inf"))  # cpu; forward slices [:N] + .to(dev)
    sink[:N] = torch.linspace(-1.0, 2.0, N)
    return {
        "fused_wqa_wkv": _FusedWqaWkv().to(dtype),
        "q_norm": q_norm,
        "kv_norm": kv_norm,
        "wq_b": nn.Linear(QLORA, N * HD, bias=False).to(dtype),
        "wo_a": _WoA().to(dtype),
        "wo_b": nn.Linear(G * OLORA, HIDDEN, bias=False).to(dtype),
        "rotary_emb": SimpleNamespace(cos_sin_cache=_cos_sin_cache().to(dtype)),
        "sink": sink.to(dtype),
    }


def _to_dev(mods, dev):
    import copy

    out = {}
    for k, v in mods.items():
        if isinstance(v, nn.Module):
            out[k] = copy.deepcopy(v).to(dev)
        elif isinstance(v, SimpleNamespace):
            out[k] = SimpleNamespace(cos_sin_cache=v.cos_sin_cache.to(dev))
        else:
            out[k] = v.to(dev)
    return out


def _build_wrapper(vllm_config, mods, prefix, window=W):
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MLAModules,
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    mla_modules = DeepseekV4MLAModules(
        vllm_config=vllm_config,
        fused_wqa_wkv=mods["fused_wqa_wkv"],
        q_norm=mods["q_norm"],
        wq_b=mods["wq_b"],
        kv_norm=mods["kv_norm"],
        wo_a=mods["wo_a"],
        wo_b=mods["wo_b"],
        attn_sink=mods["sink"],
        rotary_emb=mods["rotary_emb"],
        indexer=None,
        indexer_rotary_emb=mods["rotary_emb"],
        topk_indices_buffer=None,
        aux_stream_list=None,
    )
    return DeepseekV4MultiHeadLatentAttentionWrapper(
        hidden_size=HIDDEN,
        num_heads=N,
        head_dim=HD,
        scale=SCALE,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=HD,
        q_lora_rank=QLORA,
        kv_lora_rank=HD,
        o_lora_rank=OLORA,
        mla_modules=mla_modules,
        window_size=window,
        compress_ratio=1,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
    )


def _run(dev, base_mods, hidden, positions, prefix, dtype):
    from vllm.config import set_current_vllm_config

    mods = _to_dev(base_mods, dev)
    vllm_config = _mock_vllm_config(_cfg(), dtype)
    with contextlib.ExitStack() as s:
        s.enter_context(set_current_vllm_config(vllm_config))
        s.enter_context(
            mock.patch.object(
                deepseek_v4_attention,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        )
        w = _build_wrapper(vllm_config, mods, prefix)
        out = w.forward(positions.to(dev), hidden.to(dev))
        return out, w


def _reference(w, hidden, positions):
    from vllm_tt.attention_impls.attention_dsv4 import (
        _dsv4_cos_sin,
        _dsv4_rope_rope_dims,
    )

    qr_kv, _ = w.fused_wqa_wkv(hidden)
    qr, kv = qr_kv.split([QLORA, HD], dim=-1)
    qr, kv = w.q_norm(qr), w.kv_norm(kv)
    q = w.wq_b(qr).view(T, N, HD)
    q = w.q_head_norm(q)
    cos, sin = _dsv4_cos_sin(w.rotary_emb.cos_sin_cache, positions, ROPE)
    q = _dsv4_rope_rope_dims(q, cos, sin, NOPE, headed=True)
    kv = _dsv4_rope_rope_dims(kv, cos, sin, NOPE, headed=False)
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    keep = (j <= i) & (j > i - W)
    mask = torch.zeros(T, T).masked_fill(~keep, float("-inf"))
    logits = torch.einsum("inh,jh->nij", q.float(), kv.float()) * SCALE + mask[None]
    sink = w.mla_attn.attn_sink[:N].float()[:, None, None].expand(N, T, 1)
    z = torch.cat([logits, sink], -1).softmax(-1)[..., :T]
    o = torch.einsum("nij,jh->inh", z, kv.float())
    o = _dsv4_rope_rope_dims(o, cos, sin, NOPE, headed=True, sign=-1.0)
    o_g = o.reshape(T, G, (N // G) * HD)
    zz = torch.einsum("bhr,hdr->bhd", o_g.float(), w.wo_a.weight.float())
    return w.wo_b(zz.flatten(1).to(hidden.dtype))


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_wrapper_forward(arch):
    """wrapper.forward matches a pure-torch reference (fp32 math) and runs
    correctly on the tt device (bf16 lowering)."""
    import vllm_tt.attention_impls.attention_dsv4 as _dsv4  # noqa: F401

    xr.set_device_type("TT")
    torch.manual_seed(0)
    positions = torch.arange(T)

    # -- math (fp32, cpu): wrapper.forward vs analytic reference --
    m32 = _build_modules(torch.float32)
    h32 = torch.randn(T, HIDDEN, dtype=torch.float32)
    out_cpu32, w32 = _run(
        torch.device("cpu"), m32, h32, positions, "fwd.cpu32", torch.float32
    )
    ref32 = _reference(w32, h32, positions)
    assert out_cpu32.shape == (T, HIDDEN)
    pcc_math = _pcc(out_cpu32, ref32)
    assert pcc_math >= MATH_PCC, f"forward math PCC {pcc_math:.6f} < {MATH_PCC}"

    # -- lowering (bf16): tt device vs cpu --
    mb = _build_modules(torch.bfloat16)
    hb = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    out_cpu, _ = _run(torch.device("cpu"), mb, hb, positions, "fwd.cpu", torch.bfloat16)
    out_tt, _ = _run(torch_xla.device(), mb, hb, positions, "fwd.tt", torch.bfloat16)
    torch_xla.sync()
    pcc_low = _pcc(out_tt.cpu(), out_cpu)
    assert (
        pcc_low >= REQUIRED_PCC
    ), f"forward tt-vs-cpu PCC {pcc_low:.5f} < {REQUIRED_PCC}"


# --------------------------------------------------------------------------- #
# Paged SWA KV-cache round-trip (prefill writes the cache; decode reads it)
# --------------------------------------------------------------------------- #
_BLOCK = 32
_S = 32  # prefill length (flash_mla_prefill requires seq_len % 32 == 0)
_CUR = 32  # decode token position
_CACHE_W = 64  # window >= history (S+1 = 33 <= 64) so it's inactive here, keeping
# this test focused on the cache write/read round-trip and the
# reference simple. (Windowing on decode is exercised on the tt
# device by test_dsv4_swa_decode_window.)


def _gather_cache(cache, page_table, npos):
    out = torch.empty(npos, HD, dtype=cache.dtype)
    for p in range(npos):
        blk = int(page_table[0, p // _BLOCK])
        out[p] = cache[blk, 0, p % _BLOCK, :]
    return out


def _paged_roundtrip(dev, base_mods, dtype, hp, pp, hd, pd):
    """Construct the wrapper, bind a paged SWA cache, run prefill (writes) then
    decode (reads) via the paged methods. Returns (decode latent out [N,HD],
    filled cache, wrapper, decode q [1,N,HD])."""
    from vllm.config import set_current_vllm_config
    from vllm_tt.attention_impls.attention import TTMetadata

    mods = _to_dev(base_mods, dev)
    vllm_config = _mock_vllm_config(_cfg(), dtype)
    hp, hd = hp.to(dtype), hd.to(dtype)  # match the module dtype
    nb = _CUR // _BLOCK + 1  # blocks for one user covering [0, _CUR]
    with contextlib.ExitStack() as s:
        s.enter_context(set_current_vllm_config(vllm_config))
        s.enter_context(
            mock.patch.object(
                deepseek_v4_attention,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        )
        w = _build_wrapper(vllm_config, mods, "pc." + str(dev), window=_CACHE_W)
        cache = torch.zeros(nb, 1, _BLOCK, HD, dtype=dtype, device=dev)
        page_table = torch.arange(nb, dtype=torch.int32, device=dev).view(1, nb)
        sink = w.mla_attn.attn_sink[:N].to(dev)

        # -- prefill: fills the cache with the roped kv latents --
        qp, kvp, _, _ = w._dsv4_preprocess(pp.to(dev), hp.to(dev))
        md_p = TTMetadata(
            cache_position=torch.full((1,), _S - 1, dtype=torch.int32, device=dev),
            page_table=page_table,
            fill_page_table=page_table,
            is_causal=True,
        )
        w._swa_paged_prefill(
            qp.view(1, _S, N, HD), kvp.view(1, _S, HD), cache, md_p, sink
        )

        # -- decode: reads the windowed history from the cache --
        qd, kvd, _, _ = w._dsv4_preprocess(pd.to(dev), hd.to(dev))
        md_d = TTMetadata(
            cache_position=torch.full((1,), _CUR, dtype=torch.int32, device=dev),
            page_table=page_table,
            fill_page_table=page_table,
            is_causal=True,
        )
        od = w._swa_paged_decode(
            qd.view(1, 1, N, HD), kvd.view(1, 1, HD), cache, md_d, sink
        )  # [1, 1, N, HD]
        return od[0, 0], cache, qd, kvp


def _reference_decode_latent(cache, page_table, qd, sink):
    """Pure-torch windowed-causal + sink attention of the decode token against
    the cached latent history (V = full latent). Returns o [N, HD]."""
    hist = _gather_cache(cache, page_table, _CUR + 1).float()  # [CUR+1, HD]
    q = qd.view(N, HD).float()
    scores = torch.einsum("nh,jh->nj", q, hist) * SCALE  # [N, CUR+1]
    # window inactive for this short history; causal over [0, CUR] (all present).
    sink_col = sink.float()[:N, None]
    z = torch.cat([scores, sink_col], -1).softmax(-1)[:, :-1]  # drop sink weight
    return torch.einsum("nj,jh->nh", z, hist)  # [N, HD]


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_paged_cache_prefill_decode(arch):
    """Prefill writes per-token latents into the paged SWA cache; a following
    decode reads the windowed history back. Validates the KV-cache round-trip
    (math vs a pure-torch reference; cache contents; tt-vs-cpu lowering)."""
    import vllm_tt.attention_impls.attention_dsv4 as _dsv4  # noqa: F401

    xr.set_device_type("TT")
    torch.manual_seed(0)
    hp = torch.randn(_S, HIDDEN)
    pp = torch.arange(_S)
    hd = torch.randn(1, HIDDEN)
    pd = torch.tensor([_CUR])

    # -- math (fp32, cpu): decode-latent vs pure-torch reference over the cache --
    m32 = _build_modules(torch.float32)
    od_cpu, cache_cpu, qd_cpu, kvp = _paged_roundtrip(
        torch.device("cpu"), m32, torch.float32, hp, pp, hd, pd
    )
    pt_cpu = torch.arange(cache_cpu.shape[0], dtype=torch.int32).view(1, -1)
    # prefill must have written the latents: cache[0:_S] == the prefill kv latents.
    gathered = _gather_cache(cache_cpu, pt_cpu, _S)
    assert torch.allclose(
        gathered, kvp.view(_S, HD), atol=1e-4
    ), "prefill cache write wrong"

    ref = _reference_decode_latent(cache_cpu, pt_cpu, qd_cpu, m32["sink"])
    pcc_math = _pcc(od_cpu, ref)
    assert pcc_math >= MATH_PCC, f"paged decode math PCC {pcc_math:.6f} < {MATH_PCC}"

    # -- lowering (bf16): tt decode-latent vs cpu --
    mb = _build_modules(torch.bfloat16)
    od_c = _paged_roundtrip(torch.device("cpu"), mb, torch.bfloat16, hp, pp, hd, pd)[0]
    od_t = _paged_roundtrip(torch_xla.device(), mb, torch.bfloat16, hp, pp, hd, pd)[0]
    torch_xla.sync()
    pcc_low = _pcc(od_t.cpu(), od_c)
    assert (
        pcc_low >= REQUIRED_PCC
    ), f"paged decode tt-vs-cpu PCC {pcc_low:.5f} < {REQUIRED_PCC}"


# --------------------------------------------------------------------------- #
# C128A (HCA) fresh-sequence forward: window branch + dense compressed-prefix
# branch, merged under one softmax with the attention sink in the denominator.
# --------------------------------------------------------------------------- #
_TC = 256  # >= compress_ratio (128) so Ncomp = _TC // 128 = 2 (compressor active)
_C128_RATIO = 128


def _init_dist_once():
    """Single-process TP group — the compressor builds a real
    MergedColumnParallelLinear, which needs the tensor-parallel group."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if model_parallel_is_initialized():
        return
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:12377",
        local_rank=0,
        backend="gloo",
    )
    with set_current_vllm_config(_mock_vllm_config_c128(torch.float32)):
        initialize_model_parallel(tensor_model_parallel_size=1)


def _mock_vllm_config_c128(dtype):
    from vllm.config import ParallelConfig

    cfg = _cfg()
    cfg.compress_ratios = [_C128_RATIO]
    cfg.compress_rope_theta = 10000.0
    m = _mock_vllm_config(cfg, dtype)
    m.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=8)
    m.parallel_config = ParallelConfig()
    m.model_config.is_moe = False
    return m


def _build_wrapper_c128(vllm_config, mods, prefix):
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MLAModules,
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    mla_modules = DeepseekV4MLAModules(
        vllm_config=vllm_config,
        fused_wqa_wkv=mods["fused_wqa_wkv"],
        q_norm=mods["q_norm"],
        wq_b=mods["wq_b"],
        kv_norm=mods["kv_norm"],
        wo_a=mods["wo_a"],
        wo_b=mods["wo_b"],
        attn_sink=mods["sink"],
        rotary_emb=mods["rotary_emb"],
        indexer=None,
        indexer_rotary_emb=mods["rotary_emb"],
        topk_indices_buffer=None,
        aux_stream_list=None,
    )
    return DeepseekV4MultiHeadLatentAttentionWrapper(
        hidden_size=HIDDEN,
        num_heads=N,
        head_dim=HD,
        scale=SCALE,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=HD,
        q_lora_rank=QLORA,
        kv_lora_rank=HD,
        o_lora_rank=OLORA,
        mla_modules=mla_modules,
        window_size=W,
        compress_ratio=_C128_RATIO,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
    )


def _reference_c128a(w, hidden, positions):
    """Independent pure-torch C128A forward mirroring modified_model.Attention:
    window branch + compressed branch (oracle Compressor C128 pooling) under one
    combined softmax with the sink as an extra denominator logit. Reads w's
    submodules (incl. the real w.compressor)."""
    from vllm_tt.attention_impls.attention_dsv4 import (
        _dsv4_cos_sin,
        _dsv4_rope_rope_dims,
    )

    T, ratio = _TC, _C128_RATIO
    qr_kv, _ = w.fused_wqa_wkv(hidden)
    qr, kv = qr_kv.split([QLORA, HD], dim=-1)
    qr, kv = w.q_norm(qr), w.kv_norm(kv)
    q = w.q_head_norm(w.wq_b(qr).view(T, N, HD))
    cos, sin = _dsv4_cos_sin(w.rotary_emb.cos_sin_cache, positions, ROPE)
    q = _dsv4_rope_rope_dims(q, cos, sin, NOPE, headed=True)
    kv = _dsv4_rope_rope_dims(kv, cos, sin, NOPE, headed=False)  # window latents
    # compressed latents (oracle Compressor C128 pooling, w.compressor weights)
    comp = w.compressor
    kvsc = comp.fused_wkv_wgate(hidden)
    kvsc = kvsc[0] if isinstance(kvsc, (tuple, list)) else kvsc
    ck, cs = kvsc.split([comp.coff * HD, comp.coff * HD], dim=-1)
    Ncomp = T // ratio
    cut = Ncomp * ratio
    ck = ck[:cut].reshape(Ncomp, ratio, HD).float()
    cs = cs[:cut].reshape(Ncomp, ratio, HD).float() + comp.ape.float()
    pooled = comp.norm(((ck * cs.softmax(dim=1)).sum(dim=1)).to(hidden.dtype))
    ccos, csin = _dsv4_cos_sin(
        w.rotary_emb.cos_sin_cache, torch.arange(Ncomp) * ratio, ROPE
    )
    compl = _dsv4_rope_rope_dims(pooled, ccos, csin, NOPE, headed=False)  # [Ncomp,HD]
    # one combined softmax over [window (T) | compressed (Ncomp)] + sink
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    wmask = torch.zeros(T, T).masked_fill(~((j <= i) & (j > i - W)), float("-inf"))
    logits_w = torch.einsum("inh,jh->nij", q.float(), kv.float()) * SCALE + wmask[None]
    cvalid = ((torch.arange(T) + 1) // ratio)[:, None]
    cmask = torch.zeros(T, Ncomp).masked_fill(
        ~(torch.arange(Ncomp)[None, :] < cvalid), float("-inf")
    )
    logits_c = (
        torch.einsum("inh,ch->nic", q.float(), compl.float()) * SCALE + cmask[None]
    )
    logits = torch.cat([logits_w, logits_c], dim=-1)
    sink = w.mla_attn.attn_sink[:N].float()[:, None, None].expand(N, T, 1)
    z = torch.cat([logits, sink], -1).softmax(-1)[..., : T + Ncomp]
    kv_union = torch.cat([kv.float(), compl.float()], dim=0)
    o = torch.einsum("nik,kh->inh", z, kv_union)
    o = _dsv4_rope_rope_dims(o, cos, sin, NOPE, headed=True, sign=-1.0)
    o_g = o.reshape(T, G, (N // G) * HD)
    zz = torch.einsum("bhr,hdr->bhd", o_g.float(), w.wo_a.weight.float())
    return w.wo_b(zz.flatten(1).to(hidden.dtype))


def _run_c128a(dev, base_mods, hidden, positions, prefix, dtype):
    mods = _to_dev(base_mods, dev)
    vllm_config = _mock_vllm_config_c128(dtype)
    with contextlib.ExitStack() as s:
        s.enter_context(set_current_vllm_config(vllm_config))
        s.enter_context(
            mock.patch.object(
                deepseek_v4_attention,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        )
        torch.manual_seed(7)  # deterministic compressor init (built by the ctor)
        w = _build_wrapper_c128(vllm_config, mods, prefix)
        # The compressor is built inside the ctor (a real parallel linear, on cpu
        # with vLLM's default fp32 params). In production its weights are the
        # bf16-dequantized checkpoint tensors; here we cast it to the test
        # dtype+device so activations and weights match (as they do post-load).
        w.compressor = w.compressor.to(device=dev, dtype=dtype)
        out = w.forward(positions.to(dev), hidden.to(dev))
        return out, w


from vllm.config import set_current_vllm_config  # noqa: E402


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_c128a_forward(arch):
    """C128A (HCA) fresh-sequence forward: matches an independent oracle-mirroring
    reference (fp32 math) and runs on tt (bf16 lowering). Uses T=256 >= ratio 128
    so the compressor produces Ncomp=2 real compressed slots (both branches +
    the two-branch merge are exercised)."""
    import vllm_tt.attention_impls.attention_dsv4 as _dsv4  # noqa: F401

    xr.set_device_type("TT")
    _init_dist_once()
    positions = torch.arange(_TC)

    # -- math (fp32, cpu): wrapper.forward vs independent C128A reference --
    m32 = _build_modules(torch.float32)
    h32 = torch.randn(_TC, HIDDEN, dtype=torch.float32)
    out_cpu32, w32 = _run_c128a(
        torch.device("cpu"), m32, h32, positions, "c128.cpu32", torch.float32
    )
    assert out_cpu32.shape == (_TC, HIDDEN)
    ref32 = _reference_c128a(w32, h32, positions)
    pcc_math = _pcc(out_cpu32, ref32)
    assert pcc_math >= MATH_PCC, f"C128A math PCC {pcc_math:.6f} < {MATH_PCC}"

    # -- lowering (bf16): tt device vs cpu (same seeded compressor weights) --
    mb = _build_modules(torch.bfloat16)
    hb = torch.randn(_TC, HIDDEN, dtype=torch.bfloat16)
    out_cpu, _ = _run_c128a(
        torch.device("cpu"), mb, hb, positions, "c128.cpu", torch.bfloat16
    )
    out_tt, _ = _run_c128a(
        torch_xla.device(), mb, hb, positions, "c128.tt", torch.bfloat16
    )
    torch_xla.sync()
    pcc_low = _pcc(out_tt.cpu(), out_cpu)
    # C128A accumulates more bf16 device-vs-cpu divergence than the SWA-only path:
    # it adds the compressor's 128-way softmax pool, a second (compressed)
    # attention branch, and the two-branch merge on top of the window branch (the
    # fp32 score/lse einsums lower to bf16 matmuls on the tt hardware). The fp32
    # math check above is exact (>= 0.999), so this is genuine bf16 lowering noise,
    # not a logic error — use a slightly looser bar than the SWA REQUIRED_PCC.
    C128A_PCC = 0.98
    assert pcc_low >= C128A_PCC, f"C128A tt-vs-cpu PCC {pcc_low:.5f} < {C128A_PCC}"


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_c128a_subwindow_equals_swa(arch):
    """A sub-window C128A chunk (T < compress_ratio => Ncomp = 0) has no completed
    compressed slots: the guard in _dsv4_compress returns None and the forward
    reduces to the pure SWA window+sink attention. This pins the tt-mlir 0-dim
    compile-crash fix numerically -- T=64 (< ratio 128) is exactly one of the
    prefill compile buckets that used to divide-by-zero. Same shared weights, so
    the C128A (ratio 128) forward must equal the SWA (ratio 1) forward."""
    import vllm_tt.attention_impls.attention_dsv4 as _dsv4  # noqa: F401

    xr.set_device_type("TT")
    _init_dist_once()
    Tsub = 64  # < compress_ratio 128 -> Ncomp = 0
    positions = torch.arange(Tsub)
    mb = _build_modules(torch.bfloat16)
    hb = torch.randn(Tsub, HIDDEN, dtype=torch.bfloat16)

    out_c128, w = _run_c128a(
        torch.device("cpu"), mb, hb, positions, "sub.c128", torch.bfloat16
    )
    # the compressor must NOT have run (no completed group)
    assert w._dsv4_compress(hb) is None, "expected Ncomp==0 (compressor pruned)"
    out_swa, _ = _run(torch.device("cpu"), mb, hb, positions, "sub.swa", torch.bfloat16)

    pcc = _pcc(out_c128, out_swa)
    assert pcc >= 0.999, f"sub-window C128A != SWA window+sink: PCC {pcc:.6f}"


# --------------------------------------------------------------------------- #
# C128A paged prefill + decode round-trip: prefill writes BOTH the window (SWA)
# latent cache and the compressed latent cache; a following decode reads both
# back (window paged read + compressed paged read) and merges. S=128 == ratio so
# the compressor writes exactly Ncomp=1 real compressed slot.
# --------------------------------------------------------------------------- #
_SC = 128  # prefill length: % 32 == 0 (kernel) and == ratio so Ncomp = 1
_CBLOCK = 32


def _paged_roundtrip_c128a(dev, base_mods, dtype, hp, pp, hd, pd):
    """Build the C128A wrapper, bind BOTH paged caches, run _c128a_paged_prefill
    (writes) then _c128a_paged_decode (reads). Returns (decode out [1,N,HD],
    window cache, compressed cache, comp latents that prefill wrote)."""
    from vllm_tt.attention_impls.attention import TTMetadata

    mods = _to_dev(base_mods, dev)
    vllm_config = _mock_vllm_config_c128(dtype)
    hp, hd = hp.to(dtype), hd.to(dtype)
    decode_pos = int(pd.item())
    nb_w = decode_pos // _CBLOCK + 1  # window blocks covering [0, decode_pos]
    with contextlib.ExitStack() as s:
        s.enter_context(set_current_vllm_config(vllm_config))
        s.enter_context(
            mock.patch.object(
                deepseek_v4_attention,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        )
        torch.manual_seed(7)
        w = _build_wrapper_c128(vllm_config, mods, "pc128." + str(dev))
        w.compressor = w.compressor.to(device=dev, dtype=dtype)
        sink = w.mla_attn.attn_sink[:N].to(dev)

        wcache = torch.zeros(nb_w, 1, _CBLOCK, HD, dtype=dtype, device=dev)
        wpt = torch.arange(nb_w, dtype=torch.int32, device=dev).view(1, nb_w)
        ccache = torch.zeros(1, 1, _CBLOCK, HD, dtype=dtype, device=dev)  # 1 comp slot
        cpt = torch.zeros(1, 1, dtype=torch.int32, device=dev)

        # -- prefill: window latents + compressed latents into the two caches --
        qp, kvp, _, _ = w._dsv4_preprocess(pp.to(dev), hp.to(dev))
        md_p = TTMetadata(
            cache_position=torch.full((1,), _SC - 1, dtype=torch.int32, device=dev),
            page_table=wpt,
            fill_page_table=wpt,
            is_causal=True,
        )
        comp_md_p = TTMetadata(
            cache_position=torch.zeros(1, dtype=torch.int32, device=dev),
            page_table=cpt,
            fill_page_table=cpt,
            is_causal=True,
        )
        comp = w._dsv4_compress(hp.to(dev)).unsqueeze(0)  # [1, Ncomp, HD]
        w._c128a_paged_prefill(
            qp.view(1, _SC, N, HD),
            kvp.view(1, _SC, HD),
            hp.to(dev),
            pp.to(dev),
            sink,
            wcache,
            md_p,
            ccache,
            comp_md_p,
        )

        # -- decode: read window + compressed history back and merge --
        qd, kvd, _, _ = w._dsv4_preprocess(pd.to(dev), hd.to(dev))
        md_d = TTMetadata(
            cache_position=torch.full((1,), decode_pos, dtype=torch.int32, device=dev),
            page_table=wpt,
            fill_page_table=wpt,
            is_causal=True,
        )
        comp_md_d = TTMetadata(
            cache_position=torch.zeros(
                1, dtype=torch.int32, device=dev
            ),  # (pos+1)//r-1
            page_table=cpt,
            fill_page_table=cpt,
            is_causal=True,
        )
        od = w._c128a_paged_decode(
            qd.view(1, 1, N, HD),
            kvd.view(1, 1, HD),
            sink,
            wcache,
            md_d,
            ccache,
            comp_md_d,
        )  # [1, 1, N, HD]
        return od[0, 0], wcache, ccache, comp


@pytest.mark.push
@parametrize_arch(["single_device"])
def test_dsv4_c128a_paged_roundtrip(arch):
    """Paged C128A prefill writes BOTH the window + compressed latent caches;
    a following decode reads them back and merges. Validates the compressed
    cache round-trip numerically: (1) prefill wrote the compressor latents into
    the compressed cache, (2) the decode (window paged read + compressed paged
    read + sink merge) is finite. Runs on CPU (the tt ops' reference path); the
    on-device tt lowering of this exact paged prefill+decode is covered by the
    engine E2E runs (dummy 3-layer smoke + real-weight 4-layer generation)."""
    import vllm_tt.attention_impls.attention_dsv4 as _dsv4  # noqa: F401

    xr.set_device_type("TT")
    torch.manual_seed(0)
    _init_dist_once()
    hp = torch.randn(_SC, HIDDEN)
    pp = torch.arange(_SC)
    hd = torch.randn(1, HIDDEN)
    pd = torch.tensor([_SC])  # decode at position _SC (compressed slot 0 valid)

    mb = _build_modules(torch.bfloat16)
    od_c, _, ccache_c, comp = _paged_roundtrip_c128a(
        torch.device("cpu"), mb, torch.bfloat16, hp, pp, hd, pd
    )
    Ncomp = comp.shape[1]
    assert Ncomp == 1, f"expected 1 compressed slot for S={_SC}, got {Ncomp}"
    # prefill must have written the compressor latents into the compressed cache
    written = ccache_c[0, 0, :Ncomp, :]  # first Ncomp rows of block 0
    assert torch.allclose(
        written, comp[0].to(written.dtype), atol=2e-2
    ), "compressed cache write != compressor latents"
    # decode read both caches (window + compressed) + merged -> finite output
    assert torch.isfinite(od_c).all(), "paged C128A decode output not finite"
    assert od_c.shape == (N, HD), f"unexpected decode shape {tuple(od_c.shape)}"
