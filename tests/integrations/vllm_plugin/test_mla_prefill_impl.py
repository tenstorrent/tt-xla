# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for ``TTMLAAttentionBackendImpl.forward`` (the unified MLA prefill
and decode impl in ``vllm_tt.attention_mla``) using **real DeepSeek-V3 weights
and config** for a single layer.

What this exercises
-------------------
``TTMLAAttentionBackendImpl.forward`` is the heavy-lifting half of the MLA
prefill path: Q-absorption (``q_nope @ W_UK_T``), the latent flash attention
(``tt.flash_mla_prefill``), the ``W_UV`` projection back to physical space, and
the paged latent KV-cache write (``tt.paged_fill_cache``). The only *weights* it
consumes are ``layer.W_UK_T`` / ``layer.W_UV``, which vLLM derives from
``kv_b_proj.weight`` in ``MLAAttention.process_weights_after_loading``
(mla_attention.py:710-797). We load the real layer-0 ``kv_b_proj`` from
``deepseek-ai/DeepSeek-V3`` (FP8, block-dequantized) and derive the absorbed
weights exactly as vLLM does, and we take the MLA dims + attention scale from the
real V3 config. The activation inputs (q_nope/q_pe/kv_c_normed/k_pe) are random —
the function does not consume any projection weights to produce them.

Why test the impl directly (not the OOT wrapper)
------------------------------------------------
``forward`` only touches ``torch.ops.tt.*`` (dispatched by tensor device),
``layer.W_UK_T/W_UV`` (a stub here), and a ``TTMetadata`` we build directly. So
we can drive it without vLLM's platform/config/distributed/forward-context
machinery — keeping this a focused single-op-graph correctness check.

Sharding (tensor parallelism)
-----------------------------
On a multi-device arch (``llmbox``) the impl is run under SPMD with the N query
heads split across the mesh ``"model"`` axis — the standard head-parallel MLA
layout. The per-head Q activations (``q_nope``/``q_pe``) and the absorbed
weights (``W_UK_T``/``W_UV``, sharded on their head axis) partition along the
heads, so both einsums and the ``tt.flash_mla_prefill`` kernel run head-local
with no cross-device collectives. The compressed latent KV (``num_kv_heads ==
1``, i.e. ``kv_c_normed``/``k_pe`` and the paged latent cache) is shared by
every head and stays replicated — exactly how vLLM keeps the MLA KV cache under
TP. ``single_device`` runs the same code path unsharded (``mesh=None``).

PCC method: run the impl on CPU (custom-op CPU fallback = SDPA + manual paged
fill) as golden, then on the TT device (StableHLO custom calls → ttnn), and
compare the returned attention output with Pearson correlation. Sharding is a
layout hint, so the gathered device output must still match the replicated CPU
golden. The paged cache write is exercised but its result is discarded by
``forward`` (it returns the attention output), so only the attention math is
scored.

Decode tests
------------
``test_paged_mla_decode_impl`` drives only the decode branch (``S == 1`` per
user, dispatched via ``_infer_is_prefill``): the new token's latent K is written
into a pre-seeded paged cache via ``tt.paged_update_cache`` and attended over
``[0, cur_pos]`` via ``tt.paged_flash_mla_decode``.
``test_mla_prefill_and_decode_impl`` chains the two — prefill fills the cache,
then a decode step appends one token — and additionally asserts the paged cache
holds the exact latent KV at the page-table-mapped slots after each stage
(cache-write correctness, not just the attention output). Unlike the
prefill-only test, these inspect the persisted cache, so ``_run_impl`` returns
it alongside the attention output.

All three tests run at the real DeepSeek-V3 sizes; 128 heads don't fit one
device's L1, so each xfails its ``single_device`` case and does the real work on
``llmbox`` head-parallel (see ``_SINGLE_DEVICE_L1_XFAIL``). The paged decode
kernel additionally needed a ``k_chunk_size`` fix in the tt-mlir runtime to fit
even head-parallel.
"""

import math

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

from tests.utils import failed_runtime, parametrize_arch

REPO = "deepseek-ai/DeepSeek-V3"
LAYER = 0
BLOCK_SIZE = 32  # TTMLAAttentionBackend.get_page_size
# bf16 flash attention + fp8-derived absorbed weights; the einsums run in fp32
# inside the impl, so the residual error is dominated by the bf16 kernel.
REQUIRED_PCC = 0.99


# --------------------------------------------------------------------------- #
# Real-weight loading helpers
# --------------------------------------------------------------------------- #
def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """Mirror ``vllm.model_executor.models.deepseek_v2.yarn_get_mscale``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _dequant_fp8_blockwise(
    w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """DeepSeek FP8 block-wise dequant: ``w_bf16 = w_fp8 * weight_scale_inv``,
    with one scale per ``block x block`` tile (weight_block_size = [128, 128])."""
    w = w_fp8.to(torch.float32)
    rows, cols = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return w * s[:rows, :cols]


def _mla_scale(cfg: dict) -> float:
    """Attention scale exactly as DeepseekV2MLAAttention computes it
    (deepseek_v2.py:862, 918-939): base ``qk_head_dim**-0.5`` times the squared
    YaRN mscale when rope scaling is yarn."""
    qk_head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    scale = qk_head_dim**-0.5
    rope = cfg.get("rope_scaling") or {}
    if rope.get("type") == "yarn" or rope.get("rope_type") == "yarn":
        mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
        m = _yarn_get_mscale(rope["factor"], mscale_all_dim)
        scale = scale * m * m
    return float(scale)


# safetensors dtype tags -> torch dtypes, for the ranged-read fast path below.
_ST_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _fetch_safetensors(repo: str, shard: str, names: list[str]) -> dict:
    """Return ``{name: tensor}`` for ``names`` from ``repo``'s ``shard``,
    downloading only what's needed (and downloading nothing if already cached).

    DeepSeek-V3's shards are ~5 GB each, but we only want a single ~17 MB
    ``kv_b_proj`` weight, so pulling the whole file via ``hf_hub_download`` is
    wasteful. If a previous run already cached the full shard locally we read it
    straight from disk; otherwise we stream just the requested tensors over HTTP
    ranged reads — parse the 8-byte length prefix + JSON header for each
    tensor's dtype/shape/byte-offsets, then read only those bytes."""
    import json
    import struct

    from huggingface_hub import HfFileSystem, try_to_load_from_cache
    from safetensors import safe_open

    cached = try_to_load_from_cache(repo, shard)
    if isinstance(cached, str):  # full shard already downloaded -> read locally
        with safe_open(cached, framework="pt") as f:
            return {name: f.get_tensor(name) for name in names}

    out = {}
    with HfFileSystem().open(f"{repo}/{shard}", "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len  # tensor data offsets are relative to here
        for name in names:
            meta = header[name]
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            raw = bytearray(f.read(end - begin))  # writable -> no frombuffer warning
            out[name] = torch.frombuffer(raw, dtype=_ST_DTYPE[meta["dtype"]]).reshape(
                meta["shape"]
            )
    return out


def _load_absorbed_weights(cfg: dict, act_dtype: torch.dtype):
    """Fetch real layer-0 ``kv_b_proj`` from DeepSeek-V3, FP8-dequant it, and
    derive ``(W_UK_T, W_UV)`` exactly as
    ``MLAAttention.process_weights_after_loading`` (mla_attention.py:710-797)."""
    import json

    from huggingface_hub import hf_hub_download

    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]

    index = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    key = f"model.layers.{LAYER}.self_attn.kv_b_proj.weight"
    shard = index["weight_map"][key]
    tensors = _fetch_safetensors(REPO, shard, [key, key + "_scale_inv"])
    weight = tensors[key]  # fp8_e4m3, [N*(P+V), L]
    scale_inv = tensors[key + "_scale_inv"]  # [ceil/128, ceil/128]

    # [N*(P+V), L] -> [L, N, P+V]  (mla_attention.py:714-732)
    kv_b = _dequant_fp8_blockwise(weight, scale_inv).to(act_dtype)
    kv_b = kv_b.t().contiguous().view(L, N, P + V)
    W_UK, W_UV = kv_b.split([P, V], dim=-1)  # [L,N,P], [L,N,V]
    # (mla_attention.py:794-797)
    W_UV = W_UV.transpose(0, 1).contiguous()  # [N, L, V]
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
    return W_UK_T, W_UV


@pytest.fixture(scope="module")
def deepseek_v3_mla():
    """Load real DeepSeek-V3 config + absorbed layer-0 weights once. Skips the
    whole module if the checkpoint can't be reached (no network / no auth)."""
    act_dtype = torch.bfloat16
    try:
        import json

        from huggingface_hub import hf_hub_download

        cfg = json.load(open(hf_hub_download(REPO, "config.json")))
        W_UK_T, W_UV = _load_absorbed_weights(cfg, act_dtype)
    except Exception as e:  # network/auth/format issues -> skip, don't fail
        pytest.skip(f"DeepSeek-V3 weights/config unavailable: {type(e).__name__}: {e}")

    return {
        "cfg": cfg,
        "act_dtype": act_dtype,
        "scale": _mla_scale(cfg),
        "W_UK_T": W_UK_T,
        "W_UV": W_UV,
    }


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #
def _pcc(device_out: torch.Tensor, golden: torch.Tensor) -> float:
    """Pearson correlation, mirroring TorchComparisonEvaluator.compute_pcc
    (tests/infra/evaluators/torch_comparison_evaluator.py:160-164)."""
    x = device_out.flatten().float()
    y = golden.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


def _run_impl(device, params, inputs, mesh=None):
    """Build a fresh impl + stub layer + TTMetadata on ``device`` and run
    ``TTMLAAttentionBackendImpl.forward``; returns ``(attention_output,
    kv_cache)``.

    The KV cache is cloned before ``forward`` so each run mutates its own copy
    (``forward`` persists new tokens into it in place via ``paged_fill_cache`` /
    ``paged_update_cache``). This keeps the CPU-golden and device runs from
    contaminating each other through a shared input cache, and — since the
    passed-in cache is left untouched — lets a caller chain prefill→decode by
    feeding one run's returned cache into the next.

    When ``mesh`` is given (the multi-device path), the inputs and absorbed
    weights are head-parallel sharded across the mesh ``"model"`` axis before
    ``forward`` runs."""
    from types import SimpleNamespace

    from vllm_tt.attention import TTMetadata
    from vllm_tt.attention_mla import TTMLAAttentionBackendImpl

    cfg = params["cfg"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]

    impl = TTMLAAttentionBackendImpl(
        num_heads=cfg["num_attention_heads"],
        head_size=L + R,  # MLA latent kv-cache head dim
        scale=params["scale"],
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=cfg.get("q_lora_rank"),
        kv_lora_rank=L,
        qk_nope_head_dim=cfg["qk_nope_head_dim"],
        qk_rope_head_dim=R,
        qk_head_dim=cfg["qk_nope_head_dim"] + R,
        v_head_dim=cfg["v_head_dim"],
    )

    # Stub `layer`: forward only reads layer.W_UK_T / layer.W_UV.
    W_UK_T = params["W_UK_T"].to(device)  # [N, P, L]
    W_UV = params["W_UV"].to(device)  # [N, L, V]

    q_nope, q_pe, kv_c_normed, k_pe, kv_cache, cache_position, page_table = (
        t.to(device) for t in inputs
    )
    # Mutated in place by forward; clone so the passed-in cache is preserved.
    kv_cache = kv_cache.clone()

    if mesh is not None:
        # Head-parallel (tensor-parallel) MLA sharding: split the N query heads
        # across the mesh "model" axis. The per-head Q activations and absorbed
        # weights shard along their head axis; the compressed latent KV
        # (num_kv_heads == 1) is shared by every head, so kv_c_normed / k_pe /
        # cache_position / page_table and the paged latent cache stay replicated.
        xs.mark_sharding(q_nope, mesh, (None, "model", None))  # [tokens, N, P]
        xs.mark_sharding(q_pe, mesh, (None, "model", None))  # [tokens, N, R]
        xs.mark_sharding(W_UK_T, mesh, ("model", None, None))  # [N, P, L]
        xs.mark_sharding(W_UV, mesh, ("model", None, None))  # [N, L, V]

    layer = SimpleNamespace(W_UK_T=W_UK_T, W_UV=W_UV)

    attn_metadata = TTMetadata(
        cache_position=cache_position,
        attn_mask=None,
        page_table=page_table,
        is_causal=True,
        fill_page_table=page_table,
    )
    out = impl.forward(
        q=(q_nope, q_pe),
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        layer=layer,
        output=None,
    )
    return out, kv_cache


def _maybe_mesh(arch, num_heads):
    """Head-parallel mesh for the ``llmbox`` arch (one shard of the N heads per
    device), or ``None`` for ``single_device``. Skips when the heads don't
    divide evenly across the devices."""
    if arch != "llmbox":
        return None
    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_heads % num_devices != 0:
        pytest.skip(
            f"num_heads ({num_heads}) not divisible by num_devices ({num_devices})"
        )
    return get_mesh((1, num_devices), ("batch", "model"))


def _latent_k(kv_c_normed: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """The latent K the impl persists into the paged cache: ``cat([kv_c_normed,
    k_pe], dim=-1)``. Mirrors ``k_lat`` in ``forward`` (kv_c_normed: [tokens, L],
    k_pe: [tokens, 1, R]) -> [tokens, L + R]."""
    return torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)


def _gather_cache(
    cache: torch.Tensor, page_table: torch.Tensor, num_positions: int
) -> torch.Tensor:
    """Read logical sequence positions ``[0, num_positions)`` out of the paged
    latent cache for every user, undoing the paging via ``page_table``.

    cache: [num_blocks, 1, BLOCK_SIZE, head_dim] -> [users, num_positions, head_dim].
    Position ``p`` of user ``u`` lives in physical block
    ``page_table[u, p // BLOCK_SIZE]`` at offset ``p % BLOCK_SIZE``."""
    users = page_table.shape[0]
    out = torch.empty(users, num_positions, cache.shape[-1], dtype=cache.dtype)
    for u in range(users):
        for p in range(num_positions):
            blk = int(page_table[u, p // BLOCK_SIZE])
            out[u, p] = cache[blk, 0, p % BLOCK_SIZE, :]
    return out


# DeepSeek-V3 MLA at full scale (128 heads; latent head_dim = kv_lora_rank 512 +
# qk_rope_head_dim 64 = 576, v_head_dim 512) doesn't fit a single device's 1.5 MB
# L1 — it's meant to run head-parallel under tensor parallelism (llmbox: 16
# heads/device). So every test here xfails its ``single_device`` case and the
# real work happens on ``llmbox``. (The paged *decode* kernel additionally needed
# a ``k_chunk_size`` fix in the tt-mlir runtime —
# ``paged_flash_multi_latent_attention_decode.cpp`` — to fit even head-parallel;
# without it ``sdpa_decode``'s default dynamic chunking overflowed L1 at any head
# count.)
_SINGLE_DEVICE_L1_XFAIL = failed_runtime(
    "DeepSeek-V3 MLA (128 heads, latent head_dim=576) exceeds single-device L1; "
    "runs head-parallel under tensor parallelism (llmbox)."
)


@pytest.mark.nightly
@parametrize_arch(
    ["single_device", "llmbox"], xfail={"single_device": _SINGLE_DEVICE_L1_XFAIL}
)
@pytest.mark.parametrize("users, seq_len", [(1, 64), (2, 64), (1, 128)])
def test_mla_prefill_impl_deepseek_v3(users, seq_len, arch, deepseek_v3_mla):
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    head_dim = L + R

    mesh = _maybe_mesh(arch, N)

    assert seq_len % 32 == 0, "flash_mla_prefill requires seq_len % 32 == 0"
    tokens = users * seq_len
    blocks_per_user = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = users * blocks_per_user

    # Random activations (the impl consumes no projection weights to make these);
    # real absorbed weights come from `params`.
    q_nope = torch.randn(tokens, N, P, dtype=dtype)
    q_pe = torch.randn(tokens, N, R, dtype=dtype)
    kv_c_normed = torch.randn(tokens, L, dtype=dtype)
    k_pe = torch.randn(tokens, 1, R, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    # cache_position: one entry per user (only its shape[0] is read for `users`).
    cache_position = torch.full((users,), seq_len - 1, dtype=torch.int32)
    # Each user gets `blocks_per_user` distinct, contiguous block ids.
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    inputs = (q_nope, q_pe, kv_c_normed, k_pe, kv_cache, cache_position, page_table)

    golden, _ = _run_impl(torch.device("cpu"), params, inputs)

    device_out, _ = _run_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out = device_out.cpu()

    assert device_out.shape == golden.shape == (tokens, N * cfg["v_head_dim"])
    pcc = _pcc(device_out, golden)
    assert pcc >= REQUIRED_PCC, f"MLA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"


@pytest.mark.nightly
@parametrize_arch(
    ["single_device", "llmbox"], xfail={"single_device": _SINGLE_DEVICE_L1_XFAIL}
)
@pytest.mark.parametrize("users, cur_pos", [(1, 31), (2, 48), (1, 96)])
def test_paged_mla_decode_impl(users, cur_pos, arch, deepseek_v3_mla):
    """Run JUST the decode branch of ``TTMLAAttentionBackendImpl.forward``, with
    real DeepSeek-V3 weights + config.

    Decode is one token per user (``S == 1``), so ``_infer_is_prefill`` routes
    ``forward`` into ``_forward_decode``. We seed the whole paged latent cache
    with random KV (this is the user's prior context; positions past ``cur_pos``
    are masked out by the causal decode kernel), then run a single decode step:
    the impl writes the new token's latent K at ``cur_pos`` via
    ``tt.paged_update_cache`` and attends over ``[0, cur_pos]`` via
    ``tt.paged_flash_mla_decode``. We score the attention output (device vs CPU
    golden) and confirm the device cache update matches the CPU golden — the new
    token lands at ``cur_pos`` and the prior context is left intact.

    ``single_device`` xfails (128 heads exceed one device's L1, see
    ``_SINGLE_DEVICE_L1_XFAIL``); ``llmbox`` runs head-parallel.
    """
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]
    head_dim = L + R

    mesh = _maybe_mesh(arch, N)

    # Enough blocks per user to hold the current position cur_pos.
    blocks_per_user = cur_pos // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user

    # One decode token per user; the prior context is the pre-seeded cache.
    q_nope = torch.randn(users, N, P, dtype=dtype)
    q_pe = torch.randn(users, N, R, dtype=dtype)
    kv_c_normed = torch.randn(users, L, dtype=dtype)
    k_pe = torch.randn(users, 1, R, dtype=dtype)
    seeded_cache = torch.randn(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    cache_position = torch.full((users,), cur_pos, dtype=torch.int32)
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    inputs = (q_nope, q_pe, kv_c_normed, k_pe, seeded_cache, cache_position, page_table)

    golden, cpu_cache = _run_impl(torch.device("cpu"), params, inputs)

    device_out, device_cache = _run_impl(torch_xla.device(), params, inputs, mesh=mesh)
    torch_xla.sync()
    device_out, device_cache = device_out.cpu(), device_cache.cpu()

    # -- Decode attention output (one token per user) --
    assert device_out.shape == golden.shape == (users, N * V)
    pcc = _pcc(device_out, golden)
    assert pcc >= REQUIRED_PCC, f"MLA decode PCC {pcc:.5f} < {REQUIRED_PCC}"

    # -- Cache update: new token at cur_pos, prior context untouched --
    # `_run_impl` clones the cache, so `seeded_cache` still holds the pre-decode
    # state to compare the [0, cur_pos) prefix against.
    new_token_k = _latent_k(kv_c_normed, k_pe)  # [users, head_dim]
    after = _gather_cache(cpu_cache, page_table, cur_pos + 1)
    before = _gather_cache(seeded_cache, page_table, cur_pos + 1)
    assert torch.allclose(
        after[:, cur_pos, :], new_token_k
    ), "decode token not written at cur_pos"
    assert torch.allclose(
        after[:, :cur_pos, :], before[:, :cur_pos, :]
    ), "decode clobbered prior context"
    cache_pcc = _pcc(device_cache, cpu_cache)
    assert (
        cache_pcc >= REQUIRED_PCC
    ), f"decode cache PCC {cache_pcc:.5f} < {REQUIRED_PCC}"


@pytest.mark.nightly
@parametrize_arch(
    ["single_device", "llmbox"], xfail={"single_device": _SINGLE_DEVICE_L1_XFAIL}
)
@pytest.mark.parametrize("users, seq_len", [(1, 64), (2, 64)])
def test_mla_prefill_and_decode_impl(users, seq_len, arch, deepseek_v3_mla):
    """Run MLA prefill (fills the paged latent cache) then a decode step against
    that cache, with real DeepSeek-V3 weights + config, verifying the cache after
    each stage.

    Prefill writes positions ``[0, seq_len)`` for every user via
    ``tt.paged_fill_cache``; decode then appends the token at position
    ``seq_len`` via ``tt.paged_update_cache`` and attends over ``[0, seq_len]``.
    Both stages run on device, chained — decode reads the cache the prefill just
    produced — and we score each stage's attention output against the CPU golden
    plus assert the paged cache holds the exact latent KV at the
    page-table-mapped slots after each stage (prefill fills ``[0, seq_len)``;
    decode preserves that prefix and appends the new token at ``seq_len``).

    ``single_device`` xfails (128 heads exceed one device's L1, see
    ``_SINGLE_DEVICE_L1_XFAIL``); ``llmbox`` runs head-parallel.
    """
    xr.set_device_type("TT")
    torch.manual_seed(0)

    params = deepseek_v3_mla
    cfg = params["cfg"]
    dtype = params["act_dtype"]
    N = cfg["num_attention_heads"]
    L = cfg["kv_lora_rank"]
    R = cfg["qk_rope_head_dim"]
    P = cfg["qk_nope_head_dim"]
    V = cfg["v_head_dim"]
    head_dim = L + R

    assert seq_len % 32 == 0, "flash_mla_prefill requires seq_len % 32 == 0"
    mesh = _maybe_mesh(arch, N)

    # One extra block per user past the prefill length, to hold the decode token
    # written at position seq_len.
    blocks_per_user = seq_len // BLOCK_SIZE + 1
    num_blocks = users * blocks_per_user
    page_table = torch.arange(num_blocks, dtype=torch.int32).view(
        users, blocks_per_user
    )

    # ----- Prefill inputs (S = seq_len tokens per user) -----
    tokens = users * seq_len
    q_nope_p = torch.randn(tokens, N, P, dtype=dtype)
    q_pe_p = torch.randn(tokens, N, R, dtype=dtype)
    kv_c_p = torch.randn(tokens, L, dtype=dtype)
    k_pe_p = torch.randn(tokens, 1, R, dtype=dtype)
    kv_cache = torch.zeros(num_blocks, 1, BLOCK_SIZE, head_dim, dtype=dtype)
    cache_position_p = torch.full((users,), seq_len - 1, dtype=torch.int32)
    prefill_inputs = (
        q_nope_p,
        q_pe_p,
        kv_c_p,
        k_pe_p,
        kv_cache,
        cache_position_p,
        page_table,
    )

    # ----- Decode inputs (S = 1 token per user, written at position seq_len) ---
    q_nope_d = torch.randn(users, N, P, dtype=dtype)
    q_pe_d = torch.randn(users, N, R, dtype=dtype)
    kv_c_d = torch.randn(users, L, dtype=dtype)
    k_pe_d = torch.randn(users, 1, R, dtype=dtype)
    cache_position_d = torch.full((users,), seq_len, dtype=torch.int32)

    def _decode_inputs(cache_after_prefill):
        return (
            q_nope_d,
            q_pe_d,
            kv_c_d,
            k_pe_d,
            cache_after_prefill,
            cache_position_d,
            page_table,
        )

    # ----- CPU goldens: prefill, then decode against the prefilled cache -----
    golden_prefill, cpu_cache_prefill = _run_impl(
        torch.device("cpu"), params, prefill_inputs
    )
    golden_decode, _ = _run_impl(
        torch.device("cpu"), params, _decode_inputs(cpu_cache_prefill)
    )

    # ----- Device: prefill, then decode on device right after it -----
    # The back-to-back on-device prefill->decode used to return garbage under
    # SPMD; that program-transition bug is now fixed, so we score the on-device
    # decode. We snapshot the prefill cache to host *before* running decode and
    # feed that snapshot in: reading `device_cache_prefill` after the decode is
    # unreliable because XLA may donate the chained input buffer to the decode
    # graph (it would then show the decode's write). The decode still runs on
    # device immediately after the prefill graph — the path that used to corrupt.
    device_prefill, device_cache_prefill = _run_impl(
        torch_xla.device(), params, prefill_inputs, mesh=mesh
    )
    torch_xla.sync()
    device_prefill = device_prefill.cpu()
    device_cache_prefill = device_cache_prefill.cpu()

    device_decode, device_cache_decode = _run_impl(
        torch_xla.device(), params, _decode_inputs(device_cache_prefill), mesh=mesh
    )
    torch_xla.sync()
    device_decode = device_decode.cpu()
    device_cache_decode = device_cache_decode.cpu()

    # ===== Prefill: attention output + cache filled with the prefill tokens ====
    assert device_prefill.shape == golden_prefill.shape == (tokens, N * V)
    pcc = _pcc(device_prefill, golden_prefill)
    assert pcc >= REQUIRED_PCC, f"MLA prefill PCC {pcc:.5f} < {REQUIRED_PCC}"

    expected_prefill_k = _latent_k(kv_c_p, k_pe_p).view(users, seq_len, head_dim)
    filled = _gather_cache(device_cache_prefill, page_table, seq_len)
    assert torch.allclose(filled, expected_prefill_k), "prefill cache filled wrong"
    # The decode block (last per user) hasn't been written yet.
    assert (
        device_cache_prefill[page_table[:, -1].long()] == 0
    ).all(), "decode block not zero after prefill"

    # ===== Decode: attention output + cache appended with the new token ========
    assert device_decode.shape == golden_decode.shape == (users, N * V)
    pcc = _pcc(device_decode, golden_decode)
    assert pcc >= REQUIRED_PCC, f"MLA decode PCC {pcc:.5f} < {REQUIRED_PCC}"

    expected_decode_k = _latent_k(kv_c_d, k_pe_d)  # [users, head_dim]
    updated = _gather_cache(device_cache_decode, page_table, seq_len + 1)
    assert torch.allclose(
        updated[:, :seq_len, :], expected_prefill_k
    ), "decode clobbered prefill context"
    assert torch.allclose(
        updated[:, seq_len, :], expected_decode_k
    ), "decode token not written at seq_len"
