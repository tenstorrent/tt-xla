# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``torch.compile`` variant of ``test_mla_prefill_impl.py``.

Same three DeepSeek-V3 MLA PCC tests (prefill, decode, and the chained
prefill+decode), but the on-device run drives ``TTMLAAttentionBackendImpl.forward``
through ``torch.compile(..., backend="tt")`` instead of calling it eagerly — the
path vLLM actually takes on device.

``forward`` is a method on an attention impl, not an ``nn.Module``, so we wrap it
in ``_MLAModule``: only the activation tensors (and the in-place-mutated paged
``kv_cache``) are graph inputs, while the impl, the stub ``layer`` (absorbed
``W_UK_T`` / ``W_UV``) and the ``TTMetadata`` (page table / cache positions) are
captured on the module — mirroring how vLLM threads them through the
forward-context rather than as call args. Keeping ``kv_cache`` a graph *input* is
what lets the impl's in-place paged-cache write (``tt.paged_fill_cache`` /
``tt.paged_update_cache``) propagate back out under ``torch.compile``, which the
decode/cache assertions read.

Everything else is reused verbatim from ``test_mla_prefill_impl`` (imported
below): the real FP8 ``kv_b_proj`` loading + absorbed-weight derivation, the
head-parallel sharding on ``llmbox``, the single-device L1 xfail, the PCC helper,
and the paged-cache gather/assertions. Only ``_run_impl`` is overridden here. The
CPU golden stays eager — ``backend="tt"`` is device-only, so the golden remains
exactly the reference the eager test scores against; only the scored device run
is compiled.
"""

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from test_mla_prefill_impl import (  # noqa: F401  (pytest fixture, referenced by name in tests); noqa: I001  (sibling test module on pytest's path)
    _SINGLE_DEVICE_L1_XFAIL,
    BLOCK_SIZE,
    REQUIRED_PCC,
    _gather_cache,
    _latent_k,
    _maybe_mesh,
    _pcc,
    deepseek_v3_mla,
)


# --------------------------------------------------------------------------- #
# Arch parametrization
# --------------------------------------------------------------------------- #
def _arch_compiled():
    """Like ``parametrize_arch(["single_device", "llmbox"], xfail={single_device})``
    from the eager test, but the ``single_device`` case is xfailed with
    ``run=False`` — it is *not* executed.

    Under ``torch.compile`` the single_device case can't fit L1 (128 heads,
    latent head_dim=576) exactly like the eager test, so it carries no
    compiled-path signal. But unlike the eager path, its failed on-device run
    leaves a dangling in-flight XLA op that then breaks the *next* test's
    ``xr.use_spmd()`` (the head-parallel ``llmbox`` setup in ``_maybe_mesh``),
    cascading every ``llmbox`` case to failure. The single-device L1-OOM is
    already characterized at runtime by the eager ``test_mla_prefill_impl``, so
    here we mark it xfail-without-run: the matrix stays the same (still reported
    xfail) without corrupting the shared runtime for the ``llmbox`` cases."""
    return pytest.mark.parametrize(
        "arch",
        [
            pytest.param(
                "single_device",
                marks=[
                    pytest.mark.single_device,
                    pytest.mark.xfail(reason=_SINGLE_DEVICE_L1_XFAIL, run=False),
                ],
            ),
            pytest.param("llmbox", marks=[pytest.mark.llmbox]),
        ],
    )


# --------------------------------------------------------------------------- #
# torch.compile wrapper
# --------------------------------------------------------------------------- #
class _MLAModule(torch.nn.Module):
    """Thin ``nn.Module`` around ``TTMLAAttentionBackendImpl.forward`` so it can
    be handed to ``torch.compile``.

    The activation tensors and the in-place-mutated ``kv_cache`` are forward args
    (graph inputs) — ``kv_cache`` must stay an input for the impl's paged-cache
    write to be observable after the compiled call. The impl, the ``layer``
    (absorbed weights) and the ``TTMetadata`` are captured as attributes, matching
    how vLLM supplies them via the forward-context instead of as call args."""

    def __init__(self, impl, layer, attn_metadata):
        super().__init__()
        # Plain (non-nn.Module / non-Tensor) attributes: stored as-is, not
        # registered as submodules/params. The absorbed weights live inside
        # `layer`, so dynamo lifts them as captured tensors.
        self._impl = impl
        self._layer = layer
        self._attn_metadata = attn_metadata

    def forward(self, q_nope, q_pe, kv_c_normed, k_pe, kv_cache):
        return self._impl.forward(
            q=(q_nope, q_pe),
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=self._attn_metadata,
            layer=self._layer,
            output=None,
        )


def _run_impl(device, params, inputs, mesh=None):
    """Like ``test_mla_prefill_impl._run_impl``, but the device run goes through
    ``torch.compile(..., backend="tt")`` (wrapping ``forward`` in ``_MLAModule``);
    the CPU run stays eager so it remains the golden reference.

    Returns ``(attention_output, kv_cache)`` — ``kv_cache`` is cloned and mutated
    in place by ``forward`` (paged-cache write), so each run owns its copy and a
    caller can chain prefill→decode by feeding one run's returned cache into the
    next. When ``mesh`` is given, inputs + absorbed weights are head-parallel
    sharded across the mesh ``"model"`` axis before the (compiled) call."""
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

    if device.type == "cpu":
        # Eager CPU golden (backend="tt" is device-only): the exact reference the
        # eager test scores against.
        out = impl.forward(
            q=(q_nope, q_pe),
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer=layer,
            output=None,
        )
    else:
        # Scored device run, compiled with the Tenstorrent backend.
        module = _MLAModule(impl, layer, attn_metadata)
        compiled = torch.compile(module, backend="tt", dynamic=False)
        out = compiled(q_nope, q_pe, kv_c_normed, k_pe, kv_cache)

    return out, kv_cache


# --------------------------------------------------------------------------- #
# Tests — identical to test_mla_prefill_impl, but `_run_impl` above compiles the
# device run. The helpers/fixture/xfail/constants are imported from that module.
# --------------------------------------------------------------------------- #
@pytest.mark.nightly
@_arch_compiled()
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
@_arch_compiled()
@pytest.mark.parametrize("users, cur_pos", [(1, 31), (2, 48), (1, 96)])
def test_paged_mla_decode_impl(users, cur_pos, arch, deepseek_v3_mla):
    """Run JUST the decode branch of ``TTMLAAttentionBackendImpl.forward``, with
    real DeepSeek-V3 weights + config, under ``torch.compile``.

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
@_arch_compiled()
@pytest.mark.parametrize("users, seq_len", [(1, 64), (2, 64)])
def test_mla_prefill_and_decode_impl(users, seq_len, arch, deepseek_v3_mla):
    """Run MLA prefill (fills the paged latent cache) then a decode step against
    that cache, with real DeepSeek-V3 weights + config, under ``torch.compile``,
    verifying the cache after each stage.

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
