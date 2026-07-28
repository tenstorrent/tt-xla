# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the DeepSeek Sparse Attention (DSA) ops.

Each op's own CPU branch is the golden (``run_op_test`` runs the op on CPU and on
device, then compares), so these validate the StableHLO emission and the tt-mlir
lowering.

Architecture note: the three DSA TTNN kernels are Blackhole-only. Elsewhere the
``ttcore.composite`` falls back to inlining a primitive decomposition. Those
decompositions are faithful for ``tt.indexer_score_dsa`` and ``tt.sparse_sdpa``
(sentinels included), so most tests here are arch-agnostic by design. The one
exception is ``tt.topk_large_indices``' ``-inf`` -> sentinel contract, which
exists only in the kernel — see ``test_topk_large_indices_sentinel_contract``.
"""

import pytest
import torch

# Registers the tt:: custom ops (including the three DSA ops).
import tt_torch.custom_ops  # noqa: F401
from benchmark.utils import compute_pcc
from infra import Framework, run_op_test
from tt_torch.custom_ops import TOPK_LARGE_INDICES_SENTINEL

DTYPE = torch.bfloat16
REQUIRED_PCC = 0.99


def _finite_pcc(device_output, golden_output) -> float:
    """PCC over entries finite in both tensors.

    ``tt.indexer_score_dsa`` returns ``-inf`` at masked positions; feeding those
    to a plain PCC yields NaN.
    """
    x = device_output.flatten().float()
    y = golden_output.flatten().float()
    mask = torch.isfinite(x) & torch.isfinite(y)
    return compute_pcc(y[mask], x[mask])


def indexer_score_comparator(device_output, golden_output, args, kwargs):
    """Compare the causal mask exactly, then PCC over the finite scores.

    The mask comparison is the important half: a mismatch there means the causal
    boundary moved, which is precisely the failure that would let DSA attend to
    future tokens.
    """
    device_output = device_output.cpu()
    assert torch.equal(
        torch.isneginf(device_output), torch.isneginf(golden_output)
    ), "causal (-inf) mask differs between device and golden"

    pcc = _finite_pcc(device_output, golden_output)
    assert pcc > REQUIRED_PCC, f"finite-score PCC {pcc} (required > {REQUIRED_PCC})"


def dsa_topk_comparator(device_output, golden_output, args, kwargs):
    """Compare top-k *selections* by the values they pick, not by index equality.

    bf16 rows contain many ties, so index order is arbitrary; what must match is
    the set of scores selected. Both sides are converted to int64 first — uint32
    supports almost no torch ops (``gather`` included).
    """
    input_tensor = args[0]
    device_indices = device_output.cpu().to(torch.int64)
    golden_indices = golden_output.to(torch.int64)

    # The op declares a ui32 result and eager XLA preserves it, but the torch_xla
    # dynamo bridge downgrades uint32 *graph outputs* to int32. Either is fine here
    # -- the bit pattern is preserved and DSA keeps the indices inside the graph.
    assert device_output.dtype in (
        torch.uint32,
        torch.int32,
    ), f"expected uint32/int32 indices, got {device_output.dtype}"
    assert device_indices.shape == golden_indices.shape

    flat_dev = device_indices.reshape(-1, device_indices.shape[-1])
    for row in flat_dev:
        assert row.unique().numel() == row.numel(), f"duplicate indices in row {row}"

    device_gathered = torch.gather(input_tensor.float(), -1, device_indices)
    golden_gathered = torch.gather(input_tensor.float(), -1, golden_indices)
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0),
        golden_gathered.flatten().unsqueeze(0),
    )
    assert cos_sim > 0.99, f"Cosine similarity: {cos_sim.item()} (required > 0.99)"


def _causal_score(b, hi, sq, t, d, chunk_start_idx=0, seed=0):
    """A causally masked indexer score, i.e. the real producer for top-k."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hi, sq, d, dtype=DTYPE, generator=g)
    k = torch.randn(b, 1, t, d, dtype=DTYPE, generator=g)
    w = torch.randn(b, hi, sq, 1, dtype=DTYPE, generator=g)
    return torch.ops.tt.indexer_score_dsa(q, k, w, chunk_start_idx)


def _sorted_indices(seq_len, key_seq_len, topk, sentinel_tail=0, seed=0):
    """Distinct, in-range, ascending indices per row, optionally sentinel-padded.

    Mirrors what ``tt.topk_large_indices`` guarantees its consumer: valid indices
    below ``key_seq_len`` and any sentinels as a contiguous tail.
    """
    g = torch.Generator().manual_seed(seed)
    keep = topk - sentinel_tail
    rows = [
        torch.randperm(key_seq_len, generator=g)[:keep].sort().values
        for _ in range(seq_len)
    ]
    idx = torch.stack(rows).to(torch.int64)
    if sentinel_tail:
        pad = torch.full(
            (seq_len, sentinel_tail), TOPK_LARGE_INDICES_SENTINEL, dtype=torch.int64
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.view(1, 1, seq_len, topk).to(torch.uint32)


# --------------------------------------------------------------------------- #
# tt.indexer_score_dsa
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "num_index_heads, sq, t, index_head_dim, chunk_start_idx",
    [
        (64, 32, 32, 128, 0),  # DeepSeek-V3.2 indexer dims
        (64, 64, 64, 128, 0),
        (64, 32, 64, 128, 32),  # decode-style: cached prefix + new tokens
        (8, 32, 32, 128, 0),  # small head count
    ],
)
def test_indexer_score_dsa(num_index_heads, sq, t, index_head_dim, chunk_start_idx):
    query = torch.randn(1, num_index_heads, sq, index_head_dim, dtype=DTYPE)
    key = torch.randn(1, 1, t, index_head_dim, dtype=DTYPE)
    weights = torch.randn(1, num_index_heads, sq, 1, dtype=DTYPE)

    run_op_test(
        torch.ops.tt.indexer_score_dsa,
        [query, key, weights, chunk_start_idx],
        framework=Framework.TORCH,
        custom_comparator=indexer_score_comparator,
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_indexer_score_dsa_full_prefill():
    """DeepSeek-V3.2 indexer at a realistic prefill length."""
    query = torch.randn(1, 64, 1024, 128, dtype=DTYPE)
    key = torch.randn(1, 1, 1024, 128, dtype=DTYPE)
    weights = torch.randn(1, 64, 1024, 1, dtype=DTYPE)

    run_op_test(
        torch.ops.tt.indexer_score_dsa,
        [query, key, weights, 0],
        framework=Framework.TORCH,
        custom_comparator=indexer_score_comparator,
    )


# --------------------------------------------------------------------------- #
# tt.topk_large_indices
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "shape, k",
    [
        ((1, 1, 32, 64), 16),
        ((1, 1, 64, 256), 64),
        ((1, 1, 32, 4096), 2048),
    ],
)
def test_topk_large_indices(shape, k):
    # Finite inputs only: with no -inf the CPU reference and the non-Blackhole
    # decomposition agree, so this test is meaningful on every architecture.
    # The sentinel path is covered by test_topk_large_indices_sentinel_contract.
    input_tensor = torch.randn(*shape, dtype=DTYPE)

    run_op_test(
        torch.ops.tt.topk_large_indices,
        [input_tensor, k],
        framework=Framework.TORCH,
        custom_comparator=dsa_topk_comparator,
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_topk_large_indices_production_topk():
    """DSA's real ``index_topk`` over a long row."""
    input_tensor = torch.randn(1, 1, 64, 2048, dtype=DTYPE)
    run_op_test(
        torch.ops.tt.topk_large_indices,
        [input_tensor, 2048],
        framework=Framework.TORCH,
        custom_comparator=dsa_topk_comparator,
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_topk_large_indices_sentinel_contract():
    """The ``-inf`` -> ``0xFFFFFFFF`` contract, which only the kernel implements.

    This is the tripwire for the whole sparse-prefill correctness story: without
    sentinels, ``tt.sparse_sdpa`` attends to future tokens. On non-Blackhole the
    composite inlines a plain ``ttir.topk``, which returns ordinary indices for
    ``-inf`` ties, so the plugin repairs it with
    ``topk_large_indices_mask_invalid_slots`` instead (see
    ``test_dsa_ops_cpu.py::test_mask_invalid_slots_repairs_decomposition_style_indices``).
    """
    import torch_xla
    from tt_torch.custom_ops import dsa_kernels_available

    # Reuse the production predicate rather than get_torch_device_arch(): it is the
    # exact condition the plugin branches on, and it degrades to False instead of
    # raising when the device handle is unusual (e.g. "SPMD:0" once an earlier test
    # in the same process has enabled SPMD).
    if not dsa_kernels_available():
        pytest.skip(
            "topk_large_indices' -inf -> sentinel contract is implemented only by "
            "the Blackhole TTNN kernel; elsewhere the composite inlines a plain "
            "ttir.topk that returns ordinary indices for -inf ties."
        )

    sq, k = 64, 16
    score = _causal_score(1, 8, sq, sq, 128)
    device_idx = torch.ops.tt.topk_large_indices(score.to(torch_xla.device()), k)
    torch_xla.sync()
    signed = device_idx.cpu().to(torch.int32)

    # Row s sees s+1 keys, so it needs max(0, k - (s+1)) sentinels.
    expected = (k - torch.arange(1, sq + 1)).clamp(min=0).to(torch.int32)
    assert torch.equal(
        (signed < 0).sum(-1)[0, 0], expected
    ), "sentinel counts do not match the causal visible counts"

    for s in range(sq):
        row = signed[0, 0, s].tolist()
        sentinel_slots = [i for i, v in enumerate(row) if v < 0]
        if sentinel_slots:
            assert sentinel_slots == list(
                range(sentinel_slots[0], k)
            ), f"row {s} sentinels are not a contiguous tail: {row}"
        assert all(v <= s for v in row if v >= 0), f"row {s} selected a future key"


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("seq_len", [32, 64])
def test_dsa_chain_matches_dense_causal_on_device(seq_len):
    """The whole pipeline on device, against dense causal SDPA.

    With ``topk == seq_len`` top-k covers every causally visible key, so sparse
    attention *is* dense causal attention — the exactness argument the plugin's
    dense-fallback predicates rest on. Run end to end under ``torch.compile`` so it
    exercises the same graph topology the plugin builds, including the
    architecture-appropriate index repair.
    """
    import torch_xla
    from tt_torch.custom_ops import (
        dsa_kernels_available,
        topk_large_indices_mask_invalid_slots,
    )

    topk = seq_len
    num_index_heads, index_head_dim = 8, 128
    num_heads, head_dim, v_dim = 32, 576, 512
    dev = torch_xla.device()
    needs_repair = not dsa_kernels_available(dev)

    def chain(score, query, kv):
        indices = torch.ops.tt.topk_large_indices(score, topk)
        if needs_repair:
            visible = (
                torch.arange(seq_len, dtype=torch.int32, device=score.device) + 1
            ).view(1, 1, seq_len, 1)
            indices = topk_large_indices_mask_invalid_slots(indices, visible)
        return torch.ops.tt.sparse_sdpa(query, kv, indices, v_dim, None, 32)

    g = torch.Generator().manual_seed(0)
    iq = torch.randn(
        1, num_index_heads, seq_len, index_head_dim, dtype=DTYPE, generator=g
    )
    ik = torch.randn(1, 1, seq_len, index_head_dim, dtype=DTYPE, generator=g)
    iw = torch.randn(1, num_index_heads, seq_len, 1, dtype=DTYPE, generator=g)
    score = torch.ops.tt.indexer_score_dsa(iq, ik, iw, 0)
    query = torch.randn(1, num_heads, seq_len, head_dim, dtype=DTYPE, generator=g)
    kv = torch.randn(1, 1, seq_len, head_dim, dtype=DTYPE, generator=g)

    out = torch.compile(chain, backend="tt")(score.to(dev), query.to(dev), kv.to(dev))
    torch_xla.sync()
    out = out.cpu()

    dense = torch.nn.functional.scaled_dot_product_attention(
        query.float(),
        kv.float(),
        kv[..., :v_dim].float(),
        is_causal=True,
        scale=head_dim**-0.5,
        enable_gqa=True,
    )
    assert torch.isfinite(out.float()).all(), "sparse output has NaN/inf"
    pcc = compute_pcc(dense, out.float())
    assert pcc > REQUIRED_PCC, (
        f"DSA chain PCC {pcc} vs dense causal (required > {REQUIRED_PCC}). A low "
        "value here usually means the top-k index repair is wrong and sparse_sdpa "
        "is attending to future tokens."
    )


# --------------------------------------------------------------------------- #
# tt.sparse_sdpa
# --------------------------------------------------------------------------- #
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "num_heads, seq_len, key_seq_len, head_dim, v_dim, topk, k_chunk_size",
    [
        # DeepSeek-V3.2 latent widths: head_dim = kv_lora_rank + qk_rope_head_dim.
        (32, 32, 256, 576, 512, 128, 128),
        (64, 32, 512, 576, 512, 256, 128),
        (32, 64, 128, 576, 512, 128, 32),
    ],
)
def test_sparse_sdpa(
    num_heads, seq_len, key_seq_len, head_dim, v_dim, topk, k_chunk_size
):
    query = torch.randn(1, num_heads, seq_len, head_dim, dtype=DTYPE)
    kv = torch.randn(1, 1, key_seq_len, head_dim, dtype=DTYPE)
    indices = _sorted_indices(seq_len, key_seq_len, topk)

    run_op_test(
        torch.ops.tt.sparse_sdpa,
        [query, kv, indices, v_dim, None, k_chunk_size],
        framework=Framework.TORCH,
    )


@pytest.mark.push
@pytest.mark.single_device
def test_sparse_sdpa_with_sentinel_tail():
    """Masked slots must be ignored — the layout every DSA prefill row has once
    the sequence is shorter than ``index_topk``."""
    num_heads, seq_len, key_seq_len, head_dim, v_dim, topk = 32, 32, 256, 576, 512, 128
    query = torch.randn(1, num_heads, seq_len, head_dim, dtype=DTYPE)
    kv = torch.randn(1, 1, key_seq_len, head_dim, dtype=DTYPE)
    indices = _sorted_indices(seq_len, key_seq_len, topk, sentinel_tail=topk // 4)

    run_op_test(
        torch.ops.tt.sparse_sdpa,
        [query, kv, indices, v_dim, None, 128],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_sparse_sdpa_production_topk():
    """DSA's real ``index_topk`` and latent widths."""
    query = torch.randn(1, 32, 64, 576, dtype=DTYPE)
    kv = torch.randn(1, 1, 2048, 576, dtype=DTYPE)
    indices = _sorted_indices(64, 2048, 2048)

    run_op_test(
        torch.ops.tt.sparse_sdpa,
        [query, kv, indices, 512, None, 128],
        framework=Framework.TORCH,
    )
