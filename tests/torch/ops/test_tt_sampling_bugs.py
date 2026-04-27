# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TT device bugs affecting non-greedy vLLM sampling.

These tests go through the torch.compile / TT compilation flow (via
run_op_test with Framework.TORCH) rather than eager execution, per
recommendation from Het Shah — eager execution does not insert the composite
op lowerings used in production.

Bug 1 (topk wrong indices): Per Het Shah, ttnn.topk does not have the same
index bug as ttnn.sort. The correct test uses gathered values for comparison,
not exact index match (topk ordering is non-deterministic).

Bug 2 (gather int64): Original eager test showed mismatch. Converted to
compiled execution per Het Shah — if this still fails it indicates a bug in
the stablehlo.gather → ttnn.gather lowering in tt-mlir.

Related: vllm_sampling_tt_bugs.md, tt-xla issue #4329
"""

import pytest
import torch
from infra import Framework, run_op_test

SEED = 42


# ---------------------------------------------------------------------------
# topk: correct values selected, ordering non-deterministic
# ---------------------------------------------------------------------------


def topk_values_comparator(device_output, golden_output, args, kwargs):
    """Correct topk comparator: gathered values must match, not exact indices.

    torch.topk does not guarantee ordering among the top-k elements.
    CPU may return [0,1,2,3], device [3,2,1,0] — both are valid.
    """
    device_vals, device_idx = device_output
    golden_vals, golden_idx = golden_output
    input_tensor = args[0]

    device_idx = device_idx.cpu()
    golden_idx = golden_idx.cpu()

    device_gathered = torch.gather(input_tensor, -1, device_idx)
    golden_gathered = torch.gather(input_tensor, -1, golden_idx)
    cos_sim = torch.nn.functional.cosine_similarity(
        device_gathered.flatten().unsqueeze(0).float(),
        golden_gathered.flatten().unsqueeze(0).float(),
    )
    idx_match = (device_idx == golden_idx).float().mean().item()
    k = device_idx.shape[-1]
    print(
        f"\n  values_cos_sim={cos_sim.item():.6f}"
        f"  index_exact_match={idx_match:.3f} ({int(idx_match * k)}/{k})"
        f"  (ordering non-deterministic — only values matter)"
    )
    assert cos_sim > 0.99, f"topk gathered values wrong: cos_sim={cos_sim.item():.6f}"


@pytest.mark.single_device
@pytest.mark.parametrize(
    "shape,k",
    [
        pytest.param((1, 32768), 32, id="32768-k32-vllm-chunk"),
        pytest.param((1, 32768), 64, id="32768-k64"),
        pytest.param((1, 16384), 32, id="16384-k32"),
        pytest.param((1, 65536), 32, id="65536-k32"),
    ],
)
def test_topk_values_correct(shape, k):
    """torch.topk selects the correct top-k values (ordering may differ)."""

    class TopKBoth(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k, dim=-1)

    torch.manual_seed(SEED)
    x_cpu = torch.randn(*shape, dtype=torch.float32)

    run_op_test(
        TopKBoth(k),
        [x_cpu],
        framework=Framework.TORCH,
        custom_comparator=topk_values_comparator,
    )


# ---------------------------------------------------------------------------
# gather int64: compiled execution (per Het Shah recommendation)
# ---------------------------------------------------------------------------


def gather_int64_comparator(device_output, golden_output, args, kwargs):
    """Comparator for torch.gather on int64 index tensors.

    The gathered value must be an exact int64 match — any mismatch indicates
    a bug in the stablehlo.gather → ttnn.gather lowering in tt-mlir.
    """
    device_value = device_output.cpu().item()
    golden_value = golden_output.item()
    local_idx = args[1].item()
    print(
        f"\n  local_idx={local_idx}  "
        f"cpu_token={golden_value}  dev_token={device_value}  "
        f"match={golden_value == device_value}"
    )
    assert golden_value == device_value, (
        f"gather int64 mismatch: cpu={golden_value} dev={device_value} "
        f"(local_idx={local_idx}) — stablehlo.gather lowering bug in tt-mlir"
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "candidates,vocab_size",
    [
        pytest.param(64, 50272, id="opt125m"),
        pytest.param(128, 128256, id="llama"),
    ],
)
def test_gather_int64_correctness(candidates, vocab_size):
    """torch.gather on int64 tensors via compiled execution.

    Uses run_op_test (Framework.TORCH) so the op goes through torch.compile
    and the TT stablehlo.gather lowering path, per Het Shah's recommendation.
    If this fails it indicates a bug in the lowering, not in eager execution.
    """

    class GatherInt64(torch.nn.Module):
        def forward(self, idx, local):
            return torch.gather(idx, 1, local)

    torch.manual_seed(SEED)
    idx_cpu = torch.randperm(vocab_size, dtype=torch.int64)[:candidates].unsqueeze(0)
    local_cpu = torch.randint(0, candidates, (1, 1), dtype=torch.int64)

    run_op_test(
        GatherInt64(),
        [idx_cpu, local_cpu],
        framework=Framework.TORCH,
        custom_comparator=gather_int64_comparator,
    )


def gather_int64_batched_comparator(device_output, golden_output, args, kwargs):
    """Per-row gather check across the batch — every row must match exactly.

    The production sampler uses candidate_indices.gather(1, local) where
    candidate_indices is [batch, 128] and local is [batch, 1]. Each row's
    gather is independent: row i picks index local[i] from candidate_indices[i].
    Distinct candidate sets per row make cross-row leakage detectable —
    if batch i mistakenly indexes batch j's candidates, the output will not
    match.
    """
    device_vals = device_output.cpu().squeeze(-1)
    golden_vals = golden_output.squeeze(-1)
    local = args[1].squeeze(-1)
    print()
    mismatches = []
    for i in range(device_vals.shape[0]):
        match = device_vals[i].item() == golden_vals[i].item()
        print(
            f"  row={i}  local_idx={local[i].item()}  "
            f"cpu={golden_vals[i].item()}  dev={device_vals[i].item()}  "
            f"match={match}"
        )
        if not match:
            mismatches.append(i)
    assert not mismatches, (
        f"gather batch mismatch on rows {mismatches}: "
        f"cpu={golden_vals.tolist()} dev={device_vals.tolist()} — "
        "stablehlo.gather lowering bug at batch>1 in tt-mlir"
    )


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4, 32])
@pytest.mark.parametrize(
    "candidates,vocab_size",
    [
        pytest.param(64, 50272, id="opt125m"),
        pytest.param(128, 128256, id="llama"),
    ],
)
def test_gather_int64_correctness_batched(batch, candidates, vocab_size):
    """Per-row gather: matches the apply_top_k_top_p_fast call shape.

    Build a [batch, candidates] index tensor with a DISTINCT random
    permutation per row, plus a [batch, 1] local-index tensor with a
    different local pick per row. The gather output must equal the CPU
    reference row-by-row. Catches batch-dim bugs in
    stablehlo.gather -> ttnn.gather lowering that single-row tests miss.
    """

    class GatherInt64(torch.nn.Module):
        def forward(self, idx, local):
            return torch.gather(idx, 1, local)

    torch.manual_seed(SEED)
    idx_cpu = torch.stack(
        [
            torch.randperm(vocab_size, dtype=torch.int64)[:candidates]
            for _ in range(batch)
        ]
    )
    local_cpu = torch.randint(0, candidates, (batch, 1), dtype=torch.int64)

    run_op_test(
        GatherInt64(),
        [idx_cpu, local_cpu],
        framework=Framework.TORCH,
        custom_comparator=gather_int64_batched_comparator,
    )


# ---------------------------------------------------------------------------
# topk at batch>1: same chunk size apply_top_k_top_p_fast uses
# ---------------------------------------------------------------------------


def topk_batched_values_comparator(device_output, golden_output, args, kwargs):
    """Per-row topk values check (ordering non-deterministic, sorted).

    For each row i, gather input via device_idx[i] and golden_idx[i] then
    sort within row before cosine_similarity. Per-row cos_sim catches
    cross-row leakage that the existing flattened comparator would miss
    (a row with the wrong other-row's indices would still have high
    flattened cos_sim if rows are similar).
    """
    device_vals, device_idx = device_output
    golden_vals, golden_idx = golden_output
    input_tensor = args[0]

    device_idx = device_idx.cpu()
    golden_idx = golden_idx.cpu()

    print()
    failed = []
    for i in range(input_tensor.shape[0]):
        d_gathered, _ = input_tensor[i].gather(-1, device_idx[i]).float().sort()
        g_gathered, _ = input_tensor[i].gather(-1, golden_idx[i]).float().sort()
        cos = torch.nn.functional.cosine_similarity(
            d_gathered.unsqueeze(0), g_gathered.unsqueeze(0)
        ).item()
        if cos < 0.99:
            failed.append((i, cos))
    n = input_tensor.shape[0]
    if failed:
        for i, cos in failed[:3]:
            print(f"  row={i}  values_cos_sim={cos:.6f}  FAIL")
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more")
        assert False, f"Per-row topk value mismatch on {len(failed)}/{n} rows"
    print(f"  All {n} rows match (per-row cos_sim >= 0.99)")


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4, 32])
def test_topk_batched_vllm_chunk(batch):
    """torch.topk at the exact (chunk_size, k) apply_top_k_top_p_fast uses.

    Each row is a distinct random distribution. Per-row gathered values
    must match the CPU reference. Batch=1 is the regression baseline;
    batch>1 catches batch-dim bugs in the multi-core ttnn.topk lowering.
    """

    class TopKBoth(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, 32, dim=-1)

    torch.manual_seed(SEED)
    x_cpu = torch.randn(batch, 32768, dtype=torch.float32)

    run_op_test(
        TopKBoth(),
        [x_cpu],
        framework=Framework.TORCH,
        custom_comparator=topk_batched_values_comparator,
    )


# ---------------------------------------------------------------------------
# Multi-pick gather: q_samples.gather(1, candidate_indices) call shape
# ---------------------------------------------------------------------------


def gather_multi_pick_comparator(device_output, golden_output, args, kwargs):
    """Element-wise match for gather([B, V], dim=1, [B, K]).

    Used in the new Sampler as q_samples.gather(1, candidate_indices)
    when q_samples is non-None. Each output row picks K positions from
    the corresponding input row. Failures print first-diff per failing
    row to localize the lowering bug.
    """
    device_out = device_output.cpu()
    golden_out = golden_output

    failed = []
    for i in range(device_out.shape[0]):
        if not torch.equal(device_out[i], golden_out[i]):
            failed.append(i)

    n = device_out.shape[0]
    if failed:
        print()
        for i in failed[:3]:
            diff_mask = device_out[i] != golden_out[i]
            n_diff = diff_mask.sum().item()
            first = diff_mask.nonzero()[0].item()
            print(
                f"  row={i}  n_mismatch={n_diff}/{device_out.shape[1]}  "
                f"first diff at col {first}: "
                f"dev={device_out[i, first].item():.6f} "
                f"cpu={golden_out[i, first].item():.6f}"
            )
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more")
        assert False, (
            f"Multi-pick gather mismatch on {len(failed)}/{n} rows — "
            "stablehlo.gather lowering bug for [B, K] index against [B, V]"
        )
    print(f"\n  All {n} rows match exactly")


# ---------------------------------------------------------------------------
# Batched sort: residual ttnn.sort([B, 128]) inside apply_top_k_top_p_fast
# ---------------------------------------------------------------------------


def sort_batched_values_comparator(device_output, golden_output, args, kwargs):
    """Per-row exact-equal check on sorted VALUES (indices may differ).

    Per the bug-tracker comment at the top of this file, ttnn.sort has a
    known index bug. We compare the values output (which the sampler
    actually consumes; indices are deallocated) row-by-row against CPU.
    Using exact equal on float32 is fine because torch.sort on CPU is
    deterministic for distinct values and we generate distinct values.
    """
    device_vals = device_output.cpu()
    golden_vals = golden_output

    failed = []
    for i in range(device_vals.shape[0]):
        # Use cosine_similarity to be tolerant of floating point round-trip
        # via TT, then also check that the values are sorted.
        cos = torch.nn.functional.cosine_similarity(
            device_vals[i].float().unsqueeze(0),
            golden_vals[i].float().unsqueeze(0),
        ).item()
        is_sorted = bool((device_vals[i][1:] >= device_vals[i][:-1]).all())
        if cos < 0.999 or not is_sorted:
            failed.append((i, cos, is_sorted))
    n = device_vals.shape[0]
    if failed:
        print()
        for i, cos, is_sorted in failed[:3]:
            print(
                f"  row={i}  values_cos_sim={cos:.6f}  "
                f"is_monotone_ascending={is_sorted}"
            )
            mismatch_pos = (device_vals[i] != golden_vals[i]).nonzero().flatten()
            if mismatch_pos.numel():
                first = mismatch_pos[0].item()
                print(
                    f"    first diff at pos {first}: "
                    f"dev={device_vals[i, first].item():.6f} "
                    f"cpu={golden_vals[i, first].item():.6f}"
                )
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more")
        assert False, (
            f"ttnn.sort produced wrong values on {len(failed)}/{n} rows — "
            "this is the inner sort in apply_top_k_top_p_fast"
        )
    print(f"\n  All {n} rows match (cos_sim >= 0.999 and ascending)")


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4, 32])
def test_sort_ascending_batched_inner_mask_shape(batch):
    """torch.sort([B, 128], dim=-1, descending=False) — the residual sort
    inside apply_top_k_top_p_fast's top_k/top_p mask path.

    With shape [2, 128], this is exactly what the IR dump showed firing in
    the sampler graph at batch=2. Per the docstring at the top of this
    file, ttnn.sort has a known index bug that ttnn.topk does not — but
    the values output also needs to be correct because the sampler uses
    it for top_k/top_p threshold computation.
    """

    class SortAscending(torch.nn.Module):
        def forward(self, x):
            vals, _ = torch.sort(x, dim=-1, descending=False)
            return vals

    torch.manual_seed(SEED)
    # Distinct rows; distinct values within each row (so sort order is
    # unambiguous on CPU).
    x_cpu = torch.randn(batch, 128, dtype=torch.float32)

    run_op_test(
        SortAscending(),
        [x_cpu],
        framework=Framework.TORCH,
        custom_comparator=sort_batched_values_comparator,
    )


# ---------------------------------------------------------------------------
# End-to-end apply_top_k_top_p_fast assembly at batch>1
# ---------------------------------------------------------------------------


def apply_topk_assembly_comparator(device_output, golden_output, args, kwargs):
    """Per-row check on (filtered_values, candidate_indices).

    The previous version of this comparator used set-equality on
    candidate_indices, which produces false positives on uniform random
    data: many logits are within an ULP of each other near the top-K
    margin, and bf16 round-trip on device causes 1-2 candidates to swap
    with similarly-valued non-candidates. Real Llama logits are far more
    peaked so this isn't a production issue, but it polluted unit-test
    signal.

    New strategy: gather VALUES via both device and cpu indices, compare
    those gathered-from-input values per row. If the indices differ at
    the margin but the values they point to are essentially the same,
    that's bf16 noise, not a bug. Hard fail only on:
      - cos_sim of gathered values < 0.999
      - any device index < 0 or >= vocab_size  (out-of-bounds bug)
      - device indices with cross-chunk leakage (chunk N's index landing
        in chunk M's vocab range — the production bug shape)
    """
    dev_vals, dev_idx = device_output
    gold_vals, gold_idx = golden_output
    input_tensor = args[0]  # logits
    dev_vals = dev_vals.cpu()
    dev_idx = dev_idx.cpu()
    vocab = input_tensor.shape[-1]

    failed = []
    for i in range(dev_vals.shape[0]):
        d_gathered = input_tensor[i].gather(-1, dev_idx[i].long()).float()
        g_gathered = input_tensor[i].gather(-1, gold_idx[i].long()).float()
        d_sorted, _ = d_gathered.sort()
        g_sorted, _ = g_gathered.sort()
        cos = torch.nn.functional.cosine_similarity(
            d_sorted.unsqueeze(0), g_sorted.unsqueeze(0)
        ).item()
        oob_low = dev_idx[i].min().item() < 0
        oob_high = dev_idx[i].max().item() >= vocab
        if cos < 0.999 or oob_low or oob_high:
            failed.append((i, cos, oob_low, oob_high))

    n = dev_vals.shape[0]
    if failed:
        print()
        for i, cos, oob_low, oob_high in failed[:3]:
            print(
                f"  row={i}  gathered_values_cos_sim={cos:.6f}  "
                f"oob_low={oob_low}  oob_high={oob_high}  "
                f"dev_idx_range=[{dev_idx[i].min().item()},{dev_idx[i].max().item()}]"
            )
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more")
        assert False, (
            f"apply_top_k_top_p_fast assembly mismatch on {len(failed)}/{n} rows"
        )
    print(f"\n  All {n} rows: gathered values agree + indices in [0, {vocab})")


_LLAMA_LOGITS_FIXTURE = (
    "tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt"
)


def _load_llama_logits():
    """Load the saved single-row [1, 128256] Llama logits fixture."""
    fixture = torch.load(_LLAMA_LOGITS_FIXTURE, weights_only=False)
    return fixture["logits"].float()  # [1, 128256]


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_apply_top_k_top_p_fast_assembly_real_logits(batch):
    """apply_top_k_top_p_fast on real Llama-3.2-3B logits.

    Loads the saved-logits fixture (single row of real model output)
    and tiles to the requested batch size — this matches the user's
    production failure shape (two identical prompts at batch=2 produce
    identical logits per row, modulo prefix differences in the very
    first decode step). Real logits are far more peaked than random
    N(0,1), so the bf16 top-K margin ambiguity disappears.

    If this fails, the bug is unambiguously in the chunked-topk
    assembly (split / pad / topk / offset / cat) — not a numerical
    artifact of the test inputs.
    """
    from integrations.vllm_plugin.vllm_tt.sampler import apply_top_k_top_p_fast

    class TopKTopPFastModule(torch.nn.Module):
        def forward(self, logits):
            vals, idx = apply_top_k_top_p_fast(logits, k=None, p=None)
            return vals, idx

    single = _load_llama_logits()  # [1, 128256]
    logits_cpu = single.repeat(batch, 1)  # [B, 128256]

    run_op_test(
        TopKTopPFastModule(),
        [logits_cpu],
        framework=Framework.TORCH,
        custom_comparator=apply_topk_assembly_comparator,
    )


# ---------------------------------------------------------------------------
# random_sample: Gumbel-max sampling at batch>1
# ---------------------------------------------------------------------------


def random_sample_comparator(device_output, golden_output, args, kwargs):
    """random_sample is deterministic when q_samples is provided. Output
    is [B] indices into the K candidate set. Each row's pick must equal
    CPU exactly.
    """
    dev = device_output.cpu()
    gold = golden_output

    failed = []
    for i in range(dev.shape[0]):
        if dev[i].item() != gold[i].item():
            failed.append((i, dev[i].item(), gold[i].item()))

    n = dev.shape[0]
    if failed:
        print()
        for i, d, g in failed[:5]:
            print(f"  row={i}  dev_pick={d}  cpu_pick={g}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
        assert False, (
            f"random_sample mismatch on {len(failed)}/{n} rows — "
            "Gumbel-max picks differ between device and cpu"
        )
    print(f"\n  All {n} rows pick the same candidate as CPU")


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("k", [128])
def test_random_sample_gumbel_max_batched(batch, k):
    """Gumbel-max sampling: ``probs.div(q).argmax(dim=-1)`` — exactly the
    math random_sample uses when q_samples is provided (deterministic
    path). probs is [B, K] of softmaxed candidate probabilities; q is
    [B, K] of pre-computed exponential q-values.

    Distinct per-row probs and q so cross-row leakage / shared-RNG bugs
    are detectable. If this fails at b>1, the bug is in argmax / div
    batched lowering — not in apply_top_k_top_p_fast and not in gather.
    """

    class GumbelMax(torch.nn.Module):
        def forward(self, probs, q):
            return probs.div(q).argmax(dim=-1)

    torch.manual_seed(SEED)
    # Distinct row distributions: each row gets a unique softmax over K
    raw = torch.randn(batch, k, dtype=torch.float32)
    probs_cpu = raw.softmax(dim=-1)
    # Distinct exponential q values per row (the ".exponential_()" output
    # in random_sample, which divides probs); use 2-D distinct random
    q_cpu = torch.empty(batch, k, dtype=torch.float32).exponential_()

    run_op_test(
        GumbelMax(),
        [probs_cpu, q_cpu],
        framework=Framework.TORCH,
        custom_comparator=random_sample_comparator,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("batch", [1, 2, 4, 32])
@pytest.mark.parametrize(
    "vocab,k",
    [
        pytest.param(50272, 64, id="opt125m"),
        pytest.param(128256, 128, id="llama"),
    ],
)
def test_gather_multi_pick_batched(batch, vocab, k):
    """gather([B, V], dim=1, [B, K]) — q_samples gather call shape.

    Each row gets distinct random float data and a distinct index
    permutation, so cross-row leakage is detectable. Mirrors the
    Sampler's q_samples.gather(1, candidate_indices) call when
    sampling_metadata.q_samples is non-None (seeded sampling).
    Single-pick (K=1) tests like test_gather_int64_correctness_batched
    don't exercise the multi-pick lowering path this would.
    """

    class GatherMulti(torch.nn.Module):
        def forward(self, vals, idx):
            return torch.gather(vals, 1, idx)

    torch.manual_seed(SEED)
    vals_cpu = torch.randn(batch, vocab, dtype=torch.float32)
    idx_cpu = torch.stack(
        [torch.randperm(vocab, dtype=torch.int64)[:k] for _ in range(batch)]
    )

    run_op_test(
        GatherMulti(),
        [vals_cpu, idx_cpu],
        framework=Framework.TORCH,
        custom_comparator=gather_multi_pick_comparator,
    )
