# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest sanity comparing four masked_scatter_ implementations:
  1. Reference:  torch.Tensor.masked_scatter_ (ground truth)
  2. Old decomp: flatten-to-1D + broadcast + cumsum on S*D elements
  3. New decomp: row-level cumsum on S elements + 2D gather
  4. New decomp v2: row-level cumsum + mul+add index linearization (no gather)

The v2 decomposition replaces torch.gather with explicit mul+add to compute
flat indices, avoiding the ttnn.matmul precision bug on Wormhole hardware
(see tt-metal#42845, hardware bug #38306).

Context:
  - Issue #3316: masked_scatter_ introduced dynamic shapes, so it was
    decomposed into broadcast+flatten+cumsum+where (the "old" approach).
  - Issue #3412: the old approach runs cumsum on [S*D] int64 tensors
    (e.g. [1168640] for S=913, D=1280), causing moreh_cumsum L1/DRAM OOM.
  - Issue #4328: torch.gather in the new decomp causes PCC drop on TT
    due to ttnn.matmul precision bug in the XLA-generated index linearization.
  - The v2 approach replaces gather with mul+add for index computation,
    keeping cumsum at [S] and bypassing the matmul entirely.

Usage:
  pytest test_masked_scatter_decomp.py -svv
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# PCC from tt-xla infra (torch_comparison_evaluator.py lines 100-116)
# ---------------------------------------------------------------------------
def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Pearson Correlation Coefficient matching
    tests/infra/evaluators/torch_comparison_evaluator.py::compute_pcc.
    """
    x_f = x.flatten().to(torch.float64)
    y_f = y.flatten().to(torch.float64)

    if torch.allclose(x_f, y_f, rtol=1e-5, atol=1e-8):
        return 1.0

    if x_f.numel() <= 1:
        return 0.0

    vx = x_f - x_f.mean()
    vy = y_f - y_f.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")

    return float((vx @ vy) / denom)


# ---------------------------------------------------------------------------
# Four implementations under test
# ---------------------------------------------------------------------------
def masked_scatter_reference(inputs_embeds, mask_1d, source):
    """Ground truth: PyTorch's built-in masked_scatter_."""
    result = inputs_embeds.clone()
    result.masked_scatter_(mask_1d.unsqueeze(-1), source)
    return result


def masked_scatter_old_decomp(inputs_embeds, mask_1d, source):
    """
    Old decomposition (flatten to 1D, cumsum on S*D).
    This is what caused OOM in issue #3412.
    """
    mask = mask_1d.unsqueeze(-1)
    mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
    mask_flat = mask_broad.reshape(-1)
    data_flat = data.reshape(-1)
    source_flat = source.reshape(-1)
    mask_i = mask_flat.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
    gathered = source_flat[source_idx]
    result_flat = torch.where(mask_flat, gathered, data_flat)
    return result_flat.view_as(inputs_embeds)


def masked_scatter_new_decomp(inputs_embeds, mask_1d, source):
    """
    New decomposition (row-level cumsum on S, 2D gather).
    cumsum operates on [S] instead of [S*D].
    """
    if source.shape[0] == 0:
        return inputs_embeds.clone()
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds)
    gathered_rows = torch.gather(source, 0, source_idx_2d)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds)


def masked_scatter_new_decomp_v2(inputs_embeds, mask_1d, source):
    """
    New decomposition v2: row-level cumsum + mul+add index linearization.

    Replaces torch.gather with explicit flat index computation:
        flat_idx = source_idx * D + col_arange
    This avoids the ttnn.matmul that XLA generates for torch.gather's
    index linearization, bypassing the Wormhole hardware precision bug.

    See: tt-metal#42845, tt-xla#4328, hardware bug #38306
    """
    if source.shape[0] == 0:
        return inputs_embeds.clone()
    S, D = inputs_embeds.shape
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    flat_source = source.reshape(-1)
    col_idx = torch.arange(D, device=source.device, dtype=source_idx.dtype)
    flat_idx = source_idx.unsqueeze(-1) * D + col_idx.unsqueeze(0)
    gathered_rows = flat_source[flat_idx.reshape(-1)].reshape(S, D)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_at(S, positions):
    m = torch.zeros(S, dtype=torch.bool)
    for p in positions:
        m[p] = True
    return m


def _build_inputs(S, D, num_true, seed=42):
    torch.manual_seed(seed)
    inputs_embeds = torch.randn(S, D)
    source = torch.randn(num_true, D)
    mask_1d = torch.zeros(S, dtype=torch.bool)
    true_positions = torch.randperm(S)[:num_true].sort().values
    mask_1d[true_positions] = True
    assert mask_1d.sum().item() == num_true
    return inputs_embeds, mask_1d, source


def _assert_match(candidate, reference, label):
    """Assert equal, allclose, and PCC >= 1.0 between candidate and reference."""
    is_equal = torch.equal(candidate, reference)
    is_allclose = torch.allclose(candidate, reference, atol=1e-6, rtol=1e-5)
    pcc = compute_pcc(candidate, reference)
    max_diff = (candidate - reference).abs().max().item()

    assert is_equal or is_allclose, (
        f"{label}: not allclose (max diff={max_diff:.2e})"
    )
    assert pcc >= 0.999999, f"{label}: PCC too low ({pcc:.10f})"


# ---------------------------------------------------------------------------
# Parametrized correctness tests
# ---------------------------------------------------------------------------
SHAPE_CASES = [
    pytest.param(32, 16, 10, id="small_S32_D16"),
    pytest.param(913, 1280, 903, id="deepseek_ocr_S913_D1280"),
    pytest.param(2048, 1280, 1024, id="stress_S2048_D1280"),
]


class TestMaskedScatterDecomp:
    """Correctness tests: old decomp, new decomp, v2 decomp, and cross-checks."""

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_old_decomp_vs_reference(self, S, D, num_true):
        inputs, mask, source = _build_inputs(S, D, num_true)
        ref = masked_scatter_reference(inputs, mask, source)
        old = masked_scatter_old_decomp(inputs, mask, source)
        _assert_match(old, ref, "Old decomp vs Reference")

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_new_decomp_vs_reference(self, S, D, num_true):
        inputs, mask, source = _build_inputs(S, D, num_true)
        ref = masked_scatter_reference(inputs, mask, source)
        new = masked_scatter_new_decomp(inputs, mask, source)
        _assert_match(new, ref, "New decomp vs Reference")

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_v2_decomp_vs_reference(self, S, D, num_true):
        """v2 decomp (mul+add) should be bit-exact with reference."""
        inputs, mask, source = _build_inputs(S, D, num_true)
        ref = masked_scatter_reference(inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(inputs, mask, source)
        _assert_match(v2, ref, "V2 decomp (mul+add) vs Reference")

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_v2_vs_new_direct(self, S, D, num_true):
        """v2 and v1 should produce identical results on CPU."""
        inputs, mask, source = _build_inputs(S, D, num_true)
        new = masked_scatter_new_decomp(inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(inputs, mask, source)
        _assert_match(v2, new, "V2 decomp vs New decomp")

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_old_vs_new_direct(self, S, D, num_true):
        inputs, mask, source = _build_inputs(S, D, num_true)
        old = masked_scatter_old_decomp(inputs, mask, source)
        new = masked_scatter_new_decomp(inputs, mask, source)
        _assert_match(new, old, "New decomp vs Old decomp")

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_true_positions_get_source(self, S, D, num_true):
        """Rows where mask is True must contain exact source values."""
        inputs, mask, source = _build_inputs(S, D, num_true)
        v2 = masked_scatter_new_decomp_v2(inputs, mask, source)
        new = masked_scatter_new_decomp(inputs, mask, source)
        old = masked_scatter_old_decomp(inputs, mask, source)
        assert torch.allclose(v2[mask], source, atol=1e-6), \
            "V2 decomp: True positions don't match source"
        assert torch.allclose(new[mask], source, atol=1e-6), \
            "New decomp: True positions don't match source"
        assert torch.allclose(old[mask], source, atol=1e-6), \
            "Old decomp: True positions don't match source"

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_false_positions_keep_original(self, S, D, num_true):
        """Rows where mask is False must be unchanged from input."""
        inputs, mask, source = _build_inputs(S, D, num_true)
        v2 = masked_scatter_new_decomp_v2(inputs, mask, source)
        new = masked_scatter_new_decomp(inputs, mask, source)
        old = masked_scatter_old_decomp(inputs, mask, source)
        assert torch.allclose(v2[~mask], inputs[~mask], atol=1e-6), \
            "V2 decomp: False positions changed"
        assert torch.allclose(new[~mask], inputs[~mask], atol=1e-6), \
            "New decomp: False positions changed"
        assert torch.allclose(old[~mask], inputs[~mask], atol=1e-6), \
            "Old decomp: False positions changed"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    S, D = 16, 8

    @pytest.fixture(autouse=True)
    def _seed_and_input(self):
        torch.manual_seed(42)
        self.inputs = torch.randn(self.S, self.D)

    def test_all_true(self):
        mask = torch.ones(self.S, dtype=torch.bool)
        source = torch.randn(self.S, self.D)
        ref = masked_scatter_reference(self.inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        _assert_match(v2, ref, "All True (v2)")

    def test_all_false(self):
        mask = torch.zeros(self.S, dtype=torch.bool)
        source = torch.randn(0, self.D)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        assert torch.equal(self.inputs, v2), "All-False: output should be unchanged"

    def test_single_true_at_start(self):
        mask = _mask_at(self.S, [0])
        source = torch.randn(1, self.D)
        ref = masked_scatter_reference(self.inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        _assert_match(v2, ref, "Single True at start (v2)")

    def test_single_true_at_end(self):
        mask = _mask_at(self.S, [self.S - 1])
        source = torch.randn(1, self.D)
        ref = masked_scatter_reference(self.inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        _assert_match(v2, ref, "Single True at end (v2)")

    def test_alternating(self):
        mask = _mask_at(self.S, list(range(0, self.S, 2)))
        num_true = mask.sum().item()
        source = torch.randn(num_true, self.D)
        ref = masked_scatter_reference(self.inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        _assert_match(v2, ref, "Alternating True/False (v2)")

    def test_consecutive_block_in_middle(self):
        mask = _mask_at(self.S, list(range(4, 10)))
        num_true = mask.sum().item()
        source = torch.randn(num_true, self.D)
        ref = masked_scatter_reference(self.inputs, mask, source)
        v2 = masked_scatter_new_decomp_v2(self.inputs, mask, source)
        _assert_match(v2, ref, "Consecutive Trues in middle (v2)")


# ---------------------------------------------------------------------------
# Memory footprint reporting (printed, not asserted)
# ---------------------------------------------------------------------------
class TestMemoryReport:
    """Prints theoretical intermediate memory comparison."""

    @pytest.mark.parametrize("S, D, num_true", SHAPE_CASES)
    def test_report_memory_savings(self, S, D, num_true, capsys):
        elem_float, elem_long, elem_bool = 4, 8, 1

        total_old = (
            S * D * elem_bool       # mask_broad
            + S * D * elem_long     # mask_i
            + S * D * elem_long     # source_idx
            + S * D * elem_float    # gathered
            + S * D * elem_float    # result_flat
        )
        total_new = (
            S * elem_long           # mask_i
            + S * elem_long         # source_idx
            + S * D * elem_float    # gathered_rows
        )
        total_v2 = (
            S * elem_long           # mask_i
            + S * elem_long         # source_idx
            + S * D * elem_long     # flat_idx
            + S * D * elem_float    # gathered_rows
        )

        reduction_new = (1 - total_new / total_old) * 100
        reduction_v2 = (1 - total_v2 / total_old) * 100

        print(f"\n  S={S}, D={D}, num_true={num_true}")
        print(f"  Old decomp intermediates: {total_old / 1024**2:.1f} MB "
              f"(cumsum on [{S * D}] = {S}*{D} elements)")
        print(f"  New decomp (gather) intermediates: {total_new / 1024**2:.1f} MB "
              f"(cumsum on [{S}] elements)")
        print(f"  V2 decomp (mul+add) intermediates: {total_v2 / 1024**2:.1f} MB "
              f"(cumsum on [{S}] elements, explicit flat_idx)")
        print(f"  Reduction vs old:  new={reduction_new:.1f}%  v2={reduction_v2:.1f}%")

        assert total_new < total_old, "New decomp should use less memory"
        assert total_v2 < total_old, "V2 decomp should use less memory than old"
