# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Shardy ``sdy.op_sharding_rule`` builder and the
``@tt_torch.tt_lang_operation(sharding_rule=...)`` decorator plumbing.

Most tests here cover only the Python-side surface (string building,
decorator/dispatch plumbing) and need no hardware. The MLIR pass that
promotes the frontend attribute into a real ``sdy.op_sharding_rule`` and
hands it to Shardy is covered by lit tests in tt-mlir
(``.../shardy/op_propagation_registry/tt_lang_custom_rule*.mlir``).

The multi-device e2e tests compile a tt-lang kernel carrying a custom
``sharding_rule``, run it across a device mesh, and compare against a CPU
golden -- one case where the rule allows dim-0 propagation, and one where
``need_replication`` blocks it. They are gated behind multi-device markers,
and their device-only imports (torch_xla SPMD, infra) are done lazily inside
the tests so the Python-only tests above still collect and run on machines
without a device. (``ttl`` is imported at module scope because tt-lang
kernels must resolve it as a global; it is a required dependency, so this
does not affect the Python-only tests.)
"""

import pytest
import torch
import tt_torch  # noqa: F401  -- registers torch.ops.tt.tt_lang_op
import ttl
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import sdy
from tt_torch import make_sharding_rule
from tt_torch import tt_lang as tt_lang_mod
from tt_torch.sharding_rule import make_fully_replicated_sharding_rule
from tt_torch.tt_lang import tt_lang_operation


@pytest.fixture
def clean_registry():
    saved = list(tt_lang_mod.iter_registered_operations())
    tt_lang_mod._clear_registry_for_tests()
    try:
        yield
    finally:
        tt_lang_mod._clear_registry_for_tests()
        for entry in saved:
            tt_lang_mod._register(entry)


def assert_rule(
    text,
    *,
    factor_sizes,
    operands,
    results,
    reduction=(),
    need_replication=(),
    permutation=(),
    blocked=(),
):
    """Compare a printed rule to a structurally-built ``OpShardingRuleAttr``.

    Avoids brittle exact-string checks that break on jaxlib printer changes.
    Always expects ``is_custom=True``; ``make_sharding_rule`` hard-codes it.
    """
    with ir.Context() as ctx, ir.Location.unknown():
        sdy.register_dialect(ctx)

        def tensor_mapping(dims):
            return sdy.TensorMappingAttr.get(
                [sdy.DimMappingAttr.get([i]) for i in dims]
            )

        want = sdy.OpShardingRuleAttr.get(
            factor_sizes=list(factor_sizes),
            operand_mappings=[tensor_mapping(d) for d in operands],
            result_mappings=[tensor_mapping(d) for d in results],
            reduction_factors=list(reduction),
            need_replication_factors=list(need_replication),
            permutation_factors=list(permutation),
            blocked_propagation_factors=list(blocked),
            is_custom=True,
        )
        assert ir.Attribute.parse(text) == want


# ---------------------------------------------------------------------------
# make_sharding_rule: happy-path structure (via Shardy attr equality)
# ---------------------------------------------------------------------------


def test_make_sharding_rule_matmul_shape():
    """A canonical matmul rule matches the OpShardingRuleAttr Shardy builds."""
    rule = make_sharding_rule(
        operand_mappings=[("B", "M"), ("M", "N")],
        result_mappings=[("B", "N")],
        factor_sizes={"B": 8, "M": 16, "N": 32},
        reduction_factors=["M"],
    )
    assert_rule(
        rule,
        factor_sizes=[8, 16, 32],
        operands=[[0, 1], [1, 2]],
        results=[[0, 2]],
        reduction=[1],
    )


def test_make_sharding_rule_pointwise_no_reduction():
    rule = make_sharding_rule(
        operand_mappings=[("i", "j"), ("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": 8, "j": 8},
    )
    assert_rule(
        rule,
        factor_sizes=[8, 8],
        operands=[[0, 1], [0, 1]],
        results=[[0, 1]],
    )


def test_make_sharding_rule_all_factor_types():
    rule = make_sharding_rule(
        operand_mappings=[("a", "b", "c", "d")],
        result_mappings=[("a", "b", "c", "d")],
        factor_sizes={"a": 2, "b": 3, "c": 4, "d": 5},
        reduction_factors=["b"],
        need_replication_factors=["c"],
        permutation_factors=["d"],
        blocked_propagation_factors=["a"],
    )
    assert_rule(
        rule,
        factor_sizes=[2, 3, 4, 5],
        operands=[[0, 1, 2, 3]],
        results=[[0, 1, 2, 3]],
        reduction=[1],
        need_replication=[2],
        permutation=[3],
        blocked=[0],
    )


def test_make_sharding_rule_scalar_operand():
    rule = make_sharding_rule(
        operand_mappings=[()],
        result_mappings=[()],
        factor_sizes={},
    )
    assert_rule(rule, factor_sizes=[], operands=[[]], results=[[]])


def test_make_sharding_rule_uses_first_appearance_order_for_symbols():
    """Factor letters are assigned by first-appearance order across mappings,
    not by insertion order in ``factor_sizes``. This mirrors the underlying
    shardy convention that expects ``i, j, k, ...`` in printed factor sizes."""
    # We list N in factor_sizes first, but M appears first in operand_mappings.
    rule = make_sharding_rule(
        operand_mappings=[("M",)],
        result_mappings=[("M", "N")],
        factor_sizes={"N": 4, "M": 8},
    )
    assert_rule(rule, factor_sizes=[8, 4], operands=[[0]], results=[[0, 1]])


def test_make_sharding_rule_factor_sizes_dict_key_order_does_not_leak():
    """Even for a factor that only appears in ``factor_sizes`` and no
    mapping, the index is assigned last so ``i, j, k, ...`` ordering
    is preserved across the used prefix."""
    rule = make_sharding_rule(
        operand_mappings=[("a",)],
        result_mappings=[("a",)],
        factor_sizes={"b": 2, "a": 4},  # 'b' declared first, but unused
    )
    assert_rule(rule, factor_sizes=[4, 2], operands=[[0]], results=[[0]])


def test_make_sharding_rule_reduction_over_many_factors():
    rule = make_sharding_rule(
        operand_mappings=[("i", "j", "k")],
        result_mappings=[("i",)],
        factor_sizes={"i": 4, "j": 8, "k": 16},
        reduction_factors=["j", "k"],
    )
    assert_rule(
        rule,
        factor_sizes=[4, 8, 16],
        operands=[[0, 1, 2]],
        results=[[0]],
        reduction=[1, 2],
    )


def test_make_fully_replicated_sharding_rule_pointwise():
    # Functional SHLO form: only "in" tensors are rule operands; "out"
    # shapes appear as results.
    rule = make_fully_replicated_sharding_rule(
        [(8, 16), (8, 16), (8, 16)],
        out_indices=[2],
    )
    assert_rule(
        rule,
        factor_sizes=[8, 16],
        operands=[[0, 1], [0, 1]],
        results=[[0, 1]],
        need_replication=[0, 1],
    )


def test_make_fully_replicated_sharding_rule_distinct_shapes():
    rule = make_fully_replicated_sharding_rule(
        [(4,), (4, 8), (4, 8)],
        out_indices=[2],
    )
    assert_rule(
        rule,
        factor_sizes=[4, 4, 8],
        operands=[[0], [1, 2]],
        results=[[1, 2]],
        need_replication=[0, 1, 2],
    )


# ---------------------------------------------------------------------------
# make_sharding_rule: input validation
# ---------------------------------------------------------------------------


def test_make_sharding_rule_rejects_factor_not_in_sizes():
    with pytest.raises(ValueError, match="not present in factor_sizes"):
        make_sharding_rule(
            operand_mappings=[("i",)],
            result_mappings=[("j",)],
            factor_sizes={"i": 4},
        )


def test_make_sharding_rule_rejects_non_positive_factor_size():
    with pytest.raises(ValueError, match="must be a positive int"):
        make_sharding_rule(
            operand_mappings=[("i",)],
            result_mappings=[("i",)],
            factor_sizes={"i": 0},
        )


def test_make_sharding_rule_rejects_duplicate_in_reduction_list():
    with pytest.raises(ValueError, match="duplicate factor"):
        make_sharding_rule(
            operand_mappings=[("i", "j")],
            result_mappings=[("i",)],
            factor_sizes={"i": 4, "j": 8},
            reduction_factors=["j", "j"],
        )


def test_make_sharding_rule_rejects_unknown_factor_in_reduction_list():
    with pytest.raises(ValueError, match="not declared in factor_sizes"):
        make_sharding_rule(
            operand_mappings=[("i",)],
            result_mappings=[("i",)],
            factor_sizes={"i": 4},
            reduction_factors=["j"],
        )


def test_make_sharding_rule_string_is_repeatably_stringifiable():
    """The returned value is a string. ``str()`` returns itself, so a
    caller can safely pass it through anything that accepts either the
    builder result or arbitrary text."""
    rule = make_sharding_rule(
        operand_mappings=[("i",)],
        result_mappings=[("i",)],
        factor_sizes={"i": 4},
    )
    assert isinstance(rule, str)
    assert str(rule) == rule


# ---------------------------------------------------------------------------
# @tt_lang_operation decoration accepts the new sharding_rule kwarg
# ---------------------------------------------------------------------------


def test_decorator_stores_sharding_rule_string(clean_registry):
    rule = "#sdy.op_sharding_rule<([i]) -> ([i]) {i=2} custom>"

    @tt_lang_operation(
        operation_id="unit.sharding_rule.decorate.v1",
        arg_roles=("in", "out"),
        sharding_rule=rule,
    )
    def k(x, out): ...

    assert k._tt_lang_sharding_rule == rule


# ---------------------------------------------------------------------------
# torch.ops.tt.tt_lang_op schema still accepts the old-style positional call
# ---------------------------------------------------------------------------
def test_tt_lang_op_schema_accepts_new_sharding_rule_arg():
    """The new ``sharding_rule`` positional argument is accepted."""
    a = torch.zeros(2, 3)
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.tt.tt_lang_op(
            [a],
            "k",
            "out",
            "vt0",
            "",
            [0],
            "#sdy.op_sharding_rule<([i, j]) -> ([i, j]) {i=2, j=3} custom>",
        )


# ---------------------------------------------------------------------------
# tt_lang_op_dispatch -- keyword-only wrapper with the new kwarg
# ---------------------------------------------------------------------------


def test_tt_lang_op_dispatch_forwards_sharding_rule(monkeypatch):
    captured: list[tuple] = []

    def _fake_torch_op(*args):
        captured.append(args)
        return [args[0][i].clone() for i in args[5]]

    monkeypatch.setattr(torch.ops.tt, "tt_lang_op", _fake_torch_op)

    a = torch.zeros(2)
    from tt_torch.custom_ops import tt_lang_op_dispatch

    tt_lang_op_dispatch(
        [a],
        kernel_id="k",
        arg_roles="out",
        version_tag="vt0",
        shard_spec="",
        out_indices=[0],
        sharding_rule="#sdy.op_sharding_rule<([i]) -> ([i]) {i=2} custom>",
    )

    assert len(captured) == 1
    args = captured[0]
    # Positional layout: (tensors, kernel_id, arg_roles, version_tag,
    #                    shard_spec, out_indices, sharding_rule)
    assert args[6] == "#sdy.op_sharding_rule<([i]) -> ([i]) {i=2} custom>"


def test_tt_lang_op_dispatch_default_sharding_rule_is_empty(monkeypatch):
    captured: list[tuple] = []

    def _fake_torch_op(*args):
        captured.append(args)
        return [args[0][i].clone() for i in args[5]]

    monkeypatch.setattr(torch.ops.tt, "tt_lang_op", _fake_torch_op)

    a = torch.zeros(2)
    from tt_torch.custom_ops import tt_lang_op_dispatch

    tt_lang_op_dispatch(
        [a],
        kernel_id="k",
        arg_roles="out",
        version_tag="vt0",
        shard_spec="",
        out_indices=[0],
    )

    assert len(captured) == 1
    # Dispatch forwards ""; tt_lang_op synthesizes the replication rule.
    assert captured[0][6] == ""


def test_resolve_empty_sharding_rule_warns_and_replicates():
    """Empty sharding_rule -> warn + explicit full-replication rule."""
    from tt_torch.custom_ops import _resolve_tt_lang_sharding_rule

    a = torch.zeros(4, 8)
    b = torch.zeros(4, 8)
    out = torch.zeros(4, 8)
    with pytest.warns(UserWarning, match="no sharding_rule provided"):
        rule = _resolve_tt_lang_sharding_rule(
            "", [a, b, out], [2], kernel_id="unit.empty_rule"
        )
    assert_rule(
        rule,
        factor_sizes=[4, 8],
        operands=[[0, 1], [0, 1]],
        results=[[0, 1]],
        need_replication=[0, 1],
    )


def test_resolve_nonempty_sharding_rule_passthrough():
    from tt_torch.custom_ops import _resolve_tt_lang_sharding_rule

    a = torch.zeros(2)
    rule = "#sdy.op_sharding_rule<([i])->([i]) {i=2}, custom>"
    assert (
        _resolve_tt_lang_sharding_rule(rule, [a], [0], kernel_id="unit.passthrough")
        == rule
    )


# ---------------------------------------------------------------------------
# End-to-end hardware tests: a tt-lang kernel carrying a custom sharding_rule
# runs across a device mesh and matches the CPU golden.
#
# These are the only tests in the file that need real hardware and the
# tt-lang kernel compiler. All hardware imports are kept lazy (inside the
# builder / test) so the pure-Python tests above still collect on machines
# without a device or ``ttl``.
# ---------------------------------------------------------------------------

# Per-shard rows must be a multiple of TILE_SIZE * GRANULARITY (the kernel
# stages 2-tile row blocks), so each device's row-shard has to be % 64.
TILE_SIZE = 32
GRANULARITY = 2


def _make_sharded_eltwise_add_operation(operation_id: str, sharding_rule: str):
    """Build the tt-lang elementwise-add op wrapped with a custom sharding_rule.

    Kernel body is the 2-tile-block pipelined add from
    ``tests/torch/ops/test_tt_lang_kernel_e2e.py`` (itself cribbed from
    tt-lang's ``examples/eltwise_add.py``). The only difference is that the
    ``@tt_lang_operation`` wrapper carries ``sharding_rule=`` so Shardy can
    propagate a dim-0 sharding through the ``stablehlo.custom_call`` and run
    the kernel on row-shards.
    """

    @ttl.operation(grid="auto")
    def _ttl_add(a_in, b_in, out):
        row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
        col_tiles = a_in.shape[1] // TILE_SIZE

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows_per_node = -(-row_tiles // grid_rows)
        cols_per_node = -(-col_tiles // grid_cols)

        a_dfb = ttl.make_dataflow_buffer_like(
            a_in, shape=(GRANULARITY, 1), block_count=2
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_in, shape=(GRANULARITY, 1), block_count=2
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(GRANULARITY, 1), block_count=2
        )

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                                out_dfb.reserve() as out_blk,
                            ):
                                out_blk.store(a_blk + b_blk)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                tx_a = ttl.copy(a_in[r0:r1, col : col + 1], a_blk)
                                tx_b = ttl.copy(b_in[r0:r1, col : col + 1], b_blk)
                                tx_a.wait()
                                tx_b.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with out_dfb.wait() as out_blk:
                                tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                                tx.wait()

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "out"),
        version_tag="sharding-e2e-v1",
        sharding_rule=sharding_rule,
    )
    def add_op(a, b, out):
        return _ttl_add(a, b, out)

    return add_op


def _eltwise_add_e2e_setup(num_devices: int):
    """Shared mesh/shape setup for the multi-device eltwise-add e2e tests."""
    import numpy as np
    from torch_xla.distributed.spmd import Mesh

    # Global shape: one row-block (64 rows) per device after sharding dim 0.
    rows = TILE_SIZE * GRANULARITY * num_devices
    cols = 64
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
    return rows, cols, mesh


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.llmbox
def test_tt_lang_eltwise_add_custom_sharding_rule_reshards_unmarked_operand_e2e(
    clean_registry,
):
    """End-to-end: an unmarked operand is explicitly resharded to match the
    sharding the custom rule propagates onto its sibling and the result.

    Only ``a`` is marked sharded on dim 0; ``b`` is left unmarked
    (replicated). The rule ties both operands' dim 0 to the same factor
    with no ``need_replication``, so Shardy propagates ``a``'s sharding to
    the result -- and, since the rule links them, to ``b`` as well. ``b``
    starts replicated, so ``insert-explicit-reshards`` has to insert an
    ``sdy.reshard`` on it before the custom call runs; this is the hardware
    counterpart to the tt-mlir lit test
    ``tt_lang_custom_rule_reshard.mlir::operand_needs_reshard``.

    A correct result against the CPU golden is proof the reshard actually
    happened: without it, ``a``'s and ``b``'s per-device row blocks would be
    misaligned (one sharded, one still full-size) and the kernel would add
    mismatched rows.
    """
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
    from infra.utilities.torch_multichip_utils import enable_spmd

    enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    if num_devices < 2:
        pytest.skip(f"needs a multi-device mesh, got {num_devices} device(s)")

    rows, cols, mesh = _eltwise_add_e2e_setup(num_devices)

    # Same pointwise, no-need_replication rule as the propagation test above:
    # dim 0 ("i") is a shared factor between the two operands and the result,
    # so whatever sharding one operand gets, the others must match it.
    sharding_rule = make_sharding_rule(
        operand_mappings=[("i", "j"), ("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": rows, "j": cols},
    )

    add_op = _make_sharded_eltwise_add_operation(
        f"tt_xla.e2e.reshard_eltwise_add.{rows}x{cols}.v1", sharding_rule
    )

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    golden = a_cpu + b_cpu

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = torch.zeros_like(a_cpu).to(device)

    # Only `a` gets a host sharding annotation. `b` is left unmarked
    # (replicated) so its alignment with `a` has to come from the rule
    # driving propagation plus an explicit reshard, not a host annotation.
    xs.mark_sharding(a_xla, mesh, ("model", None))
    input_sharding = torch_xla._XLAC._get_xla_sharding_spec(a_xla)
    assert input_sharding, "expected a non-empty sharding spec on marked input a"
    assert not torch_xla._XLAC._get_xla_sharding_spec(
        b_xla
    ), "b must start unmarked so its alignment comes from the reshard, not a host annotation"

    result_xla = add_op(a_xla, b_xla, out_xla)
    torch_xla.sync()

    result_sharding = torch_xla._XLAC._get_xla_sharding_spec(result_xla)
    assert (
        result_sharding == input_sharding
    ), f"result sharding not propagated from a: {result_sharding!r} vs {input_sharding!r}"

    result = result_xla.to("cpu")
    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.llmbox
def test_tt_lang_eltwise_add_need_replication_blocks_propagation_e2e(
    clean_registry,
):
    """End-to-end: ``need_replication`` still produces a correct result.

    Same eltwise-add kernel as the sharded e2e, but the rule marks every
    factor as ``need_replication``. Operands are pre-sharded like the
    propagation test; Shardy must reshard them to replicated for the op.
    IR-level proof that the result stays replicated is in tt-mlir lit
    coverage; this test checks the hardware path still matches the golden.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
    from infra.utilities.torch_multichip_utils import enable_spmd

    enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    if num_devices < 2:
        pytest.skip(f"needs a multi-device mesh, got {num_devices} device(s)")

    rows, cols, mesh = _eltwise_add_e2e_setup(num_devices)

    sharding_rule = make_sharding_rule(
        operand_mappings=[("i", "j"), ("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": rows, "j": cols},
        need_replication_factors=["i", "j"],
    )

    add_op = _make_sharded_eltwise_add_operation(
        f"tt_xla.e2e.need_replication_eltwise_add.{rows}x{cols}.v1",
        sharding_rule,
    )

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    golden = a_cpu + b_cpu

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = torch.zeros_like(a_cpu).to(device)

    # Inputs arrive sharded; need_replication forces a reshard to
    # replicated before the kernel. Mark ``out`` for host copy_/gather.
    xs.mark_sharding(a_xla, mesh, ("model", None))
    xs.mark_sharding(b_xla, mesh, ("model", None))
    xs.mark_sharding(out_xla, mesh, ("model", None))

    result_xla = add_op(a_xla, b_xla, out_xla)
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)
