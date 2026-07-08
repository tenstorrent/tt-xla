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

The final test, ``test_tt_lang_eltwise_add_custom_sharding_rule_e2e``, is a
real multi-device hardware test: it compiles a tt-lang kernel carrying a
custom ``sharding_rule``, runs it sharded across a device mesh, and compares
the gathered result against a CPU golden. It is gated behind multi-device
markers, and its device-only imports (torch_xla SPMD, infra) are done lazily
inside the test so the Python-only tests above still collect and run on
machines without a device. (``ttl`` is imported at module scope because
tt-lang kernels must resolve it as a global; it is a required dependency, so
this does not affect the Python-only tests.)
"""

import pytest
import torch
import tt_torch  # noqa: F401  -- registers torch.ops.tt.tt_lang_op

import ttl
from tt_torch import make_sharding_rule
from tt_torch import tt_lang as tt_lang_mod
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


# ---------------------------------------------------------------------------
# make_sharding_rule: happy-path formatting
#
# make_sharding_rule builds the rule through Shardy's own MLIR bindings, so
# the expected strings below are exactly what Shardy's canonical printer
# emits (note ``->`` has no surrounding spaces, unlike hand-written text).
# ---------------------------------------------------------------------------


def test_make_sharding_rule_matmul_shape():
    """A canonical matmul rule matches the printer format Shardy uses."""
    rule = make_sharding_rule(
        operand_mappings=[("B", "M"), ("M", "N")],
        result_mappings=[("B", "N")],
        factor_sizes={"B": 8, "M": 16, "N": 32},
        reduction_factors=["M"],
    )
    assert rule == (
        "#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) "
        "{i=8, j=16, k=32} reduction={j}, custom>"
    )


def test_make_sharding_rule_pointwise_no_reduction():
    rule = make_sharding_rule(
        operand_mappings=[("i", "j"), ("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": 8, "j": 8},
    )
    assert rule == (
        "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=8}, custom>"
    )


def test_make_sharding_rule_all_factor_types_and_no_custom():
    rule = make_sharding_rule(
        operand_mappings=[("a", "b", "c", "d")],
        result_mappings=[("a", "b", "c", "d")],
        factor_sizes={"a": 2, "b": 3, "c": 4, "d": 5},
        reduction_factors=["b"],
        need_replication_factors=["c"],
        permutation_factors=["d"],
        blocked_propagation_factors=["a"],
        is_custom=False,
    )
    assert rule == (
        "#sdy.op_sharding_rule<([i, j, k, l])->([i, j, k, l]) "
        "{i=2, j=3, k=4, l=5} reduction={j} need_replication={k} "
        "permutation={l} blocked_propagation={i}>"
    )


def test_make_sharding_rule_scalar_operand():
    rule = make_sharding_rule(
        operand_mappings=[()],
        result_mappings=[()],
        factor_sizes={},
    )
    assert rule == "#sdy.op_sharding_rule<([])->([]), custom>"


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
    assert rule == "#sdy.op_sharding_rule<([i])->([i, j]) {i=8, j=4}, custom>"


def test_make_sharding_rule_factor_sizes_dict_key_order_does_not_leak():
    """Even for a factor that only appears in ``factor_sizes`` and no
    mapping, the index is assigned last so ``i, j, k, ...`` ordering
    is preserved across the used prefix."""
    rule = make_sharding_rule(
        operand_mappings=[("a",)],
        result_mappings=[("a",)],
        factor_sizes={"b": 2, "a": 4},  # 'b' declared first, but unused
    )
    assert rule == "#sdy.op_sharding_rule<([i])->([i]) {i=4, j=2}, custom>"


def test_make_sharding_rule_reduction_over_many_factors():
    rule = make_sharding_rule(
        operand_mappings=[("i", "j", "k")],
        result_mappings=[("i",)],
        factor_sizes={"i": 4, "j": 8, "k": 16},
        reduction_factors=["j", "k"],
    )
    assert rule == (
        "#sdy.op_sharding_rule<([i, j, k])->([i]) {i=4, j=8, k=16} "
        "reduction={j, k}, custom>"
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


def test_decorator_stores_sharding_rule_from_builder(clean_registry):
    rule = make_sharding_rule(
        operand_mappings=[("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": 4, "j": 4},
    )

    @tt_lang_operation(
        operation_id="unit.sharding_rule.decorate.builder.v1",
        arg_roles=("in", "out"),
        sharding_rule=rule,
    )
    def k(x, out): ...

    assert k._tt_lang_sharding_rule == rule


def test_decorator_default_sharding_rule_is_empty(clean_registry):
    """Not passing ``sharding_rule=`` is legal (backwards-compatible)."""

    @tt_lang_operation(
        operation_id="unit.sharding_rule.default.v1", arg_roles=("in", "out")
    )
    def k(x, out): ...

    assert k._tt_lang_sharding_rule == ""


# ---------------------------------------------------------------------------
# torch.ops.tt.tt_lang_op schema still accepts the old-style positional call
# ---------------------------------------------------------------------------


def test_tt_lang_op_schema_still_accepts_old_call():
    """Adding ``sharding_rule`` (default ``""``) must not break existing
    six-arg call sites that predate this parameter."""
    a = torch.zeros(2, 3)
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.ops.tt.tt_lang_op([a], "k", "out", "vt0", "", [0])


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
    assert captured[0][6] == ""


# ---------------------------------------------------------------------------
# End-to-end hardware test: a tt-lang kernel carrying a custom sharding_rule
# runs sharded across a device mesh and matches the CPU golden.
#
# This is the only test in the file that needs real hardware and the tt-lang
# kernel compiler. All hardware imports are kept lazy (inside the builder /
# test) so the pure-Python tests above still collect on machines without a
# device or ``ttl``.
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


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.llmbox
def test_tt_lang_eltwise_add_custom_sharding_rule_e2e(clean_registry):
    """End-to-end: a tt-lang kernel with a custom pointwise ``sharding_rule``
    is sharded on dim 0 across the mesh and matches the bf16 CPU golden.

    This exercises the whole custom-rule path on real hardware:
      * ``make_sharding_rule`` builds an ``sdy.op_sharding_rule`` text,
      * it rides the ``stablehlo.custom_call`` as the ``xla.sdy.sharding_rule``
        frontend attribute,
      * tt-mlir's ``register-custom-sharding-rule`` pass promotes it and hands
        it to Shardy, which propagates the dim-0 (``"model"``) sharding through
        the custom call so the kernel runs on ``rows/num_devices`` row-shards,
      * the gathered result equals the un-sharded ``a + b``.

    Without the custom rule the custom_call has no registered sharding rule and
    would fall back to replication, so a passing sharded run demonstrates the
    rule actually drove propagation.
    """
    import numpy as np
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
    from infra.utilities.torch_multichip_utils import enable_spmd
    from torch_xla.distributed.spmd import Mesh

    enable_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    if num_devices < 2:
        pytest.skip(f"needs a multi-device mesh, got {num_devices} device(s)")

    # Global shape: one row-block (64 rows) per device after sharding dim 0.
    rows = TILE_SIZE * GRANULARITY * num_devices
    cols = 64

    # Shard dim 0 across the "model" axis; dim 1 stays replicated.
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    # Pointwise rule over the *global* shape: factor "i" covers dim 0 (rows),
    # "j" covers dim 1 (cols). No reduction, so Shardy is free to shard "i".
    sharding_rule = make_sharding_rule(
        operand_mappings=[("i", "j"), ("i", "j")],
        result_mappings=[("i", "j")],
        factor_sizes={"i": rows, "j": cols},
    )

    add_op = _make_sharded_eltwise_add_operation(
        f"tt_xla.e2e.sharded_eltwise_add.{rows}x{cols}.v1", sharding_rule
    )

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    golden = a_cpu + b_cpu

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = torch.zeros_like(a_cpu).to(device)

    xs.mark_sharding(a_xla, mesh, ("model", None))
    xs.mark_sharding(b_xla, mesh, ("model", None))
    xs.mark_sharding(out_xla, mesh, ("model", None))

    result_xla = add_op(a_xla, b_xla, out_xla)
    xm.mark_step()
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)
