# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end hardware test for *broadcast/reduce* tt-lang kernels in tt-xla.

This is the block-math companion to ``test_tt_lang_kernel_e2e.py`` (per-core
elementwise add) and ``test_tt_lang_pipe_kernel_e2e.py`` (inter-core pipe +
semaphores). Where those exercise data movement, this one exercises the
``ttl.block`` / ``ttl.math`` *compute* intrinsics that lower to multiple LLK
ops and, in the reduce case, a **compiler-allocated** circular buffer.

What's exercised on top of the elementwise-add e2e
---------------------------------------------------
1. ``ttl.block.broadcast`` over both axes:
     * a **column vector** ``b`` (shape ``(N_BLOCK, 1)`` in tiles) broadcast
       along the innermost dim (``dims=[-1]``) -- an intra-tile scalar
       broadcast plus inter-tile fan-out across the M column tiles.
     * a **row vector** ``c`` (shape ``(1, M)`` in tiles) broadcast along the
       outer dim (``dims=[0]``) -- an inter-tile fan-out across the N row
       tiles.
   Mirrors ``tt-lang/examples/eltwise_broadcast_reduce.py`` (which is a
   sim-only ``xfail-compiler`` example); the spellings here are the ones the
   pinned tt-lang build accepts (``ttl.block.broadcast`` rather than
   ``ttl.math.broadcast``).
2. ``ttl.math.reduce_sum`` along the width (``dims=[-1]``). tt-lang lowers a
   reduce to an LLK reduce plus a **scaler tile** held in a compiler-allocated
   CB. That CB carries no user ``DataflowBuffer`` and tt-lang stamps it with
   ``_cb_index == -1``; tt-xla's artifact serializer must assign its
   ``buffer_index`` from the CB's *position* in ``compiled.cb_configs`` (as
   tt-lang's native ``build_cb_descriptors`` does), not from ``_cb_index``,
   or the lowering rejects ``buffer_index = -1``. This test is the regression
   guard for that path (see
   ``tt_torch.tt_lang._serialize_cb_config``).

The reduce result lands in **column 0** of each output tile (other columns
are zero), matching tt-metal ``reduce_w`` semantics, so the reduce golden is
compared on column 0 only.

Run with::

    PJRT_DEVICE=TT pytest -svv tests/torch/ops/test_tt_lang_broadcast_kernel_e2e.py
"""

from __future__ import annotations

import pytest
import torch

# Import torch_xla before tt_torch so the plugin registers on torch_xla
# startup.
import torch_xla  # noqa: F401
import torch_xla.core.xla_model as xm
import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
import ttl
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from tt_torch.tt_lang import tt_lang_operation

# ---------------------------------------------------------------------------
# Hardware / version gates
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.single_device,
]

TILE_SIZE = 32
N_BLOCK = 2  # process N row tiles in blocks of 2 -> rows must be % 64.


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------


def _uniform_per_row_tile(values: torch.Tensor) -> torch.Tensor:
    """Build a ``(n_tiles*TILE, TILE)`` column vector whose every 32x32 tile
    holds one uniform scalar ``values[n]``.

    A uniform tile makes the broadcast value-blind: the intra-tile scalar
    broadcast (which replicates column 0 across the tile) and the inter-tile
    fan-out both produce the same scalar, so the golden is just the per-tile
    value regardless of which broadcast step ran.
    """
    n = values.shape[0]
    return (
        values.view(n, 1, 1)
        .expand(n, TILE_SIZE, TILE_SIZE)
        .reshape(n * TILE_SIZE, TILE_SIZE)
        .contiguous()
    )


def _uniform_per_col_tile(values: torch.Tensor) -> torch.Tensor:
    """Build a ``(TILE, m_tiles*TILE)`` row vector whose every 32x32 tile holds
    one uniform scalar ``values[m]`` (column-vector analogue of
    :func:`_uniform_per_row_tile`)."""
    m = values.shape[0]
    return (
        values.view(1, 1, m, 1)
        .expand(1, TILE_SIZE, m, TILE_SIZE)
        .reshape(TILE_SIZE, m * TILE_SIZE)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Kernel 1: broadcast-add. ``out[n, m] = a[n, m] + b[n]`` -- ``b`` is a column
# vector broadcast across the M column tiles (``dims=[-1]``).
# ---------------------------------------------------------------------------


def _make_broadcast_add_operation(operation_id: str):
    @ttl.operation(grid=(1, 1))
    def _ttl_bcast_add(a_in, b_in, out):
        n_tiles = a_in.shape[0] // TILE_SIZE
        m_tiles = a_in.shape[1] // TILE_SIZE
        n_blocks = n_tiles // N_BLOCK

        a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(N_BLOCK, m_tiles))
        b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(N_BLOCK, 1))
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(N_BLOCK, m_tiles))

        @ttl.datamovement()
        def read():
            for nb in range(n_blocks):
                r0 = nb * N_BLOCK
                r1 = (nb + 1) * N_BLOCK
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    tx_a = ttl.copy(a_in[r0:r1, 0:m_tiles], a_blk)
                    tx_a.wait()
                    tx_b = ttl.copy(b_in[r0:r1, 0:1], b_blk)
                    tx_b.wait()

        @ttl.compute()
        def compute():
            for _nb in range(n_blocks):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    b_bcast = ttl.block.broadcast(
                        b_blk, dims=[-1], shape=(N_BLOCK, m_tiles)
                    )
                    out_blk.store(a_blk + b_bcast)

        @ttl.datamovement()
        def write():
            for nb in range(n_blocks):
                r0 = nb * N_BLOCK
                r1 = (nb + 1) * N_BLOCK
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[r0:r1, 0:m_tiles])
                    tx.wait()

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "out"),
        version_tag="e2e-bcast-v1",
    )
    def bcast_op(a, b, out):
        return _ttl_bcast_add(a, b, out)

    return bcast_op


# ---------------------------------------------------------------------------
# Kernel 2: broadcast + reduce. ``y[n] = sum_m( a[n, m] + b[n] + c[m] )`` --
# ``b`` is a column vector (``dims=[-1]``), ``c`` is a row vector
# (``dims=[0]``), then a width reduce (``dims=[-1]``) collapses the M tiles to
# a single column. ``c`` is invariant across N blocks so it is read once.
# ---------------------------------------------------------------------------


def _make_broadcast_reduce_operation(operation_id: str):
    @ttl.operation(grid=(1, 1))
    def _ttl_bcast_reduce(a_in, b_in, c_in, y_out):
        n_tiles = a_in.shape[0] // TILE_SIZE
        m_tiles = a_in.shape[1] // TILE_SIZE
        n_blocks = n_tiles // N_BLOCK

        a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(N_BLOCK, m_tiles))
        b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(N_BLOCK, 1))
        c_dfb = ttl.make_dataflow_buffer_like(c_in, shape=(1, m_tiles))
        y_dfb = ttl.make_dataflow_buffer_like(y_out, shape=(N_BLOCK, 1))

        @ttl.datamovement()
        def read():
            # c is invariant across N blocks -> read it once.
            with c_dfb.reserve() as c_blk:
                tx_c = ttl.copy(c_in[0:1, 0:m_tiles], c_blk)
                tx_c.wait()
            for nb in range(n_blocks):
                r0 = nb * N_BLOCK
                r1 = (nb + 1) * N_BLOCK
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    tx_a = ttl.copy(a_in[r0:r1, 0:m_tiles], a_blk)
                    tx_a.wait()
                    tx_b = ttl.copy(b_in[r0:r1, 0:1], b_blk)
                    tx_b.wait()

        @ttl.compute()
        def compute():
            with c_dfb.wait() as c_blk:
                c_bcast = ttl.block.broadcast(c_blk, dims=[0], shape=(N_BLOCK, m_tiles))
                for _nb in range(n_blocks):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        y_dfb.reserve() as y_blk,
                    ):
                        b_bcast = ttl.block.broadcast(
                            b_blk, dims=[-1], shape=(N_BLOCK, m_tiles)
                        )
                        s = a_blk + b_bcast + c_bcast
                        y_blk.store(ttl.math.reduce_sum(s, dims=[-1]))

        @ttl.datamovement()
        def write():
            for nb in range(n_blocks):
                r0 = nb * N_BLOCK
                r1 = (nb + 1) * N_BLOCK
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y_out[r0:r1, 0:1])
                    tx.wait()

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "in", "out"),
        version_tag="e2e-bcast-reduce-v1",
    )
    def bcr_op(a, b, c, y):
        return _ttl_bcast_reduce(a, b, c, y)

    return bcr_op


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("rows", [64, 128], ids=lambda r: f"{r}rows")
@pytest.mark.parametrize("m_tiles", [1, 3], ids=lambda m: f"{m}mtiles")
def test_tt_lang_broadcast_add_e2e(rows, m_tiles, request):
    """Compile and execute a broadcast-add tt-lang kernel
    (``out[n, m] = a[n, m] + b[n]``) on the TT device through the full
    ``@tt_torch.tt_lang_operation`` -> stablehlo.custom_call -> tt_lang_op ->
    kernel_artifact -> ttnn.generic -> flatbuffer pipeline, and verify it
    matches the bf16 torch golden.

    Exercises ``ttl.block.broadcast`` of a column vector across the M column
    tiles (``dims=[-1]``).
    """

    cols = m_tiles * TILE_SIZE

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    # b: column vector, uniform per row tile so the broadcast is value-blind.
    b_vals = torch.randn(rows // TILE_SIZE, dtype=torch.bfloat16)
    b_cpu = _uniform_per_row_tile(b_vals)  # (rows, TILE)
    out_cpu = torch.zeros_like(a_cpu)

    operation_id = f"tt_xla.e2e.bcast_add.{rows}x{cols}.v1"
    bcast_op = _make_broadcast_add_operation(operation_id)

    # b broadcasts across all M column tiles (torch broadcasts (rows, 1) over
    # (rows, cols)).
    golden = a_cpu + b_cpu[:, 0:1]

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = out_cpu.to(device)

    result_xla = bcast_op(a_xla, b_xla, out_xla)
    # mark_step triggers compile + execute. Errors from the bridge
    # (e.g. resolve_operation failure) surface here as RuntimeError.
    xm.mark_step()
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("rows", [64, 128], ids=lambda r: f"{r}rows")
@pytest.mark.parametrize("m_tiles", [3], ids=lambda m: f"{m}mtiles")
def test_tt_lang_broadcast_reduce_e2e(rows, m_tiles, request):
    """Compile and execute a broadcast+reduce tt-lang kernel
    (``y[n] = sum_m( a[n, m] + b[n] + c[m] )``) on the TT device and verify it
    matches the bf16 torch golden.

    On top of :func:`test_tt_lang_broadcast_add_e2e` this drives:
      * broadcast over *both* axes (column vector ``b`` via ``dims=[-1]`` and
        row vector ``c`` via ``dims=[0]``), and
      * ``ttl.math.reduce_sum`` along the width, whose scaler tile lives in a
        compiler-allocated CB -- the regression guard for positional
        ``buffer_index`` assignment in the artifact serializer.

    The reduce result occupies column 0 of each output tile (tt-metal
    ``reduce_w`` semantics), so the golden is compared on column 0.
    """

    cols = m_tiles * TILE_SIZE

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    # b: column vector (uniform per row tile); c: row vector (uniform per col
    # tile). Uniform tiles make both broadcasts value-blind.
    b_vals = torch.randn(rows // TILE_SIZE, dtype=torch.bfloat16)
    b_cpu = _uniform_per_row_tile(b_vals)  # (rows, TILE)
    c_vals = torch.randn(m_tiles, dtype=torch.bfloat16)
    c_cpu = _uniform_per_col_tile(c_vals)  # (TILE, cols)
    y_cpu = torch.zeros(rows, TILE_SIZE, dtype=torch.bfloat16)

    operation_id = f"tt_xla.e2e.bcast_reduce.{rows}x{cols}.v1"
    bcr_op = _make_broadcast_reduce_operation(operation_id)

    # Golden: s[r, col] = a[r, col] + b[r] + c[col]; the width reduce lands the
    # per-row sum in column 0 (other columns stay zero).
    b_col = b_cpu[:, 0:1].float()  # (rows, 1)
    c_row = c_cpu[0:1, :].float()  # (1, cols)
    s = a_cpu.float() + b_col + c_row
    golden_col0 = s.sum(dim=1, keepdim=True).to(torch.bfloat16)  # (rows, 1)

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    c_xla = c_cpu.to(device)
    y_xla = y_cpu.to(device)

    result_xla = bcr_op(a_xla, b_xla, c_xla, y_xla)
    xm.mark_step()
    result = result_xla.to("cpu")

    assert result.shape == (
        rows,
        TILE_SIZE,
    ), f"shape mismatch: {result.shape} vs {(rows, TILE_SIZE)}"

    # The reduced sum lives in column 0; compare that column.
    result_col0 = result[:, 0:1]

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result_col0, golden_col0)
