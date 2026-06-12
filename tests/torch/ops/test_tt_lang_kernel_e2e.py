# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end hardware test for the tt-lang integration in tt-xla.

What's exercised
----------------
1. A user-authored ``@ttl.operation`` kernel is wrapped with
   ``@tt_torch.tt_lang_operation`` and called from inside a ``torch.nn.Module``
   placed on the ``xla`` device.
2. ``torch_xla`` lowers the kernel call to
   ``stablehlo.custom_call @tt.tt_lang_op``.
3. tt-xla's PJRT plugin walks through SHLO -> TTIR -> TTNN, leaves a
   ``ttnn.tt_lang_op`` carrying ``kernel_id`` / ``version_tag`` /
   ``arg_roles``.
4. ``ModuleBuilder::resolveTtLangKernels`` runs tt-mlir's
   ``--ttnn-resolve-tt-lang-kernels`` pass, which calls
   ``tt_torch.tt_lang.resolve_operation`` (via pybind11, under the GIL)
   with the final shard-local shapes and printed layout encodings. The
   Python side runs tt-lang's compile path against duck-typed
   ``_StubTtnnTensor`` stand-ins (DEMO HACK -- see
   ``python_package/tt_torch/tt_lang.py``) so no ttnn device is ever
   opened in the plugin process. The artifact carries a kernel binary
   per thread, CB descriptors, and core ranges -- but **no**
   TensorAccessor compile-time args; those are filled in at launch
   time by the tt-mlir runtime.
5. ``TTNNToFlatbuffer.cpp`` translates the artifact into a
   ``GenericOp`` flatbuffer record. For each NOC kernel it writes one
   ``KernelArgTensorAccessorArgs`` marker per operand (in declaration
   order); compute kernels keep just the CB-index prefix.
6. The TTNN runtime (``runtime/lib/ttnn/operations/generic/
   generic_op.cpp``) resolves each marker to ``io_tensors[i].buffer()``
   at launch time, calls
   ``::tt::tt_metal::TensorAccessorArgs(buffer).get_compile_time_args()``,
   and splices the resulting uint32 sequence into the kernel binary's
   compile-time args. The kernel then executes on silicon with
   correct addresses, page sizes, and alignments derived from the
   real buffer.
7. The result is copied back to CPU and compared against the torch
   golden under bfloat16 atol/rtol.

Gating
------
* ``ttl`` must be importable. ``ttnn`` is no longer required in the
  plugin process: the bridge's default device-less compile path
  doesn't import it.
* A real Tenstorrent device must be visible to torch_xla (``xla:0``).

Run with::

    PJRT_DEVICE=TT pytest -svv tests/torch/ops/test_tt_lang_kernel_e2e.py
"""

from __future__ import annotations

import os

import pytest
import torch

# Import torch_xla before tt_torch so the plugin registers on torch_xla
# startup. ttl (tt-lang) is an optional dependency -- skip if absent.
import torch_xla  # noqa: F401
import torch_xla.core.xla_model as xm

ttl = pytest.importorskip("ttl", exc_type=ImportError)

import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
from tt_torch import tt_lang as tt_lang_mod
from tt_torch.tt_lang import tt_lang_operation

# ---------------------------------------------------------------------------
# Hardware / version gates
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.single_device,
]

# ---------------------------------------------------------------------------
# Kernel under test: tilewise elementwise add (cribbed from tt-lang's
# `examples/eltwise_add.py`, the smallest non-trivial kernel they ship).
# Kept *inside* a builder function so each invocation re-registers
# under a fresh operation_id -- avoids registry collisions across
# parametrised test invocations and keeps each test self-contained.
# ---------------------------------------------------------------------------


TILE_SIZE = 32
GRANULARITY = 2  # block_rows = GRANULARITY tiles -> shape[0] must be % 64.


def _make_eltwise_add_operation(operation_id: str):
    """Build the tt-lang operation + ``@tt_torch.tt_lang_operation`` wrapper.

    The operation mirrors `tt-lang/examples/eltwise_add.py`: a 2-tile-block
    pipelined elementwise add with explicit reader/compute/writer
    threads and CB-based block staging. Block count = 2 gives the
    runtime a minimal pipeline depth to exercise.
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
        version_tag="e2e-v1",
    )
    def add_op(a, b, out):
        return _ttl_add(a, b, out)

    return add_op


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 32), (128, 64)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_tt_lang_eltwise_add_e2e(shape, request):
    """Compile and execute a tt-lang elementwise-add kernel on the TT
    device through the full @tt_torch.tt_lang_operation -> stablehlo.custom_call
    -> tt_lang_op -> kernel_artifact -> flatbuffer -> GenericOp
    pipeline; verify the result matches the bf16 torch.add golden.
    """

    rows, cols = shape
    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    out_cpu = torch.zeros_like(a_cpu)

    operation_id = f"tt_xla.e2e.eltwise_add.{rows}x{cols}.v1"
    add_op = _make_eltwise_add_operation(operation_id)

    golden = a_cpu + b_cpu

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = out_cpu.to(device)

    try:
        result_xla = add_op(a_xla, b_xla, out_xla)
        # mark_step triggers compile + execute. Errors from the bridge
        # (e.g. resolve_operation failure) surface here as RuntimeError.
        xm.mark_step()
        result = result_xla.to("cpu")
    except RuntimeError as e:
        msg = str(e)
        if "undefined symbol" in msg or "_ZN2tt8tt_metal" in msg:
            pytest.skip("tt-metal ABI mismatch surfaced at compile time: " + msg[:300])
        raise

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    # bfloat16 tile add: ~1ulp per op, leave headroom.
    torch.testing.assert_close(result.float(), golden.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("shape", [(64, 32), (128, 64)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_tt_lang_add_stitched_with_torch_add_e2e(shape, request):
    """Stitch a tt-lang operation between two ``torch.add`` ops inside the
    same XLA program; verify the full graph compiles and matches the
    CPU four-operand sum golden.

    The graph forces both stitch directions to be exercised in a single
    PJRT executable::

        ab     = torch.add(a, b)             # ttnn.add   (regular)
        abc    = add_op(ab, c, out_buf)  # ttnn.tt_lang_op  (custom)
        result = torch.add(abc, d)           # ttnn.add   (regular)

    What this proves on top of ``test_tt_lang_eltwise_add_e2e``:

      * The output of a regular ``ttnn.add`` reaches the kernel through
        the layout pipeline -- the ``Layout::Tile`` workaround on
        ``ttnn.tt_lang_op`` operands (tt-mlir commit f59021024) inserts
        a ``to_layout`` if the upstream op picked row-major.
      * The kernel's output is a first-class TTNN value: the next
        ``ttnn.add`` consumes it directly, no host roundtrip between
        the two segments.
      * Both the regular and custom op paths land in the *same* PJRT
        executable; only one ``mark_step`` is called.
      * The dealloc-skip for the kernel's "out"-roled operand (tt-mlir
        commit 4d6e1bf95) doesn't leak through to the consumer ``ttnn.add``
        on the other side.
      * End-to-end correctness against ``((a+b) + c) + d`` under bf16
        tolerances.
    """

    rows, cols = shape
    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    c_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    d_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)

    # Distinct operation_id per parametrisation to avoid registry collisions.
    operation_id = f"tt_xla.e2e.stitched_add.{rows}x{cols}.v1"
    add_op = _make_eltwise_add_operation(operation_id)

    golden = ((a_cpu + b_cpu) + c_cpu) + d_cpu

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    c_xla = c_cpu.to(device)
    d_xla = d_cpu.to(device)

    ab = a_xla + b_xla
    # The kernel is mutation-style: it needs an "out" operand of the
    # right shape/dtype/device. ``torch.zeros_like(ab)`` is traced as
    # an XLA constant in the same program -- it's not a host
    # allocation. We never read out_buf after the call; the returned
    # functional result is what we chain forward.
    out_buf = torch.zeros_like(ab)
    add_op(ab, c_xla, out_buf)
    result_xla = out_buf + d_xla
    # One mark_step -- everything must compile into a single
    # executable. If torch_xla split the graph at the custom_call
    # boundary, we'd observe extra compiles in the logs.
    xm.mark_step()
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    # Three bf16 adds chained: ~3 ulp accumulation worst case, leave
    # headroom for tile rounding inside ttnn.add and the kernel.
    torch.testing.assert_close(result.float(), golden.float(), rtol=3e-2, atol=3e-2)
