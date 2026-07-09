# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Author and run a custom tt-lang kernel on Tenstorrent hardware.

This standalone example shows how to write a device kernel in Tenstorrent's
``ttl`` kernel DSL, expose it to PyTorch/XLA with the
``@tt_torch.tt_lang_operation`` decorator, and run it end-to-end on a TT
device through the PJRT plugin.

What happens under the hood
---------------------------
1. The ``@ttl.operation`` kernel is wrapped by ``@tt_torch.tt_lang_operation``
   and called from a normal PyTorch program placed on the ``xla`` device.
2. ``torch_xla`` lowers the call to ``stablehlo.custom_call @tt.tt_lang_op``.
3. The tt-xla PJRT plugin runs SHLO -> TTIR -> TTNN, leaving a
   ``ttnn.tt_lang_op`` carrying ``kernel_id`` / ``version_tag`` / ``arg_roles``.
4. tt-mlir's ``--ttnn-resolve-tt-lang-kernels`` pass calls back into
   ``tt_torch.tt_lang.resolve_operation`` to compile the kernel (device-less)
   into a ``kernel_artifact``; ``--ttnn-lower-tt-lang-to-generic`` then rewrites
   the op into an ``ttnn.generic`` that runs through the ordinary
   generic-kernel flatbuffer/runtime path.
5. The result is copied back to CPU and compared against the torch golden.

Requirements
------------
* ``ttl`` (Tenstorrent kernel DSL) must be importable.
* A real Tenstorrent device must be visible to torch_xla.

Run with::

    python examples/pytorch/tt_lang_eltwise_add.py

or under pytest::

    pytest -svv examples/pytorch/tt_lang_eltwise_add.py
"""

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Importing tt_torch registers the "tt" PJRT plugin, ``torch.ops.tt.*``, and
# the ``@tt_lang_operation`` decorator. ``ttl`` is the kernel DSL.
import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
import ttl
from tt_torch.tt_lang import tt_lang_operation

TILE_SIZE = 32
GRANULARITY = 2  # block_rows = GRANULARITY tiles -> shape[0] must be % 64.


def build_eltwise_add_operation(operation_id: str):
    """Build a tilewise elementwise-add kernel wrapped for PyTorch/XLA.

    Mirrors ``tt-lang/examples/eltwise_add.py``: a 2-tile-block pipelined
    add with explicit reader / compute / writer threads and CB-based block
    staging. The operation is built inside a function so each call registers
    under a fresh ``operation_id``, keeping invocations self-contained.
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
        version_tag="example-v1",
    )
    def add_op(a, b, out):
        return _ttl_add(a, b, out)

    return add_op


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened to float32."""
    a = actual.flatten().float()
    b = expected.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_eltwise_add(shape=(128, 64)):
    """Run the tt-lang add kernel on the TT device; return (result, golden)."""
    rows, cols = shape
    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    out_cpu = torch.zeros_like(a_cpu)
    golden = a_cpu + b_cpu

    add_op = build_eltwise_add_operation(f"tt_xla.example.eltwise_add.{rows}x{cols}.v1")

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    out_xla = out_cpu.to(device)

    result_xla = add_op(a_xla, b_xla, out_xla)
    # mark_step triggers compile + execute; bridge errors surface here.
    xm.mark_step()
    return result_xla.to("cpu"), golden


@pytest.mark.parametrize("shape", [(64, 32), (128, 64)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_tt_lang_eltwise_add(shape):
    """Compile and execute the tt-lang add kernel; compare to the torch golden."""
    xr.set_device_type("TT")

    result, golden = run_eltwise_add(shape)

    assert result.shape == golden.shape, f"{result.shape} vs {golden.shape}"
    pcc = _pcc(result, golden)
    print(f"[eltwise_add {shape}] PCC={pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


if __name__ == "__main__":
    # torch_xla defaults to CPU; point it at the TT device.
    xr.set_device_type("TT")
    test_tt_lang_eltwise_add((128, 64))
    test_tt_lang_add_stitched_with_torch_add((128, 64))
    print("tt-lang eltwise-add example passed.")
