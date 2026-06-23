# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end hardware test for a *pipe*-using tt-lang kernel in tt-xla.

This is the multi-core companion to ``test_tt_lang_kernel_e2e.py``. Where
that test exercises a self-contained per-core elementwise add, this one
adds the piece that elementwise add never touches: **inter-core
communication via a ``ttl.PipeNet``**, and therefore **semaphores**.

What's exercised on top of the elementwise-add e2e
---------------------------------------------------
1. A ``@ttl.operation`` kernel uses ``ttl.Pipe`` / ``ttl.PipeNet`` to
   *multicast* a shared operand (``c``, one ``granularity``-tall tile
   block) from the source core ``(0, 0)`` to the destination cores
   ``(0, 1)``/``(0, 2)``/``(0, 3)``. The four active cores each own one
   output column tile and compute ``a * b + c`` for it; ``c`` is read from
   DRAM once (by the source) and reaches the other three cores over the
   pipe rather than via three more DRAM reads.
2. The pipe handshake is implemented by tt-lang with **sync semaphores**.
   tt-lang's compiler reports how many it allocated as
   ``CompiledTTNNKernel.num_pipe_sync_semaphores``; tt-xla's bridge
   serializes that into the kernel artifact as ``num_pipe_nets`` (see
   ``tt_torch.tt_lang._serialize_compiled_operation``).
3. tt-mlir's ``--ttnn-lower-tt-lang-to-generic`` pass reads
   ``num_pipe_nets`` and declares that many ``#ttnn.kernel_semaphore``
   descriptors on the resulting ``#ttnn.program`` (worker core type, the
   program's full core range, initial value 0) -- mirroring tt-lang's
   native ``kernel_runner.run_kernel_on_device`` launch path. The
   generated kernel C++ references the semaphores by baked-in id literals
   (``get_semaphore(<id>)`` / ``noc_semaphore_*``), so no per-kernel
   semaphore args are needed.
4. ``TTNNToFlatbuffer`` emits the semaphore descriptors into the
   ``GenericOp`` program, and the TTNN runtime
   (``generic_op.cpp::createSemaphoreDescriptor``) materialises a real
   tt-metal semaphore per descriptor at launch time.
5. The result is copied back to CPU and compared against the torch golden
   under bf16 PCC.

If the semaphores are *not* declared on the program (the pre-fix
behaviour), the pipe handshake has no backing semaphores and the kernel
deadlocks / faults on device -- so this test is a direct regression guard
for the semaphore-plumbing path.

Run with::

    PJRT_DEVICE=TT pytest -svv tests/torch/ops/test_tt_lang_pipe_kernel_e2e.py
"""

from __future__ import annotations

import pytest
import torch

# Import torch_xla before tt_torch so the plugin registers on torch_xla
# startup.
import torch_xla  # noqa: F401
import torch_xla.core.xla_model as xm

import ttl

import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from tt_torch.tt_lang import tt_lang_operation

# ---------------------------------------------------------------------------
# Hardware / version gates
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.single_device,
]

# ---------------------------------------------------------------------------
# Kernel under test: a pipe-multicast elementwise ``a * b + c``.
#
# The pipe source core (0, 0) reads the shared addend ``c`` (one
# ``granularity``-tall tile column) from DRAM and multicasts it to the
# destination cores (0, 1..3). Each of the four active cores owns one
# output column tile (``column tile == node_row``) and computes
# ``a * b + c`` for its column, reusing the multicast ``c`` across every
# row block.
#
# Kept inside a builder function so each invocation re-registers under a
# fresh operation_id -- avoids registry collisions across parametrised
# test invocations and keeps each test self-contained.
# ---------------------------------------------------------------------------


TILE_SIZE = 32
GRANULARITY = 2  # block_rows = GRANULARITY tiles -> shape[0] must be % 64.
NUM_ACTIVE_COLS = 4  # pipe fans out to cores (0, 0..3); 4 column tiles.


def _make_pipe_add_operation(operation_id: str):
    """Build the pipe-multicast ``a * b + c`` operation + wrapper.

    The structure mirrors ``tt-lang/examples/eltwise_pipe.py`` (a 2-tile
    pipelined block transfer with explicit reader/compute/writer threads
    and a ``PipeNet`` C-multicast), adapted to the primitives available in
    the pinned tt-lang build: ``ttl.node(dims=2)`` for the 2D core index,
    inline slicing, and ``if pipe_net.is_active():`` guards (rather than
    early ``return``).
    """

    @ttl.operation(grid="full")  # NOTE: use the full device grid.
    def _ttl_pipe_add(a_in, b_in, c_in, out):
        granularity = GRANULARITY
        row_tiles = a_in.shape[0] // TILE_SIZE
        block_count = 2

        a_dfb = ttl.make_dataflow_buffer_like(
            a_in, shape=(granularity, 1), block_count=block_count
        )
        b_dfb = ttl.make_dataflow_buffer_like(
            b_in, shape=(granularity, 1), block_count=block_count
        )
        c_dfb = ttl.make_dataflow_buffer_like(
            c_in, shape=(granularity, 1), block_count=block_count
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(granularity, 1), block_count=block_count
        )

        # Multicast C from the source core (0, 0) to the destination cores
        # at column 0, rows 1..3 -> (0, 1), (0, 2), (0, 3). Together with
        # the source this is the four-core active set; each core owns the
        # output column tile equal to its row coordinate.
        pipe = ttl.Pipe((0, 0), (0, slice(1, NUM_ACTIVE_COLS)))
        pipe_net = ttl.PipeNet([pipe])

        @ttl.compute()
        def compute():
            if pipe_net.is_active():
                _node_col, _node_row = ttl.node(dims=2)
                c_block = c_dfb.wait()
                for _rt in range(row_tiles // granularity):
                    a_block = a_dfb.wait()
                    b_block = b_dfb.wait()
                    out_block = out_dfb.reserve()
                    out_block.store(a_block * b_block + c_block)
                    out_block.push()
                    a_block.pop()
                    b_block.pop()
                c_block.pop()

        @ttl.datamovement()
        def read():
            if pipe_net.is_active():
                _node_col, node_row = ttl.node(dims=2)
                col_tile = node_row
                # Pipe communication setup -- must happen before the main
                # loop. The source reads C from DRAM then pushes it onto the
                # pipe; every destination pulls it off the pipe.
                with c_dfb.reserve() as c_block:

                    def pipe_src(pipe_id):
                        tx = ttl.copy(c_in[0:granularity, 0:1], c_block)
                        tx.wait()
                        tx2 = ttl.copy(c_block, pipe_id)
                        tx2.wait()

                    def pipe_dst(pipe_id):
                        tx = ttl.copy(pipe_id, c_block)
                        tx.wait()

                    pipe_net.if_src(pipe_src)
                    pipe_net.if_dst(pipe_dst)

                for rt in range(row_tiles // granularity):
                    r0 = rt * granularity
                    r1 = (rt + 1) * granularity
                    a_block = a_dfb.reserve()
                    tx = ttl.copy(a_in[r0:r1, col_tile : col_tile + 1], a_block)
                    tx.wait()
                    a_block.push()
                    b_block = b_dfb.reserve()
                    tx = ttl.copy(b_in[r0:r1, col_tile : col_tile + 1], b_block)
                    tx.wait()
                    b_block.push()

        @ttl.datamovement()
        def write():
            if pipe_net.is_active():
                _node_col, node_row = ttl.node(dims=2)
                col_tile = node_row
                for rt in range(row_tiles // granularity):
                    r0 = rt * granularity
                    r1 = (rt + 1) * granularity
                    out_block = out_dfb.wait()
                    tx = ttl.copy(out_block, out[r0:r1, col_tile : col_tile + 1])
                    tx.wait()
                    out_block.pop()

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "in", "out"),
        version_tag="e2e-pipe-v1",
    )
    def add_op(a, b, c, out):
        return _ttl_pipe_add(a, b, c, out)

    return add_op


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "rows", [64, 128], ids=lambda r: f"{r}x{NUM_ACTIVE_COLS * TILE_SIZE}"
)
def test_tt_lang_pipe_add_e2e(rows, request):
    """Compile and execute a pipe-multicast ``a * b + c`` tt-lang kernel on
    the TT device through the full pipeline and verify it matches the bf16
    torch golden.

    This is the first e2e test to drive a kernel whose correctness depends
    on semaphores: the ``PipeNet`` C-multicast can only complete if the
    sync semaphores tt-lang allocated are declared on the lowered
    ``ttnn.generic`` program and created by the runtime.
    """

    cols = NUM_ACTIVE_COLS * TILE_SIZE  # 4 column tiles, one per active core.
    c_rows = GRANULARITY * TILE_SIZE  # C is one ``granularity``-tall block.

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    c_cpu = torch.randn(c_rows, TILE_SIZE, dtype=torch.bfloat16)
    out_cpu = torch.zeros_like(a_cpu)

    operation_id = f"tt_xla.e2e.pipe_add.{rows}x{cols}.v1"
    add_op = _make_pipe_add_operation(operation_id)

    # Golden: each output column tile is ``a * b`` plus the shared addend
    # ``c``; ``c`` (one ``granularity``-tall tile block) is reused across
    # every row block and every active column, so it tiles across the
    # whole output.
    c_tiled = c_cpu.repeat(rows // c_rows, cols // TILE_SIZE)
    golden = a_cpu * b_cpu + c_tiled

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    c_xla = c_cpu.to(device)
    out_xla = out_cpu.to(device)

    result_xla = add_op(a_xla, b_xla, c_xla, out_xla)
    # mark_step triggers compile + execute. Errors from the bridge
    # (e.g. resolve_operation failure) surface here as RuntimeError.
    xm.mark_step()
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    # bf16 tile multiply-add: compare with PCC.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)
