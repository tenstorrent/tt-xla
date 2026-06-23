# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end hardware test for PipeNet GlobalSemaphore ready counters.

Companion to ``test_tt_lang_pipe_kernel_e2e.py``. That test uses a single
PipeNet with one pipe, so tt-lang stays under the local-semaphore budget
(``num_pipe_nets + max_pipes_per_source <= 16``) and allocates only sync
semaphores. This test forces the GlobalSemaphore path:

    num_pipe_nets + max_pipes_per_source > MAX_HARDWARE_SEMAPHORE_IDS (16)

by building one PipeNet with 16 point-to-point pipes from the same source
(``max_pipes_per_source == 16`` → sync count 1, global count 16).

What's exercised on top of the local-sync pipe e2e
--------------------------------------------------
1. tt-lang's PipeLowering selects GlobalSemaphore-backed ready counters and
   bakes ``GetCommonArgVal`` indices for them into the kernel C++.
2. tt-xla serializes ``num_pipe_global_semaphores`` into the kernel artifact.
3. ``--ttnn-lower-tt-lang-to-generic`` creates that many
   ``ttnn.create_global_semaphore`` ops, passes them as GenericOp
   ``additional_args``, and appends ``#ttnn.kernel_arg_global_semaphore``
   markers to every kernel's ``common_rt_args`` after the optional SRAM
   scratch address -- matching ``build_pipe_runtime_resources``.
4. The TTNN runtime resolves each marker to a GlobalSemaphore address at
   launch; without this plumbing the ready-counter handshake deadlocks.

Run with::

    PJRT_DEVICE=TT pytest -svv \\
        tests/torch/ops/test_tt_lang_pipe_global_semaphore_e2e.py
"""

from __future__ import annotations

import pytest
import torch

# Import torch_xla before tt_torch so the plugin registers on torch_xla
# startup.
import torch_xla  # noqa: F401
import torch_xla.core.xla_model as xm

import ttl
from ttl.constants import MAX_HARDWARE_SEMAPHORE_IDS

import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from tt_torch.tt_lang import tt_lang_operation

pytestmark = [
    pytest.mark.single_device,
]

TILE_SIZE = 32
GRANULARITY = 2  # block_rows = GRANULARITY tiles -> shape[0] must be % 64.

# 16 p2p destinations from source (0, 0) on an 8x8 grid (skip the source).
# One PipeNet with 16 pipes from the same source trips
# nets + max_pipes_per_source = 1 + 16 > 16 → GlobalSemaphore path.
_GLOBAL_SEM_DSTS = tuple((i % 8, i // 8) for i in range(1, 17))
assert len(_GLOBAL_SEM_DSTS) == 16
assert (0, 0) not in _GLOBAL_SEM_DSTS
assert 1 + len(_GLOBAL_SEM_DSTS) > MAX_HARDWARE_SEMAPHORE_IDS

# Active cores: source + 16 destinations. Output is one column tile per
# active core, laid out as columns 0..16 via `row*8 + col` (source owns 0).
NUM_ACTIVE_CORES = 1 + len(_GLOBAL_SEM_DSTS)  # 17


def _make_global_sem_pipe_add_operation(operation_id: str):
    """Build a pipe-multicast ``a * b + c`` that requires GlobalSemaphores.

    Same compute pattern as the local-sync pipe e2e, but C is multicasted
    over 16 point-to-point pipes (one PipeNet) so tt-lang allocates
    GlobalSemaphore ready counters instead of local sender-ready
    semaphores.
    """

    @ttl.operation(grid="full")
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

        pipes = [ttl.Pipe((0, 0), dst) for dst in _GLOBAL_SEM_DSTS]
        pipe_net = ttl.PipeNet(pipes)

        @ttl.compute()
        def compute():
            if pipe_net.is_active():
                # Source (0, 0) -> col 0; dest (i%8, i//8) for i in 1..16
                # -> col i. Both are `row*8 + col` with no branch (avoids
                # tt-lang's if-phi i64/index type mismatch).
                node_col, node_row = ttl.node(dims=2)
                col_tile = node_row * 8 + node_col
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
                node_col, node_row = ttl.node(dims=2)
                col_tile = node_row * 8 + node_col
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
                node_col, node_row = ttl.node(dims=2)
                col_tile = node_row * 8 + node_col
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
        version_tag="e2e-pipe-global-sem-v1",
    )
    def add_op(a, b, c, out):
        return _ttl_pipe_add(a, b, c, out)

    return add_op


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("rows", [64], ids=lambda r: f"{r}x{NUM_ACTIVE_CORES * TILE_SIZE}")
def test_tt_lang_pipe_global_semaphore_e2e(rows, request):
    """Compile and execute a PipeNet that requires GlobalSemaphore ready
    counters through the full tt-xla pipeline.
    """
    cols = NUM_ACTIVE_CORES * TILE_SIZE
    c_rows = GRANULARITY * TILE_SIZE

    a_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    b_cpu = torch.randn(rows, cols, dtype=torch.bfloat16)
    c_cpu = torch.randn(c_rows, TILE_SIZE, dtype=torch.bfloat16)
    out_cpu = torch.zeros_like(a_cpu)

    operation_id = f"tt_xla.e2e.pipe_global_sem.{rows}x{cols}.v1"
    add_op = _make_global_sem_pipe_add_operation(operation_id)

    # Sanity: the registered compile must have requested GlobalSemaphores.
    # resolve_operation is exercised indirectly via mark_step; assert the
    # threshold math here so a silent local-sem fallback fails the test.
    from ttl._pipenets import NodeCoord, OperationPipeNets, PipeUse

    graph = OperationPipeNets()
    graph.add_pipe_net(
        [
            PipeUse(src=NodeCoord((0, 0)), dst=NodeCoord(dst))
            for dst in _GLOBAL_SEM_DSTS
        ]
    )
    assert graph.num_pipe_global_semaphores() == len(_GLOBAL_SEM_DSTS)
    assert graph.num_pipe_sync_semaphores() == 1

    c_tiled = c_cpu.repeat(rows // c_rows, cols // TILE_SIZE)
    golden = a_cpu * b_cpu + c_tiled

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    c_xla = c_cpu.to(device)
    out_xla = out_cpu.to(device)

    result_xla = add_op(a_xla, b_xla, c_xla, out_xla)
    xm.mark_step()
    result = result_xla.to("cpu")

    assert result.shape == golden.shape, f"shape mismatch: {result.shape} vs {golden.shape}"

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)
