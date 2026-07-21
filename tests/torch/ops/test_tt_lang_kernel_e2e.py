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
4. ``ModuleBuilder::resolveTTLangKernels`` runs tt-mlir's
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
5. tt-mlir's ``--ttnn-lower-tt-lang-to-generic`` pass rewrites the
   resolved ``ttnn.tt_lang_op`` into an equivalent ``ttnn.generic`` op.
   It parses the ``kernel_artifact`` JSON and builds a ``#ttnn.program``
   whose kernels carry their C++ body inline via the
   ``#ttnn.source_{compute,read,write}_kernel`` attributes (rather than a
   ttkernel symbol). For each NOC kernel it appends one
   ``#ttnn.kernel_arg_tensor_accessor_args`` marker per operand (in
   declaration order); compute kernels keep just the CB-index prefix.
6. ``TTNNToFlatbuffer.cpp`` emits that ``ttnn.generic`` through the
   ordinary generic-kernel path into a ``GenericOp`` flatbuffer record --
   no tt-lang-specific handling: the inline source and accessor markers
   come straight from the program attribute built in step 5.
7. The TTNN runtime (``runtime/lib/ttnn/operations/generic/
   generic_op.cpp``) resolves each marker to ``io_tensors[i].buffer()``
   at launch time, calls
   ``::tt::tt_metal::TensorAccessorArgs(buffer).get_compile_time_args()``,
   and splices the resulting uint32 sequence into the kernel binary's
   compile-time args. The kernel then executes on silicon with
   correct addresses, page sizes, and alignments derived from the
   real buffer.
8. The result is copied back to CPU and compared against the torch
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
# startup.
import torch_xla  # noqa: F401
import torch_xla.core.xla_model as xm
import tt_torch  # noqa: F401  -- registers torch.ops.tt.*
import ttl
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
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


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("shape", [(64, 32), (128, 64)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_tt_lang_eltwise_add_e2e(shape, request):
    """Compile and execute a tt-lang elementwise-add kernel on the TT
    device through the full @tt_torch.tt_lang_operation -> stablehlo.custom_call
    -> tt_lang_op -> kernel_artifact -> ttnn.generic -> flatbuffer GenericOp
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

    result_xla = add_op(a_xla, b_xla, out_xla)
    # mark_step triggers compile + execute. Errors from the bridge
    # (e.g. resolve_operation failure) surface here as RuntimeError.
    xm.mark_step()
    result = result_xla.to("cpu")

    assert (
        result.shape == golden.shape
    ), f"shape mismatch: {result.shape} vs {golden.shape}"

    # bfloat16 tile add: compare with PCC.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)


@pytest.mark.push
@pytest.mark.single_device
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

    # Three bf16 adds chained: compare with PCC.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(result, golden)


# ---------------------------------------------------------------------------
# Multi-output kernel under test: two "out" operands from a single kernel.
#
# The single-output eltwise kernel above never exercises the paths that only
# appear once a tt-lang op has >=2 "out" operands. This kernel does, and a
# matmul (rather than a plain eltwise) additionally forces the compiler to
# allocate scratch circular buffers. Between them, one kernel exercises:
#
#   * Shardy tuple decomposition -- a >=2-output op is emitted as a
#     tuple-result ``stablehlo.custom_call``; without the extended
#     ``DecomposeCustomCallTuplesPass`` compilation dies with
#     "Shardy propagation doesn't support tuples".
#   * Multi-output result de-aliasing -- both "out" operands must get their
#     own device buffer (fresh ``ttnn.empty`` per result in
#     ``TTNNLowerTTLangToGeneric``); the bug collapsed them so both results
#     read back identical (== the LAST output).
#   * ``dfb_index`` CB serialization -- the matmul's compiler-allocated
#     scratch CB carries its index on ``dfb_index`` (not ``_cb_index``);
#     mis-reading it serialized ``buffer_index = -1``, which the flatbuffer
#     emitter rejects ("out of range for uint32_t").
#
# ``out0 = a @ b`` and ``out1 = -(a @ b)`` -- two distinct outputs from one
# accumulation, so a de-aliasing regression (out0 == out1) is detectable.
# ---------------------------------------------------------------------------


def _make_matmul_multi_output_operation(operation_id: str):
    """Build a 2-input / 2-output tt-lang matmul: (a@b, -(a@b)).

    grid=(1,1) keeps the tiling trivial (single core); the K-loop accumulates
    into a compiler-allocated ``dst`` scratch CB. Both outputs are written from
    the same accumulator so the writer stages two "out" DFBs.
    """

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
    def _ttl_matmul2(a, b, pos_out, neg_out):
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        Mt, Kt, Nt = M // TILE_SIZE, K // TILE_SIZE, N // TILE_SIZE

        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        pos_dfb = ttl.make_dataflow_buffer_like(pos_out, shape=(1, 1), block_count=2)
        neg_dfb = ttl.make_dataflow_buffer_like(neg_out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for _ in range(Mt):
                for _ in range(Nt):
                    with pos_dfb.reserve() as p, neg_dfb.reserve() as q:
                        acc = ttl.block.fill(0, shape=p.shape)
                        for _ in range(Kt):
                            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                                acc += a_blk @ b_blk
                        p.store(acc)
                        q.store(ttl.block.fill(0, shape=q.shape) - acc)

        @ttl.datamovement()
        def read():
            for m in range(Mt):
                for n in range(Nt):
                    for k in range(Kt):
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(a[m, k], a_blk)
                            tx_b = ttl.copy(b[k, n], b_blk)
                            tx_a.wait()
                            tx_b.wait()

        @ttl.datamovement()
        def write():
            for m in range(Mt):
                for n in range(Nt):
                    with pos_dfb.wait() as p, neg_dfb.wait() as q:
                        tx_p = ttl.copy(p, pos_out[m, n])
                        tx_q = ttl.copy(q, neg_out[m, n])
                        tx_p.wait()
                        tx_q.wait()

    @tt_lang_operation(
        operation_id=operation_id,
        arg_roles=("in", "in", "out", "out"),
        version_tag="e2e-multi-v1",
    )
    def matmul2_op(a, b, pos_out, neg_out):
        return _ttl_matmul2(a, b, pos_out, neg_out)

    return matmul2_op


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "mkn", [(64, 64, 64), (64, 128, 96)], ids=lambda v: f"{v[0]}x{v[1]}x{v[2]}"
)
def test_tt_lang_multi_output_e2e(mkn, request):
    """A tt-lang kernel with two 'out' operands, executed on device.

    Gates the multi-output pipeline that the single-output eltwise test cannot
    reach: Shardy tuple decomposition, per-result buffer de-aliasing, and the
    matmul scratch CB's ``dfb_index`` serialization (see the builder docstring).

    Asserts each output matches its own golden AND that the two outputs are
    distinct -- a de-aliasing regression makes out0 == out1 (both == the last
    output), which the per-output PCC alone would not catch.
    """
    m, k, n = mkn
    a_cpu = torch.randn(m, k, dtype=torch.bfloat16)
    b_cpu = torch.randn(k, n, dtype=torch.bfloat16)

    operation_id = f"tt_xla.e2e.matmul2.{m}x{k}x{n}.v1"
    op = _make_matmul_multi_output_operation(operation_id)

    pos_golden = a_cpu.float() @ b_cpu.float()
    neg_golden = -pos_golden

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)
    pos_xla = torch.zeros(m, n, dtype=torch.bfloat16).to(device)
    neg_xla = torch.zeros(m, n, dtype=torch.bfloat16).to(device)

    ret = op(a_xla, b_xla, pos_xla, neg_xla)
    xm.mark_step()
    pos = ret[0].to("cpu")
    neg = ret[1].to("cpu")

    assert pos.shape == pos_golden.shape

    # bf16 matmul (fp32 accumulate): compare each output with PCC.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(pos, pos_golden.to(torch.bfloat16))
    comparator.evaluate(neg, neg_golden.to(torch.bfloat16))

    # De-aliasing regression guard: the two outputs must not collapse onto one
    # device buffer. For non-degenerate inputs out0 (a@b) and out1 (-(a@b))
    # differ everywhere; equality here means both results aliased the same
    # buffer (the multi-output write-back bug).
    frac_equal = torch.eq(pos, neg).float().mean().item()
    assert frac_equal < 0.5, (
        f"multi-output aliasing regressed: out0 == out1 for "
        f"{frac_equal:.1%} of elements (expected two distinct buffers)"
    )


@pytest.mark.push
@pytest.mark.single_device
def test_tt_lang_multi_output_reused_signature_e2e(request):
    """Call the *same* multi-output op twice with the same operand signature in
    one XLA program.

    tt-lang memoizes each compiled kernel in a per-operation cache and, on a
    hit, returns without re-running its ``_compile_kernel`` -- the only point
    the bridge captures the artifact. A process-wide operation object resolved
    twice for the same signature (e.g. a model reusing one norm/matmul width)
    therefore leaves the second resolve with nothing captured. The bridge's
    mirror cache serves the earlier artifact; without it the second resolve
    raises "tt-lang compile did not produce a CompiledTTNNKernel".

    Two chained calls of one op on identically-shaped operands force exactly
    that second same-signature resolve. Passing this test means the mirror
    cache recovered the artifact and both invocations ran correctly.
    """
    m = k = n = 64
    a_cpu = torch.randn(m, k, dtype=torch.bfloat16)
    b_cpu = torch.randn(k, n, dtype=torch.bfloat16)

    # ONE operation object, resolved twice below -> exercises tt-lang's per-op
    # cache-hit path on the second resolve.
    op = _make_matmul_multi_output_operation("tt_xla.e2e.matmul2.reused.v1")

    device = xm.xla_device()
    a_xla = a_cpu.to(device)
    b_xla = b_cpu.to(device)

    pos1 = torch.zeros(m, n, dtype=torch.bfloat16).to(device)
    neg1 = torch.zeros(m, n, dtype=torch.bfloat16).to(device)
    ret1 = op(a_xla, b_xla, pos1, neg1)  # resolve #1 (tt-lang cache miss)

    pos2 = torch.zeros(m, n, dtype=torch.bfloat16).to(device)
    neg2 = torch.zeros(m, n, dtype=torch.bfloat16).to(device)
    ret2 = op(ret1[0], b_xla, pos2, neg2)  # resolve #2, SAME signature (cache hit)

    xm.mark_step()

    golden1 = a_cpu.float() @ b_cpu.float()
    golden2 = ret1[0].to("cpu").float() @ b_cpu.float()

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(ret1[0].to("cpu"), golden1.to(torch.bfloat16))
    comparator.evaluate(ret2[0].to("cpu"), golden2.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# Customer-authored custom backward: native RMSNorm forward, tt-lang backward.
#
# This is the motivating use case for the whole mechanism -- keep a stock
# RMSNorm's forward native (so it lowers/fuses normally) and override ONLY the
# backward with a custom tt-lang kernel that runs on silicon inside the XLA
# graph. It is built on ``torch.autograd.Function`` (native forward + custom
# backward) rather than ``torch.library.register_autograd`` (which would make
# the forward opaque).
#
# The 2-pass RMSNorm-backward kernel (tt-metal PR #44516) has two "out"
# operands (dL_dinput, dL_dgamma_comp), so like the matmul test above it
# exercises the multi-output pipeline (Shardy tuple, de-aliasing) and its
# matmul-with-ones gamma-broadcast produces compiler-allocated CBs. On top of
# that it gates the kernel's own ``dL_dinput`` math: gamma is a [1, C] row
# vector, so it must be broadcast across the token/row dimension (via
# ``ones @ gamma``) -- the bug that left every gamma-dependent term correct
# only for token 0 (dL_dinput PCC ~= 0.17).
# ---------------------------------------------------------------------------

from ttl_rmsnorm_bw_2pass import make_kernel as _make_rmsnorm_bw_kernel

_rmsnorm_bw_kernel = _make_rmsnorm_bw_kernel()


@tt_lang_operation(
    operation_id="tt_xla.e2e.rmsnorm_bw_2pass.v1",
    arg_roles=("in", "in", "in", "in", "out", "out"),
    version_tag="e2e-rmsnorm-v1",
)
def _rmsnorm_bw_op(
    input_t, gamma_t, rms_t, dL_dout_t, dL_dinput_out, dL_dgamma_comp_out
):
    return _rmsnorm_bw_kernel(
        input_t, gamma_t, rms_t, dL_dout_t, dL_dinput_out, dL_dgamma_comp_out
    )


def _rmsnorm_bw_reference(x, gamma_row, rms, dL_dout):
    """Analytic RMSNorm backward -- the exact math the kernel implements.

    ``x``/``dL_dout`` are [T, C], ``gamma_row`` [1, C], ``rms`` [T, 1].
    Returns ``(dL_dinput [T, C], dL_dgamma_comp [T, C])``; grad_weight is the
    column sum of dL_dgamma_comp.
    """
    C = x.shape[-1]
    r = 1.0 / rms
    scale = (x * (gamma_row * r) * dL_dout).sum(-1, keepdim=True)  # [T, 1]
    dL_dinput = (gamma_row * r) * dL_dout - scale * x * (r * r) * (1.0 / C)
    dL_dgamma_comp = x * r * dL_dout
    return dL_dinput, dL_dgamma_comp


# How a customer wires this into training: a ``torch.autograd.Function`` with a
# native forward (stays transparent, lowers/fuses normally) and a backward that
# dispatches ``_rmsnorm_bw_op``. The test below drives the kernel directly rather
# than through ``.backward()`` -- the eager XLA autograd engine trips an
# unrelated device-ready-queue assert under pytest -- but the operands and the
# reduction are exactly what this backward would feed the kernel.
#
#   class CustomRMSNormFn(torch.autograd.Function):
#       @staticmethod
#       def forward(ctx, x, weight, eps):
#           x2d = x.reshape(-1, x.shape[-1])
#           rms = torch.sqrt(x2d.pow(2).mean(-1, keepdim=True) + eps)
#           ctx.save_for_backward(x2d, weight, rms)
#           return ((x2d / rms) * weight.reshape(1, -1)).reshape(x.shape)
#       @staticmethod
#       def backward(ctx, grad_out):
#           x2d, weight, rms = ctx.saved_tensors
#           di, dgc = torch.empty_like(x2d), torch.empty_like(x2d)
#           _rmsnorm_bw_op(x2d, weight.reshape(1, -1), rms, grad_out, di, dgc)
#           return di, dgc.sum(0), None


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "shape", [(128, 128), (64, 256)], ids=lambda s: f"{s[0]}x{s[1]}"
)
def test_tt_lang_rmsnorm_backward_e2e(shape, request):
    """Run the RMSNorm-backward tt-lang kernel on device; check both gradients.

    Dispatches the 2-output kernel with the operands a custom RMSNorm backward
    would feed it (x, gamma, saved rms, dL_dout) and compares dL_dinput,
    dL_dgamma_comp, and the reduced grad_weight to the analytic reference.

    Gates the kernel's ``dL_dinput`` math -- gamma is a [1, C] row vector that
    must be broadcast across the token/row dimension (via ``ones @ gamma``);
    the bug left every gamma term correct only for token 0 (dL_dinput PCC
    ~= 0.17). Also exercises the multi-output infra: two "out" operands (Shardy
    tuple + de-aliasing) and the matmul-with-ones' compiler-allocated CB.
    """
    torch.manual_seed(0)
    T, C = shape  # tokens x hidden; both must be multiples of 32

    x_cpu = torch.randn(T, C)
    gamma_cpu = torch.randn(1, C)  # [1, C] row vector -- must broadcast over rows
    # rms as a forward would save it: strictly positive, per token.
    rms_cpu = torch.randn(T, 1).abs() + 0.5
    dL_cpu = torch.randn(T, C)

    di_ref, dgc_ref = _rmsnorm_bw_reference(x_cpu, gamma_cpu, rms_cpu, dL_cpu)
    gw_ref = dgc_ref.sum(0)  # grad_weight [C]

    # bf16 at the kernel boundary (kernel authored for bf16 TILE_LAYOUT operands;
    # fp32_dest_acc_en=True keeps the reduction in f32).
    device = xm.xla_device()
    x_xla = x_cpu.to(torch.bfloat16).to(device)
    gamma_xla = gamma_cpu.to(torch.bfloat16).to(device)
    rms_xla = rms_cpu.to(torch.bfloat16).to(device)
    dL_xla = dL_cpu.to(torch.bfloat16).to(device)
    di_xla = torch.empty_like(x_xla)
    dgc_xla = torch.empty_like(x_xla)

    ret = _rmsnorm_bw_op(x_xla, gamma_xla, rms_xla, dL_xla, di_xla, dgc_xla)
    xm.mark_step()
    di = ret[0].to("cpu")
    dgc = ret[1].to("cpu")
    grad_weight = dgc.float().sum(0)

    # bf16 kernel: compare each output with PCC.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(di, di_ref.to(torch.bfloat16))  # the gamma-broadcast math
    comparator.evaluate(dgc, dgc_ref.to(torch.bfloat16))
    comparator.evaluate(grad_weight.to(torch.bfloat16), gw_ref.to(torch.bfloat16))

    # Multi-output de-aliasing guard: dL_dinput and dL_dgamma_comp are different
    # tensors; equality would mean both results aliased one device buffer.
    frac_equal = torch.eq(di, dgc).float().mean().item()
    assert frac_equal < 0.5, (
        f"multi-output aliasing regressed: dL_dinput == dL_dgamma_comp for "
        f"{frac_equal:.1%} of elements"
    )
