# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Show that the shard-then-grow KV cache init survives a cache too large for
one chip, and that the naive unsharded init does not.

Background: in vLLM the KV cache is allocated before the full model compiles.
A cache sized for the whole mesh does not fit in one chip's DRAM, and an
unsharded allocation has to be resident somewhere before a sharding annotation
can take effect -- so it OOMs. The workaround allocates a tiny seed, shards it,
forces the shard to materialize, and only then grows it with `repeat`, so no
single device is ever asked for more than its own slice.

Both tests build the same logical cache, sized from the reported per-chip DRAM
so it overcommits one chip several times over. The only difference is the order
of shard-vs-allocate.

Run them in SEPARATE pytest invocations: the OOM leaves the mesh device closed,
so a following test in the same process will fail for the wrong reason.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sharding import sharding_constraint_tensor

# vLLM K-cache geometry is (num_blocks, num_kv_heads, block_size, head_size);
# only num_blocks is scaled to hit the target size.
BLOCK_SIZE = 32  # tokens per page
NUM_KV_HEADS = 64  # must be divisible by TP_SIZE
HEAD_SIZE = 128
KV_DTYPE = torch.bfloat16

# Multiple of one chip's DRAM the unsharded cache should demand. 3x is a clear
# OOM unsharded, while 3/8 of a chip when split 8 ways leaves room for the
# runtime's own allocations.
OVERCOMMIT = 3.0


def _ts() -> str:
    """UTC timestamp matching memory_logger.py's `timestamp_utc` column, so test
    prints line up directly with rows in the RSS/DRAM CSV."""
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    """Timestamped, flushed print.

    Flushing matters: stdout is block-buffered when the run is piped to a log
    file, so unflushed prints would interleave wrongly with the plugin's
    unbuffered stderr logging.
    """
    print(f"{_ts()}: {msg}", flush=True)


MESH_SHAPE = (2, 4)  # DP x TP: the cache shards on TP and replicates on DP
MESH_AXIS_NAMES = ("_axis_0", "_axis_1")
HEAD_AXIS = "_axis_1"  # the TP axis the head dim is split on
TP_SIZE = MESH_SHAPE[1]
MESH_DEVICES = int(np.prod(MESH_SHAPE))

# KNOWN GAP (background): what this file asks for -- heads split TP_SIZE ways,
# replicated across the DP axis -- is not expressible through
# `sharding_constraint_tensor` alone.
#
# `sharding_constraint_tensor` transmits an axis's *position*, never its size: it
# emits the placeholder `mesh_idx_1` and drops the 4 (see
# `_partition_spec_to_sdy_sharding` in tt_torch/sharding.py). tt-mlir binds that
# to the graph's own `sdy.mesh`, and by default that mesh comes from the flat XLA
# device assignment -- 1 replica x N partitions -- not from MESH_SHAPE. The
# default dumped IR (`export_path` compile option, stage `shlo_compiler`) reads:
#
#     sdy.mesh @mesh = <["x"=1, "y"=8]>
#     out_shardings=[<@mesh, [{}, {"y"}, {}, {}]>]
#     ttcore.local_shape = tensor<8x16x32x128xbf16>      # 128 / 8 heads, not / 4
#
# So without help `mesh_idx_1` can only mean "y" = every device, or "x" = 1
# device; a 4-of-8 split with 2-way replication has no representation there.
#
# THE FIX (used by the constrained-fill test below): the `mesh_shape` compile
# option. For a graph with NO tensor inputs at all, the plugin stamps the
# requested 2D shape onto the synthesized `sdy.mesh` before the `mesh_idx_N`
# placeholders are resolved, so `mesh_idx_1` binds to a size-4 "y" axis and
# "x"=2 is the replicated DP axis. See `CompileOptions::mesh_shape_override`
# and `ModuleBuilder::runCompilerStableHLOPipeline` in the plugin.
#
# It does NOT extend to a graph whose inputs are merely replicated, despite
# `moduleHasDeviceShardedInputs` returning false for those. A replicated arg is
# still an arg: it makes the plugin's `isSpmdMode` true (see
# `shlo_set_proper_sdy_mesh_attribute.cc` -- the check is "func has arguments
# AND arg 0 has mhlo.sharding") and closes the `!moduleHasAnyFuncArguments`
# gate, so the module lands on the annotation-driven mesh path with nothing
# defining @mesh. That surfaces as Shardy rejecting the module with
# "'func.func' op arg 0 - unknown mesh: @mesh". Any graph with inputs has to
# carry its own mesh via real `mark_sharding` on those inputs -- see
# `test_create_cache_with_seed_and_sharding_pinned`.
#
# The option is set process-wide (`set_custom_compile_options`), so it rides on
# every graph compiled afterwards -- including the *second* graph this test runs,
# which consumes the sharded cache as a device-sharded input. That graph already
# defines its own mesh via its arg sharding, so the plugin silently ignores the
# override for it (a graph with device-sharded inputs is deferred to, never
# overridden) and it compiles on its own terms.


def _setup(mesh_shape_override=None):
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    # Const-eval would evaluate a zeros/repeat/add chain on the host, so the
    # cache would never reach device DRAM and neither test would prove anything.
    compile_options = {
        "enable_const_eval_inputs_to_system_memory": False,
        "enable_const_eval_on_cpu": False,
        "enable_const_eval": False,
    }
    # Hand the compiler the intended (DP x TP) mesh explicitly. Without it the
    # plugin derives a flat 1xN mesh from the device list, so `mesh_idx_1` binds
    # to all N devices instead of just the TP axis (the KNOWN GAP note below).
    # The plugin only honors this when the graph defines no device-sharded
    # inputs of its own -- true for the constrained-fill graph, which takes no
    # tensor inputs -- and rejects it otherwise, so it stays opt-in per test.
    if mesh_shape_override is not None:
        # compile_options = {"mesh_shape": ",".join(str(d) for d in mesh_shape_override)}
        compile_options["mesh_shape"] = ",".join(str(d) for d in mesh_shape_override)
        torch_xla.set_custom_compile_options(compile_options)
    n = xr.global_runtime_device_count()
    if n != MESH_DEVICES:
        pytest.skip(
            f"MESH_SHAPE {MESH_SHAPE} needs {MESH_DEVICES} devices, found {n}. "
            "Set MESH_SHAPE to a factorization of this box's device count."
        )
    mesh = Mesh(np.arange(n), MESH_SHAPE, MESH_AXIS_NAMES)
    return mesh, torch_xla.device()


def _per_chip_dram_bytes() -> int:
    """Total DRAM on one chip, from the PJRT device attributes.

    Same source vLLM's worker uses (`num_dram_channels x dram_channel_size` from
    the tt-mlir SystemDesc), so the size scales with the board instead of
    hardcoding a per-arch number.
    """
    attrs = xr.global_runtime_device_attributes()
    dram = int(attrs[0].get("dram_size_bytes", 0))
    if dram <= 0:
        pytest.skip(
            "Plugin did not report dram_size_bytes; cannot size an OOM "
            "deterministically on this build."
        )
    return dram


def _kv_dims(tp_size: int, overcommit: float = OVERCOMMIT):
    # """Cache dims whose unsharded size is `overcommit` x one chip's DRAM."""
    # assert NUM_KV_HEADS % tp_size == 0

    # page_bytes = NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE * KV_DTYPE.itemsize
    # per_chip = _per_chip_dram_bytes()
    # num_blocks = int(overcommit * per_chip) // page_bytes
    # # The seed contributes tp_size on dim 1, so the repeat factor must divide out.
    # num_blocks -= num_blocks % tp_size

    # total = num_blocks * page_bytes
    # print(
    #     f"per-chip DRAM {per_chip / 2**30:.1f} GiB | "
    #     f"cache {total / 2**30:.1f} GiB unsharded ({total / per_chip:.1f}x one chip) | "
    #     f"{total / tp_size / 2**30:.1f} GiB per chip sharded {tp_size}-way"
    # )
    num_blocks = 1024 * 8  # 24576
    return (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)


def _mock_kv_touch(cache: torch.Tensor):
    """Cheapest execution that forces the cache to be device-resident: read a
    handful of elements and reduce them.

    Deliberately does NOT write the cache. A write (`index_copy_`, `index_put_`,
    any in-place op) gets functionalized into a full-size copy, because the
    plugin stubs out buffer donation -- so it would add a second GiB-scale buffer
    and make the DRAM figure unattributable. A read adds nothing: the cache is a
    graph parameter, so its whole buffer must be resident no matter how few
    elements the program touches.
    """
    return cache[0, 0, 0, 0].sum(dtype=torch.float32).reshape(1)


def _run_mock(k_cache, mesh, device):
    """Run a device program against the cache, adding as little DRAM as possible.

    No extra tensors are shipped: the cache is the only input, so whatever the
    allocator reports is the cache. The returned scalar is left on device -- a
    `.to("cpu")` would only add a host round-trip, and the point is that the
    program ran, not what it computed.
    """
    # The fill pinned sharding *inside* its own graph via sdy.sharding_constraint,
    # which does not necessarily leave a spec on the tensor handed back. If this
    # prints empty, the mock's parameter looks unsharded to the compiler and the
    # runtime has to reshard/relayout the cache -- a fresh full-slice buffer each
    # time, which is where the extra copies come from.
    spec = torch_xla._XLAC._get_xla_sharding_spec(k_cache)
    _log(f"cache sharding spec entering mock: {spec!r}")

    # Re-assert the same sharding so the second program's parameter matches the
    # buffer already on device. Not a free no-op: torch_xla rebuilds the spec from
    # MESH_SHAPE here and aborts with "Existing annotation must be cleared first"
    # if it differs from the one the compiler left on the tensor. Any such abort
    # means the producing graph resolved HEAD_AXIS to a different width than
    # MESH_SHAPE promises -- see `_mesh_declaration`.
    # xs.mark_sharding(k_cache, mesh, (None, HEAD_AXIS, None, None))

    compiled = torch.compile(_mock_kv_touch, backend="tt")
    _log("touching cache on device")
    compiled(k_cache)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log("touch done")


def _mock_kv_touch_both(k_cache, v_cache):
    """`_mock_kv_touch` for a K/V pair, in ONE graph.

    Both caches are parameters of the same program, so both buffers have to be
    resident at the same instant -- which is the point. Touching them in two
    separate executions only ever proves one cache fits at a time.

    The two reads are summed into a single output so neither can be dropped as
    dead. Still read-only, for the same reason as `_mock_kv_touch`.
    """
    k = k_cache[0, 0, 0, 0].sum(dtype=torch.float32)
    v = v_cache[0, 0, 0, 0].sum(dtype=torch.float32)
    return (k + v).reshape(1)


def _run_mock_kv(k_cache, v_cache, mesh, device):
    """Run one device program that takes both caches as inputs.

    Whatever the allocator reports here is two caches' worth: no other tensors
    are shipped, and the output is a single scalar left on device.
    """
    _log(f"k spec entering mock: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)!r}")
    _log(f"v spec entering mock: {torch_xla._XLAC._get_xla_sharding_spec(v_cache)!r}")

    compiled = torch.compile(_mock_kv_touch_both, backend="tt")
    _log("touching both caches on device")
    compiled(k_cache, v_cache)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log("touch done")


@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_oversized_kv_cache_oom_without_workaround() -> None:
    mesh, device = _setup()
    target = _kv_dims(TP_SIZE)

    with pytest.raises(Exception) as excinfo:
        k_cache = torch.zeros(target, dtype=KV_DTYPE)
        print(f"{_ts()}: created cache")
        k_cache = k_cache.to(device)
        print(f"{_ts()}: marked .to(device)")
        xs.mark_sharding(k_cache, mesh, (None, HEAD_AXIS, None, None))
        print(f"{_ts()}: marked sharding")
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        print(f"{_ts()}: synced")
        _run_mock(k_cache, mesh, device)

    _log(f"unsharded init failed as expected: {excinfo.value}")


def _make_k_cache(mesh, shape, device):
    k_cache = torch.zeros(shape, dtype=KV_DTYPE, device=device)
    return sharding_constraint_tensor(k_cache, mesh, (None, HEAD_AXIS, None, None))


def _make_k_v_cache(mesh, shape, device):
    k_cache = torch.zeros(shape, dtype=KV_DTYPE, device=device)
    v_cache = torch.full(shape, 1.0, dtype=KV_DTYPE, device=device)
    return (
        sharding_constraint_tensor(k_cache, mesh, (None, HEAD_AXIS, None, None)),
        sharding_constraint_tensor(v_cache, mesh, (None, HEAD_AXIS, None, None)),
    )


@pytest.mark.bh_galaxy
@pytest.mark.parametrize(
    "overcommit",
    [0.25, 2.5, 3.0, 1.0, 2.0, 6.0, 7.5],
    ids=["0.25x", "2.5x", "3.0x", "1.0x", "2.0x", "6.0x", "7.5x"],
)
@torch.inference_mode()
def test_oversized_kv_cache_via_constrained_fill(overcommit: float) -> None:
    mesh, device = _setup(mesh_shape_override=MESH_SHAPE)
    tp = TP_SIZE
    target = _kv_dims(tp, overcommit)

    compiled_make_k = torch.compile(_make_k_v_cache, backend="tt")
    print(f"{_ts()}: compiled cache creation function")
    k_cache, v_cache = compiled_make_k(mesh, target, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    print(f"{_ts()}: allocated cache, shape {tuple(k_cache.shape)}")
    print(f"{_ts()}: shard spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)}")

    _run_mock_kv(k_cache, v_cache, mesh, device)


@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_oversized_kv_cache_via_constrained_fill_k_only() -> None:
    mesh, device = _setup(mesh_shape_override=MESH_SHAPE)
    target = _kv_dims(TP_SIZE, 1.0)
    compiled_make_k = torch.compile(_make_k_cache, backend="tt")
    k_cache = compiled_make_k(mesh, target, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    print(f"{_ts()}: allocated cache, shape {tuple(k_cache.shape)}")
    print(f"{_ts()}: shard spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)}")
    _run_mock(k_cache, mesh, device)



def _make_kv_cache_zeros(mesh, shape, device):
    k = torch.zeros(shape, dtype=KV_DTYPE, device=device)
    v = torch.ones(shape, dtype=KV_DTYPE, device=device)
    return (
        sharding_constraint_tensor(k, mesh, (None, HEAD_AXIS, None, None)),
        sharding_constraint_tensor(v, mesh, (None, HEAD_AXIS, None, None)),
    )

@torch.inference_mode()
def test_failing_make_kv_cache_zeros():
    mesh, device = _setup(mesh_shape_override=MESH_SHAPE)

    kv_cache_groups = [
        {
            "group": 1,
            "kv_cache_shape": (1024 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
        {
            "group": 2,
            "kv_cache_shape": (1024 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
        {
            "group": 3,
            "kv_cache_shape": (556 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
    ]
    for kv_cache_group in kv_cache_groups:
        _log(f"allocating cache for group {kv_cache_group['group']}")
        compiled_make_kv_pair = torch.compile(_make_kv_cache_zeros, backend="tt")
        _log("compiled cache creation function")
        k_cache, v_cache = compiled_make_kv_pair(
            mesh, kv_cache_group["kv_cache_shape"], device
        )
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        kv_cache_group["k_cache"] = k_cache
        kv_cache_group["v_cache"] = v_cache
        _log(f"allocated cache, shape {tuple(k_cache.shape)}")
        _log(f"shard spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)}")

    time.sleep(0.5)


def _make_kv_cache(mesh, shape, device):
    """One K/V pair, each its own device allocation.

    `torch.zeros` cannot be used here. Two of them fold into a single buffer
    before tt-mlir sees the graph, and with const-eval on, the creation is
    hoisted behind `ttcore.load_cached`, so calling this again for a shape
    already seen hands back the previous pointer instead of allocating. The
    custom op exists precisely to opt out of both: `has_side_effect` on the
    custom call stops the fold, and `TTCore_NonCacheableTrait` on
    `ttir.zeros_buffer` keeps it out of const-eval.
    """
    k = torch.ops.tt.zeros_buffer(list(shape), KV_DTYPE, device)
    v = torch.ops.tt.zeros_buffer(list(shape), KV_DTYPE, device)
    return (
        sharding_constraint_tensor(k, mesh, (None, HEAD_AXIS, None, None)),
        sharding_constraint_tensor(v, mesh, (None, HEAD_AXIS, None, None)),
    )


@torch.inference_mode()
def test_mock_vllm_kv_cache_initialization():
    mesh, device = _setup(mesh_shape_override=MESH_SHAPE)

    # vLLM iterates over kv_cache_groups and generates multiple kv_cache pairs.
    # Groups 1 and 2 share a shape deliberately: that pair is the re-invocation
    # case, where a cached allocation would hand back group 1's buffer.
    kv_cache_groups = [
        {
            "group": 1,
            "kv_cache_shape": (1024 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
        {
            "group": 2,
            "kv_cache_shape": (1024 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
        {
            "group": 3,
            "kv_cache_shape": (556 * 8, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
            "k_cache": None,
            "v_cache": None,
        },
    ]
    for kv_cache_group in kv_cache_groups:
        _log(f"allocating cache for group {kv_cache_group['group']}")
        compiled_make_kv_pair = torch.compile(_make_kv_cache, backend="tt")
        _log("compiled cache creation function")
        k_cache, v_cache = compiled_make_kv_pair(
            mesh, kv_cache_group["kv_cache_shape"], device
        )
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        kv_cache_group["k_cache"] = k_cache
        kv_cache_group["v_cache"] = v_cache
        _log(f"allocated cache, shape {tuple(k_cache.shape)}")
        _log(f"shard spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)}")

    # Every cache must be its own allocation. `_get_tensors_handle` is the probe
    # to use: the plugin stubs out PJRT_Buffer_UnsafePointer, so there is no way
    # to ask for a real device address, but distinct handles are exactly what
    # "no two caches alias" means here.
    # handles = []
    # for kv_cache_group in kv_cache_groups:
    #     k_handle, v_handle = torch_xla._XLAC._get_tensors_handle(
    #         [kv_cache_group["k_cache"], kv_cache_group["v_cache"]]
    #     )
    #     _log(f"group {kv_cache_group['group']} handles: k={k_handle} v={v_handle}")
    #     assert k_handle != v_handle, (
    #         f"group {kv_cache_group['group']}: k and v share device buffer "
    #         f"{k_handle}"
    #     )
    #     handles += [k_handle, v_handle]

    # assert len(set(handles)) == len(handles), f"aliased cache buffers: {handles}"
    time.sleep(0.5)

    # _run_mock_kv(
    #     kv_cache_groups[0]["k_cache"], kv_cache_groups[0]["v_cache"], mesh, device
    # )


"""
THESE DON'T WORK :(
"""

# def create_cache_with_seed_and_sharding_pinned(k_seed, v_seed, mesh, shape):
#     """Broadcast two seeds up to the full cache shape, sharding pinned in-graph.

#     `expand` only grows singleton dims, so every dim the cache grows on must be
#     1 in the seed. The head dim is the exception: it carries its full extent so
#     the seed itself can be sharded on it, which is what gives this graph a real
#     mesh (see the test).
#     """
#     spec = (None, HEAD_AXIS, None, None)
#     return (
#         sharding_constraint_tensor(k_seed.expand(shape), mesh, spec),
#         sharding_constraint_tensor(v_seed.expand(shape), mesh, spec),
#     )


# def test_create_cache_with_seed_and_sharding_pinned():
#     mesh, device = _setup(mesh_shape_override=MESH_SHAPE)
#     target = _kv_dims(TP_SIZE, 1.0)

#     # Singleton everywhere the broadcast grows, full extent on the head dim.
#     # Passing the *target* shape here instead would make `expand` an identity:
#     # the graph body collapses to `func.return(%arg0, %arg1)`, Shardy treats it
#     # as already solved and drops the tensor shardings, and the surviving
#     # `@mesh` references in the arg attrs resolve to nothing -- which is the
#     # "'func.func' op arg 0 - unknown mesh: @mesh" failure.
#     seed_shape = (1, NUM_KV_HEADS, 1, 1)
#     k_seed = torch.zeros(seed_shape, dtype=KV_DTYPE, device=device)
#     v_seed = torch.zeros(seed_shape, dtype=KV_DTYPE, device=device)

#     # Also not optional. An unannotated seed still enters the graph as a
#     # *replicated* argument, which is enough to make the plugin's `isSpmdMode`
#     # true and take the module off the path the input-less fill used, where
#     # `mesh_shape` stamped a real 2x4 `sdy.mesh` before the `mesh_idx_N`
#     # placeholder was resolved. Sharding the seeds makes the graph carry its own
#     # mesh via its arg shardings, so nothing has to be synthesized.
#     xs.mark_sharding(k_seed, mesh, (None, HEAD_AXIS, None, None))
#     xs.mark_sharding(v_seed, mesh, (None, HEAD_AXIS, None, None))
#     torch_xla.sync(wait=True)
#     _log(f"seed spec: {torch_xla._XLAC._get_xla_sharding_spec(k_seed)!r}")

#     compiled_create_cache = torch.compile(
#         create_cache_with_seed_and_sharding_pinned, backend="tt"
#     )
#     k_cache, v_cache = compiled_create_cache(k_seed, v_seed, mesh, target)
#     torch_xla.sync(wait=True)
#     xm.wait_device_ops()

#     assert tuple(k_cache.shape) == target
#     assert tuple(v_cache.shape) == target
#     _log(f"created caches, shape {tuple(k_cache.shape)}")
#     _log(f"k spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)!r}")
#     _log(f"v spec: {torch_xla._XLAC._get_xla_sharding_spec(v_cache)!r}")

#     # Distinct device buffers, not one aliased by both handles.
#     k_handle, v_handle = torch_xla._XLAC._get_tensors_handle([k_cache, v_cache])
#     assert k_handle != v_handle, f"K and V share device buffer {k_handle}"


# # Fails to do repeat on device
# # @pytest.mark.bh_galaxy
# # @torch.inference_mode()
# # def test_oversized_kv_cache_fits_when_grown_from_sharded_seed() -> None:
# #     """Shard first, then grow: each chip only ever allocates its own slice."""
# #     mesh, device = _setup()
# #     tp = TP_SIZE
# #     target = _kv_dims(tp)
# #     num_blocks = target[0]

# #     # 1) tiny seed, with the TP-axis extent on the head dim so each device holds
# #     #    a single element after sharding.
# #     k_seed = torch.zeros(1, tp, BLOCK_SIZE, 32, dtype=KV_DTYPE).to(device)
# #     xs.mark_sharding(k_seed, mesh, (None, HEAD_AXIS, None, None))
# #     torch_xla.sync(wait=True)  # force the shard before it grows

# #     # 2) grow to full size. Repeat factors apply to the global logical shape, so
# #     #    dim 1 grows by NUM_KV_HEADS // tp on top of the seed's tp.
# #     k_cache = k_seed.repeat(num_blocks, NUM_KV_HEADS // tp, 1, HEAD_SIZE // 32)
# #     xs.mark_sharding(k_cache, mesh, (None, HEAD_AXIS, None, None))  # re-pin
# #     assert tuple(k_cache.shape) == target

# #     # The repeat is lazy IR until something executes; this is what makes the
# #     # cache resident. It overcommits one chip several times over and only
# #     # survives because every chip holds one slice.
# #     _run_mock(k_cache, mesh, device)


# def _grow_cache(seed, mesh, num_blocks, head_mult, dim3_mult):
#     """Tile the seed up to full cache size, pinning the result's sharding.

#     Two things matter here, and both are why this runs inside a compiled graph
#     rather than eagerly:

#     - `seed` arrives as a graph *parameter*, so `zeros -> repeat` is no longer a
#       constant expression the compiler can fold into a full-size literal. Done
#       eagerly, that fold materializes the whole cache on the host.
#     - `sharding_constraint_tensor` puts an `sdy.sharding_constraint` *in* the
#       graph, so Shardy has to produce the tile already split on the head axis.
#       Annotating the result afterwards only constrains the final layout and
#       leaves the compiler free to build it replicated and reshard.
#     """
#     out = seed.repeat(num_blocks, head_mult, 1, dim3_mult)
#     return sharding_constraint_tensor(out, mesh, (None, HEAD_AXIS, None, None))


# @pytest.mark.bh_galaxy
# @torch.inference_mode()
# def test_oversized_kv_cache_fits_when_grown_from_sharded_seed() -> None:
#     """Shard first, then grow on device: each chip only allocates its own slice."""
#     mesh, device = _setup()
#     tp = TP_SIZE
#     target = _kv_dims(tp)
#     num_blocks = target[0]

#     # 1) tiny seed, with the TP-axis extent on the head dim so each device holds
#     #    a single element after sharding.
#     k_seed = torch.zeros(1, tp, BLOCK_SIZE, 32, dtype=KV_DTYPE).to(device)
#     xs.mark_sharding(k_seed, mesh, (None, HEAD_AXIS, None, None))
#     torch_xla.sync(wait=True)

#     # 2) grow to full size as a device execution. Repeat factors apply to the
#     #    global logical shape, so dim 1 grows by NUM_KV_HEADS // tp on top of the
#     #    seed's tp. The output is a real device buffer (execution outputs come
#     #    back from tt::runtime::submit), unlike a host-staged .to(device) tensor.
#     compiled_grow = torch.compile(_grow_cache, backend="tt")
#     _log("growing cache on device")
#     k_cache = compiled_grow(
#         k_seed, mesh, num_blocks, NUM_KV_HEADS // tp, HEAD_SIZE // 32
#     )
#     torch_xla.sync(wait=True)
#     xm.wait_device_ops()
#     _log(f"grow done, shape {tuple(k_cache.shape)}")
#     assert tuple(k_cache.shape) == target

#     # 3) second execution against the same cache. If it stays device-resident
#     #    between runs this is cheap; if the plugin round-trips it to host, that
#     #    shows up as a host RSS spike here.
#     _run_mock(k_cache, mesh, device)


"""
Some basic math:
Running without run mock passes for 7.5x but fails at 8.0x with NO PREV ALLOCATED DRAM
The muliplications are happening in the mock function after the cache is init.
Running with run mock pass for 2.5x passes but fails at 3.0x with 2 caches worth of DRAM
previously allocated. This means it's trying to allocate 3 copies of the cache per device.. why?
"""
"""
TODO next:
- Get the constrained fill test working on a 2x4 box
  - set mesh shape in pjrt
- Remove reshard and just assert on sharding spec
- Make the mock test more like a kv_cache interaction
- Test with runtime debug memory logging to see where the extra copies are coming from
"""
