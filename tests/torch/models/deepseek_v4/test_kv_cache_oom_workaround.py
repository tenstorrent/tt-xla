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
NUM_KV_HEADS = 128  # must be divisible by TP_SIZE
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
# option. When the graph defines no device-sharded inputs of its own -- true for
# a no-input fill, and also for a graph whose inputs are all replicated -- the
# plugin stamps the requested 2D shape onto the synthesized `sdy.mesh` before the
# `mesh_idx_N` placeholders are resolved, so `mesh_idx_1` binds to a size-4 "y"
# axis and "x"=2 is the replicated DP axis. See `CompileOptions::mesh_shape_
# override` and `ModuleBuilder::runCompilerStableHLOPipeline` in the plugin.
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
    }
    # Hand the compiler the intended (DP x TP) mesh explicitly. Without it the
    # plugin derives a flat 1xN mesh from the device list, so `mesh_idx_1` binds
    # to all N devices instead of just the TP axis (the KNOWN GAP note below).
    # The plugin only honors this when the graph defines no device-sharded
    # inputs of its own -- true for the constrained-fill graph, which takes no
    # tensor inputs -- and rejects it otherwise, so it stays opt-in per test.
    if mesh_shape_override is not None:
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
    """Cache dims whose unsharded size is `overcommit` x one chip's DRAM."""
    assert NUM_KV_HEADS % tp_size == 0

    page_bytes = NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE * KV_DTYPE.itemsize
    per_chip = _per_chip_dram_bytes()
    num_blocks = int(overcommit * per_chip) // page_bytes
    # The seed contributes tp_size on dim 1, so the repeat factor must divide out.
    num_blocks -= num_blocks % tp_size

    total = num_blocks * page_bytes
    print(
        f"per-chip DRAM {per_chip / 2**30:.1f} GiB | "
        f"cache {total / 2**30:.1f} GiB unsharded ({total / per_chip:.1f}x one chip) | "
        f"{total / tp_size / 2**30:.1f} GiB per chip sharded {tp_size}-way"
    )
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

    NOTE: the index on dim 1 is what makes this collective. Head 0 lives on one
    device, so producing a replicated scalar from it forces Shardy to all-gather
    the cache. See `_mock_kv_update` for the local-only alternative.
    """
    return cache[0, 0, 0, 0].sum(dtype=torch.float32).reshape(1)


def _mock_kv_update(cache: torch.Tensor, mesh, block_idx: int = 0):
    """Write one page, entirely within each device's own shard. No collectives.

    This is vLLM's cache write in miniature -- a scatter along the *block* dim.
    Only heads are sharded here, so the block dim is replicated: every device
    writes the same block offset into its own head slice, and the scatter lowers
    to a purely local dynamic-update-slice.

    Two rules keep it local, and breaking either reintroduces the gather:

    - Touch ONLY dim 0. Indexing, slicing, or reducing dim 1 asks for heads that
      live on other devices, and Shardy resolves that by all-gathering the whole
      cache -- exactly what `_mock_kv_touch` does.
    - Constrain the page AND the result to the cache's own spec. An
      unconstrained `ones` is replicated, and feeding a replicated operand into
      a sharded tensor makes Shardy reshard one side of the write.

    DRAM cost: this is out-of-place (donation is stubbed, see the module docstring
    and `PJRT_Buffer_DonateWithControlDependency` in the plugin's stubs), so the
    input and output caches are both live -- budget 2x the per-device slice.
    """
    # A single page: (1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE), ~1 MiB. Sharded the
    # same way as the cache so the write has no reshard on either operand.
    page = torch.ones((1, *cache.shape[1:]), dtype=cache.dtype, device=cache.device)
    page = sharding_constraint_tensor(page, mesh, (None, HEAD_AXIS, None, None))

    # slice_scatter, not `cache[block_idx] = page`: the in-place form is
    # functionalized into the same copy anyway, and the functional op is what
    # lowers cleanly to dynamic-update-slice.
    updated = torch.slice_scatter(
        cache, page, dim=0, start=block_idx, end=block_idx + 1
    )
    return sharding_constraint_tensor(updated, mesh, (None, HEAD_AXIS, None, None))


def _run_mock(k_cache, mesh, device, update: bool = False):
    """Run a device program against the cache, adding as little DRAM as possible.

    No extra tensors are shipped: the cache is the only input, so whatever the
    allocator reports is the cache. The returned scalar is left on device -- a
    `.to("cpu")` would only add a host round-trip, and the point is that the
    program ran, not what it computed.

    `update=True` swaps the read-only touch for `_mock_kv_update`, a local page
    write. That is the realistic vLLM access pattern, but it doubles per-device
    DRAM (no donation), so it cannot share an overcommit budget with the
    read-only path -- see the note at the bottom of this file.
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

    if update:
        compiled = torch.compile(_mock_kv_update, backend="tt")
        _log("updating one cache page on device")
        out = compiled(k_cache, mesh)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        # Same global shape means the write was not silently resharded into a
        # different layout; the spec is the check that it stayed distributed.
        assert tuple(out.shape) == tuple(k_cache.shape)
        _log(f"update done, out spec: {torch_xla._XLAC._get_xla_sharding_spec(out)}")
        return out

    compiled = torch.compile(_mock_kv_touch, backend="tt")
    _log("touching cache on device")
    compiled(k_cache)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log("touch done")
    return None


# Fails to do repeat on device
# @pytest.mark.bh_galaxy
# @torch.inference_mode()
# def test_oversized_kv_cache_fits_when_grown_from_sharded_seed() -> None:
#     """Shard first, then grow: each chip only ever allocates its own slice."""
#     mesh, device = _setup()
#     tp = TP_SIZE
#     target = _kv_dims(tp)
#     num_blocks = target[0]

#     # 1) tiny seed, with the TP-axis extent on the head dim so each device holds
#     #    a single element after sharding.
#     k_seed = torch.zeros(1, tp, BLOCK_SIZE, 32, dtype=KV_DTYPE).to(device)
#     xs.mark_sharding(k_seed, mesh, (None, HEAD_AXIS, None, None))
#     torch_xla.sync(wait=True)  # force the shard before it grows

#     # 2) grow to full size. Repeat factors apply to the global logical shape, so
#     #    dim 1 grows by NUM_KV_HEADS // tp on top of the seed's tp.
#     k_cache = k_seed.repeat(num_blocks, NUM_KV_HEADS // tp, 1, HEAD_SIZE // 32)
#     xs.mark_sharding(k_cache, mesh, (None, HEAD_AXIS, None, None))  # re-pin
#     assert tuple(k_cache.shape) == target

#     # The repeat is lazy IR until something executes; this is what makes the
#     # cache resident. It overcommits one chip several times over and only
#     # survives because every chip holds one slice.
#     _run_mock(k_cache, mesh, device)


def _grow_cache(seed, mesh, num_blocks, head_mult, dim3_mult):
    """Tile the seed up to full cache size, pinning the result's sharding.

    Two things matter here, and both are why this runs inside a compiled graph
    rather than eagerly:

    - `seed` arrives as a graph *parameter*, so `zeros -> repeat` is no longer a
      constant expression the compiler can fold into a full-size literal. Done
      eagerly, that fold materializes the whole cache on the host.
    - `sharding_constraint_tensor` puts an `sdy.sharding_constraint` *in* the
      graph, so Shardy has to produce the tile already split on the head axis.
      Annotating the result afterwards only constrains the final layout and
      leaves the compiler free to build it replicated and reshard.
    """
    out = seed.repeat(num_blocks, head_mult, 1, dim3_mult)
    return sharding_constraint_tensor(out, mesh, (None, HEAD_AXIS, None, None))


@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_oversized_kv_cache_fits_when_grown_from_sharded_seed() -> None:
    """Shard first, then grow on device: each chip only allocates its own slice."""
    mesh, device = _setup()
    tp = TP_SIZE
    target = _kv_dims(tp)
    num_blocks = target[0]

    # 1) tiny seed, with the TP-axis extent on the head dim so each device holds
    #    a single element after sharding.
    k_seed = torch.zeros(1, tp, BLOCK_SIZE, 32, dtype=KV_DTYPE).to(device)
    xs.mark_sharding(k_seed, mesh, (None, HEAD_AXIS, None, None))
    torch_xla.sync(wait=True)

    # 2) grow to full size as a device execution. Repeat factors apply to the
    #    global logical shape, so dim 1 grows by NUM_KV_HEADS // tp on top of the
    #    seed's tp. The output is a real device buffer (execution outputs come
    #    back from tt::runtime::submit), unlike a host-staged .to(device) tensor.
    compiled_grow = torch.compile(_grow_cache, backend="tt")
    _log("growing cache on device")
    k_cache = compiled_grow(
        k_seed, mesh, num_blocks, NUM_KV_HEADS // tp, HEAD_SIZE // 32
    )
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log(f"grow done, shape {tuple(k_cache.shape)}")
    assert tuple(k_cache.shape) == target

    # 3) second execution against the same cache. If it stays device-resident
    #    between runs this is cheap; if the plugin round-trips it to host, that
    #    shows up as a host RSS spike here.
    _run_mock(k_cache, mesh, device)


@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_oversized_kv_cache_oom_without_workaround() -> None:
    """Allocate unsharded first: expected to OOM.

    Built directly on device rather than host-then-`.to()`. At these dims a host
    tensor would exhaust system RAM instead, which is not the failure under test.
    With no sharding annotation the tensor is replicated under SPMD, so every
    chip is asked for the full size -- the vLLM failure mode, where the cache has
    to be resident before a shard annotation can apply.
    """
    mesh, device = _setup()
    target = _kv_dims(TP_SIZE)

    with pytest.raises(Exception) as excinfo:
        k_cache = torch.zeros(target, dtype=KV_DTYPE)
        print(f"{_ts()}: created cache")
        k_cache = k_cache.to(device)
        print(f"{_ts()}: marked .to(device)")
        # Annotating after the fact does not save it: the unsharded allocation
        # has to succeed first.
        xs.mark_sharding(k_cache, mesh, (None, HEAD_AXIS, None, None))
        print(f"{_ts()}: marked sharding")
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        print(f"{_ts()}: synced")
        _run_mock(k_cache, mesh, device)

    # Printed rather than pattern-matched: the allocator's wording is not a
    # stable contract, and a different failure here is still informative.
    _log(f"unsharded init failed as expected: {excinfo.value}")


def _make_cache(mesh, shape, device):
    """Allocate a zeroed cache with its sharding pinned in-graph.

    `device` is not optional: without it `torch.zeros` fills on CPU, and
    `tt::sharding_constraint` silently no-ops on a CPU tensor (it just clones),
    so the whole cache would be allocated in host RAM without an error.

    No seed and no `repeat`: just a fill. A fill has no data dependence between
    output elements, so each device can allocate its own slice and zero it
    locally -- there is no cross-shard gather to tempt the compiler into
    building a full-size intermediate first.

    `repeat` is the opposite. Under global semantics output head index `j` comes
    from seed index `j % tp_size`, so every output shard needs every seed shard;
    that interleaving is what both earlier attempts resolved by materializing the
    whole tensor on the host.
    """
    cache = torch.zeros(shape, dtype=KV_DTYPE, device=device)
    return sharding_constraint_tensor(cache, mesh, (None, HEAD_AXIS, None, None))


# Run the small size FIRST. If it materializes host-side it costs ~8-24 GiB of
# RSS and fails in under a minute; the 3.0 case would take the box down instead.
#   pytest -svv <file> -k "constrained_fill and 0.25"
@pytest.mark.bh_galaxy
@pytest.mark.parametrize(
    "overcommit",
    [0.25, 2.5, 3.0, 1.0, 2.0, 6.0, 7.5],
    ids=["0.25x", "2.5x", "3.0x", "1.0x", "2.0x", "6.0x", "7.5x"],
)
@torch.inference_mode()
def test_oversized_kv_cache_via_constrained_fill(overcommit: float) -> None:
    """Allocate the cache directly as a sharded device-side fill.

    The premise being tested: a program can declare a tensor whose *global*
    logical size exceeds one chip's DRAM, as long as the sharding is pinned
    before anything materializes it, so each chip only ever allocates its slice.
    At overcommit=3.0 this cannot succeed unless that holds.

    Watch host RSS while it runs. Flat RSS means the fill stayed distributed;
    RSS climbing by the full logical size means it did not, and no
    global-logical-tensor approach will work in this stack.
    """
    mesh, device = _setup(mesh_shape_override=MESH_SHAPE)
    tp = TP_SIZE
    target = _kv_dims(tp, overcommit)

    compiled_make = torch.compile(_make_cache, backend="tt")
    print(f"{_ts()}: compiled cache creation function")
    k_cache = compiled_make(mesh, target, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    print(f"{_ts()}: allocated cache, shape {tuple(k_cache.shape)}")
    print(f"{_ts()}: shard spec: {torch_xla._XLAC._get_xla_sharding_spec(k_cache)}")

    # Second execution against the same buffer: cheap if the cache stayed
    # device-resident between runs, a host RSS spike if the plugin round-trips it.
    _run_mock(k_cache, mesh, device)


"""
Some basic math:
Running without run mock passes for 7.5x but fails at 8.0x with NO PREV ALLOCATED DRAM
The muliplications are happening in the mock function after the cache is init. 
Running with run mock pass for 2.5x passes but fails at 3.0x with 2 caches worth of DRAM 
previously allocated. This means it's trying to allocate 3 copies of the cache per device.. why?

Note the 3-copies reading is consistent with a FLAT 1x8 mesh, where each device
holds k*D/8: three copies of 2.5x fit in one chip (0.94 D) and three of 3.0x do
not (1.13 D). Under the 2x4 override each device holds k*D/4 instead, so the same
3 copies would already fail at 1.5x -- re-measure before reusing these numbers.

DRAM budget for `_run_mock(update=True)`:
  per-device slice = overcommit * D / TP_SIZE, and the update is out-of-place
  (donation is stubbed), so input + output = 2 * overcommit * D / TP_SIZE.
  With TP_SIZE=4 that needs overcommit < 2.0 just for the two cache buffers,
  before any runtime overhead. So the update mock CANNOT be used to demonstrate
  the 3.0x premise -- keep `_mock_kv_touch` (or no mock at all) for that, and use
  the update mock at <=1.5x to test the access pattern instead. An OOM at 3.0x
  with update=True proves nothing about whether the fill stayed distributed.
"""
"""
TODO next:
- Get the constrained fill test working on a 2x4 box
  - set mesh shape in pjrt
- Remove reshard and just assert on sharding spec
- Make the mock test more like a kv_cache interaction
- Test with runtime debug memory logging to see where the extra copies are coming from
- Confirm _mock_kv_update emits no all-gather: dump IR with the `export_path`
  compile option and grep the stage for `all-gather` / `sdy.all_gather`.
"""
