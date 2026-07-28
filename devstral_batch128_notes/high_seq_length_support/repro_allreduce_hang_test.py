# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# UNRUN PROPOSAL / CANDIDATE REPRO -- DO NOT ASSUME IT PASSES OR HANGS.
# ============================================================================
# This file has NOT been executed on hardware. It is a minimal, standalone
# probe designed to reproduce (or *rule out in isolation*) the ttnn.all_reduce
# hang observed in the full Devstral-123B DP+TP chunked-prefill run on a
# Blackhole galaxy. Run it only on a CLEAN, idle 32-chip galaxy: a genuine hang
# can wedge the mesh and require a host/card reset. TT_METAL_OPERATION_TIMEOUT
# is set below so a hang fails fast (~60 s) instead of the 120 s Devstral saw.
#
# WHERE THIS FILE MUST LIVE TO RUN
# --------------------------------
# It depends on the tt-metal pytest conftest fixtures `mesh_device`,
# `device_params`, `silicon_arch_name`, and `function_level_defaults`. Those are
# only discoverable from inside the tt-metal test tree. It will NOT run from the
# devstral_batch128_notes/ directory it currently sits in. Before running:
#
#   TTM=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
#   cp devstral_batch128_notes/high_seq_length_support/repro_allreduce_hang_test.py \
#      $TTM/tests/ttnn/unit_tests/operations/ccl/
#
# then run from the tt-metal root (see repro_allreduce_hang_notes.md for the
# exact command).
#
# WHAT IT REPRODUCES (established facts from allreduce_collision_analysis.md)
# --------------------------------------------------------------------------
#   * Op:        ttnn.all_reduce (the top-level, semaphore-free composite op --
#                the SAME entry point the tt-mlir runtime AllReduceOp handler
#                calls: runtime/lib/ttnn/operations/ccl/all_reduce.cpp:36).
#                For this shape it decomposes at metal runtime to
#                ReduceScatterDeviceOperation + AllGatherDeviceOperation.
#   * Tensor:    [1, 1, 4096, 12288] bf16, TILE, DRAM (Devstral o_proj TP reduce).
#   * Axis:      cluster_axis=1 (the 8-wide TP axis). DP (axis 0) is orthogonal
#                and NOT needed to reproduce -- so we carve an 8-wide submesh.
#   * Pattern:   graph-1's all_reduce (incl. inside a trace capture) SUCCEEDS;
#                the byte-identical graph-2 all_reduce (program-cache HIT, run
#                eagerly after the trace was captured/executed) HANGS.
#
# LEADING (UNPROVEN) HYPOTHESIS
# -----------------------------
# The composite all_reduce allocates+frees its reduce_scatter INTERMEDIATE
# buffer *within a single call*. Graph-1 captures that allocation inside a trace
# (pinned/replayed); graph-2's eager cache-hit allocates the intermediate fresh.
# The fabric ring computes each peer's write target from the (correctly
# refreshed) local address ASSUMING all 8 TP peers allocated the intermediate at
# the SAME address. If the fresh post-trace allocation lands at divergent
# addresses across the 8 chips, peer writes miss -> the receiving semaphore never
# signals -> hang. The program cache's spec-only hash lets graph-2 reuse graph-1's
# program (so no recompile forces a fresh symmetric layout) -- it enables/masks
# the bug but does not itself cause it.
#
# KNOWN LIMITATION OF THIS ISOLATED REPRO (READ THIS)
# ---------------------------------------------------
# A clean mesh runs SPMD: the host issues an IDENTICAL allocation sequence to
# every chip, so uniform "churn" between calls stays SYMMETRIC across chips. The
# standard ttnn mesh API exposes NO way to inject per-chip-divergent allocation.
# Therefore Variant A may well PASS even though the mechanism is real -- that is
# a RESULT, not a bug in the test. The one asymmetry source that DOES exist on
# real silicon is per-chip core harvesting (different harvested rows ->
# different compute_with_storage_grid_size() -> divergent buffer layouts). This
# repro logs that per chip so you can see whether the hardware even offers the
# asymmetry the hypothesis needs. See repro_allreduce_hang_notes.md for how to
# read each variant's outcome.
# ============================================================================

import os

# Must be set BEFORE ttnn/tt-metal is imported so the runtime picks it up.
# Fail fast on a hang instead of wedging the mesh for 120 s.
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "60")

import math
import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ----------------------------------------------------------------------------
# The exact Devstral o_proj TP-reduction tensor.
# ----------------------------------------------------------------------------
DEVSTRAL_OPROJ_SHAPE = [1, 1, 4096, 12288]  # bf16, TILE, DRAM
TP_AXIS = 1  # cluster_axis of the hanging op (the 8-wide TP axis)
TP_DEVICES = 8  # devices along cluster_axis=1


# ----------------------------------------------------------------------------
# Observability helpers -- the whole point is to DETECT address divergence, not
# just to hang. Call these to dump per-chip state across the 8 TP peers.
# ----------------------------------------------------------------------------
def _log_mesh_topology(mesh_device):
    logger.info(f"[repro] submesh shape={mesh_device.shape} "
                f"num_devices={mesh_device.get_num_devices()} "
                f"device_ids={mesh_device.get_device_ids()}")
    # Mesh-level compute grid (always available). ttnn usually reports a common
    # min grid across the mesh here -- if so, it would SUPPRESS the per-chip
    # harvesting asymmetry the hypothesis needs; note that when reading results.
    g = mesh_device.compute_with_storage_grid_size()
    logger.info(f"[repro]   mesh compute_grid=({g.x},{g.y})")
    # Per-chip compute grid -- if these differ across chips, core harvesting is
    # asymmetric and CAN produce the divergent allocation the hypothesis needs.
    # `get_devices()` may not exist on every build; best-effort only.
    try:
        for dev in mesh_device.get_devices():
            gg = dev.compute_with_storage_grid_size()
            logger.info(f"[repro]   chip id={dev.id()} compute_grid=({gg.x},{gg.y})")
    except Exception as e:  # API shape varies across builds; don't fail the probe on it.
        logger.warning(f"[repro] per-chip compute grid enumeration unavailable "
                       f"({e}); relying on mesh-level grid above.")


def _log_per_device_addresses(tensor_mesh, tag):
    """Dump the per-chip buffer address of a mesh tensor across all 8 TP peers.

    If the 8 addresses are NOT all equal, allocation has diverged across the
    ring -- the direct signature of the leading hypothesis.
    """
    try:
        shards = ttnn.get_device_tensors(tensor_mesh)
        addrs = []
        for i, t in enumerate(shards):
            addr = None
            for attr in ("buffer_address", "buffer"):
                try:
                    obj = getattr(t, attr)
                    addr = obj() if callable(obj) else obj
                    if hasattr(addr, "address"):
                        addr = addr.address()
                    break
                except Exception:
                    continue
            addrs.append(addr)
        uniform = len(set(a for a in addrs if a is not None)) <= 1
        logger.info(f"[repro] {tag} per-chip addresses={addrs} uniform={uniform}")
        if not uniform:
            logger.error(f"[repro] {tag} ADDRESS DIVERGENCE across ring peers -- "
                         f"this is the hypothesized hang precondition.")
    except Exception as e:
        logger.warning(f"[repro] {tag} could not read per-chip addresses: {e}")


# ----------------------------------------------------------------------------
# Core: build the sharded input and run one all_reduce.
# ----------------------------------------------------------------------------
def _make_input(mesh_device, per_chip_shape, num_devices, mem_config):
    # Shard the tensor so each of the `num_devices` chips along cluster_axis=1
    # holds `per_chip_shape`. Mirrors test_all_reduce_async.py:181-198.
    shard_dims = (1, 0)          # cluster_axis == 1
    mesh_shape = (1, num_devices)  # 1 all_reduce instance x num_devices peers

    per_dev = []
    for _ in range(num_devices):
        per_dev.append(torch.rand(per_chip_shape).bfloat16())
    unfractured = torch.cat([t.view(1, -1, t.shape[2], t.shape[3]) for t in per_dev])
    unfractured = unfractured.reshape([1, num_devices, per_chip_shape[2], per_chip_shape[3]])

    tt = ttnn.from_torch(
        unfractured,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    golden = torch.sum(torch.cat([t.view(1, -1, t.shape[2], t.shape[3]) for t in per_dev]), 0, keepdim=True)
    golden = golden.view(per_chip_shape)
    return ttnn.to_device(tt, mesh_device), golden


def _all_reduce(input_mesh, mem_config, num_links, subdevice_id):
    # Top-level ttnn.all_reduce == the composite the tt-mlir runtime calls.
    # num_links=None matches the compiler (it passes nullopt; the op auto-selects
    # links exactly as in Devstral). Topology is forced to 1D (Linear) by the
    # runtime regardless of mesh dimensionality.
    kwargs = dict(
        cluster_axis=TP_AXIS,
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
    )
    if subdevice_id is not None:
        kwargs["subdevice_id"] = subdevice_id
    if num_links is not None:
        kwargs["num_links"] = num_links
    return ttnn.all_reduce(input_mesh, **kwargs)


def _setup_subdevice(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    worker = ttnn.SubDevice([crs])
    worker_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([worker], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([worker_id])
    return worker_id


def _churn(mesh_device, mem_config):
    """Allocate + free a large scratch tensor to perturb the allocator between
    calls. NOTE: on SPMD this perturbation is symmetric across chips; it can only
    surface divergence if some *other* per-chip asymmetry (e.g. harvesting)
    already exists. Kept so the probe at least exercises allocator churn."""
    scratch = ttnn.from_torch(
        torch.rand([1, 1, 4096, 12288]).bfloat16(),
        dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    scratch.deallocate()
    ttnn.synchronize_device(mesh_device)


# ----------------------------------------------------------------------------
# The variant driver. Each variant is one point in the discriminating matrix;
# see repro_allreduce_hang_notes.md for how to interpret which hangs vs passes.
# ----------------------------------------------------------------------------
def _run_variant(
    full_mesh_device,
    variant,
    num_links,
    submesh_rows=1,
    trace_region_size=90_000_000,
):
    """
    variant:
      "A_cache_trace_churn" -- HANG CANDIDATE. cache ON: compile -> trace-capture
            all_reduce -> execute trace -> churn -> byte-identical eager cache-hit
            all_reduce. Mirrors Devstral graph-1(traced) -> graph-2(eager hit).
      "B_cache_off"          -- CONTROL. cache DISABLED: every all_reduce recompiles
            (fresh, symmetric allocation). Hypothesis => passes.
      "C_cache_notrace_churn"-- DISCRIMINATOR. cache ON, churn, but NO trace capture.
            If it ALSO hangs => trace is not required (pure cache-hit + churn).
            If it passes => the trace capture is the essential perturbation.
      "D_cache_notrace_nochurn"-- NEGATIVE CONTROL. cache ON, two back-to-back eager
            all_reduces, no trace, no churn (guaranteed cache hit). Matches the
            all_gather counterexample that succeeded; expected to PASS.
    """
    # Carve a CONTIGUOUS submesh off the full [4,8] galaxy. IMPORTANT: do NOT
    # hand-pick non-contiguous device ids (e.g. 0,4,8,...) -- prior work found
    # arbitrary carve-outs FAIL at cluster init. create_submesh takes a
    # contiguous rectangular sub-region, which is a valid connected topology.
    # (submesh_rows, TP_DEVICES): (1,8) = pure TP line; (2,8) adds DP breadth.
    submesh = full_mesh_device.create_submesh(ttnn.MeshShape((submesh_rows, TP_DEVICES)))
    logger.info(f"[repro] === variant={variant} num_links={num_links} "
                f"submesh=({submesh_rows},{TP_DEVICES}) ===")
    _log_mesh_topology(submesh)

    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
    cache_on = variant != "B_cache_off"
    if cache_on:
        submesh.enable_program_cache()
    else:
        submesh.disable_and_clear_program_cache()

    worker_id = _setup_subdevice(submesh)
    outs = []
    try:
        input_mesh, golden = _make_input(submesh, DEVSTRAL_OPROJ_SHAPE, TP_DEVICES, mem_config)

        # ---- Phase 1: first all_reduce (cache miss -> compile). Always eager. ----
        logger.info("[repro] phase-1 all_reduce (compile / cache-miss)")
        out1 = _all_reduce(input_mesh, mem_config, num_links, worker_id)
        ttnn.synchronize_device(submesh)
        _log_per_device_addresses(out1, "phase-1 output")
        outs.append(out1)

        if variant == "A_cache_trace_churn":
            # ---- Phase 2: capture a trace containing the all_reduce, execute it.
            logger.info("[repro] phase-2 capture trace of all_reduce")
            tid = ttnn.begin_trace_capture(submesh, cq_id=0)
            out_traced = _all_reduce(input_mesh, mem_config, num_links, worker_id)
            ttnn.end_trace_capture(submesh, tid, cq_id=0)
            ttnn.synchronize_device(submesh)
            logger.info("[repro] phase-2 execute trace")
            ttnn.execute_trace(submesh, tid, blocking=True)
            ttnn.synchronize_device(submesh)
            _log_per_device_addresses(out_traced, "phase-2 traced output")
            ttnn.release_trace(submesh, tid)
            # ---- allocation churn between the traced call and the eager hit. ----
            logger.info("[repro] churn (perturb allocator post-trace)")
            _churn(submesh, mem_config)

        elif variant == "C_cache_notrace_churn":
            logger.info("[repro] churn (no trace)")
            _churn(submesh, mem_config)

        # ---- Final: byte-identical eager all_reduce. HANG CANDIDATE for A / C. ----
        logger.info(f"[repro] final all_reduce (cache_on={cache_on}) -- HANG CANDIDATE")
        out_final = _all_reduce(input_mesh, mem_config, num_links, worker_id)
        ttnn.synchronize_device(submesh)
        _log_per_device_addresses(out_final, "final output")
        outs.append(out_final)
        logger.info(f"[repro] variant={variant} COMPLETED WITHOUT HANG. "
                    f"program_cache_entries={submesh.num_program_cache_entries()}")

        # Correctness sanity (only meaningful if it did not hang).
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(out_final)[0])
        eq, pcc = comp_pcc(tt_out, golden)
        logger.info(f"[repro] variant={variant} pcc={pcc} eq={eq}")
    finally:
        submesh.reset_sub_device_stall_group()
        for o in outs:
            try:
                o.deallocate()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Pytest entry points. device_params carries the trace region + fabric config;
# mesh_device is opened as the full [4,8] galaxy by the tt-metal conftest.
#
# Fabric: Devstral runs on a 2D-mesh galaxy. The runtime forces the OP topology
# to 1D, but the device-level FABRIC config is set by vLLM and may be FABRIC_1D
# or FABRIC_2D. We parametrize both because a fabric-routing hang would depend on
# it. (If you can confirm which fabric_config the vLLM run used, pin it.)
# ----------------------------------------------------------------------------
_TRACE_REGION = 90_000_000  # generous; the intermediate alone is ~100+ MB/tile-region


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param({"trace_region_size": _TRACE_REGION,
                      "fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="fabric_1d"),
        pytest.param({"trace_region_size": _TRACE_REGION,
                      "fabric_config": ttnn.FabricConfig.FABRIC_2D}, id="fabric_2d"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8_galaxy")], indirect=True)
@pytest.mark.parametrize("num_links", [None, 1], ids=["links_auto", "links_1"])
@pytest.mark.parametrize("submesh_rows", [1, 2], ids=["tp_only_1x8", "dp2_tp8_2x8"])
@pytest.mark.parametrize(
    "variant",
    [
        "A_cache_trace_churn",
        "B_cache_off",
        "C_cache_notrace_churn",
        "D_cache_notrace_nochurn",
    ],
)
def test_devstral_allreduce_hang_probe(
    mesh_device,
    device_params,
    function_level_defaults,
    variant,
    num_links,
    submesh_rows,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Requires a 32-chip galaxy ([4,8]).")
    _run_variant(
        mesh_device,
        variant=variant,
        num_links=num_links,
        submesh_rows=submesh_rows,
        trace_region_size=_TRACE_REGION,
    )
